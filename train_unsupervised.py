"""Train on the fixed spatial training region; validation and test are excluded from losses.

Usage: python train_unsupervised.py --config config.json
See README.md for input formats, outputs and execution order.
"""

from __future__ import annotations
from pipeline_config import parse_arguments, configure
from pipeline_checks import preflight, record_outputs, record_training_inputs

ARGS = parse_arguments()
import random
from pathlib import Path
import datetime
import json
import hashlib
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

BASE = 32
PATCH = 256
BATCH = 12
EPOCHS = 50
SAMPLES_PER_EPOCH = 8000
LR = 0.0003
SEED = 42
IN_CH = 2
NUM_WORKERS = 6
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
CANDIDATE_STRIDE = 16
EPS = 1e-06
globals().update(
    configure(
        ARGS, "unsupervised", "train", {k: v for k, v in globals().copy().items() if k.isupper()}
    )
)


def seed_all(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_img_mat(path: Path) -> np.ndarray:
    d = loadmat(path)
    if IMAGE_KEY not in d:
        keys = [k for k in d.keys() if not k.startswith("__")]
        raise KeyError(f"{path.name} missing '{IMAGE_KEY}'. Keys: {keys}")
    x = d[IMAGE_KEY]
    x = np.transpose(x, (2, 0, 1)).astype(np.float32, copy=False)
    return x


def load_split_masks():
    if not SPLIT_MAT.exists():
        raise FileNotFoundError(f"Split file not found:\n{SPLIT_MAT}")
    d = loadmat(SPLIT_MAT)
    required = ["train_pool", "validation", "test", "buffer_mask", "valid_mask"]
    for key in required:
        if key not in d:
            raise KeyError(f"Split file missing '{key}'.")
    masks = {}
    for key in required:
        x = np.squeeze(d[key])
        if x.ndim != 2:
            raise ValueError(f"{key} must be 2-D. Got {x.shape}")
        masks[key] = x.astype(bool)
    shape = masks["train_pool"].shape
    for key, mask in masks.items():
        if mask.shape != shape:
            raise ValueError(f"{key} shape {mask.shape} != {shape}")
    train_pool = masks["train_pool"]
    validation = masks["validation"]
    test = masks["test"]
    buffer_mask = masks["buffer_mask"]
    valid_mask = masks["valid_mask"]
    if np.any(train_pool & validation):
        raise RuntimeError("train_pool overlaps validation.")
    if np.any(train_pool & test):
        raise RuntimeError("train_pool overlaps test.")
    if np.any(validation & test):
        raise RuntimeError("validation overlaps test.")
    if np.any(train_pool & buffer_mask):
        raise RuntimeError("train_pool overlaps buffer.")
    reconstructed = train_pool | validation | test | buffer_mask
    if not np.array_equal(reconstructed, valid_mask):
        raise RuntimeError("train/validation/test/buffer do not reconstruct valid_mask.")
    usable = train_pool.sum() + validation.sum() + test.sum()
    print("\n" + "=" * 72)
    print("FROZEN SAR SPLIT")
    print("=" * 72)
    print(f"Shape       : {shape}")
    print(f"Train pool  : {train_pool.sum():,} ({100 * train_pool.sum() / usable:.2f}% usable)")
    print(f"Validation  : {validation.sum():,} ({100 * validation.sum() / usable:.2f}% usable)")
    print(f"Test        : {test.sum():,} ({100 * test.sum() / usable:.2f}% usable)")
    print(f"Buffer      : {buffer_mask.sum():,}")
    return masks


def robust_norm_train_pool(x: np.ndarray, train_pool: np.ndarray, eps: float = EPS):
    out = np.empty_like(x, dtype=np.float32)
    stats = []
    for c in range(x.shape[0]):
        train_values = x[c][train_pool]
        lo = float(np.percentile(train_values, 1))
        hi = float(np.percentile(train_values, 99))
        clipped_train = np.clip(train_values, lo, hi)
        mean = float(clipped_train.mean())
        std = float(clipped_train.std())
        z = np.clip(x[c], lo, hi)
        out[c] = (z - mean) / (std + eps)
        stats.append({"lo": lo, "hi": hi, "mean": mean, "std": std})
    return (out, stats)


def aug_flip_rot(x: np.ndarray) -> np.ndarray:
    k = random.randint(0, 3)
    if k:
        x = np.rot90(x, k, axes=(1, 2)).copy()
    if random.random() < 0.5:
        x = x[:, :, ::-1].copy()
    if random.random() < 0.5:
        x = x[:, ::-1, :].copy()
    return x


def build_train_patch_locations(train_pool: np.ndarray, patch: int, stride: int):
    H, W = train_pool.shape
    integral = np.zeros((H + 1, W + 1), dtype=np.int32)
    integral[1:, 1:] = np.cumsum(np.cumsum(train_pool.astype(np.int32), axis=0), axis=1)
    locations = []
    required = patch * patch
    tops = list(range(0, H - patch + 1, stride))
    lefts = list(range(0, W - patch + 1, stride))
    if tops[-1] != H - patch:
        tops.append(H - patch)
    if lefts[-1] != W - patch:
        lefts.append(W - patch)
    print("\nSearching 256x256 patches fully inside train_pool...")
    for top in tqdm(tops, desc="Patch rows"):
        bottom = top + patch
        for left in lefts:
            right = left + patch
            count = (
                integral[bottom, right]
                - integral[top, right]
                - integral[bottom, left]
                + integral[top, left]
            )
            if count == required:
                locations.append((top, left))
    if not locations:
        raise RuntimeError("No patch locations found fully inside train_pool.")
    locations = np.asarray(locations, dtype=np.int32)
    print(f"Accepted patch locations: {len(locations):,}")
    return locations


class BeforePatchDataset(Dataset):

    def __init__(
        self,
        before_dir: Path,
        train_pool: np.ndarray,
        patch_locations: np.ndarray,
        patch: int,
        samples: int,
        train: bool = True,
    ):
        super().__init__()
        self.files = sorted(before_dir.glob("*.mat"))
        if not self.files:
            raise RuntimeError(f"No .mat files found in {before_dir}")
        self.patch = patch
        self.samples = samples
        self.train = train
        self.patch_locations = patch_locations
        self.images = []
        self.normalization_stats = {}
        print(f"\nCaching {len(self.files)} BEFORE images...")
        for p in tqdm(self.files, desc="Caching BEFORE images"):
            x = load_img_mat(p)
            if x.shape[1:] != train_pool.shape:
                raise ValueError(
                    f"{p.name} image shape {x.shape[1:]} != split shape {train_pool.shape}"
                )
            x, stats = robust_norm_train_pool(x, train_pool)
            self.images.append(x)
            self.normalization_stats[p.name] = stats
        H0 = self.images[0].shape[1]
        W0 = self.images[0].shape[2]
        for i, x in enumerate(self.images):
            if x.shape[1] != H0 or x.shape[2] != W0:
                raise ValueError(
                    f"Image shapes are not all equal. Image 0 = {(H0, W0)}, image {i} = {(x.shape[1], x.shape[2])}"
                )
        self.H = H0
        self.W = W0
        if self.H < self.patch or self.W < self.patch:
            raise ValueError(f"PATCH={self.patch} is larger than image size {(self.H, self.W)}")
        print(f"Dataset cached. Image size = ({self.H}, {self.W}), samples/epoch = {self.samples}")

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        del idx
        x = random.choice(self.images)
        location_idx = random.randrange(len(self.patch_locations))
        top = int(self.patch_locations[location_idx, 0])
        left = int(self.patch_locations[location_idx, 1])
        patch = x[:, top : top + self.patch, left : left + self.patch]
        if self.train:
            patch = aug_flip_rot(patch)
        return torch.from_numpy(patch.astype(np.float32))


class ConvAE(nn.Module):

    def __init__(self, in_ch=2, base=BASE):
        super().__init__()
        self.e1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.p2 = nn.MaxPool2d(2)
        self.e3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base, in_ch, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.p1(x1))
        x3 = self.e3(self.p2(x2))
        y = self.d2(self.u2(x3))
        y = self.d1(self.u1(y))
        return self.out(y)


def save_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    run_dir: Path,
    split_hash: str,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "run_dir": str(run_dir),
            "split_mat": str(SPLIT_MAT),
            "split_sha256": split_hash,
            "config": {
                "BASE": BASE,
                "PATCH": PATCH,
                "BATCH": BATCH,
                "EPOCHS": EPOCHS,
                "SAMPLES_PER_EPOCH": SAMPLES_PER_EPOCH,
                "LR": LR,
                "SEED": SEED,
                "IN_CH": IN_CH,
                "NUM_WORKERS": NUM_WORKERS,
            },
        },
        ckpt_path,
    )


def main():
    preflight(globals(), "unsupervised", "train")
    if RUN_DIR.exists() and not RESUME:
        raise FileExistsError(
            f"Training run already exists: {RUN_DIR}; choose a new run_name or resume."
        )
    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    masks = load_split_masks()
    train_pool = masks["train_pool"]
    split_hash = sha256_file(SPLIT_MAT)
    print("\nSplit SHA-256:")
    print(split_hash)
    patch_locations = build_train_patch_locations(train_pool, patch=PATCH, stride=CANDIDATE_STRIDE)
    start_epoch = 1
    if RESUME:
        if RESUME_CKPT_PATH is None:
            raise ValueError("RESUME=True but RESUME_CKPT_PATH is None.")
        if not Path(RESUME_CKPT_PATH).exists():
            raise FileNotFoundError(f"Checkpoint not found: {RESUME_CKPT_PATH}")
        ckpt = torch.load(RESUME_CKPT_PATH, map_location="cpu", weights_only=False)
        if ckpt.get("split_sha256") != split_hash:
            raise RuntimeError("Resume checkpoint was trained with a different split.")
        run_dir = Path(ckpt["run_dir"])
        if not run_dir.exists():
            raise FileNotFoundError(f"Saved run_dir does not exist anymore: {run_dir}")
        print("Resuming from:", RESUME_CKPT_PATH)
        print("Run dir:", run_dir)
    else:
        timestamp = RUN_NAME
        run_dir = RUN_DIR
        run_dir.mkdir(parents=True, exist_ok=False)
        record_training_inputs(globals(), "unsupervised")
    ds = BeforePatchDataset(
        BEFORE_DIR, train_pool, patch_locations, PATCH, SAMPLES_PER_EPOCH, train=True
    )
    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS and NUM_WORKERS > 0,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
    )
    if not RESUME:
        with open(run_dir / "config.json", "w") as f:
            json.dump(
                {
                    "BASE": BASE,
                    "PATCH": PATCH,
                    "BATCH": BATCH,
                    "EPOCHS": EPOCHS,
                    "SAMPLES_PER_EPOCH": SAMPLES_PER_EPOCH,
                    "LR": LR,
                    "SEED": SEED,
                    "IN_CH": IN_CH,
                    "NUM_WORKERS": NUM_WORKERS,
                    "split_mat": str(SPLIT_MAT),
                    "split_sha256": split_hash,
                    "training_region": "train_pool only",
                    "normalization": "robust normalization; statistics fitted on train_pool only",
                    "candidate_stride": CANDIDATE_STRIDE,
                    "accepted_patch_locations": len(patch_locations),
                },
                f,
                indent=2,
            )
        with open(run_dir / "normalization_stats.json", "w") as f:
            json.dump(ds.normalization_stats, f, indent=2)
    model = ConvAE(in_ch=IN_CH, base=BASE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    scaler = GradScaler()
    if RESUME:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"]) + 1
        print(f"Warm start: resumed from epoch {ckpt['epoch']}")
    else:
        print("Warm start: training from scratch")
    log_file = run_dir / "training_log.txt"
    if not RESUME:
        with open(log_file, "w") as f:
            f.write("epoch,l1_loss\n")
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        losses = []
        for x in tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast():
                xhat = model(x)
                loss = F.l1_loss(xhat, x)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss))
        mean_loss = float(np.mean(losses))
        print(f"Epoch {epoch:03d} | L1 loss: {mean_loss:.6f}")
        with open(log_file, "a") as f:
            f.write(f"{epoch},{mean_loss:.6f}\n")
        ckpt_path = run_dir / f"epoch_{epoch:03d}.pt"
        save_checkpoint(
            ckpt_path=ckpt_path,
            model=model,
            optimizer=opt,
            scaler=scaler,
            epoch=epoch,
            run_dir=run_dir,
            split_hash=split_hash,
        )
        save_checkpoint(
            ckpt_path=run_dir / "ae_last.pt",
            model=model,
            optimizer=opt,
            scaler=scaler,
            epoch=epoch,
            run_dir=run_dir,
            split_hash=split_hash,
        )
    print("\nTraining finished.")
    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()
    record_outputs(globals(), "unsupervised", "train")
