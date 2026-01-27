"""Training script for SegVAE2D model using PyTorch Lightning."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from imageio.v2 import imread
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset

from ..models.d3_swc_vae import MultiTaskLoss, build_model


class NeuronDataset(Dataset):
    """Dataset for neuron binary mask (input) + segmentation mask pairs.

    Uses mask_bw (black/white binary mask) as input, and mask (class IDs) as segmentation target.
    Depth maps are automatically loaded if present in the manifest.
    """

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        *,
        transform=None,
        image_size: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.manifest_entries = []
        with open(manifest_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self.manifest_entries.append(entry)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num} in {manifest_path}: {e}")

        if len(self.manifest_entries) == 0:
            raise ValueError(f"Manifest file is empty: {manifest_path}")

        # Auto-detect depth availability from manifest
        self.load_depth = "depth" in self.manifest_entries[0]

        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.manifest_entries)

    def __getitem__(self, idx):
        entry = self.manifest_entries[idx]

        if "mask_bw" in entry:
            input_path = self._resolve_path(entry["mask_bw"])
        elif "image" in entry:
            input_path = self._resolve_path(entry["image"])
        else:
            raise ValueError(f"Manifest entry {idx} missing both 'mask_bw' and 'image' fields: {entry}")

        mask_path = self._resolve_path(entry["mask"])

        input_img = imread(input_path)
        if len(input_img.shape) == 3:
            input_img = np.mean(input_img, axis=2, dtype=np.float32) / 255.0
        else:
            input_img = input_img.astype(np.float32) / 255.0

        mask = imread(mask_path)

        input_tensor = torch.from_numpy(input_img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long()

        # Resize if requested
        if self.image_size is not None:
            input_tensor = torch.nn.functional.interpolate(
                input_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=(self.image_size, self.image_size),
                mode="nearest",
            ).squeeze(0).squeeze(0).long()

        sample = {
            "input": input_tensor,
            "seg": mask_tensor,
        }

        if self.load_depth:
            depth = self._load_depth(entry)
            if depth is not None:
                if self.image_size is not None:
                    depth = torch.nn.functional.interpolate(
                        depth.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                sample["depth"] = depth
                sample["valid_depth_mask"] = torch.ones_like(depth)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _resolve_path(self, rel_path: str) -> Path:
        """Resolve a relative path from manifest, handling both with and without 'images/' prefix."""
        path = self.data_root / rel_path
        if path.exists():
            return path

        if not rel_path.startswith("images/"):
            path = self.data_root / "images" / rel_path
            if path.exists():
                return path

        if rel_path.startswith("images/"):
            path = self.data_root / rel_path[7:]
            if path.exists():
                return path

        # Return original path for better error message
        return self.data_root / rel_path

    def _load_depth(self, entry: Dict):
        depth_path_str = entry.get("depth", "")
        if not depth_path_str:
            return None
        depth_path = self._resolve_path(depth_path_str)
        if depth_path.exists():
            depth = imread(depth_path).astype(np.float32) / 255.0
            return torch.from_numpy(depth).unsqueeze(0)
        return None


def collate_fn(batch):
    """Collate function for batching."""
    keys = batch[0].keys()
    batched = {}
    for key in keys:
        if key == "input":
            batched["input"] = torch.stack([item["input"] for item in batch])
        elif key == "seg":
            batched["seg"] = torch.stack([item["seg"] for item in batch])
        elif key == "depth":
            batched["depth"] = torch.stack([item["depth"] for item in batch])
        elif key == "valid_depth_mask":
            batched["valid_depth_mask"] = torch.stack([item["valid_depth_mask"] for item in batch])

    return batched


class SegVAE2DLightning(LightningModule):
    """PyTorch Lightning module for SegVAE2D training."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 6,
        latent_channels: int = 128,
        skip_latent_mult: float = 0.5,
        kld_weight: float = 0.1,
        skip_mode: str = "variational",
        free_nats: float = 0.0,
        beta: float = 1.0,
        lambda_seg: float = 1.0,
        lambda_depth: float = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        lr_scheduler: str = "cosine",
        lr_warmup_steps: int = 0,
        lr_t_max: Optional[int] = None,
        lr_eta_min: float = 1e-6,
        max_steps: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(
            in_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            latent_channels=latent_channels,
            skip_latent_mult=skip_latent_mult,
            kld_weight=kld_weight,
            skip_mode=skip_mode,
            free_nats=free_nats,
            beta=beta,
        )

        self.criterion = MultiTaskLoss(
            lambda_seg=lambda_seg,
            lambda_depth=lambda_depth,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_t_max = lr_t_max
        self.lr_eta_min = lr_eta_min
        self.max_steps = max_steps

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input"])
        loss, logs = self.criterion(outputs, batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for key, value in logs.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch["input"])
        loss, logs = self.criterion(outputs, batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for key, value in logs.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        outputs = self.model(batch["input"])
        loss, logs = self.criterion(outputs, batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        for key, value in logs.items():
            self.log(f"test/{key}", value, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if self.lr_scheduler_type == "none":
            return optimizer

        # Determine T_max for cosine scheduler
        if self.lr_t_max is not None:
            t_max = self.lr_t_max
        elif self.max_steps is not None:
            t_max = self.max_steps
        else:
            t_max = 10000  # Fallback default

        if self.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=self.lr_eta_min,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
            }
        elif self.lr_scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            }
        elif self.lr_scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=5000,
                gamma=0.5,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
            }
        else:
            return optimizer

        # Wrap with warmup if requested
        if self.lr_warmup_steps > 0:
            from torch.optim.lr_scheduler import LambdaLR, SequentialLR

            def warmup_fn(step):
                if step < self.lr_warmup_steps:
                    return float(step) / float(max(1, self.lr_warmup_steps))
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, warmup_fn)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler_config["scheduler"]],
                milestones=[self.lr_warmup_steps],
            )
            scheduler_config["scheduler"] = scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }


class NeuronDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for neuron datasets."""

    def __init__(
        self,
        data_dir: Path,
        manifest_train: Path,
        manifest_val: Optional[Path] = None,
        manifest_test: Optional[Path] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        image_size: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.manifest_train = Path(manifest_train)
        self.manifest_val = Path(manifest_val) if manifest_val else None
        self.manifest_test = Path(manifest_test) if manifest_test else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        # pin_memory only helps with CUDA
        self.pin_memory = pin_memory and torch.cuda.is_available()
        # persistent_workers can cause issues on some platforms (e.g., MPS)
        self.persistent_workers = persistent_workers and num_workers > 0 and torch.cuda.is_available()

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = NeuronDataset(
                self.manifest_train,
                self.data_dir,
                image_size=self.image_size,
            )
            if self.manifest_val:
                self.val_dataset = NeuronDataset(
                    self.manifest_val,
                    self.data_dir,
                    image_size=self.image_size,
                )
            else:
                self.val_dataset = None

        if stage == "test":
            if self.manifest_test:
                self.test_dataset = NeuronDataset(
                    self.manifest_test,
                    self.data_dir,
                    image_size=self.image_size,
                )
            else:
                self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


def auto_detect_manifests(data_dir: Path) -> Dict[str, Optional[Path]]:
    """Auto-detect manifest files in data directory."""
    manifests = {
        "train": None,
        "val": None,
        "test": None,
    }

    for split in manifests.keys():
        # Try common naming patterns
        candidates = [
            data_dir / f"manifest_{split}.jsonl",
            data_dir / f"{split}.jsonl",
            data_dir / f"manifest_{split}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                manifests[split] = candidate
                break

    # Also check for unsplit manifest.jsonl (use as train if no split manifests found)
    if manifests["train"] is None:
        unsplit = data_dir / "manifest.jsonl"
        if unsplit.exists():
            manifests["train"] = unsplit

    return manifests


def generate_run_name(args) -> str:
    """Generate a descriptive run name."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [timestamp]

    # Add key hyperparameters
    parts.append(f"bc{args.base_channels}")
    parts.append(f"lc{args.latent_channels}")
    parts.append(f"bs{args.batch_size}")

    if args.skip_mode != "variational":
        parts.append(args.skip_mode)

    return "_".join(parts)


def print_config_banner(args, data_module: NeuronDataModule):
    """Print configuration summary at startup."""
    print("\n" + "=" * 70)
    print("SegVAE2D Training")
    print("=" * 70)

    print("\n[Data]")
    print(f"  Data directory:    {args.data_dir}")
    print(f"  Train manifest:    {args.manifest_train}")
    if args.manifest_val:
        print(f"  Val manifest:      {args.manifest_val}")
    if args.manifest_test:
        print(f"  Test manifest:     {args.manifest_test}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Image size:        {args.image_size if args.image_size > 0 else 'original'}")
    print(f"  Num workers:       {args.num_workers}")

    print("\n[Model]")
    print(f"  Base channels:     {args.base_channels}")
    print(f"  Latent channels:   {args.latent_channels}")
    print(f"  Num classes:       {args.num_classes}")
    print(f"  Skip mode:         {args.skip_mode}")

    print("\n[Training]")
    print(f"  Max epochs:        {args.max_epochs}")
    if args.max_steps:
        print(f"  Max steps:         {args.max_steps}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  LR scheduler:      {args.lr_scheduler}")
    if args.lr_warmup_steps > 0:
        print(f"  Warmup steps:      {args.lr_warmup_steps}")
    print(f"  KLD weight:        {args.kld_weight}")
    print(f"  Lambda seg:        {args.lambda_seg}")
    print(f"  Lambda depth:      {args.lambda_depth}")
    if args.gradient_accumulation_steps > 1:
        print(f"  Grad accum steps:  {args.gradient_accumulation_steps}")
        print(f"  Effective batch:   {args.batch_size * args.gradient_accumulation_steps}")

    print("\n[Output]")
    print(f"  Save directory:    {args.save_dir}")
    print(f"  Log directory:     {args.log_dir}")
    print(f"  Run name:          {args.run_name}")

    print("\n[Hardware]")
    print(f"  Accelerator:       {args.accelerator}")
    print(f"  Devices:           {args.devices}")
    print(f"  Precision:         {args.precision}")
    if args.compile:
        print(f"  torch.compile:     enabled")

    if args.seed is not None:
        print(f"\n[Reproducibility]")
        print(f"  Seed:              {args.seed}")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train SegVAE2D model with PyTorch Lightning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-dir", type=Path, required=True,
        help="Dataset directory (will auto-detect manifests if not specified)"
    )
    data_group.add_argument(
        "--manifest-train", type=Path, default=None,
        help="Training manifest JSONL file (auto-detected if not set)"
    )
    data_group.add_argument(
        "--manifest-val", type=Path, default=None,
        help="Validation manifest JSONL file (auto-detected if not set)"
    )
    data_group.add_argument(
        "--manifest-test", type=Path, default=None,
        help="Test manifest JSONL file (auto-detected if not set)"
    )
    data_group.add_argument("--batch-size", type=int, default=8, help="Batch size")
    data_group.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    data_group.add_argument(
        "--image-size", type=int, default=512,
        help="Resize images to this size (default: 512, use 0 for original size)"
    )

    # Model arguments
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--in-channels", type=int, default=1, help="Input channels")
    model_group.add_argument("--base-channels", type=int, default=64, help="Base channel width")
    model_group.add_argument("--num-classes", type=int, default=6, help="Segmentation classes")
    model_group.add_argument("--latent-channels", type=int, default=128, help="Latent dimension")
    model_group.add_argument(
        "--skip-latent-mult", type=float, default=0.5,
        help="Skip connection latent size multiplier"
    )
    model_group.add_argument(
        "--skip-mode", type=str, default="variational",
        choices=["variational", "raw", "drop"],
        help="Skip connection mode"
    )

    # Loss arguments
    loss_group = parser.add_argument_group("Loss Weights")
    loss_group.add_argument("--kld-weight", type=float, default=0.1, help="KL divergence weight")
    loss_group.add_argument("--free-nats", type=float, default=0.0, help="Free-bits (nats) for KL")
    loss_group.add_argument("--beta", type=float, default=1.0, help="KL annealing multiplier")
    loss_group.add_argument("--lambda-seg", type=float, default=1.0, help="Segmentation loss weight")
    loss_group.add_argument("--lambda-depth", type=float, default=1.0, help="Depth loss weight")

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--max-epochs", type=int, default=100, help="Maximum epochs")
    train_group.add_argument("--max-steps", type=int, default=None, help="Maximum steps (overrides epochs)")
    train_group.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_group.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    train_group.add_argument(
        "--lr-scheduler", type=str, default="cosine",
        choices=["cosine", "plateau", "step", "none"],
        help="LR scheduler type"
    )
    train_group.add_argument("--lr-warmup-steps", type=int, default=0, help="LR warmup steps")
    train_group.add_argument(
        "--lr-t-max", type=int, default=None,
        help="T_max for cosine scheduler (defaults to max_steps)"
    )
    train_group.add_argument("--lr-eta-min", type=float, default=1e-6, help="Minimum LR")
    train_group.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Gradient accumulation steps for effective larger batches"
    )
    train_group.add_argument(
        "--gradient-clip-val", type=float, default=1.0,
        help="Gradient clipping value (0 to disable)"
    )

    # Early stopping
    train_group.add_argument(
        "--early-stopping", action="store_true",
        help="Enable early stopping"
    )
    train_group.add_argument(
        "--early-stopping-patience", type=int, default=10,
        help="Early stopping patience (epochs)"
    )
    train_group.add_argument(
        "--early-stopping-min-delta", type=float, default=0.0,
        help="Minimum improvement for early stopping"
    )

    # Checkpointing & Logging
    output_group = parser.add_argument_group("Output & Logging")
    output_group.add_argument(
        "--save-dir", type=Path, default=Path("checkpoints"),
        help="Checkpoint directory"
    )
    output_group.add_argument(
        "--log-dir", type=Path, default=Path("logs"),
        help="TensorBoard log directory"
    )
    output_group.add_argument(
        "--run-name", type=str, default=None,
        help="Run name (auto-generated if not set)"
    )
    output_group.add_argument(
        "--checkpoint-every-n-steps", type=int, default=1000,
        help="Save checkpoint every N steps"
    )
    output_group.add_argument(
        "--val-check-interval", type=float, default=1.0,
        help="Validation frequency (steps if int, epoch fraction if float)"
    )
    output_group.add_argument(
        "--log-every-n-steps", type=int, default=50,
        help="Log metrics every N steps"
    )
    output_group.add_argument(
        "--resume", type=Path, default=None,
        help="Resume from checkpoint"
    )

    # W&B
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    wandb_group.add_argument("--wandb-project", type=str, default="axonet", help="W&B project")
    wandb_group.add_argument("--wandb-name", type=str, default=None, help="W&B run name")

    # Hardware
    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument(
        "--accelerator", type=str, default="auto",
        help="Accelerator (auto, gpu, cpu, mps)"
    )
    hw_group.add_argument("--devices", type=int, default=1, help="Number of devices")
    hw_group.add_argument(
        "--precision", type=str, default="32",
        choices=["16", "32", "bf16", "16-mixed", "bf16-mixed"],
        help="Training precision"
    )
    hw_group.add_argument(
        "--compile", action="store_true",
        help="Use torch.compile() for faster training (PyTorch 2.0+)"
    )

    # Reproducibility
    misc_group = parser.add_argument_group("Misc")
    misc_group.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    misc_group.add_argument(
        "--quiet", action="store_true",
        help="Suppress startup banner"
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # Auto-detect manifests if not specified
    if args.manifest_train is None:
        detected = auto_detect_manifests(args.data_dir)
        if detected["train"] is None:
            print(f"ERROR: No training manifest found in {args.data_dir}", file=sys.stderr)
            print("Expected one of: manifest_train.jsonl, train.jsonl, manifest.jsonl", file=sys.stderr)
            sys.exit(1)
        args.manifest_train = detected["train"]
        if args.manifest_val is None and detected["val"] is not None:
            args.manifest_val = detected["val"]
        if args.manifest_test is None and detected["test"] is not None:
            args.manifest_test = detected["test"]

    # Validate manifest files exist
    if not args.manifest_train.exists():
        print(f"ERROR: Training manifest not found: {args.manifest_train}", file=sys.stderr)
        sys.exit(1)
    if args.manifest_val and not args.manifest_val.exists():
        print(f"ERROR: Validation manifest not found: {args.manifest_val}", file=sys.stderr)
        sys.exit(1)
    if args.manifest_test and not args.manifest_test.exists():
        print(f"ERROR: Test manifest not found: {args.manifest_test}", file=sys.stderr)
        sys.exit(1)

    # Generate run name if not specified
    if args.run_name is None:
        args.run_name = generate_run_name(args)

    # Create output directories
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Create data module
    image_size = args.image_size if args.image_size > 0 else None
    data_module = NeuronDataModule(
        data_dir=args.data_dir,
        manifest_train=args.manifest_train,
        manifest_val=args.manifest_val,
        manifest_test=args.manifest_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )

    # Print configuration banner
    if not args.quiet:
        print_config_banner(args, data_module)

    # Create model
    model = SegVAE2DLightning(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        latent_channels=args.latent_channels,
        skip_latent_mult=args.skip_latent_mult,
        kld_weight=args.kld_weight,
        skip_mode=args.skip_mode,
        free_nats=args.free_nats,
        beta=args.beta,
        lambda_seg=args.lambda_seg,
        lambda_depth=args.lambda_depth,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_t_max=args.lr_t_max,
        lr_eta_min=args.lr_eta_min,
        max_steps=args.max_steps,
    )

    # Optionally compile model
    if args.compile:
        if hasattr(torch, "compile"):
            print("Compiling model with torch.compile()...")
            model.model = torch.compile(model.model)
        else:
            print("WARNING: torch.compile() not available (requires PyTorch 2.0+)")

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename=f"{args.run_name}-{{epoch:03d}}-{{step:06d}}",
            every_n_train_steps=args.checkpoint_every_n_steps,
            save_top_k=0,
            save_last=False,
        ),
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename=f"{args.run_name}-best-{{epoch:03d}}-{{step:06d}}",
            monitor="val/loss" if args.manifest_val else "train/loss_epoch",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    if args.early_stopping and args.manifest_val:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=args.early_stopping_patience,
                min_delta=args.early_stopping_min_delta,
                mode="min",
                verbose=True,
            )
        )

    if args.manifest_val:
        from .callbacks import ValidationImageLogger

        callbacks.append(
            ValidationImageLogger(
                manifest_path=args.manifest_val,
                data_root=args.data_dir,
                num_samples=3,
                image_size=image_size or 512,
            )
        )

    # Setup loggers
    loggers = []
    if args.log_dir:
        loggers.append(
            TensorBoardLogger(
                save_dir=str(args.log_dir),
                name=args.run_name,
            )
        )

    if args.use_wandb:
        loggers.append(
            WandbLogger(
                project=args.wandb_project,
                name=args.wandb_name or args.run_name,
                log_model=False,
            )
        )

    # Build trainer kwargs
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "callbacks": callbacks,
        "logger": loggers if loggers else True,
        "log_every_n_steps": args.log_every_n_steps,
        "val_check_interval": args.val_check_interval,
        "accumulate_grad_batches": args.gradient_accumulation_steps,
        "deterministic": args.seed is not None,
    }

    if args.gradient_clip_val > 0:
        trainer_kwargs["gradient_clip_val"] = args.gradient_clip_val

    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = args.max_steps

    trainer = Trainer(**trainer_kwargs)

    # Training with proper interrupt handling
    interrupted = False
    try:
        trainer.fit(
            model,
            data_module,
            ckpt_path=str(args.resume) if args.resume else None,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl-C)")
        interrupted = True

        # Save emergency checkpoint on interruption
        current_step = getattr(trainer, "global_step", 0)
        current_epoch = getattr(trainer, "current_epoch", 0)

        if current_step > 0:
            emergency_path = args.save_dir / f"{args.run_name}-emergency-epoch{current_epoch:03d}-step{current_step:06d}.ckpt"
            print(f"Saving emergency checkpoint: {emergency_path}")
            try:
                trainer.save_checkpoint(str(emergency_path))
            except Exception as e:
                print(f"Failed to save emergency checkpoint: {e}")

    # Run test if requested
    if args.manifest_test:
        print("\nRunning test evaluation...")
        trainer.test(model, data_module)


if __name__ == "__main__":
    main()
