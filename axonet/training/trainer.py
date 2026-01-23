"""Training script for SegVAE2D model using PyTorch Lightning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from imageio.v2 import imread
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, Dataset

from ..models.d3_swc_vae import SegVAE2D, MultiTaskLoss, build_model


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
    ):
        self.data_root = Path(data_root)
        self.manifest_entries = []
        with open(manifest_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                self.manifest_entries.append(entry)

        # Auto-detect depth availability from manifest
        self.load_depth = len(self.manifest_entries) > 0 and "depth" in self.manifest_entries[0]

        self.transform = transform

    def __len__(self):
        return len(self.manifest_entries)

    def __getitem__(self, idx):
        entry = self.manifest_entries[idx]
        
        if "mask_bw" in entry:
            input_path = self._resolve_path(entry["mask_bw"])
        elif "image" in entry:
            input_path = self._resolve_path(entry["image"])
        else:
            raise ValueError(f"Manifest entry missing both 'mask_bw' and 'image' fields: {entry}")
        
        mask_path = self._resolve_path(entry["mask"])
        
        input_img = imread(input_path)
        if len(input_img.shape) == 3:
            input_img = np.mean(input_img, axis=2, dtype=np.float32) / 255.0
        else:
            input_img = input_img.astype(np.float32) / 255.0
        
        mask = imread(mask_path)
        
        input_tensor = torch.from_numpy(input_img).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long()
        
        sample = {
            "input": input_tensor,
            "seg": mask_tensor,
        }

        if self.load_depth:
            depth = self._load_depth(entry)
            if depth is not None:
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
        num_classes: int = 4,
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
        lr_t_max: int = 10000,
        lr_eta_min: float = 1e-6,
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
        self.lr_scheduler = lr_scheduler
        self.lr_t_max = lr_t_max
        self.lr_eta_min = lr_eta_min

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
        
        if self.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.lr_t_max,
                eta_min=self.lr_eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        elif self.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=10,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }
        elif self.lr_scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=5000,
                gamma=0.5,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer


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
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.manifest_train = Path(manifest_train)
        self.manifest_val = Path(manifest_val) if manifest_val else None
        self.manifest_test = Path(manifest_test) if manifest_test else None
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = NeuronDataset(
                self.manifest_train,
                self.data_dir,
            )
            if self.manifest_val:
                self.val_dataset = NeuronDataset(
                    self.manifest_val,
                    self.data_dir,
                )
            else:
                self.val_dataset = None

        if stage == "test":
            if self.manifest_test:
                self.test_dataset = NeuronDataset(
                    self.manifest_test,
                    self.data_dir,
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
        )


def main():
    parser = argparse.ArgumentParser(description="Train SegVAE2D model with PyTorch Lightning")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--manifest-train", type=Path, required=True, help="Training manifest JSONL file")
    parser.add_argument("--manifest-val", type=Path, default=None, help="Validation manifest JSONL file")
    parser.add_argument("--manifest-test", type=Path, default=None, help="Test manifest JSONL file")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Log directory")
    
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of training steps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--skip-latent-mult", type=float, default=0.5, help="Multiplicative factor for skip connection latent size (default: 0.5)")
    parser.add_argument("--skip-mode", type=str, default="variational", choices=["variational", "raw", "drop"], help="Skip connection mode: variational (stochastic), raw (identity), drop (zero)")
    parser.add_argument("--free-nats", type=float, default=0.0, help="Free-bits (nats) applied to KL terms (default: 0.0)")
    parser.add_argument("--beta", type=float, default=1.0, help="KL annealing multiplier for all KL terms (default: 1.0)")
    parser.add_argument("--kld-weight", type=float, default=0.1, help="KL divergence weight (default: 0.1, lower due to variational skip connections)")

    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-depth", type=float, default=1.0)
    
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="axonet-segvae2d", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")
    
    parser.add_argument("--checkpoint-step", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--val-check-interval", type=float, default=1.0, help="Run validation every N steps (int) or fraction of epoch (float, default: 1.0 = end of epoch)")
    parser.add_argument("--log-every-n-steps", type=int, default=10, help="Log metrics every N training steps (default: 10)")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint file")
    
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "plateau", "step", "none"], help="LR scheduler type")
    parser.add_argument("--lr-t-max", type=int, default=10000, help="T_max for cosine scheduler")
    parser.add_argument("--lr-eta-min", type=float, default=1e-6, help="Minimum LR for cosine scheduler")
    
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator type (auto, gpu, cpu, mps)")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--precision", type=str, default="32", choices=["16", "32", "bf16"], help="Precision")
    
    args = parser.parse_args()
    
    data_module = NeuronDataModule(
        data_dir=args.data_dir,
        manifest_train=args.manifest_train,
        manifest_val=args.manifest_val,
        manifest_test=args.manifest_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

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
        lr_t_max=args.lr_t_max,
        lr_eta_min=args.lr_eta_min,
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="checkpoint-{epoch:03d}-{step:06d}",
            every_n_train_steps=args.checkpoint_step,
            save_top_k=0,
            save_last=False,
        ),
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="best-{epoch:03d}-{step:06d}",
            monitor="val/loss" if args.manifest_val else "train/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    loggers = []
    if args.log_dir:
        loggers.append(TensorBoardLogger(save_dir=str(args.log_dir.parent), name=args.log_dir.name))
    
    if args.use_wandb:
        loggers.append(WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            log_model=False,
        ))
    
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "precision": args.precision,
        "callbacks": callbacks,
        "logger": loggers if loggers else True,
        "log_every_n_steps": args.log_every_n_steps,
        "val_check_interval": args.val_check_interval,
    }
    
    if args.max_steps is not None:
        trainer_kwargs["max_steps"] = args.max_steps
    
    trainer = Trainer(**trainer_kwargs)
    
    training_started = False
    
    try:
        trainer.fit(
            model,
            data_module,
            ckpt_path=str(args.resume) if args.resume else None,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl-C)")
        training_started = True
    except Exception as e:
        print(f"\nTraining interrupted by exception: {e}")
        training_started = True
        raise
    finally:
        if training_started:
            current_global_step = getattr(trainer, 'global_step', 0)
            current_epoch = getattr(trainer, 'current_epoch', 0)
            
            if current_global_step > 0 or current_epoch > 0:
                print(f"\nSaving emergency checkpoint (epoch {current_epoch}, step {current_global_step})...")
                args.save_dir.mkdir(parents=True, exist_ok=True)
                emergency_checkpoint_path = args.save_dir / f"emergency-checkpoint-epoch-{current_epoch:03d}-step-{current_global_step:06d}.ckpt"
                trainer.save_checkpoint(str(emergency_checkpoint_path))
                print(f"Emergency checkpoint saved to: {emergency_checkpoint_path}")
            else:
                print("\nNo training progress detected, skipping emergency checkpoint.")
    
    if args.manifest_test:
        trainer.test(model, data_module)


if __name__ == "__main__":
    main()
