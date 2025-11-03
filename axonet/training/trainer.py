"""Training script for SegVAE2D model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from imageio.v2 import imread

from ..models.d3_swc_vae import SegVAE2D, MultiTaskLoss, build_model


class NeuronDataset(Dataset):
    """Dataset for neuron image + mask pairs."""

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        *,
        transform=None,
        load_depth: bool = False,
        load_recon: bool = False,
    ):
        self.data_root = Path(data_root)
        self.manifest_entries = []
        with open(manifest_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                self.manifest_entries.append(entry)
        
        # Auto-detect depth availability
        if not load_depth and len(self.manifest_entries) > 0 and "depth" in self.manifest_entries[0]:
            load_depth = True
        
        self.transform = transform
        self.load_depth = load_depth
        self.load_recon = load_recon

    def __len__(self):
        return len(self.manifest_entries)

    def __getitem__(self, idx):
        entry = self.manifest_entries[idx]
        
        image_path = self.data_root / entry["image"]
        mask_path = self.data_root / entry["mask"]
        
        image = imread(image_path)
        mask = imread(mask_path)
        
        # Convert to tensors
        if len(image.shape) == 3:
            # RGB to grayscale
            image = np.mean(image, axis=2, dtype=np.float32) / 255.0
        else:
            image = image.astype(np.float32) / 255.0
        
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dim
        mask = torch.from_numpy(mask).long()
        
        sample = {
            "input": image,
            "seg": mask,
        }
        
        if self.load_depth:
            depth = self._load_depth(entry)
            if depth is not None:
                sample["depth"] = depth
                sample["valid_depth_mask"] = torch.ones_like(depth)
        
        if self.load_recon:
            sample["recon"] = image.clone()
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def _load_depth(self, entry: Dict):
        depth_path = self.data_root / entry.get("depth", "")
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
        elif key == "recon":
            batched["recon"] = torch.stack([item["recon"] for item in batch])
        elif key == "depth":
            batched["depth"] = torch.stack([item["depth"] for item in batch])
        elif key == "valid_depth_mask":
            batched["valid_depth_mask"] = torch.stack([item["valid_depth_mask"] for item in batch])
    
    return batched


class Trainer:
    """Main trainer class for SegVAE2D."""

    def __init__(
        self,
        model: SegVAE2D,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: MultiTaskLoss,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler._LRScheduler | None = None,
        *,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: Path | None = None,
        log_dir: Path | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None
        self.log_dir = Path(log_dir) if log_dir else None
        
        if self.log_dir:
            self.writer = SummaryWriter(str(self.log_dir))
        else:
            self.writer = None
        
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch["input"])
            
            loss, logs = self.criterion(outputs, batch)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if self.writer:
                for key, value in logs.items():
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            
            self.global_step += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = train_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate on validation set."""
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(batch["input"])
            
            loss, logs = self.criterion(outputs, batch)
            
            val_loss += loss.item()
            num_batches += 1
            
            if self.writer:
                for key, value in logs.items():
                    self.writer.add_scalar(f"val/{key}", value, self.global_step)
                self.writer.add_scalar("val/loss", loss.item(), self.global_step)
        
        avg_loss = val_loss / num_batches if num_batches > 0 else 0.0
        
        # Save best model
        if avg_loss < self.best_val_loss and self.save_dir:
            self.best_val_loss = avg_loss
            self.save_checkpoint(epoch, is_best=True)
        
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not self.save_dir:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        if self.lr_scheduler:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / "checkpoint.pt")
        
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pt")

    def train(self, num_epochs: int):
        """Main training loop."""
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.save_dir:
                self.save_checkpoint(epoch)
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train SegVAE2D model")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSONL file")
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"), help="Checkpoint directory")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Log directory")
    
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--use-depth", action="store_true")
    parser.add_argument("--use-recon", action="store_true")
    parser.add_argument("--kld-weight", type=float, default=1.0)
    
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-depth", type=float, default=1.0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create datasets
    dataset = NeuronDataset(
        args.manifest,
        args.data_dir,
        load_depth=args.use_depth,
        load_recon=args.use_recon,
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # Create model
    model = build_model(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        latent_channels=args.latent_channels,
        use_depth=args.use_depth,
        use_recon=args.use_recon,
        kld_weight=args.kld_weight,
    )
    
    # Create criterion
    criterion = MultiTaskLoss(
        lambda_seg=args.lambda_seg,
        lambda_depth=args.lambda_depth,
        lambda_recon=args.lambda_recon,
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        verbose=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )
    
    # Train
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()

