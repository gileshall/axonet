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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ..models.d3_swc_vae import SegVAE2D, MultiTaskLoss, build_model


class NeuronDataset(Dataset):
    """Dataset for neuron binary mask (input) + segmentation mask pairs.
    
    Uses mask_bw (black/white binary mask) as input, and mask (class IDs) as segmentation target.
    """

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
        
        # Load input: use mask_bw (binary black/white mask) as input image
        # Handle paths that may or may not include 'images/' prefix
        if "mask_bw" in entry:
            input_path = self._resolve_path(entry["mask_bw"])
        elif "image" in entry:
            # Fallback to image if mask_bw not available (backward compatibility)
            input_path = self._resolve_path(entry["image"])
        else:
            raise ValueError(f"Manifest entry missing both 'mask_bw' and 'image' fields: {entry}")
        
        mask_path = self._resolve_path(entry["mask"])
        
        # Load input (mask_bw is already grayscale uint8: 0=black, 255=white)
        input_img = imread(input_path)
        if len(input_img.shape) == 3:
            # RGB to grayscale (fallback case)
            input_img = np.mean(input_img, axis=2, dtype=np.float32) / 255.0
        else:
            input_img = input_img.astype(np.float32) / 255.0
        
        # Load segmentation mask
        mask = imread(mask_path)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_img).unsqueeze(0)  # Add channel dim
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
        
        if self.load_recon:
            sample["recon"] = input_tensor.clone()
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def _resolve_path(self, rel_path: str) -> Path:
        """Resolve a relative path from manifest, handling both with and without 'images/' prefix."""
        # Try as-is first
        path = self.data_root / rel_path
        if path.exists():
            return path
        
        # If path doesn't start with 'images/', try adding it
        if not rel_path.startswith("images/"):
            path = self.data_root / "images" / rel_path
            if path.exists():
                return path
        
        # If path starts with 'images/', try without it
        if rel_path.startswith("images/"):
            path = self.data_root / rel_path[7:]  # Remove "images/" prefix
            if path.exists():
                return path
        
        # If still not found, return the original path (will raise FileNotFoundError)
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
        test_loader: DataLoader | None = None,
        device: str | None = None,
        save_dir: Path | None = None,
        log_dir: Path | None = None,
        checkpoint_step: int = 100,
        test_step: int = 100,
        max_test_samples: int = 50,
    ):
        # Auto-detect device if not provided
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else None
        self.log_dir = Path(log_dir) if log_dir else None
        self.checkpoint_step = checkpoint_step
        self.test_step = test_step
        self.max_test_samples = max_test_samples
        
        if self.log_dir:
            self.writer = SummaryWriter(str(self.log_dir))
        else:
            self.writer = None
        
        self.use_wandb = False
        self.wandb_config = None
        
        self.global_step = 0
        self.start_epoch = 1
        self.best_val_loss = float("inf")
        self.first_test_done = False
    
    def init_wandb(self, project: str, name: str | None = None, config: dict | None = None):
        """Initialize Weights & Biases logging."""
        if not WANDB_AVAILABLE:
            raise RuntimeError("wandb is not installed. Install it with: pip install wandb")
        self.use_wandb = True
        self.wandb_config = config or {}
        wandb.init(project=project, name=name, config=self.wandb_config)

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
            
            # Step-based LR scheduler (if applicable)
            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'step') and not isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if self.writer:
                for key, value in logs.items():
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            
            if self.use_wandb:
                wandb_log = {f"train/{key}": value for key, value in logs.items()}
                wandb_log["train/loss"] = loss.item()
                wandb_log["train/learning_rate"] = self.optimizer.param_groups[0]["lr"]
                wandb.log(wandb_log, step=self.global_step)
            
            self.global_step += 1
            
            # Step-based checkpointing
            if self.save_dir and self.global_step % self.checkpoint_step == 0:
                self.save_checkpoint(epoch, is_best=False, step=self.global_step)
            
            # Step-based test evaluation
            if self.test_loader is not None and self.global_step % self.test_step == 0:
                self.evaluate_test(epoch, save_outputs=not self.first_test_done)
                self.first_test_done = True
            
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
            
            if self.use_wandb:
                wandb_log = {f"val/{key}": value for key, value in logs.items()}
                wandb_log["val/loss"] = loss.item()
                wandb.log(wandb_log, step=self.global_step)
        
        avg_loss = val_loss / num_batches if num_batches > 0 else 0.0
        
        # Save best model
        if avg_loss < self.best_val_loss and self.save_dir:
            self.best_val_loss = avg_loss
            self.save_checkpoint(epoch, is_best=True)
            if self.use_wandb:
                wandb.run.summary["best_val_loss"] = avg_loss
        
        return avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False, step: int | None = None):
        """Save model checkpoint with trainer state for resuming."""
        if not self.save_dir:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        step = step if step is not None else self.global_step
        checkpoint = {
            "epoch": epoch,
            "global_step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "first_test_done": self.first_test_done,
        }
        
        if self.lr_scheduler:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, self.save_dir / "checkpoint.pt")
        
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pt")

        # Also save step-based checkpoint
        if step is not None:
            torch.save(checkpoint, self.save_dir / f"checkpoint_step_{step}.pt")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load checkpoint and resume training state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 1) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.first_test_done = checkpoint.get("first_test_done", False)
        print(f"Resumed from checkpoint: epoch={checkpoint.get('epoch', 1)}, step={self.global_step}")
    
    @torch.no_grad()
    def evaluate_test(self, epoch: int, save_outputs: bool = False):
        """Evaluate on test set and optionally save outputs."""
        if self.test_loader is None:
            return
        
        self.model.eval()
        test_loss = 0.0
        num_batches = 0
        outputs_to_save = []
        
        samples_collected = 0
        for batch_idx, batch in enumerate(self.test_loader):
            if samples_collected >= self.max_test_samples:
                break
            
            batch_size = batch["input"].shape[0]
            if samples_collected + batch_size > self.max_test_samples:
                # Trim batch to exact number needed
                trim_size = self.max_test_samples - samples_collected
                batch = {k: v[:trim_size] for k, v in batch.items()}
                batch_size = trim_size
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(batch["input"])
            loss, logs = self.criterion(outputs, batch)
            
            test_loss += loss.item()
            num_batches += 1
            samples_collected += batch_size
            
            # Save outputs for first test evaluation
            if save_outputs:
                seg_pred = torch.argmax(outputs["seg_logits"], dim=1).cpu().numpy()
                seg_gt = batch["seg"].cpu().numpy()
                for i in range(batch_size):
                    if len(outputs_to_save) >= self.max_test_samples:
                        break
                    output_dict = {
                        "input": batch["input"][i].cpu().numpy(),
                        "seg_pred": seg_pred[i],
                        "seg_gt": seg_gt[i],
                        "step": self.global_step,
                    }
                    if "depth" in outputs:
                        output_dict["depth_pred"] = outputs["depth"][i].cpu().numpy()
                    if "depth" in batch:
                        output_dict["depth_gt"] = batch["depth"][i].cpu().numpy()
                    outputs_to_save.append(output_dict)
        
        avg_loss = test_loss / num_batches if num_batches > 0 else 0.0
        
        if self.writer:
            self.writer.add_scalar("test/loss", avg_loss, self.global_step)
            for key, value in logs.items():
                self.writer.add_scalar(f"test/{key}", value, self.global_step)
        
        if self.use_wandb:
            wandb_log = {"test/loss": avg_loss}
            for key, value in logs.items():
                wandb_log[f"test/{key}"] = value
            wandb.log(wandb_log, step=self.global_step)
        
        print(f"Test Loss (step {self.global_step}): {avg_loss:.4f}")
        
        # Save outputs to disk if requested
        if save_outputs and outputs_to_save and self.save_dir:
            output_dir = self.save_dir / "test_outputs" / f"step_{self.global_step}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, output in enumerate(outputs_to_save):
                np.savez_compressed(
                    output_dir / f"sample_{idx:04d}.npz",
                    **output
                )
            print(f"Saved {len(outputs_to_save)} test outputs to {output_dir}")
        
        self.model.train()

    def train(self, num_epochs: int, max_steps: int | None = None):
        """Main training loop."""
        for epoch in range(self.start_epoch, num_epochs + 1):
            if max_steps is not None and self.global_step >= max_steps:
                print(f"\nReached max_steps={max_steps}, stopping training.")
                break
            
            print(f"\nEpoch {epoch}/{num_epochs} (Step {self.global_step})")
            print("-" * 50)
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # ReduceLROnPlateau scheduler steps on validation loss
            if self.lr_scheduler and isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "epoch/train_loss": train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/learning_rate": self.optimizer.param_groups[0]["lr"],
                }, step=epoch)
            
            # Check max_steps after epoch
            if max_steps is not None and self.global_step >= max_steps:
                print(f"\nReached max_steps={max_steps}, stopping training.")
                break
        
        if self.writer:
            self.writer.close()
        
        if self.use_wandb:
            wandb.finish()


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
    
    # Weights & Biases arguments
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="axonet-segvae2d", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name (default: auto-generated)")
    
    # Training control arguments
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio (default: 0.1)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for train/val/test split")
    parser.add_argument("--checkpoint-step", type=int, default=100, help="Save checkpoint every N steps (default: 100)")
    parser.add_argument("--test-step", type=int, default=100, help="Evaluate test set every N steps (default: 100)")
    parser.add_argument("--max-test-samples", type=int, default=50, help="Maximum test samples to evaluate (default: 50)")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint file")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum training steps (overrides epochs)")
    
    # LR scheduler arguments
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "plateau", "step", "none"], help="LR scheduler type (default: cosine)")
    parser.add_argument("--lr-warmup-steps", type=int, default=1000, help="Warmup steps for cosine scheduler (default: 1000)")
    parser.add_argument("--lr-t-max", type=int, default=10000, help="T_max for cosine scheduler (default: 10000)")
    parser.add_argument("--lr-eta-min", type=float, default=1e-6, help="Minimum LR for cosine scheduler (default: 1e-6)")
    
    args = parser.parse_args()
    
    # Device selection: prefer MPS (Apple Metal) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Create datasets
    dataset = NeuronDataset(
        args.manifest,
        args.data_dir,
        load_depth=args.use_depth,
        load_recon=args.use_recon,
    )
    
    # Create train/val/test splits with seed for reproducibility
    generator = torch.Generator().manual_seed(args.split_seed)
    total_size = len(dataset)
    test_size = int(args.test_ratio * total_size)
    val_size = int(args.val_ratio * total_size)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"Dataset splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
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
    
    test_loader = DataLoader(
        test_dataset,
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
    
    # Create LR scheduler
    lr_scheduler = None
    if args.lr_scheduler == "cosine":
        # Cosine annealing with warm restarts (step-based)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.lr_t_max,
            eta_min=args.lr_eta_min,
        )
    elif args.lr_scheduler == "plateau":
        # Reduce on plateau (epoch-based, steps on validation loss)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        )
    elif args.lr_scheduler == "step":
        # Step decay (step-based)
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5000,
            gamma=0.5,
    )
    # else: "none" - no scheduler
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        checkpoint_step=args.checkpoint_step,
        test_step=args.test_step,
        max_test_samples=args.max_test_samples,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        wandb_config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "in_channels": args.in_channels,
            "base_channels": args.base_channels,
            "num_classes": args.num_classes,
            "latent_channels": args.latent_channels,
            "use_depth": args.use_depth,
            "use_recon": args.use_recon,
            "kld_weight": args.kld_weight,
            "lambda_seg": args.lambda_seg,
            "lambda_depth": args.lambda_depth,
            "lambda_recon": args.lambda_recon,
            "device": device,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
        }
        trainer.init_wandb(project=args.wandb_project, name=args.wandb_name, config=wandb_config)
    
    # Train
    trainer.train(args.epochs, max_steps=args.max_steps)


if __name__ == "__main__":
    main()

