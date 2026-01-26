"""PyTorch Lightning module for Stage 2 CLIP training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix tokenizers parallelism issue on macOS - must be set before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.metadata_adapters import get_adapter
from ..models.clip_modules import SegVAE2D_CLIP
from ..models.d3_swc_vae import load_model
from ..models.text_encoders import HashTextEncoder, ProjectedTextEncoder, TransformerTextEncoder
from .clip_dataset import NeuronCLIPDataset, NeuronTextGenerator
from .losses.infonce import Stage2Loss

logger = logging.getLogger(__name__)


class CLIPLightning(LightningModule):
    """Lightning module for Stage 2 CLIP contrastive training."""

    def __init__(
        self,
        stage1_checkpoint: str,
        clip_embed_dim: int = 512,
        hidden_dim: int = 256,
        freeze_encoder: bool = True,
        encoder_lr_mult: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        lambda_clip: float = 1.0,
        lambda_kld: float = 0.0,
        text_encoder_name: str = "distilbert-base-uncased",
        lr_scheduler: str = "cosine",
        lr_warmup_steps: int = 0,
        lr_t_max: Optional[int] = None,
        lr_eta_min: float = 1e-6,
        max_steps: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.stage1_checkpoint = stage1_checkpoint
        self.clip_embed_dim = clip_embed_dim
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder
        self.encoder_lr_mult = encoder_lr_mult
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_t_max = lr_t_max
        self.lr_eta_min = lr_eta_min
        self.max_steps = max_steps

        base_model = load_model(
            Path(stage1_checkpoint),
            device="cpu",
            embedding_only=True,
        )

        self.image_encoder = SegVAE2D_CLIP.from_pretrained(
            base_model,
            clip_embed_dim=clip_embed_dim,
            hidden_dim=hidden_dim,
            freeze_encoder=freeze_encoder,
        )

        # Create text encoder based on name
        if text_encoder_name == "hash" or text_encoder_name.startswith("hash:"):
            # Parse embed_dim from "hash:384" format, default 384
            if ":" in text_encoder_name:
                hash_dim = int(text_encoder_name.split(":")[1])
            else:
                hash_dim = 384
            text_encoder_base = HashTextEncoder(
                embed_dim=hash_dim,
                normalize=True,
            )
        else:
            # Use transformer model
            text_encoder_base = TransformerTextEncoder(
                model_name=text_encoder_name,
                normalize=True,
            )

        self.text_encoder = ProjectedTextEncoder(
            encoder=text_encoder_base,
            output_dim=clip_embed_dim,
            hidden_dim=hidden_dim,
            freeze_encoder=True,
        )

        self.criterion = Stage2Loss(
            lambda_clip=lambda_clip,
            lambda_kld=lambda_kld,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_encoder.encode_for_clip(x)

    def get_image_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def get_text_embedding(self, texts: List[str]) -> torch.Tensor:
        return self.text_encoder(texts)

    def _shared_step(
        self, batch: Dict[str, Any], batch_idx: int, stage: str
    ) -> Dict[str, torch.Tensor]:
        images = batch["input"]
        texts = batch["text"]

        image_embeds = self.image_encoder.encode_for_clip(images)

        text_embeds = self.text_encoder(texts)
        text_embeds = text_embeds.to(images.device)

        kld = None
        if self.hparams.lambda_kld > 0:
            out = self.image_encoder(images, return_vae_outputs=True)
            kld = out["kld"]

        loss, logs = self.criterion(
            image_embeds,
            text_embeds,
            kld=kld,
            return_accuracy=True,
        )

        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in logs.items():
            self.log(f"{stage}/{k}", v, on_step=True, on_epoch=True)

        return {"loss": loss, "logs": logs}

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        param_groups = [
            {
                "params": self.image_encoder.get_projection_parameters(),
                "lr": self.lr,
            },
            {
                "params": self.text_encoder.projection.parameters(),
                "lr": self.lr,
            },
            {
                "params": [self.criterion.infonce.log_temperature],
                "lr": self.lr,
            },
        ]

        if not self.freeze_encoder:
            param_groups.append({
                "params": self.image_encoder.get_encoder_parameters(),
                "lr": self.lr * self.encoder_lr_mult,
            })

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        if self.lr_scheduler_type == "none":
            return optimizer

        # Determine T_max for cosine scheduler
        if self.lr_t_max is not None:
            t_max = self.lr_t_max
        elif self.max_steps is not None:
            t_max = self.max_steps
        elif self.trainer and self.trainer.max_epochs:
            # Estimate steps from epochs
            t_max = self.trainer.max_epochs * 1000  # Rough estimate
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
                patience=5,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
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


class CLIPDataModule(LightningDataModule):
    """DataModule for CLIP training."""

    def __init__(
        self,
        data_root: Path,
        manifest_train: Path,
        metadata_path: Path,
        manifest_val: Optional[Path] = None,
        source: str = "allen",
        id_column: str = "cell_specimen_id",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.manifest_train = Path(manifest_train)
        self.manifest_val = Path(manifest_val) if manifest_val else None
        self.metadata_path = Path(metadata_path)
        self.source = source
        self.id_column = id_column
        self.batch_size = batch_size
        self.image_size = image_size
        # On macOS (MPS/CPU), multiprocessing DataLoaders cause "too many open files" errors
        # Use num_workers=0 to avoid this
        if torch.cuda.is_available():
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.persistent_workers = persistent_workers and num_workers > 0
        else:
            self.num_workers = 0
            self.pin_memory = False
            self.persistent_workers = False

    def setup(self, stage: Optional[str] = None):
        adapter = get_adapter(self.source)
        text_gen = NeuronTextGenerator(adapter, augment=True)

        self.train_dataset = NeuronCLIPDataset(
            manifest_path=self.manifest_train,
            data_root=self.data_root,
            metadata_path=self.metadata_path,
            adapter=adapter,
            id_column=self.id_column,
            text_generator=text_gen,
            image_size=self.image_size,
        )

        if self.manifest_val:
            text_gen_val = NeuronTextGenerator(adapter, augment=False)
            self.val_dataset = NeuronCLIPDataset(
                manifest_path=self.manifest_val,
                data_root=self.data_root,
                metadata_path=self.metadata_path,
                adapter=adapter,
                id_column=self.id_column,
                text_generator=text_gen_val,
                image_size=self.image_size,
            )
        else:
            self.val_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )


def train_from_config(config: ExperimentConfig):
    """Train CLIP model from experiment config."""
    clip_cfg = config.training.clip

    model = CLIPLightning(
        stage1_checkpoint=clip_cfg.stage1_checkpoint,
        clip_embed_dim=clip_cfg.embed_dim,
        hidden_dim=clip_cfg.hidden_dim,
        freeze_encoder=clip_cfg.freeze_encoder,
        encoder_lr_mult=clip_cfg.encoder_lr_mult,
        lr=config.training.lr,
        temperature=clip_cfg.temperature,
        learnable_temperature=clip_cfg.learnable_temperature,
        lambda_clip=clip_cfg.lambda_clip,
        lambda_kld=clip_cfg.lambda_kld,
        text_encoder_name=clip_cfg.text_encoder,
    )

    data_root = Path(config.data.root)
    datamodule = CLIPDataModule(
        data_root=data_root,
        manifest_train=data_root / config.data.manifest,
        metadata_path=data_root / config.data.metadata if config.data.metadata else None,
        source=config.data.source,
        id_column=config.data.id_column,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=config.output.save_dir,
            filename="clip-{epoch:02d}-{val_loss:.4f}",
            save_top_k=config.output.save_top_k,
            monitor=config.output.monitor,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    tb_logger = TensorBoardLogger(
        save_dir=config.output.log_dir,
        name=config.name,
    )

    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        callbacks=callbacks,
        logger=tb_logger,
    )

    trainer.fit(model, datamodule)

    return model


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
    parts = [timestamp, "clip"]

    # Add key hyperparameters
    parts.append(f"emb{args.clip_embed_dim}")
    parts.append(f"bs{args.batch_size}")

    if args.unfreeze_encoder:
        parts.append("unfrz")

    return "_".join(parts)


def print_config_banner(args, data_module: CLIPDataModule, train_size: int, val_size: int):
    """Print configuration summary at startup."""
    print("\n" + "=" * 70)
    print("CLIP Fine-tuning (Stage 2)")
    print("=" * 70)

    print("\n[Data]")
    print(f"  Data directory:    {args.data_dir}")
    print(f"  Train manifest:    {args.manifest_train}")
    print(f"  Train samples:     {train_size:,}")
    if args.manifest_val:
        print(f"  Val manifest:      {args.manifest_val}")
        print(f"  Val samples:       {val_size:,}")
    print(f"  Metadata:          {args.metadata}")
    print(f"  Source:            {args.source}")
    print(f"  ID column:         {args.id_column}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Image size:        {args.image_size}")
    print(f"  Num workers:       {args.num_workers}")

    print("\n[Model]")
    print(f"  Stage 1 ckpt:      {args.stage1_checkpoint}")
    print(f"  CLIP embed dim:    {args.clip_embed_dim}")
    print(f"  Hidden dim:        {args.hidden_dim}")
    print(f"  Freeze encoder:    {not args.unfreeze_encoder}")
    if args.unfreeze_encoder:
        print(f"  Encoder LR mult:   {args.encoder_lr_mult}")
    print(f"  Text encoder:      {args.text_encoder}")

    print("\n[Training]")
    print(f"  Max epochs:        {args.max_epochs}")
    if args.max_steps:
        print(f"  Max steps:         {args.max_steps}")
    print(f"  Learning rate:     {args.lr}")
    print(f"  LR scheduler:      {args.lr_scheduler}")
    if args.lr_warmup_steps > 0:
        print(f"  Warmup steps:      {args.lr_warmup_steps}")
    print(f"  Temperature:       {args.temperature}")
    print(f"  Learnable temp:    {args.learnable_temperature}")
    print(f"  Lambda CLIP:       {args.lambda_clip}")
    print(f"  Lambda KLD:        {args.lambda_kld}")
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
        description="Stage 2 CLIP fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file option
    parser.add_argument("--config", type=Path, help="YAML config file (overrides CLI args)")

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-dir", type=Path,
        help="Data root directory (will auto-detect manifests if not specified)"
    )
    data_group.add_argument(
        "--manifest-train", type=Path, default=None,
        help="Training manifest (auto-detected if not set)"
    )
    data_group.add_argument(
        "--manifest-val", type=Path, default=None,
        help="Validation manifest (auto-detected if not set)"
    )
    data_group.add_argument(
        "--metadata", type=Path,
        help="Metadata file (JSON/JSONL/CSV)"
    )
    data_group.add_argument(
        "--source", default="neuromorpho",
        choices=["allen", "neuromorpho", "custom"],
        help="Data source type"
    )
    data_group.add_argument("--id-column", default="neuron_id", help="ID column in metadata")
    data_group.add_argument("--batch-size", type=int, default=64, help="Batch size")
    data_group.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    data_group.add_argument(
        "--image-size", type=int, default=512,
        help="Resize images to this size"
    )

    # Model arguments
    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--stage1-checkpoint", type=Path,
        help="Path to Stage 1 (VAE) checkpoint"
    )
    model_group.add_argument("--clip-embed-dim", type=int, default=512, help="CLIP embedding dimension")
    model_group.add_argument("--hidden-dim", type=int, default=256, help="Projection hidden dimension")
    model_group.add_argument(
        "--freeze-encoder", action="store_true", default=True,
        help="Freeze VAE encoder weights (default)"
    )
    model_group.add_argument(
        "--unfreeze-encoder", action="store_true",
        help="Unfreeze VAE encoder for fine-tuning"
    )
    model_group.add_argument(
        "--encoder-lr-mult", type=float, default=0.1,
        help="LR multiplier for encoder when unfrozen"
    )
    model_group.add_argument(
        "--text-encoder", type=str,
        default="distilbert-base-uncased",
        help="Text encoder: HuggingFace model name (e.g., distilbert-base-uncased)"
    )

    # CLIP loss arguments
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    loss_group.add_argument(
        "--learnable-temperature", action="store_true", default=True,
        help="Make temperature learnable"
    )
    loss_group.add_argument(
        "--no-learnable-temperature", action="store_true",
        help="Fix temperature during training"
    )
    loss_group.add_argument("--lambda-clip", type=float, default=1.0, help="CLIP loss weight")
    loss_group.add_argument("--lambda-kld", type=float, default=0.0, help="KLD regularization weight")

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--max-epochs", type=int, default=100, help="Maximum epochs")
    train_group.add_argument("--max-steps", type=int, default=None, help="Maximum steps (overrides epochs)")
    train_group.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_group.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    train_group.add_argument(
        "--lr-scheduler", type=str, default="cosine",
        choices=["cosine", "plateau", "none"],
        help="LR scheduler type"
    )
    train_group.add_argument("--lr-warmup-steps", type=int, default=100, help="LR warmup steps")
    train_group.add_argument(
        "--lr-t-max", type=int, default=None,
        help="T_max for cosine scheduler (defaults to max_steps)"
    )
    train_group.add_argument("--lr-eta-min", type=float, default=1e-6, help="Minimum LR")
    train_group.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Gradient accumulation steps"
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

    # Checkpointing & Logging
    output_group = parser.add_argument_group("Output & Logging")
    output_group.add_argument(
        "--save-dir", type=Path, default=Path("checkpoints/clip"),
        help="Checkpoint directory"
    )
    output_group.add_argument(
        "--log-dir", type=Path, default=Path("logs/clip"),
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
    wandb_group.add_argument("--wandb-project", type=str, default="axonet-clip", help="W&B project")
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

    # Handle YAML config
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        train_from_config(config)
        return

    # Set seed for reproducibility
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # Validate required arguments
    if not args.stage1_checkpoint:
        parser.error("--stage1-checkpoint is required")
    if not args.data_dir:
        parser.error("--data-dir is required")
    if not args.metadata:
        parser.error("--metadata is required")

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

    # Validate files exist
    if not args.stage1_checkpoint.exists():
        print(f"ERROR: Stage 1 checkpoint not found: {args.stage1_checkpoint}", file=sys.stderr)
        sys.exit(1)
    if not args.manifest_train.exists():
        print(f"ERROR: Training manifest not found: {args.manifest_train}", file=sys.stderr)
        sys.exit(1)
    if args.manifest_val and not args.manifest_val.exists():
        print(f"ERROR: Validation manifest not found: {args.manifest_val}", file=sys.stderr)
        sys.exit(1)
    if not args.metadata.exists():
        print(f"ERROR: Metadata file not found: {args.metadata}", file=sys.stderr)
        sys.exit(1)

    # Generate run name if not specified
    if args.run_name is None:
        args.run_name = generate_run_name(args)

    # Create output directories
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve temperature learnable flag
    learnable_temp = args.learnable_temperature and not args.no_learnable_temperature
    freeze = args.freeze_encoder and not args.unfreeze_encoder

    # Create data module
    datamodule = CLIPDataModule(
        data_root=args.data_dir,
        manifest_train=args.manifest_train,
        manifest_val=args.manifest_val,
        metadata_path=args.metadata,
        source=args.source,
        id_column=args.id_column,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Setup datamodule to get dataset sizes
    datamodule.setup("fit")
    train_size = len(datamodule.train_dataset)
    val_size = len(datamodule.val_dataset) if datamodule.val_dataset else 0

    # Print configuration banner
    if not args.quiet:
        print_config_banner(args, datamodule, train_size, val_size)

    # Create model
    model = CLIPLightning(
        stage1_checkpoint=str(args.stage1_checkpoint),
        clip_embed_dim=args.clip_embed_dim,
        hidden_dim=args.hidden_dim,
        freeze_encoder=freeze,
        encoder_lr_mult=args.encoder_lr_mult,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        learnable_temperature=learnable_temp,
        lambda_clip=args.lambda_clip,
        lambda_kld=args.lambda_kld,
        text_encoder_name=args.text_encoder,
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
            model.image_encoder = torch.compile(model.image_encoder)
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
                mode="min",
                verbose=True,
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
            datamodule,
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


if __name__ == "__main__":
    main()
