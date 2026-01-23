"""PyTorch Lightning module for Stage 2 CLIP training."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.metadata_adapters import get_adapter
from ..models.clip_modules import SegVAE2D_CLIP
from ..models.d3_swc_vae import load_model
from ..models.text_encoders import ProjectedTextEncoder, SentenceTransformerEncoder
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
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        lambda_clip: float = 1.0,
        lambda_kld: float = 0.0,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.stage1_checkpoint = stage1_checkpoint
        self.clip_embed_dim = clip_embed_dim
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder
        self.encoder_lr_mult = encoder_lr_mult
        self.lr = lr
        
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
        
        text_encoder_base = SentenceTransformerEncoder(
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
        
        self.log(f"{stage}_loss", loss, prog_bar=True)
        for k, v in logs.items():
            self.log(f"{stage}_{k}", v)
        
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
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
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
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.manifest_train = Path(manifest_train)
        self.manifest_val = Path(manifest_val) if manifest_val else None
        self.metadata_path = Path(metadata_path)
        self.source = source
        self.id_column = id_column
        self.batch_size = batch_size
        self.num_workers = num_workers

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
            )
        else:
            self.val_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
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
            pin_memory=True,
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


def main():
    parser = argparse.ArgumentParser(description="Stage 2 CLIP training")
    parser.add_argument("--config", type=Path, help="YAML config file")
    
    parser.add_argument("--data-dir", type=Path, help="Data root directory")
    parser.add_argument("--manifest-train", type=Path, help="Training manifest")
    parser.add_argument("--manifest-val", type=Path, help="Validation manifest")
    parser.add_argument("--metadata", type=Path, help="Metadata file (JSON/CSV)")
    parser.add_argument("--source", default="allen", choices=["allen", "neuromorpho", "custom"])
    parser.add_argument("--id-column", default="cell_specimen_id")
    
    parser.add_argument("--stage1-checkpoint", type=Path, required=False)
    parser.add_argument("--clip-embed-dim", type=int, default=512)
    parser.add_argument("--freeze-encoder", action="store_true", default=True)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-clip", type=float, default=1.0)
    parser.add_argument("--lambda-kld", type=float, default=0.0)
    
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", default="32")
    
    parser.add_argument("--save-dir", type=Path, default="checkpoints/clip")
    parser.add_argument("--log-dir", type=Path, default="logs/clip")
    
    args = parser.parse_args()
    
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        train_from_config(config)
        return
    
    if not args.stage1_checkpoint:
        parser.error("--stage1-checkpoint is required when not using --config")
    if not args.data_dir:
        parser.error("--data-dir is required when not using --config")
    if not args.manifest_train:
        parser.error("--manifest-train is required when not using --config")
    if not args.metadata:
        parser.error("--metadata is required when not using --config")
    
    freeze = args.freeze_encoder and not args.unfreeze_encoder
    
    model = CLIPLightning(
        stage1_checkpoint=str(args.stage1_checkpoint),
        clip_embed_dim=args.clip_embed_dim,
        freeze_encoder=freeze,
        lr=args.lr,
        temperature=args.temperature,
        lambda_clip=args.lambda_clip,
        lambda_kld=args.lambda_kld,
    )
    
    datamodule = CLIPDataModule(
        data_root=args.data_dir,
        manifest_train=args.manifest_train,
        manifest_val=args.manifest_val,
        metadata_path=args.metadata,
        source=args.source,
        id_column=args.id_column,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    callbacks = [
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename="clip-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name="clip")
    
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=tb_logger,
    )
    
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
