#!/usr/bin/env python3
"""Entrypoint for model training in cloud environments."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_dataset(
    storage,
    remote_prefix: str,
    local_dir: Path,
    manifest_name: str = "manifest.jsonl",
):
    """Download training dataset from remote storage."""
    logger.info(f"Downloading dataset from {remote_prefix}")
    
    local_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_remote = f"{remote_prefix.rstrip('/')}/{manifest_name}"
    manifest_local = local_dir / manifest_name
    storage.download(manifest_remote, manifest_local)
    
    for f in storage.list_files(remote_prefix):
        if f.endswith(".png") or f.endswith(".npz"):
            rel = f.split(remote_prefix.rstrip("/"))[-1].lstrip("/")
            local = local_dir / rel
            storage.download(f, local)
    
    return manifest_local


def download_checkpoint(storage, remote_path: str, local_dir: Path) -> Path:
    """Download a checkpoint from remote storage."""
    logger.info(f"Downloading checkpoint from {remote_path}")
    local_path = local_dir / "checkpoint.ckpt"
    storage.download(remote_path, local_path)
    return local_path


def upload_checkpoints(storage, local_dir: Path, remote_prefix: str):
    """Upload training checkpoints to remote storage."""
    logger.info(f"Uploading checkpoints to {remote_prefix}")
    
    for ckpt in local_dir.rglob("*.ckpt"):
        rel = ckpt.relative_to(local_dir)
        remote = f"{remote_prefix.rstrip('/')}/{rel}"
        storage.upload(ckpt, remote)
    
    for log_dir in ["lightning_logs", "logs", "tensorboard"]:
        log_path = local_dir / log_dir
        if log_path.exists():
            storage.sync_up(log_path, f"{remote_prefix.rstrip('/')}/{log_dir}")


def train_stage1(
    config_path: Optional[Path],
    data_dir: Path,
    manifest: str,
    output_dir: Path,
    **kwargs,
):
    """Run Stage 1 VAE training."""
    from axonet.training.trainer import main as trainer_main
    import sys
    
    args = [
        "--data-dir", str(data_dir),
        "--manifest-train", manifest,
        "--save-dir", str(output_dir / "checkpoints"),
        "--log-dir", str(output_dir / "logs"),
    ]
    
    if config_path:
        args = ["--config", str(config_path)] + args
    
    for k, v in kwargs.items():
        if v is not None:
            args.extend([f"--{k.replace('_', '-')}", str(v)])
    
    sys.argv = ["trainer"] + args
    trainer_main()


def train_stage2(
    config_path: Optional[Path],
    data_dir: Path,
    manifest: str,
    metadata: str,
    stage1_checkpoint: Path,
    output_dir: Path,
    **kwargs,
):
    """Run Stage 2 CLIP training."""
    from axonet.training.clip_trainer import main as clip_main
    import sys
    
    args = [
        "--data-dir", str(data_dir),
        "--manifest-train", manifest,
        "--metadata", str(data_dir / metadata),
        "--stage1-checkpoint", str(stage1_checkpoint),
        "--save-dir", str(output_dir / "checkpoints"),
        "--log-dir", str(output_dir / "logs"),
    ]
    
    if config_path:
        args = ["--config", str(config_path)] + args
    
    for k, v in kwargs.items():
        if v is not None:
            args.extend([f"--{k.replace('_', '-')}", str(v)])
    
    sys.argv = ["clip_trainer"] + args
    clip_main()


def main():
    parser = argparse.ArgumentParser(description="Train axonet models")
    parser.add_argument("--config", type=Path, help="Config YAML file")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1, help="Training stage")
    
    parser.add_argument("--data-dir", required=True, help="Dataset location (local or gs://)")
    parser.add_argument("--manifest", default="manifest.jsonl")
    parser.add_argument("--metadata", default="metadata.jsonl", help="Metadata for Stage 2")
    parser.add_argument("--output", required=True, help="Output location (local or gs://)")
    
    parser.add_argument("--stage1-checkpoint", help="Stage 1 checkpoint for Stage 2 (local or gs://)")
    
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--precision", default="32")
    
    parser.add_argument("--provider", default="local", choices=["local", "google"])
    parser.add_argument("--local-dir", default="/tmp/axonet_train")
    
    args = parser.parse_args()
    
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    use_cloud = args.data_dir.startswith("gs://") or args.output.startswith("gs://")
    
    storage = None
    if use_cloud:
        from axonet.cloud import get_provider
        provider = get_provider(args.provider)
        storage = provider.storage
    
    if args.data_dir.startswith("gs://"):
        data_dir = local_dir / "data"
        download_dataset(storage, args.data_dir, data_dir, args.manifest)
    else:
        data_dir = Path(args.data_dir)
    
    if args.output.startswith("gs://"):
        output_dir = local_dir / "output"
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stage1_ckpt = None
    if args.stage == 2:
        if not args.stage1_checkpoint:
            parser.error("--stage1-checkpoint required for Stage 2")
        
        if args.stage1_checkpoint.startswith("gs://"):
            stage1_ckpt = download_checkpoint(storage, args.stage1_checkpoint, local_dir)
        else:
            stage1_ckpt = Path(args.stage1_checkpoint)
    
    train_kwargs = {}
    if args.batch_size:
        train_kwargs["batch_size"] = args.batch_size
    if args.lr:
        train_kwargs["lr"] = args.lr
    if args.max_epochs:
        train_kwargs["max_epochs"] = args.max_epochs
    if args.precision:
        train_kwargs["precision"] = args.precision
    
    logger.info(f"Starting Stage {args.stage} training")
    
    if args.stage == 1:
        train_stage1(
            config_path=args.config,
            data_dir=data_dir,
            manifest=args.manifest,
            output_dir=output_dir,
            **train_kwargs,
        )
    else:
        train_stage2(
            config_path=args.config,
            data_dir=data_dir,
            manifest=args.manifest,
            metadata=args.metadata,
            stage1_checkpoint=stage1_ckpt,
            output_dir=output_dir,
            **train_kwargs,
        )
    
    if storage and args.output.startswith("gs://"):
        upload_checkpoints(storage, output_dir, args.output)
    
    logger.info("Training complete")


if __name__ == "__main__":
    main()
