"""YAML-based experiment configuration for reproducible training."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any

import yaml


@dataclass
class DataConfig:
    source: Literal["allen", "neuromorpho", "custom"] = "custom"
    root: str = ""
    manifest: str = "manifest.jsonl"
    metadata: Optional[str] = None
    id_column: str = "cell_specimen_id"
    base_url: Optional[str] = None
    manifest_url: Optional[str] = None


@dataclass
class RenderingConfig:
    width: int = 512
    height: int = 512
    views_per_neuron: int = 24
    projection: Literal["ortho", "perspective"] = "ortho"
    cache_dir: Optional[str] = None


@dataclass
class ModelConfig:
    base_channels: int = 64
    latent_channels: int = 128
    num_classes: int = 6
    skip_mode: Literal["variational", "raw", "drop"] = "variational"


@dataclass
class CLIPConfig:
    enabled: bool = False
    embed_dim: int = 512
    hidden_dim: int = 256
    freeze_encoder: bool = True
    encoder_lr_mult: float = 0.1
    temperature: float = 0.07
    learnable_temperature: bool = True
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    stage1_checkpoint: Optional[str] = None
    lambda_clip: float = 1.0
    lambda_kld: float = 0.0


@dataclass
class TrainingConfig:
    stage: Literal[1, 2] = 1
    batch_size: int = 8
    lr: float = 1e-4
    max_epochs: int = 100
    kld_weight: float = 0.1
    beta: float = 1.0
    free_nats: float = 0.0
    clip: CLIPConfig = field(default_factory=CLIPConfig)
    gradient_clip_val: Optional[float] = 1.0
    precision: str = "32"
    num_workers: int = 4


@dataclass
class OutputConfig:
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_top_k: int = 3
    monitor: str = "val_loss"


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    description: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        data = DataConfig(**d.get("data", {}))
        rendering = RenderingConfig(**d.get("rendering", {}))
        model = ModelConfig(**d.get("model", {}))
        
        training_dict = d.get("training", {})
        clip_dict = training_dict.pop("clip", {})
        clip = CLIPConfig(**clip_dict)
        training = TrainingConfig(**training_dict, clip=clip)
        
        output = OutputConfig(**d.get("output", {}))
        
        return cls(
            name=d.get("name", d.get("experiment", {}).get("name", "experiment")),
            description=d.get("description", d.get("experiment", {}).get("description", "")),
            data=data,
            rendering=rendering,
            model=model,
            training=training,
            output=output,
        )

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": {"name": self.name, "description": self.description},
            "data": asdict(self.data),
            "rendering": asdict(self.rendering),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "output": asdict(self.output),
        }
