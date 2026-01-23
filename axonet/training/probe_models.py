"""Probe model architectures for linear probing."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    """Simple linear probe for classification or regression."""
    
    def __init__(self, embedding_dim: int, num_classes: int = None, regression: bool = False):
        """Initialize linear probe.
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of classes (for classification)
            regression: If True, output single value (for regression)
        """
        super().__init__()
        if regression:
            self.head = nn.Linear(embedding_dim, 1)
        else:
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification")
            self.head = nn.Linear(embedding_dim, num_classes)
        self.regression = regression
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, embedding_dim) input embeddings
        
        Returns:
            (B, num_classes) logits for classification or (B, 1) for regression
        """
        return self.head(x)


class LinearProbeWithNorm(nn.Module):
    """Linear probe with LayerNorm."""
    
    def __init__(self, embedding_dim: int, num_classes: int = None, regression: bool = False):
        """Initialize linear probe with LayerNorm.
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of classes (for classification)
            regression: If True, output single value (for regression)
        """
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        if regression:
            self.head = nn.Linear(embedding_dim, 1)
        else:
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification")
            self.head = nn.Linear(embedding_dim, num_classes)
        self.regression = regression
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, embedding_dim) input embeddings
        
        Returns:
            (B, num_classes) logits for classification or (B, 1) for regression
        """
        x = self.norm(x)
        return self.head(x)


class MLPProbe(nn.Module):
    """Tiny MLP probe (not strictly linear probing, but useful for comparison)."""
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int = None,
        regression: bool = False,
    ):
        """Initialize MLP probe.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension
            num_classes: Number of classes (for classification)
            regression: If True, output single value (for regression)
        """
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        if regression:
            self.fc2 = nn.Linear(hidden_dim, 1)
        else:
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification")
            self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.regression = regression
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: (B, embedding_dim) input embeddings
        
        Returns:
            (B, num_classes) logits for classification or (B, 1) for regression
        """
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


class MultiTaskProbe(nn.Module):
    """Multi-task probe with multiple heads for different tasks."""
    
    def __init__(
        self,
        embedding_dim: int,
        task_configs: Dict[str, Dict],
        probe_type: str = "linear",
        use_norm: bool = False,
        hidden_dim: Optional[int] = None,
    ):
        """Initialize multi-task probe.
        
        Args:
            embedding_dim: Dimension of input embeddings
            task_configs: Dict mapping task names to config dicts with:
                - "num_classes": for classification tasks
                - "regression": True for regression tasks
            probe_type: "linear", "linear_norm", or "mlp"
            use_norm: If True, add LayerNorm before heads (for linear probes)
            hidden_dim: Hidden dimension for MLP probe
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.task_configs = task_configs
        self.probe_type = probe_type
        self.use_norm = use_norm
        
        if use_norm and probe_type == "linear":
            self.norm = nn.LayerNorm(embedding_dim)
        else:
            self.norm = None
        
        self.heads = nn.ModuleDict()
        
        for task_name, config in task_configs.items():
            regression = config.get("regression", False)
            num_classes = config.get("num_classes", None)
            
            if probe_type == "linear":
                if use_norm:
                    self.heads[task_name] = LinearProbeWithNorm(
                        embedding_dim, num_classes, regression
                    )
                else:
                    self.heads[task_name] = LinearProbe(
                        embedding_dim, num_classes, regression
                    )
            elif probe_type == "mlp":
                if hidden_dim is None:
                    hidden_dim = embedding_dim
                self.heads[task_name] = MLPProbe(
                    embedding_dim, hidden_dim, num_classes, regression
                )
            else:
                raise ValueError(f"Unknown probe_type: {probe_type}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: (B, embedding_dim) input embeddings
        
        Returns:
            Dict mapping task names to outputs (logits or regression values)
        """
        if self.norm is not None:
            x = self.norm(x)
        
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(x)
        
        return outputs


def build_probe(
    embedding_dim: int,
    task_configs: Dict[str, Dict],
    probe_type: str = "linear",
    use_norm: bool = False,
    hidden_dim: Optional[int] = None,
) -> MultiTaskProbe:
    """Build a multi-task probe model.
    
    Args:
        embedding_dim: Dimension of input embeddings
        task_configs: Dict mapping task names to config dicts
        probe_type: "linear", "linear_norm", or "mlp"
        use_norm: If True, add LayerNorm (for linear probes)
        hidden_dim: Hidden dimension for MLP probe
    
    Returns:
        MultiTaskProbe model
    """
    return MultiTaskProbe(
        embedding_dim=embedding_dim,
        task_configs=task_configs,
        probe_type=probe_type,
        use_norm=use_norm,
        hidden_dim=hidden_dim,
    )

