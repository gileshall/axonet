"""Text encoders for CLIP-style contrastive learning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import torch
import torch.nn as nn


class TextEncoderBase(ABC):
    """Abstract base class for text encoders."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Output embedding dimension."""
        ...

    @abstractmethod
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode text strings to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of shape (len(texts), embed_dim)
        """
        ...

    def __call__(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)


class SentenceTransformerEncoder(TextEncoderBase):
    """Text encoder using sentence-transformers library.
    
    Recommended model: sentence-transformers/all-MiniLM-L6-v2
    - 22M parameters
    - 384-dim output
    - Good performance on semantic similarity
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        from sentence_transformers import SentenceTransformer
        
        self.model_name = model_name
        self.normalize = normalize
        self._device = device
        self.model = SentenceTransformer(model_name, device=device)
        self._embed_dim = self.model.get_sentence_embedding_dimension()

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: Union[str, torch.device]) -> "SentenceTransformerEncoder":
        self.model = self.model.to(device)
        self._device = str(device)
        return self

    def encode(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=self.normalize,
        )
        return embeddings


class TextProjectionHead(nn.Module):
    """Projects text encoder output to CLIP embedding space."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return nn.functional.normalize(x, p=2, dim=-1)


class ProjectedTextEncoder(nn.Module):
    """Combines text encoder with learnable projection head."""

    def __init__(
        self,
        encoder: TextEncoderBase,
        output_dim: int = 512,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.projection = TextProjectionHead(
            input_dim=encoder.embed_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    @property
    def embed_dim(self) -> int:
        return self.projection.net[-2].normalized_shape[0]

    def encode(self, texts: List[str]) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze_encoder):
            text_embeds = self.encoder.encode(texts)
        return self.projection(text_embeds)

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)

    def get_text_embedding(self, texts: List[str]) -> torch.Tensor:
        """Alias for encode."""
        return self.encode(texts)
