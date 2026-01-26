"""Text encoders for CLIP-style contrastive learning."""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

# Fix tokenizers parallelism issue on macOS - must be set before import
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class HashTextEncoder(TextEncoderBase):
    """Simple hash-based text encoder using feature hashing.

    This is a lightweight fallback that doesn't require any ML libraries.
    Uses n-gram hashing to create sparse-then-dense embeddings.

    Not as semantically rich as transformer models, but:
    - Zero dependencies beyond PyTorch
    - Very fast
    - Deterministic
    - Works well for structured text like neuron descriptions
    """

    def __init__(
        self,
        embed_dim: int = 384,
        n_gram_range: tuple = (1, 3),
        normalize: bool = True,
        device: Optional[str] = None,
    ):
        self._embed_dim = embed_dim
        self.n_gram_range = n_gram_range
        self.normalize = normalize
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def to(self, device: Union[str, torch.device]) -> "HashTextEncoder":
        self._device = str(device)
        return self

    def _get_ngrams(self, text: str) -> List[str]:
        """Extract word n-grams from text."""
        words = text.lower().split()
        ngrams = []
        for n in range(self.n_gram_range[0], self.n_gram_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i:i + n]))
        return ngrams if ngrams else [text.lower()]

    def _hash_to_index(self, ngram: str) -> int:
        """Hash an n-gram to an index."""
        h = hashlib.md5(ngram.encode()).hexdigest()
        return int(h, 16) % self._embed_dim

    def _hash_to_sign(self, ngram: str) -> int:
        """Hash an n-gram to a sign (+1 or -1)."""
        h = hashlib.sha1(ngram.encode()).hexdigest()
        return 1 if int(h, 16) % 2 == 0 else -1

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using feature hashing."""
        embeddings = torch.zeros(len(texts), self._embed_dim)

        for i, text in enumerate(texts):
            ngrams = self._get_ngrams(text)
            for ngram in ngrams:
                idx = self._hash_to_index(ngram)
                sign = self._hash_to_sign(ngram)
                embeddings[i, idx] += sign

        embeddings = embeddings.to(self._device)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


class TransformerTextEncoder(TextEncoderBase, nn.Module):
    """Text encoder using HuggingFace transformers.

    Uses mean pooling over token embeddings for sentence representation.

    Recommended models:
    - "bert-base-uncased" (110M params, 768-dim)
    - "distilbert-base-uncased" (66M params, 768-dim, faster)
    - "sentence-transformers/all-MiniLM-L6-v2" (22M params, 384-dim)
    - "prajjwal1/bert-tiny" (4M params, 128-dim, very fast)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: Optional[str] = None,
        normalize: bool = True,
        max_length: int = 128,
    ):
        nn.Module.__init__(self)
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.normalize = normalize
        self.max_length = max_length
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self._embed_dim = self.model.config.hidden_size

        if device:
            self.model = self.model.to(device)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: Union[str, torch.device]) -> "TransformerTextEncoder":
        self.model = self.model.to(device)
        self._device = str(device)
        return self

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling over token embeddings, weighted by attention mask."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings using mean pooling."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to model device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            model_output = self.model(**encoded)

        embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


# Alias for backwards compatibility
SentenceTransformerEncoder = TransformerTextEncoder


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
        return F.normalize(x, p=2, dim=-1)


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
