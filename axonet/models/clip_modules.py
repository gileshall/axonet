"""CLIP-style projection heads and extended VAE model for contrastive learning."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .d3_swc_vae import SegVAE2D


class CLIPProjectionHead(nn.Module):
    """Projects embeddings to CLIP-compatible space with L2 normalization."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
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


class SegVAE2D_CLIP(nn.Module):
    """Extends SegVAE2D with CLIP projection head for contrastive learning.

    Wraps the base VAE encoder and adds a projection head that maps
    the bottleneck mu to CLIP embedding space.
    """

    def __init__(
        self,
        base_model: Optional[SegVAE2D] = None,
        clip_embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
        latent_channels: int = 128,  # Default if no base_model provided
    ):
        super().__init__()
        self.freeze_encoder = freeze_encoder

        if base_model is not None:
            self.base_model = base_model
            # Infer latent_channels from the mu layer output channels
            latent_channels = base_model.mu.out_channels
        else:
            # Create default base model (weights will be loaded from checkpoint)
            self.base_model = SegVAE2D(latent_channels=latent_channels)

        self.image_proj = CLIPProjectionHead(
            input_dim=latent_channels,
            hidden_dim=hidden_dim,
            output_dim=clip_embed_dim,
            dropout=dropout,
        )

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze encoder weights."""
        for name, param in self.base_model.named_parameters():
            if not name.startswith("head_"):
                param.requires_grad = False

    def _unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.base_model.parameters():
            param.requires_grad = True

    @classmethod
    def from_pretrained(
        cls,
        base_model: SegVAE2D,
        clip_embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ) -> "SegVAE2D_CLIP":
        """Create CLIP model from pretrained VAE."""
        return cls(
            base_model=base_model,
            clip_embed_dim=clip_embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode image to latent space.
        
        Returns:
            z: sampled latent
            mu: mean
            logvar: log variance
        """
        z, mu, logvar, _, _, _ = self.base_model.encode(x)
        return z, mu, logvar

    def encode_for_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image directly to CLIP embedding space.
        
        Args:
            x: Input image tensor (B, C, H, W)
            
        Returns:
            L2-normalized CLIP embedding (B, clip_embed_dim)
        """
        _, mu, _ = self.encode(x)
        mu_pooled = mu.mean(dim=(2, 3))
        return self.image_proj(mu_pooled)

    def get_image_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for encode_for_clip for cleaner API."""
        return self.encode_for_clip(x)

    def forward(
        self,
        x: torch.Tensor,
        return_vae_outputs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass returning CLIP embeddings and optional VAE outputs.
        
        Args:
            x: Input image tensor (B, C, H, W)
            return_vae_outputs: If True, also return full VAE outputs
            
        Returns:
            Dict with 'image_embed' and optionally VAE outputs
        """
        image_embed = self.encode_for_clip(x)
        out = {"image_embed": image_embed}
        
        if return_vae_outputs:
            z, mu, logvar = self.encode(x)
            out["mu"] = mu
            out["logvar"] = logvar
            out["z"] = z
            out["kld"] = self.base_model.kld(mu, logvar)
        
        return out

    def get_encoder_parameters(self):
        """Get encoder parameters (for differential learning rates)."""
        return self.base_model.parameters()

    def get_projection_parameters(self):
        """Get projection head parameters."""
        return self.image_proj.parameters()
