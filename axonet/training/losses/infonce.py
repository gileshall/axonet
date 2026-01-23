"""InfoNCE and contrastive loss functions for CLIP-style training."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning (CLIP-style).
    
    Computes symmetric cross-entropy between image and text embeddings.
    Both embeddings should be L2-normalized.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
        min_temperature: float = 0.01,
        max_temperature: float = 1.0,
    ):
        super().__init__()
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        
        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor([1.0 / temperature]).log()
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor([1.0 / temperature]).log()
            )

    @property
    def temperature(self) -> float:
        temp = 1.0 / self.log_temperature.exp()
        return temp.clamp(self.min_temperature, self.max_temperature).item()

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        return_accuracy: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute InfoNCE loss.
        
        Args:
            image_embeds: (B, D) L2-normalized image embeddings
            text_embeds: (B, D) L2-normalized text embeddings
            return_accuracy: Whether to compute retrieval accuracy
            
        Returns:
            loss: Scalar loss tensor
            logs: Dict with loss components and optional accuracy
        """
        batch_size = image_embeds.shape[0]
        device = image_embeds.device
        
        logit_scale = self.log_temperature.exp()
        logits = image_embeds @ text_embeds.T * logit_scale
        
        labels = torch.arange(batch_size, device=device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        logs = {
            "loss_i2t": loss_i2t.item(),
            "loss_t2i": loss_t2i.item(),
            "temperature": self.temperature,
        }
        
        if return_accuracy:
            with torch.no_grad():
                i2t_pred = logits.argmax(dim=1)
                t2i_pred = logits.argmax(dim=0)
                acc_i2t = (i2t_pred == labels).float().mean().item()
                acc_t2i = (t2i_pred == labels).float().mean().item()
                logs["acc_i2t"] = acc_i2t
                logs["acc_t2i"] = acc_t2i
        
        return loss, logs


class Stage2Loss(nn.Module):
    """Combined loss for Stage 2 CLIP fine-tuning.
    
    Combines InfoNCE contrastive loss with optional KL regularization
    to prevent the encoder from drifting too far from Stage 1.
    """

    def __init__(
        self,
        lambda_clip: float = 1.0,
        lambda_kld: float = 0.0,
        temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.lambda_clip = lambda_clip
        self.lambda_kld = lambda_kld
        
        self.infonce = InfoNCELoss(
            temperature=temperature,
            learnable_temperature=learnable_temperature,
        )

    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        kld: Optional[torch.Tensor] = None,
        return_accuracy: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute Stage 2 loss.
        
        Args:
            image_embeds: (B, D) L2-normalized image embeddings
            text_embeds: (B, D) L2-normalized text embeddings
            kld: Optional KL divergence from VAE encoder
            return_accuracy: Whether to compute retrieval accuracy
            
        Returns:
            total_loss: Scalar loss tensor
            logs: Dict with loss components
        """
        clip_loss, clip_logs = self.infonce(
            image_embeds, text_embeds, return_accuracy=return_accuracy
        )
        
        total = self.lambda_clip * clip_loss
        logs = {f"clip_{k}": v for k, v in clip_logs.items()}
        logs["clip_loss"] = clip_loss.item()
        
        if kld is not None and self.lambda_kld > 0:
            kld_term = self.lambda_kld * kld
            total = total + kld_term
            logs["kld"] = kld.item()
            logs["kld_weighted"] = kld_term.item()
        
        logs["total_loss"] = total.item()
        
        return total, logs
