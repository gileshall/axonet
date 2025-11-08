# model.py
# ------------------------------------------------------------
# Segmentation-VAE for SWC neuron renderings (2-D).
# One encoder with a variational bottleneck; a shared UNet
# decoder; and three lightweight heads for:
#   (1) part segmentation (bg, soma, dendrite, axon),
#   (2) monocular depth regression (0..1 target),
#   (3) image reconstruction (for inpainting / jigsaw).
#
# The forward pass returns a dict with logits/predictions and
# the KL term so you can compose the multi-task loss outside.
#
# Author: ChatGPT (BICAN Grant)
# ------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Building blocks
# ----------------------------

def conv3x3(in_ch: int, out_ch: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)


class ConvBlock(nn.Module):
    """
    Two 3x3 convs with GroupNorm + SiLU.
    Safer than BatchNorm for small batch sizes.
    """
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g1 = min(groups, out_ch)
        g2 = min(groups, out_ch)
        self.block = nn.Sequential(
            conv3x3(in_ch, out_ch),
            nn.GroupNorm(g1, out_ch),
            nn.SiLU(inplace=True),
            conv3x3(out_ch, out_ch),
            nn.GroupNorm(g2, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Downsampling with strided conv followed by a ConvBlock."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(inplace=True),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    """Upsampling with bilinear + 1x1 conv, concatenate skip, then ConvBlock."""
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int | None = None):
        super().__init__()
        # Project input to match skip connection channels, then concatenate
        # If skip_ch not provided, assume input and skip have same channels (in_ch // 2 each)
        if skip_ch is None:
            skip_ch = in_ch // 2
        self.proj = nn.Conv2d(in_ch, skip_ch, kernel_size=1, bias=False)
        self.conv = ConvBlock(skip_ch + skip_ch, out_ch)  # concat: skip + projected input

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ----------------------------
# Seg-VAE
# ----------------------------

class SegVAE2D(nn.Module):
    """
    Multi-task UNet with a variational bottleneck.

    Args:
        in_channels:  number of input channels (1 for grayscale, 3 for RGB).
        base_channels: width multiplier for the UNet (64 is a good default).
        num_classes:   number of segmentation classes (4→ bg/soma/dendrite/axon).
        latent_channels: channels at the bottleneck used for VAE sampling.
        use_depth:     if True, expose a depth head (1 channel).
        use_recon:     if True, expose a reconstruction head with in_channels.
        kld_weight:    convenience default for weighting KL at training time.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 4,
        latent_channels: int = 128,
        use_depth: bool = True,
        use_recon: bool = True,
        kld_weight: float = 1.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_depth = use_depth
        self.use_recon = use_recon
        self.kld_weight = kld_weight

        ch1, ch2, ch3, ch4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # Encoder
        self.enc0 = ConvBlock(in_channels, ch1)
        self.down1 = Down(ch1, ch2)
        self.down2 = Down(ch2, ch3)
        self.down3 = Down(ch3, ch4)

        # Bottleneck → variational parameters
        self.bottleneck = ConvBlock(ch4, ch4)
        self.mu = nn.Conv2d(ch4, latent_channels, kernel_size=1)
        self.logvar = nn.Conv2d(ch4, latent_channels, kernel_size=1)

        # Project sampled z back to decoder width
        self.post_z = nn.Conv2d(latent_channels, ch4, kernel_size=1)

        # Decoder (shared trunk for all heads)
        # up3: projects ch4 -> ch3, concat with skip2 (ch3) -> ch3+ch3 -> ch3
        self.up3 = Up(ch4, ch3, skip_ch=ch3)
        # up2: projects ch3 -> ch2, concat with skip1 (ch2) -> ch2+ch2 -> ch2
        self.up2 = Up(ch3, ch2, skip_ch=ch2)
        # up1: projects ch2 -> ch1, concat with skip0 (ch1) -> ch1+ch1 -> ch1
        self.up1 = Up(ch2, ch1, skip_ch=ch1)
        self.dec_out = ConvBlock(ch1, ch1)

        # Heads
        self.head_seg = nn.Conv2d(ch1, num_classes, kernel_size=1)
        self.head_depth = nn.Conv2d(ch1, 1, kernel_size=1) if use_depth else None
        self.head_recon = nn.Conv2d(ch1, in_channels, kernel_size=1) if use_recon else None

        self._init_weights()

    # ---- utilities ----
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample with reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        KL(q(z|x) || N(0,1)) for diagonal Gaussians.
        Computed per-element then reduced across all dims.
        """
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if reduction == "mean":
            return kld.mean()
        elif reduction == "sum":
            return kld.sum()
        elif reduction == "none":
            return kld
        else:
            raise ValueError("reduction must be 'mean' | 'sum' | 'none'.")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # ---- forward ----
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e0 = self.enc0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        b = self.bottleneck(e3)
        mu = self.mu(b)
        logvar = self.logvar(b)
        z = self.reparameterize(mu, logvar)
        z = self.post_z(z)
        return z, mu, logvar, e2, e1  # keep skips

    def decode_shared(self, z: torch.Tensor, skip2: torch.Tensor, skip1: torch.Tensor, skip0: torch.Tensor) -> torch.Tensor:
        d3 = self.up3(z, skip2)   # -> ch3
        d2 = self.up2(d3, skip1)  # -> ch2
        d1 = self.up1(d2, skip0)  # -> ch1
        y = self.dec_out(d1)
        return y

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
              'seg_logits': (B, num_classes, H, W),
              'depth':      (B,1,H,W)        [if use_depth],
              'recon':      (B,C,H,W)        [if use_recon],
              'kld':        scalar KL term,
              'mu','logvar','z','feat': optional for diagnostics
            }
        """
        # Encoder
        e0 = self.enc0(x)
        z, mu, logvar, skip2, skip1 = self.encode(x)
        # Shared decoder trunk
        shared = self.decode_shared(z, skip2, skip1, e0)

        # Heads
        seg_logits = self.head_seg(shared)
        out: Dict[str, torch.Tensor] = {"seg_logits": seg_logits}

        if self.use_depth:
            out["depth"] = self.head_depth(shared)
        if self.use_recon:
            out["recon"] = self.head_recon(shared)

        # KL
        out["kld"] = self.kl_divergence(mu, logvar) * self.kld_weight

        if return_latent:
            out["mu"] = mu
            out["logvar"] = logvar
            out["z"] = z
            out["feat"] = shared

        return out


# ----------------------------
# Convenience loss wrapper
# ----------------------------

class MultiTaskLoss(nn.Module):
    """
    Compose the standard losses used for training.

    Use:
        criterion = MultiTaskLoss(lambda_seg=1.0, lambda_depth=1.0, lambda_recon=1.0)
        outputs = model(x)
        loss, logs = criterion(outputs, targets)
    """
    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_depth: float = 1.0,
        lambda_recon: float = 1.0,
        dice_smooth: float = 1.0,
        depth_huber_delta: float = 0.1,
        recon_loss: str = "l1",   # 'l1' or 'l2'
        ignore_index: int = -100  # optional for seg labels
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_depth = lambda_depth
        self.lambda_recon = lambda_recon
        self.dice_smooth = dice_smooth
        self.depth_huber_delta = depth_huber_delta
        self.recon_loss = recon_loss
        self.ignore_index = ignore_index

    @staticmethod
    def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
        """
        Multi-class soft dice computed from logits against integer labels.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target.clamp_min(0), num_classes).permute(0, 3, 1, 2).float()

        # Mask out ignore_index
        if (target == -100).any():
            mask = (target != -100).float().unsqueeze(1)  # (B,1,H,W)
            probs = probs * mask
            target_1h = target_1h * mask

        dims = (0, 2, 3)
        intersect = (probs * target_1h).sum(dims)
        denom = probs.sum(dims) + target_1h.sum(dims)
        dice = (2 * intersect + smooth) / (denom + smooth + eps)
        return 1.0 - dice.mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: dict from the model forward()
            targets: dict with optional keys:
                'seg'   → LongTensor (B,H,W)
                'depth' → FloatTensor (B,1,H,W)
                'recon' → FloatTensor (B,C,H,W) (ground truth / original image)
                'valid_depth_mask' → optional (B,1,H,W) where 1 means supervised pixel
                'inpaint_mask' → optional (B,1,H,W); if provided, only compute recon loss on masked regions.

        Returns:
            total_loss, logs (scalars)
        """
        logs = {}
        total = outputs.get("kld", torch.tensor(0.0, device=outputs["seg_logits"].device))
        logs["kld"] = float(total.detach())

        # Segmentation loss
        if "seg" in targets and self.lambda_seg > 0:
            seg_ce = F.cross_entropy(outputs["seg_logits"], targets["seg"], ignore_index=self.ignore_index)
            seg_dice = self.soft_dice_loss(outputs["seg_logits"], targets["seg"], smooth=self.dice_smooth)
            seg_loss = seg_ce + seg_dice
            total = total + self.lambda_seg * seg_loss
            logs["seg_ce"] = float(seg_ce.detach())
            logs["seg_dice"] = float(seg_dice.detach())

        # Depth loss (Huber on valid foreground)
        if "depth" in targets and "depth" in outputs and self.lambda_depth > 0:
            pred = outputs["depth"]
            target = targets["depth"]
            if "valid_depth_mask" in targets:
                mask = targets["valid_depth_mask"].float()
                pred = pred * mask
                target = target * mask
                denom = mask.sum().clamp_min(1.0)
            else:
                denom = torch.tensor(pred.numel(), device=pred.device, dtype=pred.dtype)

            depth_loss = F.huber_loss(pred, target, delta=self.depth_huber_delta, reduction="sum") / denom
            total = total + self.lambda_depth * depth_loss
            logs["depth"] = float(depth_loss.detach())

        # Reconstruction loss (masked if inpaint mask provided)
        if "recon" in outputs and self.lambda_recon > 0:
            pred = outputs["recon"]
            target = targets.get("recon", None)
            if target is None:
                # default to the input image if not provided
                target = targets.get("input", None)
                if target is None:
                    raise ValueError("Reconstruction target missing: provide targets['recon'] or targets['input'].")

            if "inpaint_mask" in targets:
                mask = targets["inpaint_mask"].float()
                pred = pred * mask
                target = target * mask
                denom = mask.sum().clamp_min(1.0)
                reduction = "sum"
            else:
                denom = torch.tensor(pred.numel(), device=pred.device, dtype=pred.dtype)
                reduction = "sum"

            if self.recon_loss == "l2":
                rec = F.mse_loss(pred, target, reduction=reduction) / denom
            else:
                rec = F.l1_loss(pred, target, reduction=reduction) / denom

            total = total + self.lambda_recon * rec
            logs["recon"] = float(rec.detach())

        return total, logs


# ----------------------------
# Factory
# ----------------------------

def build_model(
    in_channels: int = 1,
    base_channels: int = 64,
    num_classes: int = 4,
    latent_channels: int = 128,
    use_depth: bool = True,
    use_recon: bool = True,
    kld_weight: float = 1.0,
) -> SegVAE2D:
    """
    Helper to create a default SegVAE2D consistent with the design doc.
    """
    return SegVAE2D(
        in_channels=in_channels,
        base_channels=base_channels,
        num_classes=num_classes,
        latent_channels=latent_channels,
        use_depth=use_depth,
        use_recon=use_recon,
        kld_weight=kld_weight,
    )


# ----------------------------
# Quick self-test
# ----------------------------

if __name__ == "__main__":
    B, C, H, W = 2, 1, 512, 512
    x = torch.randn(B, C, H, W)

    model = build_model(in_channels=C, base_channels=32, num_classes=4, latent_channels=64)
    out = model(x, return_latent=True)

    print("seg_logits:", tuple(out["seg_logits"].shape))
    if "depth" in out:
        print("depth:", tuple(out["depth"].shape))
    if "recon" in out:
        print("recon:", tuple(out["recon"].shape))
    print("kld:", float(out["kld"]))
