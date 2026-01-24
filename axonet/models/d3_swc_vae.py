# model.py
# ------------------------------------------------------------
# Segmentation-VAE with *variational skip connections*.
# Adds per-level latent variables on the U-Net skips so that
# information cannot bypass the stochastic bottleneck.
#
# Key changes vs v1:
#  - VariationalSkip module on each skip (e2, e1, e0)
#  - Sum of KL terms: bottleneck + per-skip KLs
#  - Optional free-bits (free_nats) and KL-anneal factor (beta)
#  - Switchable skip_mode: "variational" | "raw" | "drop"
#  - Minor cleanups (weight init, diagnostics)
# ------------------------------------------------------------

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
        if skip_ch is None:
            skip_ch = in_ch // 2
        self.proj = nn.Conv2d(in_ch, skip_ch, kernel_size=1, bias=False)
        self.conv = ConvBlock(skip_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ----------------------------
# Variational skip module
# ----------------------------

class VariationalSkip(nn.Module):
    """
    Turns a deterministic skip feature S into a stochastic latent at the same
    spatial resolution. Prevents information leak around the global bottleneck.

    Flow: S -> pre (1x1 conv) -> (mu, logvar) -> z = mu + eps*exp(0.5*logvar)
          -> post (1x1 conv) to match decoder expected channels.

    If skip_mode == "raw", returns S (identity) and KL=0.
    If skip_mode == "drop", returns zeros_like(S) and KL=0.
    """
    def __init__(
        self,
        in_ch: int,
        latent_ch: int,
        out_ch: int,
        skip_mode: Literal["variational", "raw", "drop"] = "variational",
    ):
        super().__init__()
        self.skip_mode = skip_mode
        self.pre = nn.Conv2d(in_ch, latent_ch, kernel_size=1)
        self.mu = nn.Conv2d(latent_ch, latent_ch, kernel_size=1)
        self.logvar = nn.Conv2d(latent_ch, latent_ch, kernel_size=1)
        self.post = nn.Conv2d(latent_ch, out_ch, kernel_size=1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kld(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        k = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if reduction == "mean":
            return k.mean()
        elif reduction == "sum":
            return k.sum()
        elif reduction == "none":
            return k
        else:
            raise ValueError("reduction must be 'mean'|'sum'|'none'")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.skip_mode == "raw":
            mu = torch.zeros_like(s[:, :1])
            logvar = torch.zeros_like(s[:, :1])
            z = s
            out = s
            kld = torch.zeros((), device=s.device)
            return out, mu, logvar, kld
        if self.skip_mode == "drop":
            mu = torch.zeros_like(s[:, :1])
            logvar = torch.zeros_like(s[:, :1])
            z = torch.zeros_like(s)
            out = torch.zeros_like(s)
            kld = torch.zeros((), device=s.device)
            return out, mu, logvar, kld

        h = self.pre(s)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        out = self.post(z)
        kld = self.kld(mu, logvar)
        return out, mu, logvar, kld


# ----------------------------
# Seg-VAE
# ----------------------------

class SegVAE2D(nn.Module):
    """
    Multi-task UNet with a *global* variational bottleneck + *variational skips*.

    Args:
        in_channels:       input channels.
        base_channels:     UNet width.
        num_classes:       segmentation classes.
        latent_channels:   channels at the global bottleneck.
        skip_latent_mult:  multiplicative factor vs skip width for per-skip latent size.
        kld_weight:        global weight for KL.
        skip_mode:         'variational' | 'raw' | 'drop'.
        free_nats:         free-bits (nats) applied to KL terms (per mean reduction).
        beta:              KL anneal multiplier applied to *all* KL terms.
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_classes: int = 4,
        latent_channels: int = 128,
        skip_latent_mult: float = 0.5,
        kld_weight: float = 0.1,
        skip_mode: Literal["variational", "raw", "drop"] = "variational",
        free_nats: float = 0.0,
        beta: float = 1.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kld_weight = kld_weight
        self.skip_mode = skip_mode
        self.free_nats = free_nats
        self.beta = beta

        ch1, ch2, ch3, ch4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # Encoder
        self.enc0 = ConvBlock(in_channels, ch1)
        self.down1 = Down(ch1, ch2)
        self.down2 = Down(ch2, ch3)
        self.down3 = Down(ch3, ch4)

        # Global bottleneck → variational parameters
        self.bottleneck = ConvBlock(ch4, ch4)
        self.mu = nn.Conv2d(ch4, latent_channels, kernel_size=1)
        self.logvar = nn.Conv2d(ch4, latent_channels, kernel_size=1)
        self.post_z = nn.Conv2d(latent_channels, ch4, kernel_size=1)

        # Variational skips at three resolutions: e2 (ch3), e1 (ch2), e0 (ch1)
        s3_lat = max(1, int(ch3 * skip_latent_mult))
        s2_lat = max(1, int(ch2 * skip_latent_mult))
        s1_lat = max(1, int(ch1 * skip_latent_mult))
        self.vskip2 = VariationalSkip(in_ch=ch3, latent_ch=s3_lat, out_ch=ch3, skip_mode=skip_mode)
        self.vskip1 = VariationalSkip(in_ch=ch2, latent_ch=s2_lat, out_ch=ch2, skip_mode=skip_mode)
        self.vskip0 = VariationalSkip(in_ch=ch1, latent_ch=s1_lat, out_ch=ch1, skip_mode=skip_mode)

        # Decoder (shared trunk for all heads)
        self.up3 = Up(ch4, ch3, skip_ch=ch3)
        self.up2 = Up(ch3, ch2, skip_ch=ch2)
        self.up1 = Up(ch2, ch1, skip_ch=ch1)
        self.dec_out = ConvBlock(ch1, ch1)

        # Heads
        self.head_seg = nn.Conv2d(ch1, num_classes, kernel_size=1)
        self.head_depth = nn.Conv2d(ch1, 1, kernel_size=1)

        self._init_weights()

    # ---- utilities ----
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kld(mu: torch.Tensor, logvar: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        if reduction == "mean":
            return kld.mean()
        elif reduction == "sum":
            return kld.sum()
        elif reduction == "none":
            return kld
        else:
            raise ValueError("reduction must be 'mean' | 'sum' | 'none'.")

    def _apply_freebits(self, kld_val: torch.Tensor) -> torch.Tensor:
        if self.free_nats > 0.0:
            return torch.clamp(kld_val, min=self.free_nats)
        return kld_val

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Initialize logvar layers to output ~0 (variance ~1) at start
        # This prevents KLD explosion from huge exp(logvar) terms
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -2.0)  # Start with small variance

        # Same for variational skip logvar layers
        for vskip in [self.vskip0, self.vskip1, self.vskip2]:
            nn.init.zeros_(vskip.logvar.weight)
            nn.init.constant_(vskip.logvar.bias, -2.0)

    # ---- forward ----
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e0 = self.enc0(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        b = self.bottleneck(e3)
        mu = self.mu(b)
        logvar = self.logvar(b)
        z = self.reparameterize(mu, logvar)
        z = self.post_z(z)
        return z, mu, logvar, e2, e1, e0

    def decode_shared(self, z: torch.Tensor, skip2: torch.Tensor, skip1: torch.Tensor, skip0: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logs: Dict[str, torch.Tensor] = {}

        v2, mu2, lv2, k2 = self.vskip2(skip2)
        v1, mu1, lv1, k1 = self.vskip1(skip1)
        v0, mu0, lv0, k0 = self.vskip0(skip0)

        d3 = self.up3(z, v2)
        d2 = self.up2(d3, v1)
        d1 = self.up1(d2, v0)
        y = self.dec_out(d1)

        logs.update({
            "mu2": mu2, "logvar2": lv2, "kld_skip2": k2,
            "mu1": mu1, "logvar1": lv1, "kld_skip1": k1,
            "mu0": mu0, "logvar0": lv0, "kld_skip0": k0,
        })
        return y, logs

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
              'seg_logits': (B, num_classes, H, W),
              'depth':      (B,1,H,W),
              'kld':        scalar KL term (beta*(freebits(KL_b) + sum KL_skips))*kld_weight,
              + optional diagnostics if return_latent=True
            }
        """
        z, mu, logvar, s2, s1, s0 = self.encode(x)

        shared, slog = self.decode_shared(z, s2, s1, s0)

        seg_logits = self.head_seg(shared)
        out: Dict[str, torch.Tensor] = {"seg_logits": seg_logits}
        out["depth"] = self.head_depth(shared)

        kld_b = self.kld(mu, logvar)
        kld_b = self._apply_freebits(kld_b)
        kld_s2 = self._apply_freebits(slog["kld_skip2"]) if isinstance(slog["kld_skip2"], torch.Tensor) else torch.tensor(0.0, device=x.device)
        kld_s1 = self._apply_freebits(slog["kld_skip1"]) if isinstance(slog["kld_skip1"], torch.Tensor) else torch.tensor(0.0, device=x.device)
        kld_s0 = self._apply_freebits(slog["kld_skip0"]) if isinstance(slog["kld_skip0"], torch.Tensor) else torch.tensor(0.0, device=x.device)

        kld_total = self.beta * (kld_b + kld_s2 + kld_s1 + kld_s0)
        out["kld_bottleneck"] = kld_b.detach()
        out["kld_skips"] = (kld_s2 + kld_s1 + kld_s0).detach()
        out["kld"] = kld_total * self.kld_weight

        if return_latent:
            out.update({
                "mu": mu, "logvar": logvar, "z": z, "feat": shared,
                "mu2": slog["mu2"], "logvar2": slog["logvar2"],
                "mu1": slog["mu1"], "logvar1": slog["logvar1"],
                "mu0": slog["mu0"], "logvar0": slog["logvar0"],
            })

        return out


# ----------------------------
# Convenience loss wrapper
# ----------------------------

class MultiTaskLoss(nn.Module):
    """
    Compose the standard losses used for training.

    Handles multiple KL divergence terms from variational skip connections.
    Automatically sums all KL terms (from 'kld' and 'kld_*' keys in outputs).

    Use:
        criterion = MultiTaskLoss(lambda_seg=1.0, lambda_depth=1.0)
        outputs = model(x)
        loss, logs = criterion(outputs, targets)
    """
    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_depth: float = 1.0,
        dice_smooth: float = 1.0,
        depth_huber_delta: float = 0.1,
        ignore_index: int = -100
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_depth = lambda_depth
        self.dice_smooth = dice_smooth
        self.depth_huber_delta = depth_huber_delta
        self.ignore_index = ignore_index

    @staticmethod
    def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
        """
        Multi-class soft dice computed from logits against integer labels.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_1h = F.one_hot(target.clamp_min(0), num_classes).permute(0, 3, 1, 2).float()

        if (target == -100).any():
            mask = (target != -100).float().unsqueeze(1)
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
                'valid_depth_mask' → optional (B,1,H,W) where 1 means supervised pixel

        Returns:
            total_loss, logs (scalars)
        """
        logs = {}
        total = outputs.get("kld", torch.tensor(0.0, device=outputs["seg_logits"].device))
        logs["kld"] = float(total.detach())
        if "kld_bottleneck" in outputs:
            logs["kld_bottleneck"] = float(outputs["kld_bottleneck"])
        if "kld_skips" in outputs:
            logs["kld_skips"] = float(outputs["kld_skips"])

        if "seg" in targets and self.lambda_seg > 0:
            seg_ce = F.cross_entropy(outputs["seg_logits"], targets["seg"], ignore_index=self.ignore_index)
            seg_dice = self.soft_dice_loss(outputs["seg_logits"], targets["seg"], smooth=self.dice_smooth)
            seg_loss = seg_ce + seg_dice
            total = total + self.lambda_seg * seg_loss
            logs["seg_ce"] = float(seg_ce.detach())
            logs["seg_dice"] = float(seg_dice.detach())

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

        return total, logs


# ----------------------------
# Factory
# ----------------------------

def build_model(
    in_channels: int = 1,
    base_channels: int = 64,
    num_classes: int = 4,
    latent_channels: int = 128,
    skip_latent_mult: float = 0.5,
    kld_weight: float = 0.1,
    skip_mode: Literal["variational", "raw", "drop"] = "variational",
    free_nats: float = 0.0,
    beta: float = 1.0,
) -> SegVAE2D:
    """
    Helper to create a default SegVAE2D consistent with the design doc.
    """
    return SegVAE2D(
        in_channels=in_channels,
        base_channels=base_channels,
        num_classes=num_classes,
        latent_channels=latent_channels,
        skip_latent_mult=skip_latent_mult,
        kld_weight=kld_weight,
        skip_mode=skip_mode,
        free_nats=free_nats,
        beta=beta,
    )


def load_model(checkpoint_path: Path, device: str, embedding_only: bool = False, **model_kwargs) -> SegVAE2D:
    """Load trained model from checkpoint.

    Auto-detects model configuration from checkpoint state dict if not provided.
    Handles both PyTorch Lightning checkpoints (with 'state_dict' key and 'model.' prefix)
    and direct state dict checkpoints.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device string
        embedding_only: If True, skip loading decoder head weights (segmentation, depth)
                       for optimization when only extracting embeddings
        **model_kwargs: Model initialization arguments (will be overridden by checkpoint if present)

    Returns:
        Loaded SegVAE2D model
    """
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        logger.debug("Using 'state_dict' from checkpoint")
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.debug("Using 'model_state_dict' from checkpoint")
    else:
        state_dict = checkpoint
        logger.debug("Using checkpoint directly as state_dict")

    if not state_dict:
        raise ValueError(f"Checkpoint {checkpoint_path} contains no state_dict")

    if len(state_dict) == 0:
        raise ValueError(f"Checkpoint {checkpoint_path} state_dict is empty")

    logger.debug(f"State dict has {len(state_dict)} keys, first few: {list(state_dict.keys())[:5]}")

    if any(key.startswith("model.") for key in state_dict.keys()):
        logger.debug("Stripping 'model.' prefix from state dict keys")
        state_dict = {key[6:] if key.startswith("model.") else key: value for key, value in state_dict.items()}

    if not any(key.startswith("enc0.") or key.startswith("head_") for key in state_dict.keys()):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain valid model weights. "
                        f"State dict keys: {list(state_dict.keys())[:10]}...")

    # Filter out legacy head_recon weights from old checkpoints
    if "head_recon.weight" in state_dict or "head_recon.bias" in state_dict:
        logger.info("Filtering out legacy head_recon weights from checkpoint")
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head_recon.")}

    if "latent_channels" not in model_kwargs:
        if "post_z.0.weight" in state_dict:
            latent_channels = state_dict["post_z.0.weight"].shape[0]
            model_kwargs["latent_channels"] = latent_channels

    if "num_classes" not in model_kwargs:
        if "head_seg.weight" in state_dict:
            num_classes = state_dict["head_seg.weight"].shape[0]
            model_kwargs["num_classes"] = num_classes

    if "in_channels" not in model_kwargs:
        if "enc0.block.0.weight" in state_dict:
            in_channels = state_dict["enc0.block.0.weight"].shape[1]
            model_kwargs["in_channels"] = in_channels

    if embedding_only:
        logger.info("Embedding-only mode: skipping decoder head weights")
        filtered_state_dict = {}
        head_keys = ["head_seg", "head_depth"]
        for key, value in state_dict.items():
            if not any(key.startswith(head_prefix + ".") for head_prefix in head_keys):
                filtered_state_dict[key] = value
        state_dict = filtered_state_dict
        logger.debug(f"Filtered state dict: {len(state_dict)} keys (removed head weights)")

    logger.info(f"Model configuration: {model_kwargs}")

    model = SegVAE2D(**model_kwargs)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        if embedding_only:
            missing_head_keys = [k for k in missing_keys if any(k.startswith(h) for h in ["head_seg", "head_depth"])]
            if len(missing_head_keys) == len(missing_keys):
                logger.debug(f"All missing keys are head weights (expected in embedding-only mode): {len(missing_keys)}")
            else:
                critical_missing = [k for k in missing_keys if k not in missing_head_keys]
                logger.error(f"FATAL: Missing {len(critical_missing)} critical keys. First 10: {critical_missing[:10]}")
                raise ValueError(f"Checkpoint {checkpoint_path} is missing critical model weights. "
                                f"Missing keys: {critical_missing[:10]}... (total: {len(critical_missing)})")
        else:
            logger.error(f"FATAL: Missing {len(missing_keys)} keys in checkpoint. First 10: {missing_keys[:10]}")
            raise ValueError(f"Checkpoint {checkpoint_path} is missing critical model weights. "
                            f"Missing keys: {missing_keys[:10]}... (total: {len(missing_keys)})")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint (ignored): {unexpected_keys[:5]}... (total: {len(unexpected_keys)})")

    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully on {device}" + (" (embedding-only mode)" if embedding_only else ""))

    return model


# ----------------------------
# Quick self-test
# ----------------------------

if __name__ == "__main__":
    B, C, H, W = 2, 1, 256, 256
    x = torch.randn(B, C, H, W)

    model = build_model(
        in_channels=C, base_channels=32, num_classes=4, latent_channels=64,
        skip_latent_mult=0.5, skip_mode="variational", free_nats=0.0, beta=1.0,
    )
    out = model(x, return_latent=True)

    print("seg_logits:", tuple(out["seg_logits"].shape))
    print("depth:", tuple(out["depth"].shape))
    print("kld:", float(out["kld"]))
    print("kld_bottleneck:", float(out.get("kld_bottleneck", torch.tensor(0.0))))
    print("kld_skips:", float(out.get("kld_skips", torch.tensor(0.0))))
