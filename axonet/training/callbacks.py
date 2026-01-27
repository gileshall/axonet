"""Validation image logging callback for Stage 1 VAE training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch_lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)

# Segmentation class palette (uint8 RGB)
# 0=Background(black), 1=Soma(red), 2=Axon(green),
# 3=BasalDendrite(blue), 4=ApicalDendrite(magenta), 5=Other(orange)
SEG_PALETTE = np.array(
    [
        [0, 0, 0],  # 0 Background
        [255, 0, 0],  # 1 Soma
        [0, 255, 0],  # 2 Axon
        [0, 0, 255],  # 3 Basal Dendrite
        [255, 0, 255],  # 4 Apical Dendrite
        [255, 165, 0],  # 5 Other
    ],
    dtype=np.uint8,
)

# Build viridis LUT once at import time (256 entries, uint8 RGB)
try:
    from matplotlib.cm import viridis as _viridis_cm

    _VIRIDIS_LUT = (_viridis_cm(np.arange(256) / 255.0)[:, :3] * 255).astype(np.uint8)
except ImportError:
    # Fallback: grayscale ramp
    _v = np.arange(256, dtype=np.uint8)
    _VIRIDIS_LUT = np.stack([_v, _v, _v], axis=1)


def _colorize_seg(seg: np.ndarray) -> np.ndarray:
    """Map class-ID mask (H,W) -> RGB (H,W,3) using SEG_PALETTE."""
    seg = np.clip(seg, 0, len(SEG_PALETTE) - 1)
    return SEG_PALETTE[seg]


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Map uint8 depth (H,W) [0..255] -> RGB (H,W,3) via viridis LUT."""
    return _VIRIDIS_LUT[depth]


def _gray_placeholder(size: int) -> np.ndarray:
    """Return a uniform gray (H,W,3) placeholder image."""
    return np.full((size, size, 3), 128, dtype=np.uint8)


class ValidationImageLogger(Callback):
    """Log a fixed panel of sample predictions each validation epoch.

    Panel layout per row: Input | Pred Seg | GT Seg | Pred Depth | GT Depth
    A header row with column labels is drawn at the top.
    """

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        num_samples: int = 3,
        image_size: int = 512,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.num_samples = num_samples
        self.image_size = image_size

        # Populated in setup()
        self._entries: List[dict] = []
        self._inputs: List[torch.Tensor] = []  # each (1,H,W) float32 [0,1]
        self._gt_segs: List[Optional[np.ndarray]] = []  # each (H,W) uint8
        self._gt_depths: List[Optional[np.ndarray]] = []  # each (H,W) uint8

    # ------------------------------------------------------------------
    # setup: load manifest, pre-load GT images
    # ------------------------------------------------------------------
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        from imageio.v2 import imread

        entries: List[dict] = []
        with open(self.manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
                if len(entries) >= self.num_samples:
                    break

        self._entries = entries
        n = len(entries)
        if n == 0:
            logger.warning("ValidationImageLogger: manifest is empty, callback disabled")
            return
        if n < self.num_samples:
            logger.warning(
                "ValidationImageLogger: only %d entries available (requested %d)",
                n,
                self.num_samples,
            )

        for entry in entries:
            self._load_entry(entry, imread)

        logger.info("ValidationImageLogger: loaded %d samples for visualization", len(self._inputs))

    def _load_entry(self, entry: dict, imread) -> None:
        """Load input, GT seg, and GT depth for a single manifest entry."""
        sz = self.image_size

        # --- input image ---
        if "mask_bw" in entry:
            input_path = self.data_root / entry["mask_bw"]
        elif "image" in entry:
            input_path = self.data_root / entry["image"]
        else:
            logger.warning("Entry missing 'mask_bw'/'image': %s", entry.get("swc", "?"))
            self._inputs.append(torch.zeros(1, sz, sz))
            self._gt_segs.append(None)
            self._gt_depths.append(None)
            return

        try:
            img = imread(input_path)
            if img.ndim == 3:
                img = np.mean(img, axis=2, dtype=np.float32) / 255.0
            else:
                img = img.astype(np.float32) / 255.0
            t = torch.from_numpy(img).unsqueeze(0)  # (1,H,W)
            if t.shape[1] != sz or t.shape[2] != sz:
                t = torch.nn.functional.interpolate(
                    t.unsqueeze(0), size=(sz, sz), mode="bilinear", align_corners=False
                ).squeeze(0)
            self._inputs.append(t)
        except Exception as e:
            logger.warning("Failed to load input %s: %s", input_path, e)
            self._inputs.append(torch.zeros(1, sz, sz))

        # --- GT segmentation mask ---
        self._gt_segs.append(self._load_gt_mask(entry, "mask", imread, Image.NEAREST))

        # --- GT depth map ---
        self._gt_depths.append(self._load_gt_mask(entry, "depth", imread, Image.BILINEAR))

    def _load_gt_mask(self, entry: dict, key: str, imread, resample) -> Optional[np.ndarray]:
        """Load a GT image (seg or depth) from *entry[key]*, resize if needed."""
        if key not in entry:
            return None
        path = self.data_root / entry[key]
        try:
            arr = imread(path)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if arr.shape[0] != self.image_size or arr.shape[1] != self.image_size:
                pil = Image.fromarray(arr)
                pil = pil.resize((self.image_size, self.image_size), resample)
                arr = np.array(pil)
            return arr.astype(np.uint8)
        except Exception as e:
            logger.warning("Failed to load GT %s from %s: %s", key, path, e)
            return None

    # ------------------------------------------------------------------
    # on_validation_epoch_end: run inference, assemble panel, log
    # ------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._inputs:
            return

        device = pl_module.device
        base_model = pl_module.model

        col_headers = ["Input", "Pred Seg", "GT Seg", "Pred Depth", "GT Depth"]
        n_cols = len(col_headers)
        cell = self.image_size
        pad = 4
        header_h = 24

        n_rows = len(self._inputs)
        panel_w = n_cols * cell + (n_cols + 1) * pad
        panel_h = header_h + n_rows * cell + (n_rows + 1) * pad

        panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
        draw = ImageDraw.Draw(panel)

        # Draw column headers (centered in each column)
        for c, header in enumerate(col_headers):
            x = pad + c * (cell + pad) + cell // 2
            # Use default font; approximate centering by offsetting half the text width
            bbox = draw.textbbox((0, 0), header)
            tw = bbox[2] - bbox[0]
            draw.text((x - tw // 2, 4), header, fill=(0, 0, 0))

        # Run inference on all samples
        with torch.no_grad():
            for row_idx in range(n_rows):
                cells_rgb = self._predict_row(row_idx, base_model, device)
                y = header_h + pad + row_idx * (cell + pad)
                for c, cell_img in enumerate(cells_rgb):
                    x = pad + c * (cell + pad)
                    panel.paste(Image.fromarray(cell_img), (x, y))

        self._log_panel(trainer, panel)

    def _predict_row(
        self, row_idx: int, base_model: torch.nn.Module, device: torch.device
    ) -> List[np.ndarray]:
        """Run base_model on one sample, return list of 5 RGB arrays."""
        sz = self.image_size
        inp = self._inputs[row_idx].unsqueeze(0).to(device)  # (1,1,H,W)

        out = base_model(inp)
        seg_logits = out["seg_logits"]  # (1,C,H,W)
        pred_depth_raw = out["depth"]  # (1,1,H,W)

        # Pred seg -> class IDs -> colorize
        pred_seg = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_seg_rgb = _colorize_seg(pred_seg)

        # Pred depth -> normalize to [0,255] -> colorize
        d = pred_depth_raw.squeeze(0).squeeze(0).cpu().float().numpy()
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            d = np.zeros_like(d, dtype=np.uint8)
        pred_depth_rgb = _colorize_depth(d)

        # Input as RGB (grayscale repeated 3x)
        inp_np = self._inputs[row_idx].squeeze(0).numpy()
        inp_u8 = np.clip(inp_np * 255, 0, 255).astype(np.uint8)
        inp_rgb = np.stack([inp_u8, inp_u8, inp_u8], axis=-1)

        # GT seg
        gt_seg = self._gt_segs[row_idx]
        gt_seg_rgb = _colorize_seg(gt_seg) if gt_seg is not None else _gray_placeholder(sz)

        # GT depth
        gt_depth = self._gt_depths[row_idx]
        gt_depth_rgb = _colorize_depth(gt_depth) if gt_depth is not None else _gray_placeholder(sz)

        return [inp_rgb, pred_seg_rgb, gt_seg_rgb, pred_depth_rgb, gt_depth_rgb]

    @staticmethod
    def _log_panel(trainer: Trainer, panel: Image.Image) -> None:
        """Dispatch the assembled panel image to all active loggers."""
        from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

        panel_np = np.array(panel)
        step = trainer.global_step

        for lg in trainer.loggers:
            try:
                if isinstance(lg, WandbLogger):
                    import wandb

                    lg.experiment.log(
                        {"val/predictions": wandb.Image(panel)},
                        step=step,
                    )
                elif isinstance(lg, TensorBoardLogger):
                    lg.experiment.add_image(
                        "val/predictions",
                        panel_np,
                        global_step=step,
                        dataformats="HWC",
                    )
            except Exception as e:
                logger.warning("Failed to log validation panel to %s: %s", type(lg).__name__, e)
