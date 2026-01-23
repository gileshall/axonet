"""
Complete dataset generator for SegVAE2D training.

Generates paired images + segmentation masks from SWC neuron data.
Integrates with render.py for high-quality rendering.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from imageio.v2 import imwrite

from ..visualization.render import OffscreenContext, NeuroRenderCore, COLORS
from .sampling import fibonacci_sphere, random_sphere


def place_camera_on_sphere(core: NeuroRenderCore, direction: np.ndarray, *, distance: float | None = None) -> None:
    """Place camera on sphere looking at target."""
    direction = np.asarray(direction, dtype=np.float32)
    direction /= np.linalg.norm(direction) + 1e-12
    target = core.camera.target.copy()
    curr_dir = core.camera.eye - target
    curr_dist = float(np.linalg.norm(curr_dir))
    dist = curr_dist if distance is None else float(distance)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(direction, up))) > 0.95:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    core.camera.eye = target + (-direction) * dist
    core.camera.up = up


def ensure_qc(core: NeuroRenderCore, min_qc: float, *, retries: int, sampler: List[np.ndarray], auto_margin: bool, margin_step: float = 1.05) -> Tuple[float, np.ndarray]:
    """Try camera directions until QC met or retries exhausted."""
    used_dir = None
    qc_val = 0.0
    for k, d in enumerate(sampler):
        place_camera_on_sphere(core, d)
        qc = core.qc_fraction_in_frame()
        if qc >= min_qc:
            used_dir = d
            qc_val = qc
            break
        if auto_margin and not core.camera.perspective:
            old = core.camera.ortho_scale
            core.camera.ortho_scale = old * margin_step
            qc = core.qc_fraction_in_frame()
            if qc >= min_qc:
                used_dir = d
                qc_val = qc
                break
        if k + 1 >= retries:
            used_dir = d
            qc_val = qc
            break
    return qc_val, used_dir


def _render_worker(args_tuple):
    """Worker function for multiprocessing."""
    swc_path, out_dir, render_args = args_tuple
    return render_with_masks(
        swc_path,
        out_dir,
        **render_args
    )


def _print_progress(current: int, total: int, start_time: float, prefix: str = ""):
    """Print progress information."""
    elapsed = time.time() - start_time
    if current > 0:
        avg_time = elapsed / current
        remaining = avg_time * (total - current)
        percent = (current / total) * 100
        print(f"{prefix}[{current}/{total}] ({percent:.1f}%) "
              f"Elapsed: {elapsed:.1f}s, Remaining: ~{remaining:.1f}s")
    else:
        print(f"{prefix}[{current}/{total}] (0.0%)")


def render_with_masks(
    swc_path: Path,
    out_dir: Path,
    *,
    root_dir: Path,
    views: int,
    width: int,
    height: int,
    segments: int,
    radius_scale: float,
    radius_adaptive_alpha: float,
    radius_ref_percentile: float,
    projection: str,
    fovy: float,
    margin: float,
    depth_shading: bool,
    background: Tuple[float, float, float, float],
    min_qc: float,
    qc_retries: int,
    sampling: str,
    auto_margin: bool,
    seed: int | None,
    supersample_factor: int = 2,
) -> Tuple[Path, List[dict]]:
    """Render image + segmentation mask pairs."""
    from ..io import NeuronClass
    
    # Formal class name -> value mapping used for all outputs
    CLASS_MAP = {
        "BACKGROUND": 0,
        NeuronClass.SOMA.name: 1,
        NeuronClass.AXON.name: 2,
        NeuronClass.BASAL_DENDRITE.name: 3,
        NeuronClass.APICAL_DENDRITE.name: 4,
        NeuronClass.OTHER.name: 5,
    }

    COLOR_MAP = {
        "BACKGROUND": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        NeuronClass.SOMA.name: COLORS['red'],
        NeuronClass.AXON.name: COLORS['green'],
        NeuronClass.BASAL_DENDRITE.name: COLORS['blue'],
        NeuronClass.APICAL_DENDRITE.name: COLORS['purple'],
        NeuronClass.OTHER.name: COLORS['orange'],
    }
    
    rng = random.Random(seed)

    rel_name = swc_path.stem
    img_dir = out_dir / rel_name
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[dict] = []

    with OffscreenContext(width, height, visible=False, samples=0) as ctx:
        core = NeuroRenderCore(ctx)
        core.load_swc(swc_path, segments=segments, radius_scale=radius_scale, radius_adaptive_alpha=radius_adaptive_alpha, radius_ref_percentile=radius_ref_percentile)
        core.set_projection(perspective=(projection == "perspective"), fovy=fovy)
        core.depth_shading = depth_shading
        core.color_mode = 1
        core.fit_camera(margin=margin)

        target = core.camera.target.copy()
        dist = float(np.linalg.norm(core.camera.eye - target))

        if sampling == "fibonacci":
            dirs = fibonacci_sphere(views, jitter=0.0, rng=rng)
        elif sampling == "random":
            dirs = random_sphere(views, rng=rng)
        else:
            raise ValueError("sampling must be 'fibonacci' or 'random'")

        for i in range(views):
            base_dir = dirs[i]
            cand_dirs = [base_dir]
            for _ in range(max(0, qc_retries - 1)):
                jitter = np.array([rng.gauss(0, 0.12), rng.gauss(0, 0.12), rng.gauss(0, 0.12)], dtype=np.float32)
                cand = base_dir + jitter
                cand /= np.linalg.norm(cand) + 1e-12
                cand_dirs.append(cand)

            place_camera_on_sphere(core, cand_dirs[0], distance=dist)
            qc, used_dir = ensure_qc(
                core,
                min_qc,
                retries=len(cand_dirs),
                sampler=cand_dirs,
                auto_margin=auto_margin,
            )

            # Render depth map (supersampled internally for determinism)
            depth = core.render_depth(factor=supersample_factor)
            
            # Depth is already in [0,1] range where closer = brighter (no inversion needed)
            depth_visual = depth
            
            # Find valid pixels (exclude background at far plane)
            valid_mask = depth < 0.999
            
            if valid_mask.any():
                valid_depths = depth_visual[valid_mask]
                if len(valid_depths) > 0:
                    dmin, dmax = valid_depths.min(), valid_depths.max()
                    
                    # Remap depth to full 0-255 range: near objects become white, far objects become darker gray
                    if dmax > dmin:
                        depth_normalized = np.where(
                            valid_mask,
                            ((depth_visual - dmin) / (dmax - dmin)) * 255.0,
                            0.0
                        )
                        depth_output = np.clip(depth_normalized, 0, 255).astype(np.uint8)
                    else:
                        # All at same depth, just set to medium gray
                        depth_output = np.where(valid_mask, 128, 0).astype(np.uint8)
                else:
                    depth_output = np.zeros_like(depth_visual, dtype=np.uint8)
            else:
                depth_output = np.zeros_like(depth_visual, dtype=np.uint8)
            
            # Debug output
            if i == 0:
                valid_pixels = np.sum(valid_mask)
                print(f"  Depth stats: valid={valid_pixels}/{depth.size}, raw_min={depth.min():.3f}, raw_max={depth.max():.3f}")
            depth_path = img_dir / f"{rel_name}_{i:04d}_depth.png"
            imwrite(depth_path, depth_output)

            # Render class id mask with supersampling + majority pooling
            mask = core.render_class_id_mask_supersampled({
                NeuronClass.SOMA.name: CLASS_MAP[NeuronClass.SOMA.name],
                NeuronClass.AXON.name: CLASS_MAP[NeuronClass.AXON.name],
                NeuronClass.BASAL_DENDRITE.name: CLASS_MAP[NeuronClass.BASAL_DENDRITE.name],
                NeuronClass.APICAL_DENDRITE.name: CLASS_MAP[NeuronClass.APICAL_DENDRITE.name],
                NeuronClass.OTHER.name: CLASS_MAP[NeuronClass.OTHER.name],
            }, factor=supersample_factor)
                
            # Debug: check mask values
            if i == 0:
                unique_vals = np.unique(mask)
                print(f"  Mask debug: unique values={unique_vals}, shape={mask.shape}, dtype={mask.dtype}")
                print(f"  Mask stats: min={mask.min()}, max={mask.max()}, non_zero={np.count_nonzero(mask)}")

            mask_path = img_dir / f"{rel_name}_{i:04d}_mask.png"
            imwrite(mask_path, mask)

            # Save colorized segmentation receipt image for auditing
            colorized = np.zeros((height, width, 3), dtype=np.uint8)
            for cls_name, cls_val in CLASS_MAP.items():
                if cls_name == "BACKGROUND":
                    continue
                rgb = (np.clip(COLOR_MAP[cls_name], 0.0, 1.0) * 255.0).astype(np.uint8)
                colorized[mask == cls_val] = rgb
            color_path = img_dir / f"{rel_name}_{i:04d}_mask_color.png"
            imwrite(color_path, colorized)

            # Save binary (black/white) mask: non-background -> white(255), background -> black(0)
            mask_bw = np.where(mask != 0, 255, 0).astype(np.uint8)
            mask_bw_path = img_dir / f"{rel_name}_{i:04d}_mask_bw.png"
            imwrite(mask_bw_path, mask_bw)

            entry = {
                "swc": str(swc_path.name),
                "mask": str(mask_path.relative_to(root_dir)),
                "depth": str(depth_path.relative_to(root_dir)),
                "mask_color": str(color_path.relative_to(root_dir)),
                "mask_bw": str(mask_bw_path.relative_to(root_dir)),
                "idx": i,
                "qc_fraction": float(qc),
                "projection": "perspective" if core.camera.perspective else "ortho",
                "camera": {
                    "eye": [float(x) for x in core.camera.eye],
                    "target": [float(x) for x in core.camera.target],
                    "up": [float(x) for x in core.camera.up],
                    "fovy": float(core.camera.fovy),
                    "ortho_scale": float(core.camera.ortho_scale),
                    "near": float(core.camera.near),
                    "far": float(core.camera.far),
                },
                "classes": {
                    "mapping": CLASS_MAP,
                },
            }
            manifest_entries.append(entry)

    return (swc_path, manifest_entries)


def main():
    p = argparse.ArgumentParser(description="Generate dataset for SegVAE2D training")
    p.add_argument("--swc-dir", type=Path, required=True, help="Directory containing .swc files")
    p.add_argument("--out", type=Path, required=True, help="Output dataset directory")
    p.add_argument("--views", type=int, default=24, help="Number of viewpoints per SWC")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--segments", type=int, default=32, help="Cylinder segments")
    p.add_argument("--radius-scale", type=float, default=1.0, help="Global factor applied to neurite radii (soma unchanged)")
    p.add_argument("--radius-adaptive-alpha", type=float, default=0.0, help="If >0, adapt scaling stronger for thin neurites. 0 disables adaptive scaling")
    p.add_argument("--radius-ref-percentile", type=float, default=50.0, help="Reference percentile of neurite radii for adaptive scaling (e.g., 50 for median)")
    p.add_argument("--projection", choices=["ortho", "perspective"], default="ortho")
    p.add_argument("--fovy", type=float, default=55.0)
    p.add_argument("--margin", type=float, default=0.40)
    p.add_argument("--depth-shading", action="store_true")
    p.add_argument("--bg", type=float, nargs=4, default=(0, 0, 0, 1), metavar=("R", "G", "B", "A"))
    p.add_argument("--min-qc", type=float, default=0.7)
    p.add_argument("--qc-retries", type=int, default=5)
    p.add_argument("--sampling", choices=["fibonacci", "random"], default="fibonacci")
    p.add_argument("--auto-margin", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--supersample-factor", type=int, default=2, help="Supersampling factor for depth and mask rendering (2, 3, or 4 recommended)")
    p.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel jobs (1 = sequential)")
    args = p.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    swc_files = sorted([p for p in Path(args.swc_dir).rglob("*.swc")])
    if not swc_files:
        raise SystemExit(f"No .swc files found under {args.swc_dir}")

    manifest_path = out_dir / "manifest.jsonl"
    print(f"Found {len(swc_files)} SWC files. Writing to {manifest_path}")
    
    render_args = {
        "views": args.views,
        "width": args.width,
        "height": args.height,
        "segments": args.segments,
        "radius_scale": args.radius_scale,
        "projection": args.projection,
        "fovy": args.fovy,
        "margin": args.margin,
        "depth_shading": args.depth_shading,
        "background": tuple(args.bg),
        "min_qc": args.min_qc,
        "qc_retries": args.qc_retries,
        "sampling": args.sampling,
        "auto_margin": args.auto_margin,
        "seed": args.seed,
        "radius_adaptive_alpha": args.radius_adaptive_alpha,
        "radius_ref_percentile": args.radius_ref_percentile,
        "supersample_factor": args.supersample_factor,
    }

    total = 0
    start_time = time.time()
    
    if args.jobs > 1:
        print(f"\nRendering {len(swc_files)} SWC files with {args.jobs} parallel workers...")
        print(f"Expected ~{len(swc_files) * args.views} total samples\n")
        
        # Use imap_unordered for progress updates
        with multiprocessing.Pool(args.jobs) as pool:
            tasks = [(swc, out_dir / "images", {**render_args, "root_dir": out_dir}) for swc in swc_files]
            completed = 0
            with open(manifest_path, "w", encoding="utf-8") as fout:
                for swc_path, manifest_entries in pool.imap_unordered(_render_worker, tasks):
                    completed += 1
                    for e in manifest_entries:
                        e["swc"] = str(Path(e["swc"]).as_posix())
                        e["mask"] = str(Path(e["mask"]).as_posix())
                        e["depth"] = str(Path(e["depth"]).as_posix())
                        fout.write(json.dumps(e, ensure_ascii=False) + "\n")
                        total += 1
                    _print_progress(completed, len(swc_files), start_time, 
                                  f"  {swc_path.name} ({len(manifest_entries)} views): ")
    else:
        print(f"\nRendering {len(swc_files)} SWC files sequentially...")
        print(f"Expected ~{len(swc_files) * args.views} total samples\n")
        
        with open(manifest_path, "w", encoding="utf-8") as fout:
            for idx, swc in enumerate(swc_files):
                _print_progress(idx, len(swc_files), start_time, 
                              f"Rendering {swc.name}: ")
                
                swc_path, manifest_entries = render_with_masks(
                    swc,
                    out_dir / "images",
                    root_dir=out_dir,
                    **render_args
                )
                for e in manifest_entries:
                    e["swc"] = str(Path(e["swc"]).as_posix())
                    e["mask"] = str(Path(e["mask"]).as_posix())
                    e["depth"] = str(Path(e["depth"]).as_posix())
                    fout.write(json.dumps(e, ensure_ascii=False) + "\n")
                    total += 1

    elapsed = time.time() - start_time
    print(f"\n✓ Done! Generated {total} samples across {len(swc_files)} neurons in {elapsed:.1f}s")
    print(f"  Average: {total / len(swc_files):.1f} samples per neuron")


if __name__ == "__main__":
    main()

