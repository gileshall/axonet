"""
Complete dataset generator for SegVAE2D training.

Generates paired images + segmentation masks from SWC neuron data.
Integrates with render.py for high-quality rendering.

Splits datasets at the SWC (sample) level, not camera pose level.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from imageio.v2 import imwrite

from ..visualization.render import OffscreenContext, NeuroRenderCore, COLORS
from .sampling import fibonacci_sphere, random_sphere, pca_guided_sampling, compute_projected_extent, compute_neuron_pca


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
    cache: bool = True,
    cache_dir: Path | None = None,
    adaptive_framing: bool = False,
    canonical_views: int = 6,
    biased_views: int = 12,
    random_views: int = 6,
) -> Tuple[Path, List[dict]]:
    """Render image + segmentation mask pairs."""
    from ..io import NeuronClass
    
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
        core.load_swc(
            swc_path, segments=segments, radius_scale=radius_scale, 
            radius_adaptive_alpha=radius_adaptive_alpha, radius_ref_percentile=radius_ref_percentile,
            cache=cache, cache_dir=cache_dir
        )
        core.set_projection(perspective=(projection == "perspective"), fovy=fovy)
        core.depth_shading = depth_shading
        core.color_mode = 1
        core.fit_camera(margin=margin)

        target = core.camera.target.copy()
        dist = float(np.linalg.norm(core.camera.eye - target))

        # Extract node positions for PCA and adaptive framing
        node_positions = np.array([n.position for n in core.neuron.nodes.values()], dtype=np.float64)

        pca_eigenvalues = None
        if sampling == "pca":
            dirs, tiers = pca_guided_sampling(
                node_positions,
                n_canonical=canonical_views,
                n_biased=biased_views,
                n_random=random_views,
                rng=rng,
            )
            views = len(dirs)
            if len(node_positions) >= 10:
                _, pca_eigenvalues, _ = compute_neuron_pca(node_positions)
        elif sampling == "fibonacci":
            dirs = fibonacci_sphere(views, jitter=0.0, rng=rng)
            tiers = ["fibonacci"] * views
        elif sampling == "random":
            dirs = random_sphere(views, rng=rng)
            tiers = ["random"] * views
        else:
            raise ValueError("sampling must be 'pca', 'fibonacci', or 'random'")

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

            # Adaptive framing: tighten ortho_scale per view based on projected extent
            if adaptive_framing and not core.camera.perspective:
                proj_extent = compute_projected_extent(node_positions, used_dir)
                diag = core.diag
                clamped_extent = max(proj_extent, 0.15 * diag)
                core.camera.ortho_scale = margin * 0.5 * clamped_extent

            depth = core.render_depth(factor=supersample_factor)
            
            depth_visual = depth
            
            valid_mask = depth < 0.999
            
            if valid_mask.any():
                valid_depths = depth_visual[valid_mask]
                if len(valid_depths) > 0:
                    dmin, dmax = valid_depths.min(), valid_depths.max()
                    
                    if dmax > dmin:
                        depth_normalized = np.where(
                            valid_mask,
                            ((depth_visual - dmin) / (dmax - dmin)) * 255.0,
                            0.0
                        )
                        depth_output = np.clip(depth_normalized, 0, 255).astype(np.uint8)
                    else:
                        depth_output = np.where(valid_mask, 128, 0).astype(np.uint8)
                else:
                    depth_output = np.zeros_like(depth_visual, dtype=np.uint8)
            else:
                depth_output = np.zeros_like(depth_visual, dtype=np.uint8)
            
            if i == 0:
                valid_pixels = np.sum(valid_mask)
                print(f"  Depth stats: valid={valid_pixels}/{depth.size}, raw_min={depth.min():.3f}, raw_max={depth.max():.3f}")
            depth_path = img_dir / f"{rel_name}_{i:04d}_depth.png"
            imwrite(depth_path, depth_output)

            mask = core.render_class_id_mask_supersampled({
                NeuronClass.SOMA.name: CLASS_MAP[NeuronClass.SOMA.name],
                NeuronClass.AXON.name: CLASS_MAP[NeuronClass.AXON.name],
                NeuronClass.BASAL_DENDRITE.name: CLASS_MAP[NeuronClass.BASAL_DENDRITE.name],
                NeuronClass.APICAL_DENDRITE.name: CLASS_MAP[NeuronClass.APICAL_DENDRITE.name],
                NeuronClass.OTHER.name: CLASS_MAP[NeuronClass.OTHER.name],
            }, factor=supersample_factor)
                
            if i == 0:
                unique_vals = np.unique(mask)
                print(f"  Mask debug: unique values={unique_vals}, shape={mask.shape}, dtype={mask.dtype}")
                print(f"  Mask stats: min={mask.min()}, max={mask.max()}, non_zero={np.count_nonzero(mask)}")

            mask_path = img_dir / f"{rel_name}_{i:04d}_mask.png"
            imwrite(mask_path, mask)

            colorized = np.zeros((height, width, 3), dtype=np.uint8)
            for cls_name, cls_val in CLASS_MAP.items():
                if cls_name == "BACKGROUND":
                    continue
                rgb = (np.clip(COLOR_MAP[cls_name], 0.0, 1.0) * 255.0).astype(np.uint8)
                colorized[mask == cls_val] = rgb
            color_path = img_dir / f"{rel_name}_{i:04d}_mask_color.png"
            imwrite(color_path, colorized)

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
                "view_tier": tiers[i],
                "occupancy_fraction": float(np.count_nonzero(mask) / mask.size),
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
            if pca_eigenvalues is not None and i == 0:
                entry["pca_eigenvalues"] = [float(ev) for ev in pca_eigenvalues]
                entry["aspect_ratio"] = float(pca_eigenvalues[0] / max(pca_eigenvalues[2], 1e-12))
            manifest_entries.append(entry)

    return (swc_path, manifest_entries)


def split_swc_files(swc_files: List[Path], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split SWC files into train/val/test sets at the sample level."""
    rng = random.Random(seed)
    shuffled = swc_files.copy()
    rng.shuffle(shuffled)
    
    total = len(shuffled)
    test_size = int(test_ratio * total) if test_ratio > 0 else 0
    val_size = int(val_ratio * total) if val_ratio > 0 else 0
    train_size = total - val_size - test_size
    
    test_files = shuffled[:test_size] if test_size > 0 else []
    val_files = shuffled[test_size:test_size + val_size] if val_size > 0 else []
    train_files = shuffled[test_size + val_size:] if train_size > 0 else []
    
    return train_files, val_files, test_files


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
    p.add_argument("--sampling", choices=["pca", "fibonacci", "random"], default="pca")
    p.add_argument("--auto-margin", action="store_true")
    p.add_argument("--adaptive-framing", action="store_true", default=False,
                   help="Per-view adaptive ortho_scale based on projected extent")
    p.add_argument("--canonical-views", type=int, default=6, help="PCA canonical views (+/-PC1,PC2,PC3)")
    p.add_argument("--biased-views", type=int, default=12, help="PCA biased views near PC1-PC2 plane")
    p.add_argument("--random-views", type=int, default=6, help="Random views for diversity")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--supersample-factor", type=int, default=2, help="Supersampling factor for depth and mask rendering (2, 3, or 4 recommended)")
    p.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel jobs (1 = sequential)")
    
    p.add_argument("--val-ratio", type=float, default=0.0, help="Validation set ratio (0.0 = no validation set, default: 0.0)")
    p.add_argument("--test-ratio", type=float, default=0.0, help="Test set ratio (0.0 = no test set, default: 0.0)")
    p.add_argument("--split-seed", type=int, default=42, help="Random seed for SWC file splitting")
    
    p.add_argument("--no-cache", action="store_true", help="Disable mesh caching (caching enabled by default)")
    p.add_argument("--cache-dir", type=Path, default=None, help="Directory for mesh cache (default: mesh_cache/ next to SWC files)")
    
    args = p.parse_args()

    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.test_ratio < 0 or args.test_ratio >= 1:
        raise ValueError("--test-ratio must be in [0, 1)")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio + --test-ratio must be < 1")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    swc_files = sorted([p for p in Path(args.swc_dir).rglob("*.swc", case_sensitive=False)])
    if not swc_files:
        raise SystemExit(f"No .swc files found under {args.swc_dir}")

    train_files, val_files, test_files = split_swc_files(swc_files, args.val_ratio, args.test_ratio, args.split_seed)
    
    print(f"Found {len(swc_files)} SWC files")
    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    cache_enabled = not args.no_cache
    if cache_enabled:
        cache_loc = args.cache_dir or "mesh_cache/ next to SWC files"
        print(f"Mesh caching: enabled ({cache_loc})")
    else:
        print("Mesh caching: disabled")
    
    # When using PCA sampling, total views = canonical + biased + random
    if args.sampling == "pca":
        effective_views = args.canonical_views + args.biased_views + args.random_views
        print(f"PCA sampling: {args.canonical_views} canonical + {args.biased_views} biased + {args.random_views} random = {effective_views} views")
        if args.adaptive_framing:
            print("Adaptive framing: enabled")
    else:
        effective_views = args.views

    render_args = {
        "views": effective_views,
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
        "cache": cache_enabled,
        "cache_dir": args.cache_dir,
        "adaptive_framing": args.adaptive_framing,
        "canonical_views": args.canonical_views,
        "biased_views": args.biased_views,
        "random_views": args.random_views,
    }

    def process_split(split_name: str, swc_list: List[Path], manifest_path: Path):
        """Process a single split (train/val/test)."""
        if not swc_list:
            return 0
        
        print(f"\nProcessing {split_name} set ({len(swc_list)} SWC files)...")
        total = 0
        start_time = time.time()
        
        if args.jobs > 1:
            print(f"Rendering with {args.jobs} parallel workers...")
            print(f"Expected ~{len(swc_list) * effective_views} total samples\n")
            
            with multiprocessing.Pool(args.jobs) as pool:
                tasks = [(swc, out_dir / "images", {**render_args, "root_dir": out_dir}) for swc in swc_list]
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
                        _print_progress(completed, len(swc_list), start_time, 
                                      f"  {swc_path.name} ({len(manifest_entries)} views): ")
        else:
            print(f"Rendering sequentially...")
            print(f"Expected ~{len(swc_list) * effective_views} total samples\n")
            
            with open(manifest_path, "w", encoding="utf-8") as fout:
                for idx, swc in enumerate(swc_list):
                    _print_progress(idx, len(swc_list), start_time, 
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
        print(f"\n✓ {split_name.capitalize()} set: Generated {total} samples from {len(swc_list)} neurons in {elapsed:.1f}s")
        return total

    train_total = process_split("train", train_files, out_dir / "manifest_train.jsonl")
    val_total = process_split("val", val_files, out_dir / "manifest_val.jsonl") if val_files else 0
    test_total = process_split("test", test_files, out_dir / "manifest_test.jsonl") if test_files else 0

    print(f"\n{'='*60}")
    print(f"✓ Complete! Generated {train_total + val_total + test_total} total samples")
    print(f"  Train: {train_total} samples from {len(train_files)} neurons")
    if val_files:
        print(f"  Val:   {val_total} samples from {len(val_files)} neurons")
    if test_files:
        print(f"  Test:  {test_total} samples from {len(test_files)} neurons")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

