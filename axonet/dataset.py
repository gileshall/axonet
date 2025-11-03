"""
Training Set Generator for SWC Neurons (OpenGL Offscreen)
=========================================================

Builds a dataset of rendered neuron images + a manifest of camera/QC metadata.
Designed to be UI-agnostic and to reuse NeuroRenderCore from render.py.

Features
--------
• Offscreen, headless rendering via pyglet.
• Orthographic or perspective projection.
• Uniform camera sampling on a sphere (Fibonacci lattice) or random.
• Per-view QC (fraction of model in frame) with retries and/or auto-margin.
• Saves PNGs and a JSONL manifest.
• Deterministic with --seed.

Usage
-----
python neuro_dataset_generator.py \
  --swc-dir /path/to/swc \
  --out /path/to/dataset \
  --views 24 --width 1024 --height 768 \
  --segments 32 --projection ortho --margin 0.85 \
  --min-qc 0.75 --qc-retries 5 --auto-margin

Manifest (JSONL)
----------------
Each line is one rendered sample with fields:
{
  "swc": "<relative swc path>",
  "image": "<relative png path>",
  "idx": <integer>,
  "qc_fraction": <float in [0,1]>,
  "projection": "ortho"|"perspective",
  "camera": {
    "eye": [x,y,z], "target": [x,y,z], "up": [x,y,z],
    "fovy": <deg>, "ortho_scale": <float>, "near": <float>, "far": <float>
  },
  "render": { "width": W, "height": H, "background": [r,g,b,a], "depth_shading": bool, "color_by_type": bool }
}

Notes
-----
• Requires the module with OffscreenContext and NeuroRenderCore in import path.
• This file intentionally does not import project-local io/mesh directly; it
  relies on NeuroRenderCore doing that.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from imageio.v2 import imwrite

# Import the engine
from .visualization.render import OffscreenContext, NeuroRenderCore

# -------------------------- Sampling helpers --------------------------

def fibonacci_sphere(n: int, *, jitter: float = 0.0, rng: random.Random | None = None) -> np.ndarray:
    """Return (n,3) unit vectors approximately uniformly distributed on S^2.
    Optional small random jitter in [-jitter, +jitter] added to phi/theta.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if rng is None:
        rng = random
    # Golden-angle lattice
    ga = math.pi * (3.0 - math.sqrt(5.0))
    z = np.linspace(1 - 1/n, -1 + 1/n, n, dtype=np.float32)
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    theta = np.arange(n, dtype=np.float32) * ga
    if jitter > 0:
        theta = theta + np.array([rng.uniform(-jitter, jitter) for _ in range(n)], dtype=np.float32)
        z = np.clip(z + np.array([rng.uniform(-jitter, jitter) for _ in range(n)], dtype=np.float32) * (1.0 / n), -1.0, 1.0)
        r = np.sqrt(np.maximum(0.0, 1 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    dirs = np.stack([x, y, z], axis=1)
    # Normalize just in case
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    return dirs.astype(np.float32)


def random_sphere(n: int, rng: random.Random | None = None) -> np.ndarray:
    """Naive random unit vectors on S^2 using Gaussian sampling and normalization."""
    if rng is None:
        rng = random
    v = np.array([[rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(n)], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


# --------------------------- Camera placement ---------------------------

def place_camera_on_sphere(core: NeuroRenderCore, direction: np.ndarray, *, distance: float | None = None) -> None:
    """Place the camera so that it looks at the current core.camera.target from a given direction.
    If distance is None, keep current camera-target distance.
    """
    direction = np.asarray(direction, dtype=np.float32)
    direction /= np.linalg.norm(direction) + 1e-12
    target = core.camera.target.copy()
    curr_dir = core.camera.eye - target
    curr_dist = float(np.linalg.norm(curr_dir))
    dist = curr_dist if distance is None else float(distance)

    # Robust up vector: try world +Y, fall back if nearly parallel
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(direction, up))) > 0.95:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    core.camera.eye = target + (-direction) * dist  # eye is opposite of viewing dir
    core.camera.up = up


# ------------------------------ QC helpers ------------------------------

def ensure_qc(core: NeuroRenderCore, min_qc: float, *, retries: int, sampler: Iterable[np.ndarray], auto_margin: bool, margin_step: float = 1.05) -> Tuple[float, np.ndarray]:
    """Try camera directions from `sampler` until QC >= min_qc or retries exhausted.
    May optionally increase ortho margin automatically to help pass QC.
    Returns (qc_value, used_direction).
    """
    used_dir = None
    qc_val = 0.0
    for k, d in enumerate(sampler):
        place_camera_on_sphere(core, d)
        qc = core.qc_fraction_in_frame()
        if qc >= min_qc:
            used_dir = d
            qc_val = qc
            break
        # If orthographic and auto_margin requested, try inflating ortho scale progressively
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


# ------------------------------ Main logic ------------------------------

def gather_swc_files(swc_dir: Path) -> List[Path]:
    return sorted([p for p in swc_dir.rglob("*.swc")])


def tolist(a: np.ndarray):
    return [float(x) for x in np.asarray(a).reshape(-1)]


def render_views_for_swc(
    swc_path: Path,
    out_dir: Path,
    *,
    views: int,
    width: int,
    height: int,
    segments: int,
    projection: str,
    fovy: float,
    margin: float,
    depth_shading: bool,
    color_by_type: bool,
    background: Tuple[float, float, float, float],
    min_qc: float,
    qc_retries: int,
    sampling: str,
    auto_margin: bool,
    seed: int | None,
) -> list[dict]:
    rng = random.Random(seed)

    # Prepare per-SWC output directory
    rel_name = swc_path.stem
    img_dir = out_dir / rel_name
    img_dir.mkdir(parents=True, exist_ok=True)

    with OffscreenContext(width, height, visible=False) as ctx:
        core = NeuroRenderCore(ctx)
        core.load_swc(swc_path, segments=segments)
        core.set_projection(perspective=(projection == "perspective"), fovy=fovy)
        core.depth_shading = depth_shading
        core.color_mode = 1 if color_by_type else 0
        core.fit_camera(margin=margin)

        # Compute a sensible camera distance (keep what fit_camera set)
        target = core.camera.target.copy()
        dist = float(np.linalg.norm(core.camera.eye - target))

        # Precompute directions
        if sampling == "fibonacci":
            dirs = fibonacci_sphere(views, jitter=0.0, rng=rng)
        elif sampling == "random":
            dirs = random_sphere(views, rng=rng)
        else:
            raise ValueError("sampling must be 'fibonacci' or 'random'")

        manifest_entries: list[dict] = []

        for i in range(views):
            # Provide multiple candidate directions around the i-th direction (small random yaw/pitch)
            base_dir = dirs[i]
            # Candidates: base plus a few random perturbations
            cand_dirs = [base_dir]
            for _ in range(max(0, qc_retries - 1)):
                jitter = np.array([rng.gauss(0, 0.12), rng.gauss(0, 0.12), rng.gauss(0, 0.12)], dtype=np.float32)
                cand = base_dir + jitter
                cand /= np.linalg.norm(cand) + 1e-12
                cand_dirs.append(cand)

            # Place at fixed distance (for ortho this distance is mostly irrelevant to scale)
            place_camera_on_sphere(core, cand_dirs[0], distance=dist)

            # QC loop
            qc, used_dir = ensure_qc(
                core,
                min_qc,
                retries=len(cand_dirs),
                sampler=cand_dirs,
                auto_margin=auto_margin,
            )

            # Render
            rgba = core.render_rgba(background=background)
            out_name = f"{rel_name}_{i:04d}.png"
            out_path = img_dir / out_name
            imwrite(out_path, rgba)

            entry = {
                "swc": str(swc_path.relative_to(swc_path.parents[0]) if swc_path.is_relative_to(swc_path.parents[0]) else swc_path.name),
                "image": str(out_path.relative_to(out_dir)),
                "idx": i,
                "qc_fraction": float(qc),
                "projection": "perspective" if core.camera.perspective else "ortho",
                "camera": {
                    "eye": tolist(core.camera.eye),
                    "target": tolist(core.camera.target),
                    "up": tolist(core.camera.up),
                    "fovy": float(core.camera.fovy),
                    "ortho_scale": float(core.camera.ortho_scale),
                    "near": float(core.camera.near),
                    "far": float(core.camera.far),
                    "used_dir": tolist(used_dir) if used_dir is not None else None,
                },
                "render": {
                    "width": width,
                    "height": height,
                    "background": [float(x) for x in background],
                    "depth_shading": bool(depth_shading),
                    "color_by_type": bool(color_by_type),
                },
            }
            manifest_entries.append(entry)

        return manifest_entries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--swc-dir", type=Path, required=True, help="Directory containing .swc files (searched recursively)")
    p.add_argument("--out", type=Path, required=True, help="Output dataset directory")
    p.add_argument("--views", type=int, default=24, help="Number of viewpoints per SWC")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--segments", type=int, default=32, help="Cylinder segments when tessellating neurites")
    p.add_argument("--projection", choices=["ortho", "perspective"], default="ortho")
    p.add_argument("--fovy", type=float, default=55.0, help="Perspective vertical FOV in degrees (if projection=perspective)")
    p.add_argument("--margin", type=float, default=0.85, help="Fit-camera margin multiplier (affects ortho scale or perspective distance)")
    p.add_argument("--depth-shading", action="store_true", help="Enable depth-based shading in fragment shader")
    p.add_argument("--no-color-by-type", dest="color_by_type", action="store_false", help="Render as single color instead of per-class colors")
    p.add_argument("--bg", type=float, nargs=4, default=(0, 0, 0, 0), metavar=("R","G","B","A"), help="Background RGBA in 0..1")
    p.add_argument("--min-qc", type=float, default=0.7, help="Minimum fraction of AABB points that must be in frame")
    p.add_argument("--qc-retries", type=int, default=5, help="Max attempts per view to meet QC")
    p.add_argument("--sampling", choices=["fibonacci", "random"], default="fibonacci")
    p.add_argument("--auto-margin", action="store_true", help="If QC fails in ortho, progressively increase ortho scale")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)

    # Discover SWCs
    swc_files = gather_swc_files(args.swc_dir)
    if not swc_files:
        raise SystemExit(f"No .swc files found under {args.swc_dir}")

    manifest_path = out_dir / "manifest.jsonl"

    print(f"Found {len(swc_files)} SWC files. Writing to {manifest_path}")

    total = 0
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for swc in swc_files:
            print(f"Rendering {swc} ...")
            entries = render_views_for_swc(
                swc,
                out_dir / "images",
                views=args.views,
                width=args.width,
                height=args.height,
                segments=args.segments,
                projection=args.projection,
                fovy=args.fovy,
                margin=args.margin,
                depth_shading=args.depth_shading,
                color_by_type=args.color_by_type,
                background=tuple(args.bg),
                min_qc=args.min_qc,
                qc_retries=max(1, args.qc_retries),
                sampling=args.sampling,
                auto_margin=args.auto_margin,
                seed=args.seed,
            )
            for e in entries:
                # Make paths relative to dataset root for portability
                e["swc"] = str(Path(e["swc"]).as_posix())
                e["image"] = str(Path(e["image"]).as_posix())
                fout.write(json.dumps(e, ensure_ascii=False) + "\n")
            total += len(entries)

    print(f"Done. Wrote {total} samples across {len(swc_files)} neurons.")


if __name__ == "__main__":
    main()
