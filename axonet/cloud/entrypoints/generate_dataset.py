#!/usr/bin/env python3
"""Entrypoint for dataset generation in cloud environments.

Designed to run as a batch task where BATCH_TASK_INDEX indicates which
chunk of work to process.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_swc_files(manifest_path: Path) -> List[dict]:
    """Load SWC entries from manifest."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_task_range(total: int, task_index: int, total_tasks: int) -> tuple:
    """Calculate start/end indices for this task."""
    chunk_size = (total + total_tasks - 1) // total_tasks
    start = task_index * chunk_size
    end = min(start + chunk_size, total)
    return start, end


def download_inputs(
    storage,
    remote_manifest: str,
    remote_swc_prefix: str,
    local_dir: Path,
    entries: List[dict],
) -> Path:
    """Download required input files."""
    swc_dir = local_dir / "swc"
    swc_dir.mkdir(parents=True, exist_ok=True)
    
    for entry in entries:
        swc_path = entry.get("swc") or entry.get("path")
        if swc_path:
            remote = f"{remote_swc_prefix.rstrip('/')}/{Path(swc_path).name}"
            local = swc_dir / Path(swc_path).name
            if not local.exists():
                storage.download(remote, local)
            entry["local_swc"] = str(local)
    
    return swc_dir


def generate_renders(
    entries: List[dict],
    output_dir: Path,
    width: int = 1024,
    height: int = 1024,
    views: int = 24,
    segments: int = 32,
    supersample_factor: int = 4,
    margin: float = 0.40,
    projection: str = "ortho",
    auto_margin: bool = True,
    cache: bool = True,
):
    """Generate rendered views for each neuron."""
    from axonet.training.dataset_generator import render_with_masks
    
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    
    for entry in entries:
        swc_path = Path(entry.get("local_swc") or entry.get("swc") or entry.get("path"))
        if not swc_path.exists():
            logger.warning(f"SWC not found: {swc_path}")
            continue
        
        neuron_id = entry.get("neuron_id") or swc_path.stem
        
        logger.info(f"Processing: {neuron_id}")
        
        neuron_out_dir, render_entries = render_with_masks(
            swc_path=swc_path,
            out_dir=output_dir,
            root_dir=output_dir,
            views=views,
            width=width,
            height=height,
            segments=segments,
            radius_scale=1.0,
            radius_adaptive_alpha=0.0,
            radius_ref_percentile=50.0,
            projection=projection,
            fovy=45.0,
            margin=margin,
            depth_shading=True,
            background=(0.0, 0.0, 0.0, 1.0),
            min_qc=0.1,
            qc_retries=3,
            sampling="fibonacci",
            auto_margin=auto_margin,
            seed=None,
            supersample_factor=supersample_factor,
            cache=cache,
            cache_dir=None,
        )
        
        for re in render_entries:
            re["neuron_id"] = neuron_id
            if "neuron_name" in entry:
                re["neuron_name"] = entry["neuron_name"]
            manifest_entries.append(re)
    
    return manifest_entries


def upload_outputs(
    storage,
    local_dir: Path,
    remote_prefix: str,
    manifest_entries: List[dict],
    task_index: int,
    save_cache: bool = False,
    swc_dir: Optional[Path] = None,
):
    """Upload generated outputs to remote storage."""
    for ext in ["*.png", "*.npz"]:
        for f in local_dir.rglob(ext):
            rel = f.relative_to(local_dir)
            remote = f"{remote_prefix.rstrip('/')}/{rel}"
            storage.upload(f, remote)
    
    # Upload mesh cache if requested
    if save_cache and swc_dir:
        cache_dir = swc_dir / "mesh_cache"
        if cache_dir.exists():
            logger.info(f"Uploading mesh cache from {cache_dir}")
            for f in cache_dir.glob("*.npz"):
                remote = f"{remote_prefix.rstrip('/')}/mesh_cache/{f.name}"
                storage.upload(f, remote)
    
    manifest_path = local_dir / f"manifest_{task_index}.jsonl"
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")
    
    storage.upload(manifest_path, f"{remote_prefix.rstrip('/')}/manifests/manifest_{task_index}.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset renders")
    parser.add_argument("--manifest", required=True, help="Input manifest (local or gs://)")
    parser.add_argument("--swc-prefix", required=True, help="SWC files prefix (local or gs://)")
    parser.add_argument("--output", required=True, help="Output location (local or gs://)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--supersample-factor", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.40)
    parser.add_argument("--projection", choices=["ortho", "persp"], default="ortho")
    parser.add_argument("--auto-margin", action="store_true", default=True)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--save-cache", action="store_true", help="Upload mesh cache with outputs")
    parser.add_argument("--task-index", type=int, default=None, help="Task index (or BATCH_TASK_INDEX)")
    parser.add_argument("--total-tasks", type=int, default=1)
    parser.add_argument("--provider", default="local", choices=["local", "google"])
    parser.add_argument("--local-dir", default="/tmp/axonet_dataset")
    
    args = parser.parse_args()
    
    task_index = args.task_index
    if task_index is None:
        task_index = int(os.environ.get("BATCH_TASK_INDEX", 0))
    
    logger.info(f"Task {task_index}/{args.total_tasks}")
    
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    use_cloud = args.manifest.startswith("gs://") or args.output.startswith("gs://")
    
    storage = None
    if use_cloud:
        from axonet.cloud import get_provider
        provider = get_provider(args.provider)
        storage = provider.storage
    
    if args.manifest.startswith("gs://"):
        local_manifest = local_dir / "manifest.jsonl"
        storage.download(args.manifest, local_manifest)
        manifest_path = local_manifest
    else:
        manifest_path = Path(args.manifest)
    
    entries = get_swc_files(manifest_path)
    logger.info(f"Total entries in manifest: {len(entries)}")
    
    start, end = get_task_range(len(entries), task_index, args.total_tasks)
    task_entries = entries[start:end]
    logger.info(f"Processing entries {start}-{end} ({len(task_entries)} entries)")
    
    if not task_entries:
        logger.info("No entries to process for this task")
        return
    
    swc_dir = local_dir / "swc"
    if storage and args.swc_prefix.startswith("gs://"):
        download_inputs(storage, args.manifest, args.swc_prefix, local_dir, task_entries)
    
    output_dir = local_dir / "output"
    manifest_entries = generate_renders(
        entries=task_entries,
        output_dir=output_dir,
        width=args.width,
        height=args.height,
        views=args.views,
        segments=args.segments,
        supersample_factor=args.supersample_factor,
        margin=args.margin,
        projection=args.projection,
        auto_margin=args.auto_margin,
        cache=not args.no_cache,
    )
    
    logger.info(f"Generated {len(manifest_entries)} render entries")
    
    if storage and args.output.startswith("gs://"):
        upload_outputs(
            storage, output_dir, args.output, manifest_entries, task_index,
            save_cache=args.save_cache, swc_dir=swc_dir,
        )
    else:
        out_manifest = Path(args.output) / f"manifest_{task_index}.jsonl"
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        with open(out_manifest, "w") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + "\n")
    
    logger.info("Dataset generation complete")


if __name__ == "__main__":
    main()
