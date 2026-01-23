#!/usr/bin/env python3
"""Export curated neuron datasets for sharing."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import ExperimentConfig


def collect_dataset_files(
    data_dir: Path,
    manifest_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    include_meshes: bool = True,
    include_swcs: bool = False,
    include_images: bool = False,
) -> Dict[str, List[Path]]:
    """Collect files to include in export.
    
    Returns:
        Dict mapping category names to list of paths
    """
    files: Dict[str, List[Path]] = {
        "manifest": [],
        "metadata": [],
        "meshes": [],
        "swcs": [],
        "images": [],
    }
    
    if manifest_path and manifest_path.exists():
        files["manifest"].append(manifest_path)
    else:
        for pattern in ["manifest.jsonl", "train.jsonl", "download_log.jsonl"]:
            found = list(data_dir.glob(pattern))
            files["manifest"].extend(found)
    
    if metadata_path and metadata_path.exists():
        files["metadata"].append(metadata_path)
    else:
        for pattern in ["metadata.jsonl", "metadata.json", "metadata.csv"]:
            found = list(data_dir.glob(pattern))
            files["metadata"].extend(found)
    
    if include_meshes:
        mesh_dirs = list(data_dir.rglob("mesh_cache"))
        for mesh_dir in mesh_dirs:
            files["meshes"].extend(mesh_dir.glob("*.npz"))
    
    if include_swcs:
        files["swcs"].extend(data_dir.rglob("*.swc"))
    
    if include_images:
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            files["images"].extend(data_dir.rglob(ext))
    
    return files


def create_provenance(
    data_dir: Path,
    source: str,
    description: str,
    files: Dict[str, List[Path]],
) -> Dict[str, Any]:
    """Create provenance metadata for the export."""
    return {
        "export_date": datetime.utcnow().isoformat() + "Z",
        "source": source,
        "description": description,
        "original_data_dir": str(data_dir),
        "file_counts": {k: len(v) for k, v in files.items()},
        "axonet_version": "1.0.0",
    }


def export_dataset(
    data_dir: Path,
    output_path: Path,
    manifest_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    include_meshes: bool = True,
    include_swcs: bool = False,
    include_images: bool = False,
    source: str = "custom",
    description: str = "",
    compression: str = "gz",
) -> Path:
    """Export a curated dataset to a tar archive.
    
    Args:
        data_dir: Root directory containing the dataset
        output_path: Output tar file path
        manifest_path: Optional explicit manifest path
        metadata_path: Optional explicit metadata path
        config_path: Optional config YAML to include
        include_meshes: Include pre-computed mesh cache
        include_swcs: Include raw SWC files
        include_images: Include rendered images
        source: Data source identifier (allen, neuromorpho, custom)
        description: Dataset description
        compression: Compression type (gz, bz2, xz, or none)
        
    Returns:
        Path to created archive
    """
    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path)
    
    files = collect_dataset_files(
        data_dir,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        include_meshes=include_meshes,
        include_swcs=include_swcs,
        include_images=include_images,
    )
    
    provenance = create_provenance(data_dir, source, description, files)
    
    mode = "w:" if compression == "none" else f"w:{compression}"
    suffix = ".tar" if compression == "none" else f".tar.{compression}"
    if not str(output_path).endswith(suffix):
        output_path = output_path.with_suffix(suffix)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(output_path, mode) as tar:
        for manifest in files["manifest"]:
            tar.add(manifest, arcname=manifest.name)
        
        for metadata in files["metadata"]:
            tar.add(metadata, arcname=metadata.name)
        
        if config_path and config_path.exists():
            tar.add(config_path, arcname="config.yaml")
        
        if files["meshes"]:
            for mesh in files["meshes"]:
                rel = mesh.relative_to(data_dir)
                tar.add(mesh, arcname=str(rel))
        
        if files["swcs"]:
            for swc in files["swcs"]:
                rel = swc.relative_to(data_dir)
                tar.add(swc, arcname=f"swc/{rel}")
        
        if files["images"]:
            for img in files["images"]:
                rel = img.relative_to(data_dir)
                tar.add(img, arcname=f"images/{rel}")
        
        import io
        prov_bytes = json.dumps(provenance, indent=2).encode("utf-8")
        prov_info = tarfile.TarInfo(name="provenance.json")
        prov_info.size = len(prov_bytes)
        tar.addfile(prov_info, io.BytesIO(prov_bytes))
    
    print(f"Exported dataset to: {output_path}")
    print(f"  Manifests: {len(files['manifest'])}")
    print(f"  Metadata: {len(files['metadata'])}")
    print(f"  Meshes: {len(files['meshes'])}")
    print(f"  SWCs: {len(files['swcs'])}")
    print(f"  Images: {len(files['images'])}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export curated neuron dataset")
    parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output archive path")
    parser.add_argument("--manifest", type=Path, help="Explicit manifest path")
    parser.add_argument("--metadata", type=Path, help="Explicit metadata path")
    parser.add_argument("--config", type=Path, help="Config YAML to include")
    parser.add_argument("--include-meshes", action="store_true", default=True)
    parser.add_argument("--no-meshes", action="store_false", dest="include_meshes")
    parser.add_argument("--include-swcs", action="store_true", default=False)
    parser.add_argument("--include-images", action="store_true", default=False)
    parser.add_argument("--source", default="custom", choices=["allen", "neuromorpho", "custom"])
    parser.add_argument("--description", default="")
    parser.add_argument("--compression", default="gz", choices=["gz", "bz2", "xz", "none"])
    
    args = parser.parse_args()
    
    export_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        manifest_path=args.manifest,
        metadata_path=args.metadata,
        config_path=args.config,
        include_meshes=args.include_meshes,
        include_swcs=args.include_swcs,
        include_images=args.include_images,
        source=args.source,
        description=args.description,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
