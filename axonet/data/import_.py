#!/usr/bin/env python3
"""Import shared neuron datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests


def download_file(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Download a file from URL."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return dest


def extract_archive(archive_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Extract archive and return provenance info.
    
    Returns:
        Provenance dict if found, else empty dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    provenance = {}
    
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.name == "provenance.json":
                f = tar.extractfile(member)
                if f:
                    provenance = json.load(f)
            else:
                tar.extract(member, output_dir)
    
    return provenance


def download_swcs_from_manifest(
    manifest_path: Path,
    output_dir: Path,
    source: str = "neuromorpho",
    insecure: bool = False,
    timeout: float = 60.0,
    max_workers: int = 4,
) -> int:
    """Download SWC files referenced in manifest using neuromorpho_bulk helpers.
    
    Returns:
        Number of files downloaded
    """
    from ..utils.neuromorpho_bulk import (
        build_standardized_swc_url_pattern,
        download_file as nm_download,
        safe_filename,
    )
    
    swc_dir = output_dir / "swc"
    swc_dir.mkdir(parents=True, exist_ok=True)
    
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    
    sess = requests.Session()
    sess.headers.update({"User-Agent": "axonet-import/1.0"})
    verify = not insecure
    
    downloaded = 0
    
    for entry in entries:
        neuron_id = str(entry.get("neuron_id", ""))
        neuron_name = str(entry.get("neuron_name", ""))
        archive = str(entry.get("archive", ""))
        
        if not neuron_name:
            continue
        
        fname = safe_filename(
            f"{neuron_id}_{neuron_name}.CNG.swc" if neuron_id else f"{neuron_name}.CNG.swc"
        )
        out_path = swc_dir / fname
        
        if out_path.exists():
            continue
        
        if source == "neuromorpho" and archive:
            url = build_standardized_swc_url_pattern(
                "https://neuromorpho.org", archive, neuron_name
            )
            nm_download(sess, url, str(out_path), verify=verify, timeout=timeout)
            downloaded += 1
    
    return downloaded


def import_dataset(
    source: str,
    output_dir: Path,
    archive_path: Optional[Path] = None,
    archive_url: Optional[str] = None,
    download_swcs: bool = False,
    insecure: bool = False,
) -> Path:
    """Import a shared dataset.
    
    Args:
        source: Either path to local archive or URL
        output_dir: Where to extract/place the dataset
        archive_path: Explicit archive path (overrides source)
        archive_url: Explicit archive URL (overrides source)
        download_swcs: Download missing SWC files from source
        insecure: Disable SSL verification for downloads
        
    Returns:
        Path to imported dataset directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if archive_url:
        parsed = urlparse(archive_url)
        fname = Path(parsed.path).name or "dataset.tar.gz"
        archive_path = output_dir / fname
        print(f"Downloading archive from {archive_url}...")
        download_file(archive_url, archive_path)
    elif archive_path:
        archive_path = Path(archive_path)
    else:
        source_path = Path(source)
        if source_path.exists():
            archive_path = source_path
        elif source.startswith(("http://", "https://")):
            parsed = urlparse(source)
            fname = Path(parsed.path).name or "dataset.tar.gz"
            archive_path = output_dir / fname
            print(f"Downloading archive from {source}...")
            download_file(source, archive_path)
        else:
            raise ValueError(f"Invalid source: {source}")
    
    print(f"Extracting archive to {output_dir}...")
    provenance = extract_archive(archive_path, output_dir)
    
    if provenance:
        print(f"Dataset info:")
        print(f"  Source: {provenance.get('source', 'unknown')}")
        print(f"  Description: {provenance.get('description', '')}")
        print(f"  Export date: {provenance.get('export_date', 'unknown')}")
        counts = provenance.get("file_counts", {})
        for k, v in counts.items():
            print(f"  {k}: {v}")
    
    if download_swcs:
        manifest_path = None
        for name in ["manifest.jsonl", "train.jsonl", "download_log.jsonl", "metadata.jsonl"]:
            candidate = output_dir / name
            if candidate.exists():
                manifest_path = candidate
                break
        
        if manifest_path:
            data_source = provenance.get("source", "neuromorpho")
            print(f"\nDownloading SWC files from {data_source}...")
            count = download_swcs_from_manifest(
                manifest_path,
                output_dir,
                source=data_source,
                insecure=insecure,
            )
            print(f"Downloaded {count} SWC files")
    
    print(f"\nDataset imported to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Import shared neuron dataset")
    parser.add_argument("source", help="Archive path or URL")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument("--archive-url", help="Explicit archive URL")
    parser.add_argument("--download-swcs", action="store_true", help="Download missing SWC files")
    parser.add_argument("--insecure", action="store_true", help="Disable SSL verification")
    
    args = parser.parse_args()
    
    import_dataset(
        source=args.source,
        output_dir=args.output_dir,
        archive_url=args.archive_url,
        download_swcs=args.download_swcs,
        insecure=args.insecure,
    )


if __name__ == "__main__":
    main()
