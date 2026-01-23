#!/usr/bin/env python3
"""
allen_bulk.py

Bulk download Allen Brain Institute SWC files from Brain Image Library.

Supported datasets:
- mouse_motor_cortex: Tolias lab Patch-seq morphologies
- mouse_visual_cortex: Patch-seq morphologies (200526)

Usage:
  # Download mouse motor cortex data
  python -m axonet.utils.allen_bulk --dataset mouse_motor_cortex --out allen_motor

  # Download mouse visual cortex data  
  python -m axonet.utils.allen_bulk --dataset mouse_visual_cortex --out allen_visual

  # Download from custom URL
  python -m axonet.utils.allen_bulk --url https://download.brainimagelibrary.org/path/ --out custom_data

  # List available datasets
  python -m axonet.utils.allen_bulk --list
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from html.parser import HTMLParser

import requests

DATASETS = {
    "mouse_motor_cortex": {
        "name": "Mouse Motor Cortex Patch-seq",
        "url": "https://download.brainimagelibrary.org/biccn/zeng/tolias/pseq/morph/",
        "source": "https://knowledge.brain-map.org/data/FTX8VQ3E93P26HK79V3",
        "description": "Tolias lab mouse motor cortex Patch-seq morphologies",
    },
    "mouse_visual_cortex": {
        "name": "Mouse Visual Cortex Patch-seq", 
        "url": "https://download.brainimagelibrary.org/biccn/zeng/pseq/morph/200526/",
        "source": "https://knowledge.brain-map.org/data/1HEYEW7GMUKWIQW37BO",
        "description": "Mouse visual cortex Patch-seq morphologies (200526)",
    },
}


class LinkExtractor(HTMLParser):
    """Extract links from HTML directory listing."""
    
    def __init__(self):
        super().__init__()
        self.links = []
    
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value and not value.startswith("?"):
                    self.links.append(value)


def get_directory_listing(session: requests.Session, url: str, timeout: float = 30.0) -> List[str]:
    """Get list of files/directories from a URL."""
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    
    parser = LinkExtractor()
    parser.feed(resp.text)
    
    links = []
    for link in parser.links:
        if link.startswith("/") or link.startswith(".."):
            continue
        links.append(link)
    
    return links


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    timeout: float = 60.0,
    retries: int = 3,
) -> bool:
    """Download a file with retries."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()
            
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed to download {url}: {e}", file=sys.stderr)
                return False
    return False


def extract_cell_id(filename: str) -> Optional[str]:
    """Extract cell specimen ID from filename."""
    match = re.search(r"(\d{9,})", filename)
    if match:
        return match.group(1)
    stem = Path(filename).stem
    return stem


def crawl_and_download(
    session: requests.Session,
    base_url: str,
    out_dir: Path,
    extensions: List[str] = [".swc"],
    recursive: bool = True,
    max_depth: int = 3,
    _depth: int = 0,
) -> List[Dict]:
    """Crawl directory and download matching files.
    
    Returns list of download records.
    """
    if _depth > max_depth:
        return []
    
    records = []
    
    try:
        links = get_directory_listing(session, base_url)
    except Exception as e:
        print(f"  Error listing {base_url}: {e}", file=sys.stderr)
        return records
    
    for link in links:
        full_url = base_url.rstrip("/") + "/" + link
        
        if link.endswith("/"):
            if recursive:
                records.extend(
                    crawl_and_download(
                        session, full_url, out_dir, extensions, recursive, max_depth, _depth + 1
                    )
                )
        else:
            ext = Path(link).suffix.lower()
            if ext in extensions:
                cell_id = extract_cell_id(link)
                dest = out_dir / "swc" / link
                
                if dest.exists():
                    print(f"  Exists: {link}")
                    records.append({
                        "status": "exists",
                        "cell_id": cell_id,
                        "filename": link,
                        "url": full_url,
                        "path": str(dest),
                    })
                else:
                    print(f"  Downloading: {link}")
                    if download_file(session, full_url, dest):
                        records.append({
                            "status": "ok",
                            "cell_id": cell_id,
                            "filename": link,
                            "url": full_url,
                            "path": str(dest),
                        })
                    else:
                        records.append({
                            "status": "failed",
                            "cell_id": cell_id,
                            "filename": link,
                            "url": full_url,
                            "error": "download failed",
                        })
    
    return records


def download_dataset(
    dataset_key: str,
    out_dir: Path,
    recursive: bool = True,
) -> List[Dict]:
    """Download a predefined dataset."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASETS.keys())}")
    
    dataset = DATASETS[dataset_key]
    print(f"Downloading: {dataset['name']}")
    print(f"  Source: {dataset['source']}")
    print(f"  URL: {dataset['url']}")
    print(f"  Output: {out_dir}")
    print()
    
    session = requests.Session()
    session.headers.update({"User-Agent": "axonet-allen-bulk/1.0"})
    
    records = crawl_and_download(
        session,
        dataset["url"],
        out_dir,
        extensions=[".swc"],
        recursive=recursive,
    )
    
    for rec in records:
        rec["dataset"] = dataset_key
        rec["dataset_name"] = dataset["name"]
        rec["dataset_source"] = dataset["source"]
    
    return records


def download_from_url(
    url: str,
    out_dir: Path,
    recursive: bool = True,
) -> List[Dict]:
    """Download from a custom URL."""
    print(f"Downloading from: {url}")
    print(f"  Output: {out_dir}")
    print()
    
    session = requests.Session()
    session.headers.update({"User-Agent": "axonet-allen-bulk/1.0"})
    
    return crawl_and_download(
        session,
        url,
        out_dir,
        extensions=[".swc"],
        recursive=recursive,
    )


def write_manifest(records: List[Dict], path: Path) -> None:
    """Write download manifest as JSONL."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download Allen Brain Institute SWC files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Predefined dataset to download")
    parser.add_argument("--url", help="Custom URL to download from")
    parser.add_argument("--out", type=Path, required=False, help="Output directory")
    parser.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirectories")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available datasets:")
        print()
        for key, info in DATASETS.items():
            print(f"  {key}")
            print(f"    Name: {info['name']}")
            print(f"    Description: {info['description']}")
            print(f"    URL: {info['url']}")
            print(f"    Source: {info['source']}")
            print()
        return 0
    
    if not args.dataset and not args.url:
        parser.error("Either --dataset or --url is required")
    
    if not args.out:
        if args.dataset:
            args.out = Path(f"allen_{args.dataset}")
        else:
            parser.error("--out is required when using --url")
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    if args.dataset:
        records = download_dataset(
            args.dataset,
            args.out,
            recursive=not args.no_recursive,
        )
    else:
        records = download_from_url(
            args.url,
            args.out,
            recursive=not args.no_recursive,
        )
    
    manifest_path = args.out / "download_log.jsonl"
    write_manifest(records, manifest_path)
    
    ok = sum(1 for r in records if r["status"] == "ok")
    exists = sum(1 for r in records if r["status"] == "exists")
    failed = sum(1 for r in records if r["status"] == "failed")
    
    print()
    print(f"Download complete!")
    print(f"  New: {ok}")
    print(f"  Existing: {exists}")
    print(f"  Failed: {failed}")
    print(f"  Manifest: {manifest_path}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
