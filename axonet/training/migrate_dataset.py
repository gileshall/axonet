"""
One-off script to migrate old dataset format to new format with SWC-level splitting.

Reads old manifest.jsonl and creates separate manifest files for train/val/test
splits at the SWC (sample) level.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def group_entries_by_swc(manifest_path: Path) -> Dict[str, List[dict]]:
    """Group manifest entries by SWC file name."""
    swc_groups = defaultdict(list)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            swc_name = entry["swc"]
            swc_groups[swc_name].append(entry)
    
    return dict(swc_groups)


def split_swc_files(swc_files: List[str], val_ratio: float, test_ratio: float, seed: int) -> tuple[List[str], List[str], List[str]]:
    """Split SWC files into train/val/test sets."""
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


def write_manifest(entries: List[dict], output_path: Path):
    """Write manifest entries to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate old dataset format to new format with SWC-level splitting"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory containing the old dataset (with manifest.jsonl)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Validation set ratio (0.0 = no validation set, default: 0.0)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Test set ratio (0.0 = no test set, default: 0.0)"
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for SWC file splitting (default: 42)"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of original manifest.jsonl"
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "manifest.jsonl"
    
    if not manifest_path.exists():
        raise SystemExit(f"Manifest file not found: {manifest_path}")
    
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.test_ratio < 0 or args.test_ratio >= 1:
        raise ValueError("--test-ratio must be in [0, 1)")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("--val-ratio + --test-ratio must be < 1")
    
    print(f"Reading manifest from {manifest_path}...")
    swc_groups = group_entries_by_swc(manifest_path)
    
    unique_swc_files = sorted(swc_groups.keys())
    print(f"Found {len(unique_swc_files)} unique SWC files")
    
    total_entries = sum(len(entries) for entries in swc_groups.values())
    print(f"Total entries: {total_entries}")
    
    train_swc, val_swc, test_swc = split_swc_files(
        unique_swc_files,
        args.val_ratio,
        args.test_ratio,
        args.split_seed
    )
    
    print(f"\nSplit: train={len(train_swc)}, val={len(val_swc)}, test={len(test_swc)}")
    
    train_entries = []
    for swc_name in train_swc:
        train_entries.extend(swc_groups[swc_name])
    
    val_entries = []
    for swc_name in val_swc:
        val_entries.extend(swc_groups[swc_name])
    
    test_entries = []
    for swc_name in test_swc:
        test_entries.extend(swc_groups[swc_name])
    
    print(f"\nEntry counts: train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}")
    
    if args.backup:
        backup_path = dataset_dir / "manifest.jsonl.backup"
        print(f"\nCreating backup: {backup_path}")
        import shutil
        shutil.copy2(manifest_path, backup_path)
    
    print("\nWriting new manifest files...")
    write_manifest(train_entries, dataset_dir / "manifest_train.jsonl")
    print(f"  ✓ manifest_train.jsonl ({len(train_entries)} entries)")
    
    if val_entries:
        write_manifest(val_entries, dataset_dir / "manifest_val.jsonl")
        print(f"  ✓ manifest_val.jsonl ({len(val_entries)} entries)")
    
    if test_entries:
        write_manifest(test_entries, dataset_dir / "manifest_test.jsonl")
        print(f"  ✓ manifest_test.jsonl ({len(test_entries)} entries)")
    
    print(f"\n{'='*60}")
    print("✓ Migration complete!")
    print(f"  Train: {len(train_entries)} entries from {len(train_swc)} SWC files")
    if val_swc:
        print(f"  Val:   {len(val_entries)} entries from {len(val_swc)} SWC files")
    if test_swc:
        print(f"  Test:  {len(test_entries)} entries from {len(test_swc)} SWC files")
    print(f"{'='*60}")
    print(f"\nNote: Original manifest.jsonl is preserved.")
    if args.backup:
        print(f"Backup created at: manifest.jsonl.backup")


if __name__ == "__main__":
    main()

