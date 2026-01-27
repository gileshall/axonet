"""Curate NeuroMorpho dataset for CLIP training.

Reads metadata.jsonl, applies CLIP-optimized quality filters, computes
archive-stratified train/val/test splits, and derives percentile-based
morphometric bins from the training set.

Usage:
    python -m axonet.training.curate_neuromorpho \
        --metadata-dir neuromorpho_all_species \
        --output-dir curated \
        --val-ratio 0.1 --test-ratio 0.1

If a rendered dataset directory (--render-dir) is provided, also filters
and splits the manifest.jsonl from that directory.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Morphometric key mapping (metadata keys -> canonical names)
# ---------------------------------------------------------------------------

MORPH_KEYS = {
    "length": "length",
    "n_bifs": "n_bifs",
    "n_branch": "n_branch",
    "n_stems": "n_stems",
    "width": "width",
    "height": "height",
    "depth": "depth",
    "surface": "surface",
    "volume": "volume",
    "soma_Surface": "soma_Surface",
    "diameter": "diameter",
    "eucDistance": "eucDistance",
    "pathDistance": "pathDistance",
    "branch_Order": "branch_Order",
    "fragmentation": "fragmentation",
    "partition_asymmetry": "partition_asymmetry",
    "pk_classic": "pk_classic",
    "bif_ampl_local": "bif_ampl_local",
    "bif_ampl_remote": "bif_ampl_remote",
    "fractal_Dim": "fractal_Dim",
    "contraction": "contraction",
}


# ---------------------------------------------------------------------------
# Morphometric bin definitions
# ---------------------------------------------------------------------------

# Each entry: (morph_key, bin_labels, percentile_boundaries)
# Boundaries are percentile thresholds; labels correspond to ranges between them.
MORPH_BIN_SPECS = {
    "size": {
        "key": "length",
        "labels": ["small", "medium", "large", "very large", "extremely large"],
        "percentiles": [20, 40, 60, 80],
    },
    "branching": {
        "key": "n_bifs",
        "labels": ["sparsely branched", "moderately branched", "densely branched", "highly branched"],
        "percentiles": [25, 50, 75],
    },
    "extent": {
        "key": "_max_extent",  # computed: max(width, height, depth)
        "labels": ["compact", "extended", "sprawling"],
        "percentiles": [33, 67],
    },
    "straightness": {
        "key": "contraction",
        "labels": ["tortuous", "moderately straight", "straight"],
        "percentiles": [33, 67],
    },
    "stems": {
        "key": "n_stems",
        "labels": ["few primary processes", "several primary processes", "many primary processes"],
        "percentiles": [33, 67],
    },
    "complexity": {
        "key": "fractal_Dim",
        "labels": ["sparse", "moderately space-filling", "space-filling"],
        "percentiles": [33, 67],
    },
    "aspect": {
        "key": "_aspect_ratio",  # computed: width / height
        "labels": ["tall and narrow", "roughly symmetric", "wide and flat"],
        "percentiles": [33, 67],
    },
}


def _get_morph_value(entry: Dict[str, Any], key: str) -> Optional[float]:
    """Extract a morphometric value from metadata entry."""
    morph = entry.get("morphometry", {})
    if not morph or not isinstance(morph, dict):
        return None

    if key == "_max_extent":
        w = morph.get("width")
        h = morph.get("height")
        d = morph.get("depth")
        vals = [v for v in [w, h, d] if v is not None and v > 0]
        return max(vals) if vals else None

    if key == "_aspect_ratio":
        w = morph.get("width")
        h = morph.get("height")
        if w and h and h > 0:
            return w / h
        return None

    val = morph.get(key)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return None


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _has_valid_cell_type(entry: Dict[str, Any]) -> bool:
    ct = entry.get("cell_type", [])
    if isinstance(ct, str):
        return bool(ct and ct.lower() not in ("", "unknown", "not reported"))
    if isinstance(ct, list):
        return any(
            c.lower() not in ("", "unknown", "not reported")
            for c in ct if isinstance(c, str)
        )
    return False


def _has_valid_region(entry: Dict[str, Any]) -> bool:
    br = entry.get("brain_region", [])
    if isinstance(br, str):
        return bool(br and br.lower() not in ("", "unknown", "not reported"))
    if isinstance(br, list):
        return any(
            r.lower() not in ("", "unknown", "not reported")
            for r in br if isinstance(r, str)
        )
    return False


def filter_degenerate(entry: Dict[str, Any]) -> Optional[str]:
    """Filter 1: Degenerate morphology (won't render meaningfully)."""
    morph = entry.get("morphometry", {})
    if not morph or not isinstance(morph, dict):
        return None  # can't check, allow through

    n_bifs = morph.get("n_bifs")
    if n_bifs is not None and float(n_bifs) == 0:
        return "zero_bifurcations"

    n_branch = morph.get("n_branch")
    if n_branch is not None and float(n_branch) < 4:
        return "too_few_branches"

    width = morph.get("width")
    if width is not None and float(width) == 0:
        return "zero_width"

    depth = morph.get("depth")
    if depth is not None and float(depth) == 0:
        return "zero_depth"

    length = morph.get("length")
    if length is not None and float(length) < 200:
        return "too_short"

    return None


def filter_text_impoverished(entry: Dict[str, Any]) -> Optional[str]:
    """Filter 2: No cell type AND no brain region (nothing meaningful to say)."""
    if not _has_valid_cell_type(entry) and not _has_valid_region(entry):
        return "no_cell_type_or_region"
    return None


def filter_metadata_broken(entry: Dict[str, Any], failed_ids: set) -> Optional[str]:
    """Filter 3: Broken metadata or failed download."""
    integrity = entry.get("physical_Integrity", "")
    if isinstance(integrity, str) and "ERROR" in integrity.upper():
        return "integrity_error"

    nid = str(entry.get("neuron_id", entry.get("neuron_name", "")))
    if nid in failed_ids:
        return "download_failed"

    return None


def filter_rare_species(entry: Dict[str, Any], species_counts: Dict[str, int],
                        min_count: int = 10) -> Optional[str]:
    """Filter 4: Species with fewer than min_count neurons."""
    species = entry.get("species", "unknown")
    if species_counts.get(species, 0) < min_count:
        return f"rare_species_{species}"
    return None


# ---------------------------------------------------------------------------
# Archive-stratified splitting
# ---------------------------------------------------------------------------

def stratified_split(
    entries: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split entries into train/val/test stratified by archive.

    All neurons from the same archive stay proportionally distributed
    across splits. Splitting is done at the neuron (not view) level.
    """
    rng = random.Random(seed)

    # Group by archive
    archive_groups: Dict[str, List[Dict]] = defaultdict(list)
    for e in entries:
        archive = e.get("archive", "unknown")
        archive_groups[archive].append(e)

    train, val, test = [], [], []

    for archive, neurons in sorted(archive_groups.items()):
        rng.shuffle(neurons)
        n = len(neurons)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        test.extend(neurons[:n_test])
        val.extend(neurons[n_test:n_test + n_val])
        train.extend(neurons[n_test + n_val:])

    return train, val, test


# ---------------------------------------------------------------------------
# Morphometric bin computation
# ---------------------------------------------------------------------------

def compute_morph_bins(
    train_entries: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute percentile-based morphometric bins from training set.

    Returns a dict keyed by bin name, each containing:
        - thresholds: list of float boundary values
        - labels: list of string labels for each bin
        - key: the morphometry key used
    """
    import numpy as np

    bins = {}

    for bin_name, spec in MORPH_BIN_SPECS.items():
        key = spec["key"]
        values = []
        for e in train_entries:
            v = _get_morph_value(e, key)
            if v is not None and v > 0:
                values.append(v)

        if len(values) < 100:
            # Not enough data to compute meaningful percentiles
            continue

        arr = np.array(values)
        thresholds = [float(np.percentile(arr, p)) for p in spec["percentiles"]]

        bins[bin_name] = {
            "key": key,
            "labels": spec["labels"],
            "thresholds": thresholds,
            "percentiles": spec["percentiles"],
            "n_samples": len(values),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }

    return bins


def assign_morph_bin(value: Optional[float], bin_spec: Dict[str, Any]) -> Optional[str]:
    """Assign a single value to its morphometric bin label."""
    if value is None:
        return None
    thresholds = bin_spec["thresholds"]
    labels = bin_spec["labels"]
    for i, t in enumerate(thresholds):
        if value <= t:
            return labels[i]
    return labels[-1]


# ---------------------------------------------------------------------------
# Manifest filtering
# ---------------------------------------------------------------------------

def filter_manifest(
    manifest_path: Path,
    passing_ids: set,
    split_ids: Dict[str, set],
) -> Dict[str, List[Dict]]:
    """Filter and split a render manifest by neuron ID.

    Returns dict with keys 'train', 'val', 'test', each a list of manifest entries.
    """
    import re

    split_manifests: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            swc = entry.get("swc", "")
            # Extract numeric ID from SWC filename
            match = re.search(r"(\d+)", Path(swc).stem)
            nid = match.group(1) if match else Path(swc).stem

            if nid not in passing_ids:
                continue

            for split_name in ("train", "val", "test"):
                if nid in split_ids[split_name]:
                    split_manifests[split_name].append(entry)
                    break

    return split_manifests


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    report_path: Path,
    total: int,
    filter_counts: Dict[str, int],
    passing: int,
    species_counts: Dict[str, int],
    split_counts: Dict[str, int],
    morph_bins: Dict[str, Dict],
    manifest_split_counts: Optional[Dict[str, int]] = None,
):
    lines = []
    lines.append("NEUROMORPHO CURATION REPORT FOR CLIP TRAINING")
    lines.append("=" * 65)

    lines.append(f"\nTotal neurons in metadata: {total:,}")
    lines.append(f"\nFILTERS APPLIED")
    lines.append("-" * 50)
    for reason, count in sorted(filter_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {reason:40s} {count:>6,}")
    total_filtered = sum(filter_counts.values())
    lines.append(f"  {'TOTAL FILTERED':40s} {total_filtered:>6,}")
    lines.append(f"  {'PASSING':40s} {passing:>6,} ({passing/total*100:.1f}%)")

    lines.append(f"\nSPECIES DISTRIBUTION (passing)")
    lines.append("-" * 50)
    for sp, ct in sorted(species_counts.items(), key=lambda x: -x[1])[:30]:
        lines.append(f"  {sp:35s} {ct:>6,}")
    if len(species_counts) > 30:
        lines.append(f"  ... and {len(species_counts) - 30} more species")

    lines.append(f"\nSPLIT SUMMARY (neuron-level)")
    lines.append("-" * 50)
    for split, count in split_counts.items():
        lines.append(f"  {split:10s} {count:>6,} neurons")

    if manifest_split_counts:
        lines.append(f"\nMANIFEST SPLIT (view-level)")
        lines.append("-" * 50)
        for split, count in manifest_split_counts.items():
            lines.append(f"  {split:10s} {count:>6,} views")

    lines.append(f"\nMORPHOMETRIC BINS (computed from training set)")
    lines.append("-" * 50)
    for bin_name, spec in sorted(morph_bins.items()):
        lines.append(f"  {bin_name} (key={spec['key']}, n={spec['n_samples']:,}):")
        for i, label in enumerate(spec["labels"]):
            if i == 0:
                lines.append(f"    {label:30s} <= {spec['thresholds'][0]:.2f}")
            elif i < len(spec["thresholds"]):
                lines.append(f"    {label:30s} <= {spec['thresholds'][i]:.2f}")
            else:
                lines.append(f"    {label:30s} > {spec['thresholds'][-1]:.2f}")

    lines.append("\n" + "=" * 65)

    report = "\n".join(lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def curate(
    metadata_dir: Path,
    output_dir: Path,
    render_dir: Optional[Path] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_species: int = 10,
    seed: int = 42,
):
    """Run full curation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_path = metadata_dir / "metadata.jsonl"
    print(f"Loading metadata from {metadata_path}...")
    entries = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"  Loaded {len(entries):,} neurons")

    # Load download log for failed IDs
    failed_ids = set()
    log_path = metadata_dir / "download_log.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") == "failed":
                    nid = str(rec.get("neuron_id", rec.get("neuron_name", "")))
                    if nid:
                        failed_ids.add(nid)
        print(f"  Found {len(failed_ids):,} failed downloads")

    # Count species before filtering (for rare-species filter)
    species_counter: Counter = Counter()
    for e in entries:
        species_counter[e.get("species", "unknown")] += 1

    # Apply filters
    print("\nApplying filters...")
    filter_counts: Counter = Counter()
    passing = []

    for e in entries:
        # Filter 1: degenerate morphology
        reason = filter_degenerate(e)
        if reason:
            filter_counts[reason] += 1
            continue

        # Filter 2: text-impoverished
        reason = filter_text_impoverished(e)
        if reason:
            filter_counts[reason] += 1
            continue

        # Filter 3: broken metadata
        reason = filter_metadata_broken(e, failed_ids)
        if reason:
            filter_counts[reason] += 1
            continue

        # Filter 4: rare species
        reason = filter_rare_species(e, species_counter, min_count=min_species)
        if reason:
            filter_counts[reason] += 1
            continue

        passing.append(e)

    print(f"  {len(entries):,} -> {len(passing):,} neurons ({len(passing)/len(entries)*100:.1f}% retained)")

    # Archive-stratified split
    print("\nSplitting (archive-stratified)...")
    train, val, test = stratified_split(passing, val_ratio, test_ratio, seed)

    split_counts = {"train": len(train), "val": len(val), "test": len(test)}
    print(f"  train={len(train):,}, val={len(val):,}, test={len(test):,}")

    # Species distribution of passing neurons
    passing_species: Counter = Counter()
    for e in passing:
        passing_species[e.get("species", "unknown")] += 1

    # Compute morphometric bins from training set
    print("\nComputing morphometric bins from training set...")
    morph_bins = compute_morph_bins(train)
    print(f"  Computed {len(morph_bins)} bin dimensions")

    # Write outputs
    print("\nWriting outputs...")

    # Filtered metadata per split
    for name, split_entries in [("train", train), ("val", val), ("test", test)]:
        out = output_dir / f"metadata_{name}.jsonl"
        with open(out, "w") as f:
            for e in split_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  {out.name}: {len(split_entries):,} entries")

    # Morph bins
    bins_path = output_dir / "morph_bins.json"
    with open(bins_path, "w") as f:
        json.dump(morph_bins, f, indent=2)
    print(f"  {bins_path.name}")

    # Build ID sets for manifest filtering
    def _ids(split_list):
        ids = set()
        for e in split_list:
            nid = str(e.get("neuron_id", e.get("neuron_name", "")))
            if nid:
                ids.add(nid)
        return ids

    passing_ids = _ids(passing)
    split_ids = {
        "train": _ids(train),
        "val": _ids(val),
        "test": _ids(test),
    }

    # Filter manifest if render dir provided
    manifest_split_counts = None
    if render_dir:
        manifest_path = render_dir / "manifest.jsonl"
        if manifest_path.exists():
            print(f"\nFiltering manifest from {manifest_path}...")
            split_manifests = filter_manifest(manifest_path, passing_ids, split_ids)

            manifest_split_counts = {}
            for split_name, views in split_manifests.items():
                if views:
                    out = output_dir / f"manifest_{split_name}.jsonl"
                    with open(out, "w") as f:
                        for v in views:
                            f.write(json.dumps(v, ensure_ascii=False) + "\n")
                    manifest_split_counts[split_name] = len(views)
                    print(f"  manifest_{split_name}.jsonl: {len(views):,} views")
        else:
            print(f"\n  No manifest.jsonl found in {render_dir}")

    # Write report
    report_path = output_dir / "curation_report.txt"
    write_report(
        report_path,
        total=len(entries),
        filter_counts=dict(filter_counts),
        passing=len(passing),
        species_counts=dict(passing_species),
        split_counts=split_counts,
        morph_bins=morph_bins,
        manifest_split_counts=manifest_split_counts,
    )

    # Write passing IDs (useful for external tools)
    ids_path = output_dir / "passing_ids.txt"
    with open(ids_path, "w") as f:
        for nid in sorted(passing_ids):
            f.write(nid + "\n")
    print(f"\n  {ids_path.name}: {len(passing_ids):,} IDs")

    print(f"\nCuration complete. Outputs in {output_dir}/")
    return {
        "passing": len(passing),
        "filtered": len(entries) - len(passing),
        "splits": split_counts,
        "morph_bins": morph_bins,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Curate NeuroMorpho dataset for CLIP training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata-dir", type=Path, required=True,
        help="Directory containing metadata.jsonl and download_log.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for curated splits and bins",
    )
    parser.add_argument(
        "--render-dir", type=Path, default=None,
        help="Rendered dataset directory containing manifest.jsonl (optional)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-species", type=int, default=10,
                        help="Minimum neurons per species to include")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    curate(
        metadata_dir=args.metadata_dir,
        output_dir=args.output_dir,
        render_dir=args.render_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_species=args.min_species,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
