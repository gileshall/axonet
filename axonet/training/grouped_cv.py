"""Grouped cross-validation utilities for leakage-safe splits."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)


def create_grouped_splits(
    samples: List[Dict],
    n_splits: int = 5,
    group_key: str = "group_key",
) -> List[Tuple[List[int], List[int]]]:
    """Create grouped cross-validation splits.
    
    Args:
        samples: List of sample dicts with group_key field
        n_splits: Number of folds
        group_key: Key in sample dict to use for grouping
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    groups = [sample[group_key] for sample in samples]
    indices = np.arange(len(samples))
    
    group_kfold = GroupKFold(n_splits=n_splits)
    splits = []
    
    for train_idx, val_idx in group_kfold.split(indices, groups=groups):
        splits.append((train_idx.tolist(), val_idx.tolist()))
    
    logger.info(f"Created {len(splits)} grouped CV splits")
    logger.info(f"Group distribution: {len(set(groups))} unique groups")
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_groups = set(groups[i] for i in train_idx)
        val_groups = set(groups[i] for i in val_idx)
        overlap = train_groups & val_groups
        if overlap:
            logger.warning(f"Fold {fold_idx}: {len(overlap)} groups overlap between train and val!")
        logger.debug(f"Fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val")
    
    return splits


def get_group_statistics(samples: List[Dict], group_key: str = "group_key") -> Dict:
    """Get statistics about groups in samples.
    
    Args:
        samples: List of sample dicts
        group_key: Key in sample dict to use for grouping
    
    Returns:
        Dict with group statistics
    """
    groups = [sample[group_key] for sample in samples]
    unique_groups = set(groups)
    
    group_counts = {}
    for group in groups:
        group_counts[group] = group_counts.get(group, 0) + 1
    
    stats = {
        "n_samples": len(samples),
        "n_unique_groups": len(unique_groups),
        "samples_per_group": {
            "mean": np.mean(list(group_counts.values())),
            "std": np.std(list(group_counts.values())),
            "min": min(group_counts.values()),
            "max": max(group_counts.values()),
        },
    }
    
    return stats

