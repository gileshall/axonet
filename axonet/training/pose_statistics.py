"""Pose statistics for analyzing intra- and inter-pose embedding variance."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def compute_pose_statistics(
    embeddings: np.ndarray,
    swc_names: List[str],
    cell_specimen_ids: Optional[List[str]] = None,
) -> Dict:
    """Compute intra- and inter-pose embedding statistics.
    
    Args:
        embeddings: (N, D) numpy array of embeddings
        swc_names: List of SWC filenames (N elements)
        cell_specimen_ids: Optional list of cell specimen IDs (N elements)
    
    Returns:
        Dict with per-neuron and aggregate statistics
    """
    logger.info(f"Computing pose statistics for {len(embeddings)} embeddings")
    
    grouped_by_neuron = defaultdict(list)
    
    for idx, swc_name in enumerate(swc_names):
        cell_id = cell_specimen_ids[idx] if cell_specimen_ids else extract_cell_id(swc_name)
        if not cell_id:
            cell_id = swc_name
        grouped_by_neuron[cell_id].append((idx, embeddings[idx]))
    
    logger.info(f"Grouped into {len(grouped_by_neuron)} unique neurons")
    
    per_neuron_stats = {}
    intra_variances = []
    intra_stds = []
    intra_cosine_sims = []
    
    for cell_id, pose_embeddings in grouped_by_neuron.items():
        indices = [idx for idx, _ in pose_embeddings]
        embs = np.array([emb for _, emb in pose_embeddings])
        
        n_poses = len(embs)
        mean_embedding = np.mean(embs, axis=0)
        
        variances = np.var(embs, axis=0)
        intra_variance = np.mean(variances)
        intra_std = np.std(embs)
        
        pairwise_cosine = []
        if n_poses > 1:
            for i in range(n_poses):
                for j in range(i + 1, n_poses):
                    sim = cosine_similarity([embs[i]], [embs[j]])[0, 0]
                    pairwise_cosine.append(float(sim))
        
        per_neuron_stats[cell_id] = {
            "n_poses": n_poses,
            "mean_embedding": mean_embedding.tolist(),
            "intra_pose_variance": float(intra_variance),
            "intra_pose_std": float(intra_std),
            "pairwise_cosine_similarities": pairwise_cosine,
        }
        
        intra_variances.append(intra_variance)
        intra_stds.append(intra_std)
        intra_cosine_sims.extend(pairwise_cosine)
    
    mean_embeddings = np.array([stats["mean_embedding"] for stats in per_neuron_stats.values()])
    
    inter_neuron_cosine = []
    inter_neuron_distances = []
    
    if len(mean_embeddings) > 1:
        cosine_matrix = cosine_similarity(mean_embeddings)
        n_neurons = len(mean_embeddings)
        
        for i in range(n_neurons):
            for j in range(i + 1, n_neurons):
                sim = cosine_matrix[i, j]
                dist = 1.0 - sim
                inter_neuron_cosine.append(float(sim))
                inter_neuron_distances.append(float(dist))
    
    aggregate_stats = {
        "mean_intra_pose_variance": float(np.mean(intra_variances)) if intra_variances else 0.0,
        "std_intra_pose_variance": float(np.std(intra_variances)) if intra_variances else 0.0,
        "mean_intra_pose_std": float(np.mean(intra_stds)) if intra_stds else 0.0,
        "std_intra_pose_std": float(np.std(intra_stds)) if intra_stds else 0.0,
        "mean_intra_pose_cosine_similarity": float(np.mean(intra_cosine_sims)) if intra_cosine_sims else 0.0,
        "std_intra_pose_cosine_similarity": float(np.std(intra_cosine_sims)) if intra_cosine_sims else 0.0,
        "mean_inter_neuron_cosine_similarity": float(np.mean(inter_neuron_cosine)) if inter_neuron_cosine else 0.0,
        "std_inter_neuron_cosine_similarity": float(np.std(inter_neuron_cosine)) if inter_neuron_cosine else 0.0,
        "mean_inter_neuron_distance": float(np.mean(inter_neuron_distances)) if inter_neuron_distances else 0.0,
        "std_inter_neuron_distance": float(np.std(inter_neuron_distances)) if inter_neuron_distances else 0.0,
    }
    
    result = {
        "per_neuron": per_neuron_stats,
        "aggregate": aggregate_stats,
    }
    
    logger.info(f"Computed statistics for {len(per_neuron_stats)} neurons")
    logger.info(f"Mean intra-pose variance: {aggregate_stats['mean_intra_pose_variance']:.6f}")
    logger.info(f"Mean inter-neuron distance: {aggregate_stats['mean_inter_neuron_distance']:.6f}")
    
    return result


def extract_cell_id(swc_name: str) -> Optional[str]:
    """Extract cell ID from SWC filename."""
    if not swc_name:
        return None
    parts = swc_name.split("_")
    if parts:
        return parts[0]
    return None


def save_pose_statistics(stats: Dict, output_path: Path) -> None:
    """Save pose statistics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved pose statistics to {output_path}")


def load_pose_statistics(input_path: Path) -> Dict:
    """Load pose statistics from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)

