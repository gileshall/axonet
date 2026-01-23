"""Depth dominance diagnostics for linear probing."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .probe_metrics import compute_classification_metrics, compute_regression_metrics

logger = logging.getLogger(__name__)


def extract_depth_direction(
    depth_probe: nn.Module,
    embedding_dim: int,
    device: str,
) -> np.ndarray:
    """Extract depth prediction direction from trained depth probe.
    
    Args:
        depth_probe: Trained linear probe for depth regression
        embedding_dim: Dimension of embeddings
        device: Device string
    
    Returns:
        Normalized weight vector w_hat
    """
    if hasattr(depth_probe, "head"):
        weight = depth_probe.head.weight.data.cpu().numpy()
    elif hasattr(depth_probe, "fc2"):
        weight = depth_probe.fc2.weight.data.cpu().numpy()
    else:
        raise ValueError("Cannot extract weights from depth probe")
    
    w = weight[0]
    w_norm = np.linalg.norm(w)
    if w_norm > 0:
        w_hat = w / w_norm
    else:
        w_hat = w
    
    logger.info(f"Extracted depth direction vector (norm: {np.linalg.norm(w_hat):.6f})")
    
    return w_hat


def project_out_depth(
    embeddings: np.ndarray,
    depth_direction: np.ndarray,
) -> np.ndarray:
    """Project out depth direction from embeddings.
    
    Args:
        embeddings: (N, D) embeddings
        depth_direction: (D,) normalized depth direction vector
    
    Returns:
        (N, D) embeddings with depth direction removed
    """
    depth_direction = depth_direction.reshape(1, -1)
    
    projections = np.dot(embeddings, depth_direction.T)
    embeddings_projected = embeddings - projections * depth_direction
    
    logger.info(f"Projected out depth direction from {len(embeddings)} embeddings")
    
    return embeddings_projected


def compute_depth_diagnostics(
    embeddings: np.ndarray,
    depth_labels: np.ndarray,
    classification_embeddings: Optional[np.ndarray] = None,
    classification_labels: Optional[np.ndarray] = None,
    classification_task_name: str = "ttype",
) -> Dict:
    """Compute depth dominance diagnostics.
    
    Args:
        embeddings: (N, D) embeddings
        depth_labels: (N,) depth labels
        classification_embeddings: Optional (N, D) embeddings for classification
        classification_labels: Optional (N,) classification labels
        classification_task_name: Name of classification task
    
    Returns:
        Dict with diagnostics
    """
    logger.info("Computing depth dominance diagnostics")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score
    
    regressor = LinearRegression()
    regressor.fit(embeddings, depth_labels)
    depth_pred = regressor.predict(embeddings)
    depth_metrics = compute_regression_metrics(depth_labels, depth_pred)
    
    w = regressor.coef_
    w_norm = np.linalg.norm(w)
    if w_norm > 0:
        w_hat = w / w_norm
    else:
        w_hat = w
    
    diagnostics = {
        "depth_prediction": {
            "r2": depth_metrics["r2"],
            "rmse": depth_metrics["rmse"],
            "mae": depth_metrics["mae"],
        },
        "depth_direction": {
            "norm": float(w_norm),
            "direction_norm": float(np.linalg.norm(w_hat)),
        },
    }
    
    if classification_embeddings is not None and classification_labels is not None:
        logger.info("Computing classification metrics before/after depth removal")
        
        from sklearn.linear_model import LogisticRegression
        
        clf_before = LogisticRegression(max_iter=1000, random_state=42)
        clf_before.fit(classification_embeddings, classification_labels)
        pred_before = clf_before.predict(classification_embeddings)
        metrics_before = compute_classification_metrics(
            classification_labels,
            pred_before,
        )
        
        embeddings_no_depth = project_out_depth(classification_embeddings, w_hat)
        
        clf_after = LogisticRegression(max_iter=1000, random_state=42)
        clf_after.fit(embeddings_no_depth, classification_labels)
        pred_after = clf_after.predict(embeddings_no_depth)
        metrics_after = compute_classification_metrics(
            classification_labels,
            pred_after,
        )
        
        diagnostics["classification"] = {
            "before_depth_removal": {
                "accuracy": metrics_before["accuracy"],
                "macro_f1": metrics_before["macro_f1"],
            },
            "after_depth_removal": {
                "accuracy": metrics_after["accuracy"],
                "macro_f1": metrics_after["macro_f1"],
            },
            "accuracy_change": metrics_after["accuracy"] - metrics_before["accuracy"],
            "macro_f1_change": metrics_after["macro_f1"] - metrics_before["macro_f1"],
        }
        
        logger.info(f"Classification accuracy: {metrics_before['accuracy']:.4f} -> {metrics_after['accuracy']:.4f}")
        logger.info(f"Classification macro-F1: {metrics_before['macro_f1']:.4f} -> {metrics_after['macro_f1']:.4f}")
    
    depth_dominance = diagnostics["depth_prediction"]["r2"] > 0.5
    diagnostics["depth_dominance"] = {
        "is_dominant": depth_dominance,
        "r2_threshold": 0.5,
    }
    
    logger.info(f"Depth prediction R²: {diagnostics['depth_prediction']['r2']:.4f}")
    logger.info(f"Depth dominance: {depth_dominance}")
    
    return diagnostics

