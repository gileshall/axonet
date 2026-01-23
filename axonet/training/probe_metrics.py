"""Evaluation metrics for linear probing."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
) -> Dict:
    """Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of all label values
    
    Returns:
        Dict with metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    present_labels = sorted(list(set(y_true) | set(y_pred)))
    
    if labels is None:
        labels = present_labels
    else:
        labels = [l for l in labels if l in present_labels]
        if not labels:
            labels = present_labels
    
    if not labels:
        labels = [0]
    
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1_dict = {str(label): float(f1) for label, f1 in zip(labels, per_class_f1)}
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_class_f1_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [str(l) for l in labels],
    }
    
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict:
    """Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dict with metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }
    
    return metrics


def compute_hierarchical_metrics(
    y_true: List[str],
    y_pred: List[str],
    extract_family: callable,
) -> Dict:
    """Compute hierarchical metrics (family-level and leaf-level).
    
    Args:
        y_true: True labels (full labels)
        y_pred: Predicted labels (full labels)
        extract_family: Function to extract family from full label
    
    Returns:
        Dict with family and leaf metrics
    """
    y_true_family = [extract_family(label) for label in y_true]
    y_pred_family = [extract_family(label) for label in y_pred]
    
    y_true_array = np.array(y_true)
    y_pred_array = np.array(y_pred)
    y_true_family_array = np.array(y_true_family)
    y_pred_family_array = np.array(y_pred_family)
    
    leaf_metrics = compute_classification_metrics(y_true_array, y_pred_array)
    family_metrics = compute_classification_metrics(y_true_family_array, y_pred_family_array)
    
    return {
        "leaf": leaf_metrics,
        "family": family_metrics,
    }


def aggregate_metrics_across_folds(fold_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across cross-validation folds.
    
    Args:
        fold_metrics: List of metric dicts from each fold
    
    Returns:
        Dict with mean and std of metrics
    """
    if not fold_metrics:
        return {}
    
    aggregated = {}
    
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics if key in m]
        if values:
            if isinstance(values[0], (int, float)):
                aggregated[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
            elif isinstance(values[0], dict):
                aggregated[key] = aggregate_metrics_across_folds(values)
            elif isinstance(values[0], list):
                aggregated[key] = {
                    "mean": np.mean(values, axis=0).tolist(),
                    "std": np.std(values, axis=0).tolist(),
                }
            else:
                aggregated[key] = values[0]
    
    return aggregated


def compute_per_neuron_metrics(
    predictions: Dict[str, List],
    labels: Dict[str, List],
    swc_names: List[str],
    task_name: str,
    is_regression: bool = False,
) -> Dict:
    """Compute metrics aggregated per neuron (across poses).
    
    Args:
        predictions: Dict mapping swc_name to list of predictions
        labels: Dict mapping swc_name to list of labels
        swc_names: List of SWC names
        task_name: Name of the task
        is_regression: Whether this is a regression task
    
    Returns:
        Dict with per-neuron metrics
    """
    grouped_by_neuron = defaultdict(lambda: {"preds": [], "labels": []})
    
    for swc_name in swc_names:
        cell_id = extract_cell_id(swc_name)
        if not cell_id:
            cell_id = swc_name
        
        if swc_name in predictions and swc_name in labels:
            grouped_by_neuron[cell_id]["preds"].extend(predictions[swc_name])
            grouped_by_neuron[cell_id]["labels"].extend(labels[swc_name])
    
    per_neuron_metrics = {}
    
    for cell_id, data in grouped_by_neuron.items():
        if not data["preds"]:
            continue
        
        preds = np.array(data["preds"])
        labels_array = np.array(data["labels"])
        
        if is_regression:
            preds_mean = np.mean(preds)
            labels_mean = np.mean(labels_array)
            metrics = compute_regression_metrics(
                np.array([labels_mean]),
                np.array([preds_mean]),
            )
        else:
            preds_mode = most_common(preds)
            labels_mode = most_common(labels_array)
            metrics = compute_classification_metrics(
                np.array([labels_mode]),
                np.array([preds_mode]),
            )
        
        per_neuron_metrics[cell_id] = metrics
    
    return per_neuron_metrics


def extract_cell_id(swc_name: str) -> Optional[str]:
    """Extract cell ID from SWC filename."""
    if not swc_name:
        return None
    parts = swc_name.split("_")
    if parts:
        return parts[0]
    return None


def most_common(arr: np.ndarray):
    """Get most common value in array."""
    values, counts = np.unique(arr, return_counts=True)
    return values[np.argmax(counts)]

