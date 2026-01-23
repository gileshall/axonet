"""Main orchestration module for linear probing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..models.d3_swc_vae import load_model
from .depth_diagnostics import compute_depth_diagnostics
from .grouped_cv import create_grouped_splits, get_group_statistics
from .pose_statistics import compute_pose_statistics, save_pose_statistics
from .probe_metrics import aggregate_metrics_across_folds, compute_classification_metrics, compute_regression_metrics
from .probe_models import build_probe
from .probe_trainer import build_label_mappings, train_probe
from .probing_dataset import ProbingDataset

logger = logging.getLogger(__name__)


def run_linear_probing(
    vae_checkpoint: Path,
    swc_paths: Optional[List[Path]] = None,
    manifest_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    swc_root: Optional[Path] = None,
    metadata_csv: Path = None,
    output_dir: Path = None,
    *,
    tasks: List[str] = None,
    probe_type: str = "linear",
    use_norm: bool = False,
    hidden_dim: Optional[int] = None,
    n_splits: int = 5,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    task_weights: Dict[str, float] = None,
    use_class_weights: bool = True,
    early_stopping_patience: int = 10,
    early_stopping_metric: str = "macro_f1",
    use_mu: bool = True,
    embedding_reduce: str = "mean",
    width: int = 1024,
    height: int = 1024,
    pose_sampling: str = "random",
    n_poses_per_neuron: int = 1,
    compute_pose_stats: bool = True,
    compute_depth_diag: bool = True,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> Dict:
    """Run linear probing with cross-validation.
    
    Args:
        vae_checkpoint: Path to VAE checkpoint
        swc_paths: List of SWC file paths (if using direct paths)
        manifest_path: Path to manifest.jsonl (if using manifest)
        data_dir: Data root directory (required if using manifest)
        metadata_csv: Path to PatchSeq_metadata.csv
        output_dir: Output directory for results
        tasks: List of tasks to probe
        probe_type: Probe architecture ("linear", "mlp")
        use_norm: Whether to use LayerNorm (for linear probes)
        hidden_dim: Hidden dimension for MLP probe
        n_splits: Number of CV folds
        batch_size: Batch size
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        task_weights: Dict mapping task names to loss weights
        use_class_weights: Whether to use class weights
        early_stopping_patience: Early stopping patience
        early_stopping_metric: Metric to monitor
        use_mu: Use mu (deterministic) instead of z (stochastic)
        embedding_reduce: How to reduce spatial dimensions
        width: Render width
        height: Render height
        pose_sampling: How to sample poses
        n_poses_per_neuron: Number of poses per neuron
        compute_pose_stats: Whether to compute pose statistics
        compute_depth_diag: Whether to compute depth diagnostics
        device: Device string
        seed: Random seed
    
    Returns:
        Dict with results
    """
    logger.info("="*60)
    logger.info("Linear Probing Starting")
    logger.info("="*60)
    
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    logger.info(f"Loading VAE model from {vae_checkpoint}")
    vae_model = load_model(
        vae_checkpoint,
        device,
        embedding_only=True,
    )
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False
    
    logger.info(f"Creating probing dataset")
    dataset = ProbingDataset(
        model=vae_model,
        device=device,
        swc_paths=swc_paths,
        manifest_path=manifest_path,
        data_dir=data_dir,
        swc_root=swc_root,
        metadata_csv=metadata_csv,
        tasks=tasks,
        use_mu=use_mu,
        embedding_reduce=embedding_reduce,
        width=width,
        height=height,
        pose_sampling=pose_sampling,
        n_poses_per_neuron=n_poses_per_neuron,
        seed=seed,
    )
    
    logger.info(f"Dataset size: {len(dataset)} samples")
    
    if compute_pose_stats:
        logger.info("Computing pose statistics (sampling first 50 samples)")
        embeddings_list = []
        swc_names_list = []
        cell_ids_list = []
        
        n_samples_for_stats = min(50, len(dataset))
        logger.info(f"Processing {n_samples_for_stats} samples for pose statistics...")
        for i in range(n_samples_for_stats):
            if (i + 1) % 10 == 0:
                logger.info(f"  Pose stats: {i+1}/{n_samples_for_stats}")
            sample = dataset[i]
            embeddings_list.append(sample["embedding"].numpy())
            swc_names_list.append(sample["swc_name"])
            cell_ids_list.append(sample.get("cell_specimen_id", ""))
        
        if embeddings_list:
            embeddings_array = np.array(embeddings_list)
            pose_stats = compute_pose_statistics(
                embeddings_array,
                swc_names_list,
                cell_ids_list,
            )
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                save_pose_statistics(pose_stats, output_dir / "pose_statistics.json")
        else:
            pose_stats = None
    else:
        pose_stats = None
    
    logger.info("Building label mappings")
    label_mappings = build_label_mappings(dataset, tasks)
    
    embedding_dim = dataset[0]["embedding"].shape[0]
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    task_configs = {}
    for task in tasks:
        if task == "depth":
            task_configs[task] = {"regression": True}
        else:
            if task in label_mappings:
                task_configs[task] = {
                    "num_classes": label_mappings[task]["num_classes"],
                    "regression": False,
                }
            else:
                logger.warning(f"No label mapping for task {task}, skipping")
    
    logger.info("Creating CV splits")
    logger.info("Extracting group keys from dataset samples (without rendering)...")
    samples = []
    for i in range(len(dataset)):
        sample_dict = dataset.samples[i]
        metadata = sample_dict["metadata"]
        cell_specimen_id = sample_dict.get("cell_specimen_id", "")
        
        donor_id = metadata.get("donor_id", "")
        if donor_id:
            group_key = f"donor_{donor_id}"
        elif cell_specimen_id:
            group_key = f"cell_{cell_specimen_id}"
        else:
            group_key = f"unknown_{sample_dict['swc_name']}"
        
        sample = {
            "group_key": group_key,
            "swc_name": sample_dict["swc_name"],
            "cell_specimen_id": cell_specimen_id,
        }
        samples.append(sample)
    logger.info("Creating grouped CV splits...")
    splits = create_grouped_splits(samples, n_splits=n_splits, group_key="group_key")
    
    group_stats = get_group_statistics(samples, group_key="group_key")
    logger.info(f"Group statistics: {group_stats}")
    
    fold_metrics = []
    fold_models = []
    
    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        logger.info(f"\nFold {fold_idx + 1}/{n_splits}")
        logger.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
        
        from torch.utils.data import Subset
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        probe = build_probe(
            embedding_dim=embedding_dim,
            task_configs=task_configs,
            probe_type=probe_type,
            use_norm=use_norm,
            hidden_dim=hidden_dim,
        )
        
        trained_probe, history = train_probe(
            model=probe,
            train_dataset=train_subset,
            val_dataset=val_subset,
            tasks=tasks,
            label_mappings=label_mappings,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            task_weights=task_weights,
            use_class_weights=use_class_weights,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            device=device,
        )
        
        fold_models.append(trained_probe.state_dict())
        
        val_metrics = history["val_metrics"][-1]
        fold_metrics.append(val_metrics)
        
        logger.info(f"Fold {fold_idx + 1} metrics:")
        for task in tasks:
            if task in val_metrics:
                if task == "depth":
                    logger.info(f"  {task}: R² = {val_metrics[task]['r2']:.4f}, RMSE = {val_metrics[task]['rmse']:.4f}")
                else:
                    logger.info(f"  {task}: Accuracy = {val_metrics[task]['accuracy']:.4f}, Macro-F1 = {val_metrics[task]['macro_f1']:.4f}")
    
    logger.info("\nAggregating metrics across folds")
    aggregated_metrics = {}
    for task in tasks:
        task_fold_metrics = [m[task] for m in fold_metrics if task in m]
        if task_fold_metrics:
            aggregated_metrics[task] = aggregate_metrics_across_folds(task_fold_metrics)
    
    results = {
        "tasks": tasks,
        "probe_type": probe_type,
        "n_splits": n_splits,
        "group_statistics": group_stats,
        "fold_metrics": fold_metrics,
        "aggregated_metrics": aggregated_metrics,
        "pose_statistics": pose_stats,
    }
    
    if compute_depth_diag and "depth" in tasks:
        logger.info("Computing depth diagnostics")
        all_embeddings = []
        all_depth_labels = []
        all_classification_embeddings = None
        all_classification_labels = None
        
        classification_task = None
        for task in tasks:
            if task != "depth" and task in label_mappings:
                classification_task = task
                break
        
        for i in range(len(dataset)):
            sample = dataset[i]
            all_embeddings.append(sample["embedding"].numpy())
            if "depth" in sample["labels"]:
                all_depth_labels.append(sample["labels"]["depth"])
            
            if classification_task and classification_task in sample["labels"]:
                if all_classification_embeddings is None:
                    all_classification_embeddings = []
                    all_classification_labels = []
                all_classification_embeddings.append(sample["embedding"].numpy())
                label = sample["labels"][classification_task]
                if classification_task in label_mappings:
                    label_to_idx = label_mappings[classification_task]["label_to_idx"]
                    idx = label_to_idx.get(str(label), 0)
                    all_classification_labels.append(idx)
        
        if all_embeddings and all_depth_labels:
            embeddings_array = np.array(all_embeddings)
            depth_labels_array = np.array(all_depth_labels)
            
            classification_embeddings_array = None
            classification_labels_array = None
            if all_classification_embeddings and all_classification_labels:
                classification_embeddings_array = np.array(all_classification_embeddings)
                classification_labels_array = np.array(all_classification_labels)
            
            depth_diag = compute_depth_diagnostics(
                embeddings_array,
                depth_labels_array,
                classification_embeddings_array,
                classification_labels_array,
                classification_task or "ttype",
            )
            results["depth_diagnostics"] = depth_diag
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving results to {output_dir}")
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        best_fold_idx = np.argmax([
            max([m.get(task, {}).get("macro_f1", 0) for task in tasks if task in m])
            for m in fold_metrics
        ])
        
        best_model_state = fold_models[best_fold_idx]
        checkpoint_path = output_dir / "probe_checkpoint.pt"
        torch.save({
            "model_state_dict": best_model_state,
            "embedding_dim": embedding_dim,
            "task_configs": task_configs,
            "probe_type": probe_type,
            "use_norm": use_norm,
            "hidden_dim": hidden_dim,
            "label_mappings": label_mappings,
        }, checkpoint_path)
        
        logger.info(f"Saved best model from fold {best_fold_idx + 1} to {checkpoint_path}")
        logger.info(f"  Model size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    logger.info("="*60)
    logger.info("Linear Probing Complete")
    logger.info("="*60)
    
    return results

