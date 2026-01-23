"""Training loop for linear probes."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .probe_models import MultiTaskProbe
from .probing_dataset import ProbingDataset

logger = logging.getLogger(__name__)


def collate_probing_batch(batch):
    """Custom collate function for probing dataset batches."""
    embeddings = torch.stack([item["embedding"] for item in batch])
    labels = [item["labels"] for item in batch]
    group_keys = [item["group_key"] for item in batch]
    swc_names = [item["swc_name"] for item in batch]
    cell_specimen_ids = [item.get("cell_specimen_id", "") for item in batch]
    pose_indices = [item.get("pose_idx", 0) for item in batch]
    n_poses = [item.get("n_poses", 1) for item in batch]
    
    return {
        "embedding": embeddings,
        "labels": labels,
        "group_key": group_keys,
        "swc_name": swc_names,
        "cell_specimen_id": cell_specimen_ids,
        "pose_idx": pose_indices,
        "n_poses": n_poses,
    }


class MultiTaskLoss(nn.Module):
    """Multi-task loss for linear probing."""
    
    def __init__(
        self,
        task_weights: Dict[str, float] = None,
        class_weights: Dict[str, Dict] = None,
    ):
        """Initialize multi-task loss.
        
        Args:
            task_weights: Dict mapping task names to loss weights
            class_weights: Dict mapping task names to class weight dicts
        """
        super().__init__()
        self.task_weights = task_weights or {}
        self.class_weights = class_weights or {}
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task loss.
        
        Args:
            outputs: Dict mapping task names to predictions
            labels: Dict mapping task names to targets
        
        Returns:
            Total loss and per-task losses
        """
        total_loss = 0.0
        task_losses = {}
        
        for task_name, pred in outputs.items():
            if task_name not in labels:
                continue
            
            target = labels[task_name]
            weight = self.task_weights.get(task_name, 1.0)
            
            if task_name in self.class_weights:
                class_weights_dict = self.class_weights[task_name]
                num_classes = pred.shape[-1] if pred.shape[-1] > 1 else max(class_weights_dict.keys()) + 1
                class_weights_list = [class_weights_dict.get(i, 1.0) for i in range(num_classes)]
                class_weights_tensor = torch.tensor(
                    class_weights_list,
                    device=pred.device,
                    dtype=pred.dtype,
                )
            else:
                class_weights_tensor = None
            
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
                loss = F.mse_loss(pred, target.float())
            else:
                loss = F.cross_entropy(
                    pred,
                    target.long(),
                    weight=class_weights_tensor,
                )
            
            weighted_loss = weight * loss
            total_loss += weighted_loss
            task_losses[task_name] = float(loss.detach())
        
        return total_loss, task_losses


def build_label_mappings(
    dataset: ProbingDataset,
    tasks: List[str],
) -> Dict[str, Dict]:
    """Build label-to-index mappings for classification tasks.
    
    Args:
        dataset: ProbingDataset
        tasks: List of task names
    
    Returns:
        Dict mapping task names to label mappings
    """
    label_mappings = {}
    
    for task in tasks:
        if task == "depth":
            continue
        
        all_labels = []
        for sample in dataset.samples:
            labels = dataset.get_labels(sample)
            if task in labels:
                label = labels[task]
                if isinstance(label, str):
                    all_labels.append(label)
                elif isinstance(label, (int, float)):
                    all_labels.append(str(label))
        
        unique_labels = sorted(set(all_labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        label_mappings[task] = {
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "num_classes": len(unique_labels),
        }
        
        logger.info(f"Task {task}: {len(unique_labels)} classes")
    
    return label_mappings


def compute_class_weights(
    dataset,
    task: str,
    label_mapping: Dict,
) -> Optional[Dict[int, float]]:
    """Compute class weights for imbalanced classification.
    
    Args:
        dataset: ProbingDataset or Subset
        task: Task name
        label_mapping: Label mapping dict
    
    Returns:
        Dict mapping class index to weight, or None
    """
    if task == "depth":
        return None
    
    from torch.utils.data import Subset
    
    label_to_idx = label_mapping["label_to_idx"]
    class_counts = defaultdict(int)
    
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        indices = dataset.indices
        for idx in indices:
            sample = base_dataset.samples[idx]
            labels = base_dataset.get_labels(sample)
            if task in labels:
                label = labels[task]
                if isinstance(label, str):
                    idx_val = label_to_idx[label]
                    class_counts[idx_val] += 1
    else:
        for sample in dataset.samples:
            labels = dataset.get_labels(sample)
            if task in labels:
                label = labels[task]
                if isinstance(label, str):
                    idx = label_to_idx[label]
                    class_counts[idx] += 1
    
    if not class_counts:
        return None
    
    total = sum(class_counts.values())
    n_classes = len(class_counts)
    
    weights = {}
    for idx, count in class_counts.items():
        weights[idx] = total / (n_classes * count)
    
    return weights


def train_probe(
    model: MultiTaskProbe,
    train_dataset: ProbingDataset,
    val_dataset: ProbingDataset,
    tasks: List[str],
    label_mappings: Dict[str, Dict],
    *,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    task_weights: Dict[str, float] = None,
    use_class_weights: bool = True,
    early_stopping_patience: int = 10,
    early_stopping_metric: str = "macro_f1",
    device: str = "cpu",
) -> Tuple[MultiTaskProbe, Dict]:
    """Train a multi-task probe.
    
    Args:
        model: MultiTaskProbe model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tasks: List of task names
        label_mappings: Label mappings for classification tasks
        batch_size: Batch size
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        task_weights: Dict mapping task names to loss weights
        use_class_weights: Whether to use class weights for imbalanced classes
        early_stopping_patience: Early stopping patience
        early_stopping_metric: Metric to monitor for early stopping
        device: Device string
    
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    
    class_weights = {}
    if use_class_weights:
        for task in tasks:
            if task in label_mappings:
                weights = compute_class_weights(train_dataset, task, label_mappings[task])
                if weights:
                    class_weights[task] = weights
                    logger.info(f"Computed class weights for {task}: {weights}")
    
    criterion = MultiTaskLoss(
        task_weights=task_weights,
        class_weights=class_weights,
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_probing_batch,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_probing_batch,
    )
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": [],
    }
    
    best_val_metric = -np.inf
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            embeddings = batch["embedding"].to(device)
            labels_dict = {}
            
            for task in tasks:
                if task == "depth":
                    labels = [batch["labels"][i].get("depth") for i in range(len(batch["labels"]))]
                    labels_dict[task] = torch.tensor(labels, device=device, dtype=torch.float32)
                else:
                    if task in label_mappings:
                        label_to_idx = label_mappings[task]["label_to_idx"]
                        labels = []
                        for i in range(len(batch["labels"])):
                            label = batch["labels"][i].get(task)
                            if label is not None:
                                idx = label_to_idx.get(str(label), 0)
                                labels.append(idx)
                            else:
                                labels.append(0)
                        labels_dict[task] = torch.tensor(labels, device=device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss, task_losses = criterion(outputs, labels_dict)
            loss.backward()
            optimizer.step()
            
            train_losses.append(float(loss.detach()))
        
        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)
        
        model.eval()
        val_losses = []
        val_outputs = defaultdict(list)
        val_labels = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch["embedding"].to(device)
                labels_dict = {}
                
                for task in tasks:
                    if task == "depth":
                        labels = [batch["labels"][i].get("depth") for i in range(len(batch["labels"]))]
                        labels_dict[task] = torch.tensor(labels, device=device, dtype=torch.float32)
                    else:
                        if task in label_mappings:
                            label_to_idx = label_mappings[task]["label_to_idx"]
                            labels = []
                            for i in range(len(batch["labels"])):
                                label = batch["labels"][i].get(task)
                                if label is not None:
                                    idx = label_to_idx.get(str(label), 0)
                                    labels.append(idx)
                                else:
                                    labels.append(0)
                            labels_dict[task] = torch.tensor(labels, device=device, dtype=torch.long)
                
                outputs = model(embeddings)
                loss, _ = criterion(outputs, labels_dict)
                val_losses.append(float(loss.detach()))
                
                for task in tasks:
                    if task in outputs:
                        pred = outputs[task]
                        if pred.shape[-1] == 1:
                            pred = pred.squeeze(-1).cpu().numpy()
                        else:
                            pred = pred.argmax(dim=-1).cpu().numpy()
                        val_outputs[task].extend(pred)
                        
                        if task in labels_dict:
                            target = labels_dict[task].cpu().numpy()
                            val_labels[task].extend(target)
        
        avg_val_loss = np.mean(val_losses)
        history["val_loss"].append(avg_val_loss)
        
        from .probe_metrics import compute_classification_metrics, compute_regression_metrics
        
        val_metrics = {}
        for task in tasks:
            if task in val_outputs and task in val_labels:
                pred = np.array(val_outputs[task])
                target = np.array(val_labels[task])
                
                if task == "depth":
                    metrics = compute_regression_metrics(target, pred)
                else:
                    if task in label_mappings:
                        num_classes = label_mappings[task]["num_classes"]
                        labels = list(range(num_classes))
                        metrics = compute_classification_metrics(target, pred, labels=labels)
                    else:
                        metrics = compute_classification_metrics(target, pred)
                
                val_metrics[task] = metrics
        
        history["val_metrics"].append(val_metrics)
        
        val_metric_value = -np.inf
        if early_stopping_metric == "macro_f1":
            for task in tasks:
                if task in val_metrics and "macro_f1" in val_metrics[task]:
                    val_metric_value = max(val_metric_value, val_metrics[task]["macro_f1"])
        elif early_stopping_metric == "accuracy":
            for task in tasks:
                if task in val_metrics and "accuracy" in val_metrics[task]:
                    val_metric_value = max(val_metric_value, val_metrics[task]["accuracy"])
        elif early_stopping_metric == "r2":
            for task in tasks:
                if task in val_metrics and "r2" in val_metrics[task]:
                    val_metric_value = max(val_metric_value, val_metrics[task]["r2"])
        
        if val_metric_value > best_val_metric:
            best_val_metric = val_metric_value
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"Epoch {epoch+1}/{num_epochs}: New best {early_stopping_metric} = {best_val_metric:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Epoch {epoch+1}/{num_epochs}: {early_stopping_metric} = {val_metric_value:.4f} (patience: {patience_counter}/{early_stopping_patience})")
        
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model (best {early_stopping_metric} = {best_val_metric:.4f})")
    
    return model, history

