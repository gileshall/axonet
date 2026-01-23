"""Dataset for linear probing that loads SWC files, renders on-the-fly, and extracts embeddings."""

from __future__ import annotations

import csv
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from imageio.v2 import imread
from torch.utils.data import Dataset

from ..models.d3_swc_vae import SegVAE2D
from .vae_evaluator import extract_embeddings, render_swc_to_input

logger = logging.getLogger(__name__)


def load_metadata_csv(csv_path: Path) -> Dict[str, Dict]:
    """Load metadata CSV and index by cell_specimen_id."""
    logger.info(f"Loading metadata CSV from {csv_path}")
    metadata_dict = {}
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_specimen_id = row.get("cell_specimen_id", "")
            if cell_specimen_id:
                metadata_dict[cell_specimen_id] = row
    
    logger.info(f"Loaded metadata for {len(metadata_dict)} entries")
    return metadata_dict


def extract_cell_specimen_id(swc_name: str) -> Optional[str]:
    """Extract cell_specimen_id from SWC filename."""
    if not swc_name:
        return None
    parts = swc_name.split("_")
    if parts:
        return parts[0]
    return None


def extract_ttype_family(ttype_label: str) -> Optional[str]:
    """Extract t-type family from full t-type label (e.g., 'Vip Gpc3 Slc18a3' -> 'Vip')."""
    if not ttype_label or ttype_label == "NULL":
        return None
    parts = ttype_label.split()
    if parts:
        return parts[0]
    return None


class ProbingDataset(Dataset):
    """Dataset for linear probing that renders SWC files and extracts embeddings on-the-fly.
    
    Supports both:
    - Direct SWC file paths (from directory)
    - Manifest.jsonl entries (with multiple poses per SWC)
    """
    
    def __init__(
        self,
        model: SegVAE2D,
        device: str,
        *,
        swc_paths: Optional[List[Path]] = None,
        manifest_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        swc_root: Optional[Path] = None,
        metadata_csv: Optional[Path] = None,
        tasks: List[str] = None,
        use_mu: bool = True,
        embedding_reduce: str = "mean",
        width: int = 1024,
        height: int = 1024,
        pose_sampling: str = "random",
        n_poses_per_neuron: int = 1,
        seed: Optional[int] = None,
    ):
        """Initialize probing dataset.
        
        Args:
            model: Frozen VAE model for embedding extraction
            device: Device string for model
            swc_paths: List of SWC file paths (if using direct paths)
            manifest_path: Path to manifest.jsonl (if using manifest)
            data_dir: Data root directory (required if using manifest)
            metadata_csv: Path to PatchSeq_metadata.csv
            tasks: List of tasks to include (e.g., ['ttype', 'met', 'layer', 'depth'])
            use_mu: Use mu (deterministic) instead of z (stochastic) for embeddings
            embedding_reduce: How to reduce spatial dimensions ("mean", "max", "flatten")
            width: Render width
            height: Render height
            pose_sampling: How to sample poses ("random", "all", "first")
            n_poses_per_neuron: Number of poses per neuron (if pose_sampling="random")
            seed: Random seed for pose sampling
        """
        self.model = model
        self.device = device
        self.use_mu = use_mu
        self.embedding_reduce = embedding_reduce
        self.width = width
        self.height = height
        self.pose_sampling = pose_sampling
        self.n_poses_per_neuron = n_poses_per_neuron
        self.seed = seed
        self.swc_root = Path(swc_root) if swc_root else None
        
        if tasks is None:
            tasks = ["ttype", "met", "layer", "depth"]
        self.tasks = tasks
        
        self.metadata_dict = {}
        if metadata_csv and metadata_csv.exists():
            self.metadata_dict = load_metadata_csv(metadata_csv)
        
        self.samples = []
        
        if manifest_path and manifest_path.exists():
            self._load_from_manifest(manifest_path, data_dir)
        elif swc_paths:
            self._load_from_swc_paths(swc_paths)
        else:
            raise ValueError("Must provide either swc_paths or manifest_path")
        
        logger.info(f"Initialized ProbingDataset with {len(self.samples)} samples")
        logger.info(f"Tasks: {self.tasks}")
        logger.info(f"Embedding reduce: {embedding_reduce}, use_mu: {use_mu}")
    
    def _load_from_manifest(self, manifest_path: Path, data_dir: Path):
        """Load samples from manifest.jsonl."""
        logger.info(f"Loading from manifest: {manifest_path}")
        
        entries = []
        with open(manifest_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                entries.append(entry)
        
        grouped = {}
        for idx, entry in enumerate(entries):
            swc_name = entry.get("swc", "")
            if not swc_name:
                continue
            if swc_name not in grouped:
                grouped[swc_name] = []
            grouped[swc_name].append((idx, entry))
        
        logger.info(f"Grouped into {len(grouped)} unique SWC files")
        
        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        
        for swc_name, swc_entries in grouped.items():
            cell_specimen_id = extract_cell_specimen_id(swc_name)
            metadata = self.metadata_dict.get(cell_specimen_id, {}) if cell_specimen_id else {}
            
            if not self._has_required_labels(metadata):
                continue
            
            if self.pose_sampling == "random":
                selected_indices = rng.sample(range(len(swc_entries)), min(self.n_poses_per_neuron, len(swc_entries)))
                selected_entries = [swc_entries[i] for i in selected_indices]
            elif self.pose_sampling == "all":
                selected_entries = swc_entries
            elif self.pose_sampling == "first":
                selected_entries = [swc_entries[0]]
            else:
                raise ValueError(f"Unknown pose_sampling: {self.pose_sampling}")
            
            for pose_idx, (entry_idx, entry) in enumerate(selected_entries):
                sample = {
                    "swc_name": swc_name,
                    "swc_path": None,
                    "manifest_entry": entry,
                    "manifest_entry_idx": entry_idx,
                    "data_dir": data_dir,
                    "cell_specimen_id": cell_specimen_id,
                    "metadata": metadata,
                    "pose_idx": pose_idx,
                    "n_poses": len(selected_entries),
                }
                self.samples.append(sample)
    
    def _load_from_swc_paths(self, swc_paths: List[Path]):
        """Load samples from direct SWC file paths."""
        logger.info(f"Loading from {len(swc_paths)} SWC file paths")
        
        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        
        for swc_path in swc_paths:
            if not swc_path.exists():
                logger.warning(f"SWC file not found: {swc_path}")
                continue
            
            swc_name = swc_path.name
            cell_specimen_id = extract_cell_specimen_id(swc_name)
            metadata = self.metadata_dict.get(cell_specimen_id, {}) if cell_specimen_id else {}
            
            if not self._has_required_labels(metadata):
                continue
            
            n_poses = self.n_poses_per_neuron if self.pose_sampling == "random" else 1
            
            for pose_idx in range(n_poses):
                sample = {
                    "swc_name": swc_name,
                    "swc_path": swc_path,
                    "manifest_entry": None,
                    "manifest_entry_idx": None,
                    "data_dir": None,
                    "cell_specimen_id": cell_specimen_id,
                    "metadata": metadata,
                    "pose_idx": pose_idx,
                    "n_poses": n_poses,
                }
                self.samples.append(sample)
    
    def _has_required_labels(self, metadata: Dict) -> bool:
        """Check if metadata has required labels for requested tasks."""
        for task in self.tasks:
            if task == "ttype":
                if not metadata.get("T-type Label") or metadata.get("T-type Label") == "NULL":
                    return False
            elif task == "met":
                if not metadata.get("MET-type Label") or metadata.get("MET-type Label") == "NULL":
                    return False
            elif task == "layer":
                if not metadata.get("structure"):
                    return False
            elif task == "depth":
                depth_str = metadata.get("cell_soma_normalized_depth", "")
                if not depth_str or depth_str == "":
                    return False
        return True
    
    def _get_group_key(self, sample: Dict) -> str:
        """Get grouping key for cross-validation (donor_id or cell_specimen_id)."""
        metadata = sample["metadata"]
        donor_id = metadata.get("donor_id", "")
        if donor_id:
            return f"donor_{donor_id}"
        cell_specimen_id = sample.get("cell_specimen_id", "")
        if cell_specimen_id:
            return f"cell_{cell_specimen_id}"
        return f"unknown_{sample['swc_name']}"
    
    def get_labels(self, sample: Dict) -> Dict:
        """Extract labels from metadata for requested tasks."""
        metadata = sample["metadata"]
        labels = {}
        
        if "ttype" in self.tasks:
            ttype_label = metadata.get("T-type Label", "")
            if ttype_label and ttype_label != "NULL":
                labels["ttype"] = ttype_label
                labels["ttype_family"] = extract_ttype_family(ttype_label)
        
        if "met" in self.tasks:
            met_label = metadata.get("MET-type Label", "")
            if met_label and met_label != "NULL":
                labels["met"] = met_label
        
        if "layer" in self.tasks:
            structure = metadata.get("structure", "")
            if structure:
                labels["layer"] = structure
        
        if "depth" in self.tasks:
            depth_str = metadata.get("cell_soma_normalized_depth", "")
            if depth_str and depth_str != "":
                labels["depth"] = float(depth_str)
        
        return labels
    
    def _resolve_swc_path(self, swc_name: str, data_dir: Optional[Path]) -> Path:
        """Resolve SWC file path using swc_root if provided."""
        if not swc_name:
            raise ValueError("SWC name is empty")
        
        if self.swc_root:
            swc_path = self.swc_root / swc_name
            if swc_path.exists():
                return swc_path
            raise FileNotFoundError(f"SWC file not found in swc_root: {swc_path}")
        
        if data_dir:
            swc_path = data_dir / swc_name
            if swc_path.exists():
                return swc_path
        
        swc_path = Path(swc_name)
        if swc_path.exists():
            return swc_path
        
        raise FileNotFoundError(f"SWC file not found: {swc_name} (swc_root={self.swc_root}, data_dir={data_dir})")
    
    def _resolve_image_path(self, rel_path: str, data_dir: Optional[Path]) -> Path:
        """Resolve image path from manifest relative path."""
        if not rel_path:
            raise ValueError("Image path is empty")
        
        if data_dir:
            image_path = data_dir / rel_path
            if image_path.exists():
                return image_path
            
            if not rel_path.startswith("images/"):
                image_path = data_dir / "images" / rel_path
                if image_path.exists():
                    return image_path
        
        image_path = Path(rel_path)
        if image_path.exists():
            return image_path
        
        raise FileNotFoundError(f"Image file not found: {rel_path} (data_dir={data_dir})")
    
    def _render_swc(self, sample: Dict) -> np.ndarray:
        """Render SWC file to input image, or load pre-rendered image from manifest if available."""
        if sample["manifest_entry"]:
            entry = sample["manifest_entry"]
            if "mask_bw" in entry:
                image_path = self._resolve_image_path(entry["mask_bw"], sample["data_dir"])
                logger.debug(f"Loading cached render from manifest: {image_path}")
                input_img = imread(image_path)
                if len(input_img.shape) == 3:
                    input_img = np.mean(input_img, axis=2, dtype=np.float32) / 255.0
                else:
                    input_img = input_img.astype(np.float32) / 255.0
                return input_img
            elif "image" in entry:
                image_path = self._resolve_image_path(entry["image"], sample["data_dir"])
                logger.debug(f"Loading cached render from manifest: {image_path}")
                input_img = imread(image_path)
                if len(input_img.shape) == 3:
                    input_img = np.mean(input_img, axis=2, dtype=np.float32) / 255.0
                else:
                    input_img = input_img.astype(np.float32) / 255.0
                return input_img
        
        if sample["swc_path"]:
            swc_path = sample["swc_path"]
        elif sample["manifest_entry"]:
            swc_name = sample["manifest_entry"].get("swc", "")
            if not swc_name:
                raise ValueError(f"Manifest entry missing swc field: {sample['manifest_entry']}")
            swc_path = self._resolve_swc_path(swc_name, sample["data_dir"])
        else:
            raise ValueError("Sample has neither swc_path nor manifest_entry")
        
        if not swc_path.exists():
            raise FileNotFoundError(f"SWC file not found: {swc_path}")
        
        logger.debug(f"Rendering SWC file: {swc_path}")
        input_img, _, _ = render_swc_to_input(
            swc_path,
            width=self.width,
            height=self.height,
        )
        return input_img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample with embedding and labels."""
        sample = self.samples[idx]
        
        if idx % 10 == 0 and idx > 0:
            logger.info(f"Processing sample {idx}/{len(self.samples)}: {sample['swc_name']}")
        
        input_img = self._render_swc(sample)
        input_tensor = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0).float()
        
        embeddings_dict = extract_embeddings(
            self.model,
            input_tensor,
            self.device,
            reduce=self.embedding_reduce,
        )
        
        embedding = embeddings_dict["mu"] if self.use_mu else embeddings_dict["z"]
        embedding = torch.from_numpy(embedding[0]).float()
        
        labels = self.get_labels(sample)
        group_key = self._get_group_key(sample)
        
        return {
            "embedding": embedding,
            "labels": labels,
            "group_key": group_key,
            "swc_name": sample["swc_name"],
            "cell_specimen_id": sample.get("cell_specimen_id", ""),
            "pose_idx": sample["pose_idx"],
            "n_poses": sample["n_poses"],
        }

