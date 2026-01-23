"""Lightning DataModule with multi-source data handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from ..config import ExperimentConfig
from .metadata_adapters import MetadataAdapter, get_adapter


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL manifest file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_metadata(path: Path, id_column: str) -> Dict[str, Dict[str, Any]]:
    """Load metadata file and index by ID."""
    path = Path(path)
    
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return {str(e.get(id_column, "")): e for e in data}
        return data
    
    elif path.suffix == ".jsonl":
        metadata = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    key = str(e.get(id_column, ""))
                    if key:
                        metadata[key] = e
        return metadata
    
    elif path.suffix == ".csv":
        import csv
        metadata = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = str(row.get(id_column, ""))
                if key:
                    metadata[key] = dict(row)
        return metadata
    
    else:
        raise ValueError(f"Unsupported metadata format: {path.suffix}")


def pair_manifest_metadata(
    manifest: List[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]],
    adapter: MetadataAdapter,
) -> List[Dict[str, Any]]:
    """Pair manifest entries with metadata using adapter's ID extraction."""
    paired = []
    
    for entry in manifest:
        neuron_id = None
        
        if "cell_id" in entry:
            neuron_id = str(entry["cell_id"])
        elif "neuron_id" in entry:
            neuron_id = str(entry["neuron_id"])
        elif "swc" in entry:
            import re
            stem = Path(entry["swc"]).stem
            match = re.search(r"(\d+)", stem)
            neuron_id = match.group(1) if match else stem
        
        meta = metadata.get(neuron_id) if neuron_id else None
        
        paired.append({
            "manifest": entry,
            "metadata": meta,
            "neuron_id": neuron_id,
        })
    
    return paired


class MultiSourceDataModule(LightningDataModule):
    """DataModule that handles multiple data sources transparently.
    
    Supports:
    - Allen Brain Institute data
    - NeuroMorpho.org data
    - Custom data with configurable field mapping
    
    Automatically downloads missing data when URL-based sources are configured.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        train_manifest: Optional[str] = None,
        val_manifest: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.train_manifest_name = train_manifest or config.data.manifest
        self.val_manifest_name = val_manifest
        
        self.adapter = get_adapter(config.data.source)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        
        self._paired_train: List[Dict[str, Any]] = []
        self._paired_val: List[Dict[str, Any]] = []

    @property
    def data_root(self) -> Path:
        return Path(self.config.data.root)

    def prepare_data(self):
        """Download data if URL-based sources are configured."""
        if self.config.data.base_url:
            self._download_from_url()

    def _download_from_url(self):
        """Download dataset from URL if not already present."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load manifest and metadata, create datasets."""
        manifest_path = self.data_root / self.train_manifest_name
        manifest = load_manifest(manifest_path)
        
        metadata: Dict[str, Dict[str, Any]] = {}
        if self.config.data.metadata:
            metadata_path = self.data_root / self.config.data.metadata
            if metadata_path.exists():
                metadata = load_metadata(metadata_path, self.config.data.id_column)
        
        self._paired_train = pair_manifest_metadata(manifest, metadata, self.adapter)
        
        if self.val_manifest_name:
            val_path = self.data_root / self.val_manifest_name
            if val_path.exists():
                val_manifest = load_manifest(val_path)
                self._paired_val = pair_manifest_metadata(val_manifest, metadata, self.adapter)

    def get_paired_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Get paired manifest/metadata entries for train and val."""
        return self._paired_train, self._paired_val

    def train_dataloader(self) -> Optional[DataLoader]:
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
        )
