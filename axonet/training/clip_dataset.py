"""Dataset classes for CLIP-style contrastive training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from imageio.v2 import imread
from torch.utils.data import Dataset

from ..data.metadata_adapters import MetadataAdapter, get_adapter


class NeuronTextGenerator:
    """Generate text descriptions from neuron metadata using an adapter."""

    def __init__(
        self,
        adapter: MetadataAdapter,
        style: str = "detailed",
        augment: bool = True,
    ):
        self.adapter = adapter
        self.style = style
        self.augment = augment

    def generate(self, metadata: Dict[str, Any]) -> str:
        """Generate text description for a neuron."""
        base = self.adapter.to_text_description(metadata)
        
        if not self.augment:
            return base
        
        return self._augment_text(base)

    def _augment_text(self, text: str) -> str:
        """Apply simple text augmentation."""
        import random
        
        variants = [
            text,
            text.lower(),
            f"a {text}",
            f"morphology of {text}",
        ]
        return random.choice(variants)

    def __call__(self, metadata: Dict[str, Any]) -> str:
        return self.generate(metadata)


def _extract_id_from_path(path: Union[str, Path]) -> str:
    """Extract numeric ID from SWC path (e.g., '601506507.swc' -> '601506507')."""
    stem = Path(path).stem
    match = re.search(r"(\d+)", stem)
    return match.group(1) if match else stem


class NeuronCLIPDataset(Dataset):
    """Dataset for CLIP-style contrastive training with neuron images and text.
    
    Loads image renders and pairs them with text descriptions generated
    from metadata using an adapter.
    """

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        metadata_path: Path,
        adapter: MetadataAdapter,
        id_column: str = "cell_specimen_id",
        text_generator: Optional[NeuronTextGenerator] = None,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.adapter = adapter
        self.id_column = id_column
        self.transform = transform
        
        if text_generator is None:
            text_generator = NeuronTextGenerator(adapter)
        self.text_generator = text_generator
        
        manifest_entries = self._load_manifest(manifest_path)
        metadata_dict = self._load_metadata(metadata_path, id_column)
        
        self.entries = self._pair_manifest_metadata(manifest_entries, metadata_dict)
        
        if len(self.entries) == 0:
            raise ValueError(
                f"No matching entries found between manifest and metadata. "
                f"Manifest has {len(manifest_entries)} entries, "
                f"metadata has {len(metadata_dict)} entries."
            )

    def _load_manifest(self, path: Path) -> List[Dict[str, Any]]:
        """Load manifest JSONL file."""
        entries = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _load_metadata(
        self, path: Path, id_column: str
    ) -> Dict[str, Dict[str, Any]]:
        """Load metadata file (JSON or CSV) and index by ID."""
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

    def _pair_manifest_metadata(
        self,
        manifest: List[Dict[str, Any]],
        metadata: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Pair manifest entries with metadata by ID."""
        paired = []
        
        for entry in manifest:
            neuron_id = None
            
            if "cell_id" in entry:
                neuron_id = str(entry["cell_id"])
            elif "neuron_id" in entry:
                neuron_id = str(entry["neuron_id"])
            elif "swc" in entry:
                neuron_id = _extract_id_from_path(entry["swc"])
            
            if neuron_id and neuron_id in metadata:
                paired.append({
                    "manifest": entry,
                    "metadata": metadata[neuron_id],
                    "neuron_id": neuron_id,
                })
        
        return paired

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        manifest = entry["manifest"]
        metadata = entry["metadata"]
        
        if "mask_bw" in manifest:
            input_path = self.data_root / manifest["mask_bw"]
        elif "image" in manifest:
            input_path = self.data_root / manifest["image"]
        else:
            raise ValueError(f"Entry missing 'mask_bw' or 'image': {manifest}")
        
        img = imread(input_path)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2, dtype=np.float32) / 255.0
        else:
            img = img.astype(np.float32) / 255.0
        
        input_tensor = torch.from_numpy(img).unsqueeze(0)
        
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
        
        text = self.text_generator(metadata)
        
        return {
            "input": input_tensor,
            "text": text,
            "neuron_id": entry["neuron_id"],
        }


class NeuronCLIPDatasetFromPaired(Dataset):
    """CLIP dataset from pre-paired list of (image_path, metadata) tuples."""

    def __init__(
        self,
        paired_entries: List[Dict[str, Any]],
        data_root: Path,
        adapter: MetadataAdapter,
        text_generator: Optional[NeuronTextGenerator] = None,
        transform: Optional[Callable] = None,
    ):
        self.entries = paired_entries
        self.data_root = Path(data_root)
        self.adapter = adapter
        self.transform = transform
        
        if text_generator is None:
            text_generator = NeuronTextGenerator(adapter)
        self.text_generator = text_generator

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        
        img_path = self.data_root / entry["image_path"]
        img = imread(img_path)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2, dtype=np.float32) / 255.0
        else:
            img = img.astype(np.float32) / 255.0
        
        input_tensor = torch.from_numpy(img).unsqueeze(0)
        
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
        
        text = self.text_generator(entry["metadata"])
        
        return {
            "input": input_tensor,
            "text": text,
            "neuron_id": entry.get("neuron_id", ""),
        }
