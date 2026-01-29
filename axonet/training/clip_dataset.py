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
    """Generate text descriptions from neuron metadata using an adapter.

    Supports two styles:
        "legacy"     - original augmentation (lowercase, "a X", "morphology of X")
        "multilevel" - stochastic level selection with species, morphometric bins

    When style="multilevel", text is generated at one of four levels per call:
        broad       (10%) - "a mouse neuron"
        standard    (30%) - "a mouse pyramidal neuron from neocortex"
        detailed    (40%) - "a large, densely branched mouse pyramidal neuron from neocortex"
        morphometric(20%) - "a mouse pyramidal neuron from neocortex, 10mm total length, 340 bifurcations"
    """

    # Default probability weights for each text level
    DEFAULT_LEVEL_WEIGHTS = {
        "broad": 0.10,
        "standard": 0.30,
        "detailed": 0.40,
        "morphometric": 0.20,
    }

    def __init__(
        self,
        adapter: MetadataAdapter,
        style: str = "legacy",
        augment: bool = True,
        morph_bins: Optional[Dict[str, Any]] = None,
        level_weights: Optional[Dict[str, float]] = None,
    ):
        self.adapter = adapter
        self.style = style
        self.augment = augment
        self.morph_bins = morph_bins or {}
        self.level_weights = level_weights or self.DEFAULT_LEVEL_WEIGHTS

    def generate(self, metadata: Dict[str, Any]) -> str:
        """Generate text description for a neuron."""
        if self.style == "multilevel":
            return self._generate_multilevel(metadata)

        # Legacy behavior
        base = self.adapter.to_text_description(metadata)
        if not self.augment:
            return base
        return self._augment_text(base)

    def _augment_text(self, text: str) -> str:
        """Apply simple text augmentation (legacy mode)."""
        import random

        variants = [
            text,
            text.lower(),
            f"a {text}",
            f"morphology of {text}",
        ]
        return random.choice(variants)

    # ------------------------------------------------------------------
    # Multi-level generation
    # ------------------------------------------------------------------

    def _generate_multilevel(self, metadata: Dict[str, Any]) -> str:
        """Generate text at a stochastically chosen detail level."""
        import random

        levels = list(self.level_weights.keys())
        weights = [self.level_weights[l] for l in levels]
        level = random.choices(levels, weights=weights, k=1)[0]

        parts = self.adapter.get_description_parts(metadata)
        species = parts.get("species")
        cell_type = parts.get("cell_type")
        region = parts.get("brain_region")
        layer = parts.get("layer")

        if level == "broad":
            return self._build_broad(species)
        elif level == "standard":
            return self._build_standard(species, cell_type, region, layer)
        elif level == "detailed":
            return self._build_detailed(species, cell_type, region, layer, metadata)
        elif level == "morphometric":
            return self._build_morphometric(species, cell_type, region, layer, metadata)
        else:
            return self._build_standard(species, cell_type, region, layer)

    def _add_article(self, text: str) -> str:
        if text and text[0].lower() in "aeiou":
            return f"an {text}"
        return f"a {text}"

    def _build_broad(self, species: Optional[str]) -> str:
        """Level 1: species only. E.g. 'a mouse neuron'"""
        if species:
            return self._add_article(f"{species} neuron")
        return "a neuron"

    def _build_standard(
        self,
        species: Optional[str],
        cell_type: Optional[str],
        region: Optional[str],
        layer: Optional[str],
    ) -> str:
        """Level 2: species + cell type + region. E.g. 'a mouse pyramidal neuron from neocortex'"""
        parts = []
        if species:
            parts.append(species)
        if cell_type:
            parts.append(cell_type)
        else:
            parts.append("neuron")
        desc = " ".join(parts)
        if region:
            location = region
            if layer:
                location = f"{region} {layer}"
            desc = f"{desc} from {location}"
        return self._add_article(desc)

    def _build_detailed(
        self,
        species: Optional[str],
        cell_type: Optional[str],
        region: Optional[str],
        layer: Optional[str],
        metadata: Dict[str, Any],
    ) -> str:
        """Level 3: morphometric adjectives + standard. E.g. 'a large, densely branched mouse pyramidal neuron from neocortex'"""
        adjectives = self._get_morph_adjectives(metadata)

        parts = []
        if adjectives:
            parts.append(", ".join(adjectives))
        if species:
            parts.append(species)
        if cell_type:
            parts.append(cell_type)
        else:
            parts.append("neuron")
        desc = " ".join(parts)
        if region:
            location = region
            if layer:
                location = f"{region} {layer}"
            desc = f"{desc} from {location}"
        return self._add_article(desc)

    def _build_morphometric(
        self,
        species: Optional[str],
        cell_type: Optional[str],
        region: Optional[str],
        layer: Optional[str],
        metadata: Dict[str, Any],
    ) -> str:
        """Level 4: standard + numeric measurements. E.g. 'a mouse pyramidal neuron from neocortex, 10mm total length, 340 bifurcations'"""
        # Start with standard description
        base = self._build_standard(species, cell_type, region, layer)

        # Append numeric morphometric facts
        morph = metadata.get("morphometry", {})
        if not morph or not isinstance(morph, dict):
            return base

        facts = []
        length = morph.get("length")
        if length is not None:
            facts.append(f"{float(length):.0f} um total length")

        n_bifs = morph.get("n_bifs")
        if n_bifs is not None:
            facts.append(f"{int(float(n_bifs))} bifurcations")

        n_stems = morph.get("n_stems")
        if n_stems is not None:
            facts.append(f"{int(float(n_stems))} stems")

        width = morph.get("width")
        height = morph.get("height")
        if width is not None and height is not None:
            facts.append(f"{float(width):.0f} x {float(height):.0f} um span")

        if not facts:
            return base

        import random
        # Include 2-3 facts to avoid overly long descriptions
        k = min(len(facts), random.choice([2, 3]))
        selected = random.sample(facts, k)
        return f"{base}, {', '.join(selected)}"

    def _get_morph_adjectives(self, metadata: Dict[str, Any]) -> List[str]:
        """Get morphometric adjective phrases from bin assignments."""
        if not self.morph_bins:
            return []

        import random

        morph = metadata.get("morphometry", {})
        if not morph or not isinstance(morph, dict):
            return []

        available = []
        for bin_name, spec in self.morph_bins.items():
            key = spec["key"]
            value = self._get_morph_value(morph, key)
            label = self._assign_bin(value, spec)
            if label is not None:
                available.append(label)

        if not available:
            return []

        # Pick 1-2 adjectives to keep descriptions varied
        k = min(len(available), random.choice([1, 2]))
        return random.sample(available, k)

    @staticmethod
    def _get_morph_value(morph: Dict[str, Any], key: str) -> Optional[float]:
        """Extract a morphometric value, handling computed keys."""
        if key == "_max_extent":
            w = morph.get("width")
            h = morph.get("height")
            d = morph.get("depth")
            vals = [float(v) for v in [w, h, d] if v is not None and float(v) > 0]
            return max(vals) if vals else None
        if key == "_aspect_ratio":
            w = morph.get("width")
            h = morph.get("height")
            if w and h and float(h) > 0:
                return float(w) / float(h)
            return None
        val = morph.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _assign_bin(value: Optional[float], spec: Dict[str, Any]) -> Optional[str]:
        """Assign a value to its bin label."""
        if value is None:
            return None
        thresholds = spec["thresholds"]
        labels = spec["labels"]
        for i, t in enumerate(thresholds):
            if value <= t:
                return labels[i]
        return labels[-1]

    def __call__(self, metadata: Dict[str, Any]) -> str:
        return self.generate(metadata)


def _extract_id_from_path(path: Union[str, Path]) -> str:
    """Extract numeric ID from SWC path (e.g., '601506507.swc' -> '601506507')."""
    stem = Path(path).stem
    match = re.search(r"(\d+)", stem)
    return match.group(1) if match else stem


def _normalize_neuromorpho_id(neuron_id: str) -> List[str]:
    """Generate candidate IDs for matching NeuroMorpho-style neuron IDs.

    Handles format like '130683_S2995-x1-25.CNG' -> ['130683_S2995-x1-25.CNG', 'S2995-x1-25', '130683']
    Returns list of candidates to try for matching.
    """
    candidates = [neuron_id]

    # Remove .CNG suffix if present
    base = neuron_id
    if base.endswith('.CNG'):
        base = base[:-4]
        candidates.append(base)

    # Extract name part after numeric prefix (e.g., '130683_S2995-x1-25' -> 'S2995-x1-25')
    if '_' in base:
        parts = base.split('_', 1)
        if parts[0].isdigit() and len(parts) > 1:
            candidates.append(parts[1])  # The name part
            candidates.append(parts[0])  # The numeric ID

    return candidates


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
        image_size: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.adapter = adapter
        self.id_column = id_column
        self.transform = transform
        self.image_size = image_size

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

            if not neuron_id:
                continue

            # Try multiple candidate IDs (handles NeuroMorpho format like '130683_Name.CNG')
            matched_id = None
            for candidate in _normalize_neuromorpho_id(neuron_id):
                if candidate in metadata:
                    matched_id = candidate
                    break

            if matched_id:
                paired.append({
                    "manifest": entry,
                    "metadata": metadata[matched_id],
                    "neuron_id": matched_id,
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

        # Resize if image_size is specified
        if self.image_size is not None:
            input_tensor = torch.nn.functional.interpolate(
                input_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

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
