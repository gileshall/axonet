"""Metadata adapters for normalizing data from different neuron sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class MetadataAdapter(ABC):
    """Abstract adapter for normalizing metadata from different sources."""

    @abstractmethod
    def get_neuron_id(self, entry: Dict[str, Any]) -> str:
        """Extract unique neuron identifier."""
        ...

    @abstractmethod
    def get_cell_type(self, entry: Dict[str, Any]) -> str:
        """Extract cell type label."""
        ...

    @abstractmethod
    def get_brain_region(self, entry: Dict[str, Any]) -> str:
        """Extract brain region."""
        ...

    @abstractmethod
    def get_species(self, entry: Dict[str, Any]) -> str:
        """Extract species."""
        ...

    @abstractmethod
    def to_text_description(self, entry: Dict[str, Any]) -> str:
        """Generate text description for CLIP training."""
        ...

    def get_metadata_fields(self, entry: Dict[str, Any]) -> Dict[str, str]:
        """Get all normalized metadata fields."""
        return {
            "neuron_id": self.get_neuron_id(entry),
            "cell_type": self.get_cell_type(entry),
            "brain_region": self.get_brain_region(entry),
            "species": self.get_species(entry),
        }


class AllenAdapter(MetadataAdapter):
    """Adapter for Allen Brain Institute mIVSCC-MET data."""

    def get_neuron_id(self, entry: Dict[str, Any]) -> str:
        return str(entry.get("cell_specimen_id", ""))

    def get_cell_type(self, entry: Dict[str, Any]) -> str:
        t_type = entry.get("T-type Label", "")
        if t_type:
            return t_type
        met_type = entry.get("MET-type", "")
        if met_type:
            return met_type
        dendrite = entry.get("dendrite_type", "")
        if dendrite:
            return dendrite
        return "unknown"

    def get_brain_region(self, entry: Dict[str, Any]) -> str:
        return entry.get("structure", "cortex")

    def get_species(self, entry: Dict[str, Any]) -> str:
        return entry.get("species", "mouse")

    def to_text_description(self, entry: Dict[str, Any]) -> str:
        parts = []
        
        dendrite = entry.get("dendrite_type", "")
        if dendrite:
            parts.append(dendrite)
        
        structure = entry.get("structure", "")
        if structure:
            parts.append(structure)
        
        t_type = entry.get("T-type Label", "")
        if t_type:
            parts.append(t_type)
        
        if not parts:
            parts.append("neuron")
        
        return " ".join(parts)


class NeuroMorphoAdapter(MetadataAdapter):
    """Adapter for NeuroMorpho.org data."""

    def get_neuron_id(self, entry: Dict[str, Any]) -> str:
        return str(entry.get("neuron_id", entry.get("neuron_name", "")))

    def get_cell_type(self, entry: Dict[str, Any]) -> str:
        cell_types = entry.get("cell_type", [])
        if isinstance(cell_types, list):
            return " ".join(cell_types) if cell_types else "unknown"
        return str(cell_types) if cell_types else "unknown"

    def get_brain_region(self, entry: Dict[str, Any]) -> str:
        regions = entry.get("brain_region", [])
        if isinstance(regions, list):
            return " ".join(regions) if regions else "unknown"
        return str(regions) if regions else "unknown"

    def get_species(self, entry: Dict[str, Any]) -> str:
        return entry.get("species", "unknown")

    def to_text_description(self, entry: Dict[str, Any]) -> str:
        parts = []
        
        species = self.get_species(entry)
        if species and species != "unknown":
            parts.append(species)
        
        cell_type = self.get_cell_type(entry)
        if cell_type and cell_type != "unknown":
            parts.append(cell_type)
        
        region = self.get_brain_region(entry)
        if region and region != "unknown":
            parts.append(f"from {region}")
        
        if not parts:
            parts.append("neuron")
        
        return " ".join(parts)


class GenericAdapter(MetadataAdapter):
    """Generic adapter with configurable field mapping."""

    def __init__(
        self,
        id_field: str = "id",
        cell_type_field: str = "cell_type",
        brain_region_field: str = "brain_region",
        species_field: str = "species",
        text_template: Optional[str] = None,
    ):
        self.id_field = id_field
        self.cell_type_field = cell_type_field
        self.brain_region_field = brain_region_field
        self.species_field = species_field
        self.text_template = text_template

    def get_neuron_id(self, entry: Dict[str, Any]) -> str:
        return str(entry.get(self.id_field, ""))

    def get_cell_type(self, entry: Dict[str, Any]) -> str:
        val = entry.get(self.cell_type_field, "unknown")
        if isinstance(val, list):
            return " ".join(val)
        return str(val)

    def get_brain_region(self, entry: Dict[str, Any]) -> str:
        val = entry.get(self.brain_region_field, "unknown")
        if isinstance(val, list):
            return " ".join(val)
        return str(val)

    def get_species(self, entry: Dict[str, Any]) -> str:
        return entry.get(self.species_field, "unknown")

    def to_text_description(self, entry: Dict[str, Any]) -> str:
        if self.text_template:
            return self.text_template.format(**entry)
        
        parts = []
        species = self.get_species(entry)
        if species and species != "unknown":
            parts.append(species)
        
        cell_type = self.get_cell_type(entry)
        if cell_type and cell_type != "unknown":
            parts.append(cell_type)
        
        region = self.get_brain_region(entry)
        if region and region != "unknown":
            parts.append(f"from {region}")
        
        return " ".join(parts) if parts else "neuron"


ADAPTERS: Dict[str, type] = {
    "allen": AllenAdapter,
    "neuromorpho": NeuroMorphoAdapter,
    "custom": GenericAdapter,
}


def get_adapter(source: str, **kwargs) -> MetadataAdapter:
    """Get adapter for a data source."""
    adapter_cls = ADAPTERS.get(source, GenericAdapter)
    if adapter_cls == GenericAdapter:
        return adapter_cls(**kwargs)
    return adapter_cls()
