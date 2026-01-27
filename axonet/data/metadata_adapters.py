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

    def get_description_parts(self, entry: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Return structured description parts for multi-level text generation.

        Returns a dict with keys: species, cell_type, brain_region, layer.
        Values are None when the field is missing or uninformative.
        """
        species = self.get_species(entry)
        cell_type = self.get_cell_type(entry)
        region = self.get_brain_region(entry)
        return {
            "species": species if species and species != "unknown" else None,
            "cell_type": cell_type if cell_type and cell_type != "unknown" else None,
            "brain_region": region if region and region != "unknown" else None,
            "layer": None,
        }

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

    def get_description_parts(self, entry: Dict[str, Any]) -> Dict[str, Optional[str]]:
        species = entry.get("species", "mouse")
        dendrite = entry.get("dendrite_type", "")
        structure = entry.get("structure", "")
        t_type = entry.get("T-type Label", "")

        cell_desc = None
        if dendrite:
            cell_desc = dendrite.lower()
            if not any(w in cell_desc for w in ("cell", "neuron")):
                cell_desc = f"{cell_desc} neuron"
        elif t_type:
            cell_desc = t_type.lower()

        return {
            "species": species if species else None,
            "cell_type": cell_desc,
            "brain_region": structure.lower() if structure else None,
            "layer": None,
        }

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

    # Priority order for cell types (more specific morphological types first)
    CELL_TYPE_PRIORITY = [
        # Specific morphological types
        "basket",
        "chandelier",
        "martinotti",
        "bitufted",
        "bipolar",
        "multipolar",
        "stellate",
        "granule",
        "purkinje",
        "mitral",
        "medium spiny",
        # Broad excitatory/inhibitory
        "pyramidal",
        "interneuron",
        # Molecular markers
        "parvalbumin",
        "somatostatin",
        "VIP",
        # Generic fallback
        "principal cell",
    ]

    # Primary brain regions (broader categories first)
    PRIMARY_REGIONS = [
        "neocortex",
        "hippocampus",
        "cerebellum",
        "thalamus",
        "striatum",
        "amygdala",
        "olfactory bulb",
        "brainstem",
        "spinal cord",
        "retina",
    ]

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

    def _extract_primary_cell_type(self, cell_types: List[str]) -> Optional[str]:
        """Extract the most informative cell type from a list."""
        if not cell_types:
            return None

        cell_types_lower = [ct.lower() for ct in cell_types]

        # Find highest priority match
        for priority_type in self.CELL_TYPE_PRIORITY:
            for i, ct in enumerate(cell_types_lower):
                if priority_type in ct:
                    return cell_types[i]

        # Fall back to first non-generic type
        for ct in cell_types:
            if ct.lower() not in ("principal cell", "neuron"):
                return ct

        return cell_types[0] if cell_types else None

    def _extract_primary_region(self, regions: List[str]) -> Optional[str]:
        """Extract the primary brain region from a list."""
        if not regions:
            return None

        regions_lower = [r.lower() for r in regions]

        # Find primary region
        for primary in self.PRIMARY_REGIONS:
            for i, r in enumerate(regions_lower):
                if primary in r or r in primary:
                    return regions[i]

        # Fall back to first region
        return regions[0] if regions else None

    def _extract_layer(self, regions: List[str]) -> Optional[str]:
        """Extract cortical layer from brain region list."""
        import re

        for region in regions:
            # Match patterns like "layer 4", "layer 2-3", "layer 5/6"
            match = re.search(r"layer\s*(\d+(?:[/-]\d+)?)", region.lower())
            if match:
                return f"layer {match.group(1)}"

        return None

    def get_description_parts(self, entry: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Return structured description parts for multi-level text generation.

        Uses priority-based extraction to return the most informative
        cell type, region, and layer from the (often multi-valued) metadata.
        """
        cell_types = entry.get("cell_type", [])
        if isinstance(cell_types, str):
            cell_types = [cell_types]

        regions = entry.get("brain_region", [])
        if isinstance(regions, str):
            regions = [regions]

        primary_type = self._extract_primary_cell_type(cell_types)
        primary_region = self._extract_primary_region(regions)
        layer = self._extract_layer(regions)

        species = entry.get("species", None)
        if species and species.lower() in ("unknown", "not reported", ""):
            species = None

        # Clean cell type into natural phrasing
        cell_desc = None
        if primary_type:
            cell_desc = primary_type.lower()
            if not any(w in cell_desc for w in ("cell", "neuron")):
                cell_desc = f"{cell_desc} neuron"

        region_desc = None
        if primary_region:
            region_desc = primary_region.lower()

        return {
            "species": species,
            "cell_type": cell_desc,
            "brain_region": region_desc,
            "layer": layer,
        }

    def to_text_description(self, entry: Dict[str, Any]) -> str:
        """Generate natural language description for CLIP training.

        Examples:
        - "a stellate cell from neocortex layer 4"
        - "a pyramidal neuron from hippocampus"
        - "a parvalbumin-positive interneuron from neocortex layer 5"
        """
        parts_dict = self.get_description_parts(entry)

        # Build description
        parts = []

        # Cell type
        if parts_dict["cell_type"]:
            parts.append(parts_dict["cell_type"])
        else:
            parts.append("neuron")

        # Brain region and layer
        if parts_dict["brain_region"]:
            location = parts_dict["brain_region"]
            if parts_dict["layer"]:
                location = f"{location} {parts_dict['layer']}"
            parts.append(f"from {location}")

        # Build final description with article
        description = " ".join(parts)

        # Add appropriate article
        if description[0] in "aeiou":
            return f"an {description}"
        return f"a {description}"


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
