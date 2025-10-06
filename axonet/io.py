"""
Input/Output utilities for SWC files and other formats.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from .core import Neuron, SWCNode


class SWCParser:
    """Parse SWC neuronal morphology files."""
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.header_lines: List[str] = []
        self.metadata: Dict[str, str] = {}
    
    def parse(self) -> Tuple[List[SWCNode], Dict[str, str]]:
        """
        Parse the SWC file and return nodes and metadata.
        
        Returns:
            Tuple of (nodes, metadata)
        """
        nodes = []
        
        with open(self.filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Handle header lines
                if line.startswith('#'):
                    self.header_lines.append(line)
                    self._parse_header_line(line)
                    continue
                
                # Parse data lines
                parts = line.split()
                if len(parts) != 7:
                    continue
                
                try:
                    node = SWCNode(
                        index=int(parts[0]),
                        type_id=int(parts[1]),
                        x=float(parts[2]),
                        y=float(parts[3]),
                        z=float(parts[4]),
                        radius=float(parts[5]),
                        parent=int(parts[6])
                    )
                    nodes.append(node)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid line {line_num}: {e}")
                    continue
        
        return nodes, self.metadata
    
    def _parse_header_line(self, line: str):
        """Parse metadata from header line."""
        if '=' in line:
            key, value = line[1:].split('=', 1)
            self.metadata[key.strip()] = value.strip()
        else:
            # Store as general comment
            comment = line[1:].strip()
            if comment:
                self.metadata[f'comment_{len(self.header_lines)}'] = comment


def load_swc(filepath: Union[str, Path], validate: bool = True) -> Neuron:
    """
    Load a neuron from an SWC file.
    
    Args:
        filepath: Path to SWC file
        validate: Whether to validate the loaded data
        
    Returns:
        Neuron object
        
    Raises:
        ValueError: If file is invalid or empty
    """
    parser = SWCParser(filepath)
    nodes, metadata = parser.parse()
    
    if not nodes:
        raise ValueError(f"No valid nodes found in {filepath}")
    
    if validate:
        _validate_swc_data(nodes)
    
    return Neuron(nodes, metadata)


def _validate_swc_data(nodes: List[SWCNode]) -> None:
    """Validate SWC data for common issues."""
    node_indices = {node.index for node in nodes}
    
    # Check for root node
    root_nodes = [node for node in nodes if node.parent == -1]
    if not root_nodes:
        raise ValueError("No root node found (parent = -1)")
    
    if len(root_nodes) > 1:
        print(f"Warning: Multiple root nodes found: {[n.index for n in root_nodes]}")
    
    # Check for orphaned nodes
    for node in nodes:
        if node.parent != -1 and node.parent not in node_indices:
            print(f"Warning: Node {node.index} has invalid parent {node.parent}")
    
    # Check for self-loops
    for node in nodes:
        if node.parent == node.index:
            print(f"Warning: Node {node.index} has self-loop")


def save_swc(neuron: Neuron, filepath: Union[str, Path]) -> None:
    """
    Save a neuron to an SWC file.
    
    Args:
        neuron: Neuron object to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    
    with open(filepath, 'w') as f:
        # Write header
        for key, value in neuron.metadata.items():
            if key.startswith('comment_'):
                f.write(f"# {value}\n")
            else:
                f.write(f"# {key} = {value}\n")
        
        # Write nodes
        for node in sorted(neuron.nodes.values(), key=lambda n: n.index):
            f.write(f"{node.index} {node.type_id} {node.x:.6f} {node.y:.6f} "
                   f"{node.z:.6f} {node.radius:.6f} {node.parent}\n")


def load_multiple_swc(filepaths: List[Union[str, Path]], 
                     validate: bool = True) -> List[Neuron]:
    """
    Load multiple SWC files.
    
    Args:
        filepaths: List of SWC file paths
        validate: Whether to validate loaded data
        
    Returns:
        List of Neuron objects
    """
    neurons = []
    for filepath in filepaths:
        try:
            neuron = load_swc(filepath, validate=validate)
            neurons.append(neuron)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    return neurons
