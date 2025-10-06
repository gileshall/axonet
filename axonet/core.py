"""
Core data structures for representing neuron morphology.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import networkx as nx


@dataclass
class SWCNode:
    """Represents a single node in an SWC file."""
    index: int
    type_id: int
    x: float
    y: float
    z: float
    radius: float
    parent: int
    children: List[int] = field(default_factory=list)
    
    @property
    def position(self) -> np.ndarray:
        """Get 3D position as numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @property
    def is_soma(self) -> bool:
        """Check if this node is part of the soma."""
        return self.type_id == 1
    
    @property
    def is_axon(self) -> bool:
        """Check if this node is part of an axon."""
        return self.type_id == 2
    
    @property
    def is_dendrite(self) -> bool:
        """Check if this node is part of a dendrite."""
        return self.type_id in [3, 4]  # basal or apical dendrite
    
    @property
    def is_terminal(self) -> bool:
        """Check if this node is a terminal (leaf) node."""
        return len(self.children) == 0
    
    @property
    def is_branch_point(self) -> bool:
        """Check if this node is a branch point."""
        return len(self.children) > 1


class Neuron:
    """
    Represents a complete neuron morphology with graph structure and analysis capabilities.
    """
    
    def __init__(self, nodes: List[SWCNode], metadata: Optional[Dict] = None):
        """
        Initialize neuron from list of SWC nodes.
        
        Args:
            nodes: List of SWCNode objects
            metadata: Optional metadata dictionary
        """
        self.nodes = {node.index: node for node in nodes}
        self.metadata = metadata or {}
        self._graph = None
        self._soma_index = None
        self._build_relationships()
    
    def _build_relationships(self):
        """Build parent-child relationships between nodes."""
        for node in self.nodes.values():
            if node.parent != -1 and node.parent in self.nodes:
                self.nodes[node.parent].children.append(node.index)
    
    @property
    def graph(self) -> nx.DiGraph:
        """Get NetworkX directed graph representation."""
        if self._graph is None:
            self._graph = nx.DiGraph()
            
            # Add nodes
            for node in self.nodes.values():
                self._graph.add_node(
                    node.index,
                    type_id=node.type_id,
                    position=node.position,
                    radius=node.radius,
                    is_soma=node.is_soma,
                    is_axon=node.is_axon,
                    is_dendrite=node.is_dendrite
                )
            
            # Add edges
            for node in self.nodes.values():
                if node.parent != -1 and node.parent in self.nodes:
                    self._graph.add_edge(node.parent, node.index)
        
        return self._graph
    
    @property
    def soma_index(self) -> Optional[int]:
        """Get the index of the soma node."""
        if self._soma_index is None:
            # Look for soma nodes (type 1)
            soma_nodes = [idx for idx, node in self.nodes.items() if node.is_soma]
            if soma_nodes:
                self._soma_index = soma_nodes[0]  # Take first soma node
            else:
                # If no soma, use root node
                root_nodes = [idx for idx, node in self.nodes.items() if node.parent == -1]
                if root_nodes:
                    self._soma_index = root_nodes[0]
        
        return self._soma_index
    
    @property
    def soma_nodes(self) -> List[SWCNode]:
        """Get all soma nodes."""
        return [node for node in self.nodes.values() if node.is_soma]
    
    @property
    def axon_nodes(self) -> List[SWCNode]:
        """Get all axon nodes."""
        return [node for node in self.nodes.values() if node.is_axon]
    
    @property
    def dendrite_nodes(self) -> List[SWCNode]:
        """Get all dendrite nodes."""
        return [node for node in self.nodes.values() if node.is_dendrite]
    
    @property
    def terminal_nodes(self) -> List[SWCNode]:
        """Get all terminal (leaf) nodes."""
        return [node for node in self.nodes.values() if node.is_terminal]
    
    @property
    def branch_points(self) -> List[SWCNode]:
        """Get all branch point nodes."""
        return [node for node in self.nodes.values() if node.is_branch_point]
    
    def get_children(self, node_index: int) -> List[SWCNode]:
        """Get children of a specific node."""
        return [self.nodes[child_idx] for child_idx in self.nodes[node_index].children]
    
    def get_parent(self, node_index: int) -> Optional[SWCNode]:
        """Get parent of a specific node."""
        parent_idx = self.nodes[node_index].parent
        return self.nodes.get(parent_idx) if parent_idx != -1 else None
    
    def get_path_to_root(self, node_index: int) -> List[SWCNode]:
        """Get path from node to root (soma)."""
        path = []
        current = node_index
        
        while current != -1 and current in self.nodes:
            path.append(self.nodes[current])
            current = self.nodes[current].parent
        
        return path
    
    def get_subtree(self, root_index: int) -> List[SWCNode]:
        """Get all nodes in subtree rooted at given node."""
        subtree = []
        to_visit = [root_index]
        
        while to_visit:
            current = to_visit.pop()
            if current in self.nodes:
                subtree.append(self.nodes[current])
                to_visit.extend(self.nodes[current].children)
        
        return subtree
    
    def get_branch_order(self, node_index: int) -> int:
        """Get branch order (depth from soma) of a node."""
        path = self.get_path_to_root(node_index)
        return len(path) - 1
    
    def get_euclidean_distance(self, node1_index: int, node2_index: int) -> float:
        """Get Euclidean distance between two nodes."""
        pos1 = self.nodes[node1_index].position
        pos2 = self.nodes[node2_index].position
        return np.linalg.norm(pos1 - pos2)
    
    def get_path_distance(self, node1_index: int, node2_index: int) -> float:
        """Get path distance (along tree) between two nodes."""
        path1 = self.get_path_to_root(node1_index)
        path2 = self.get_path_to_root(node2_index)
        
        # Find common ancestor
        common_ancestor = None
        for i, node in enumerate(path1):
            if node in path2:
                common_ancestor = i
                break
        
        if common_ancestor is None:
            return float('inf')
        
        # Distance is sum of distances from each node to common ancestor
        distance = 0
        for i in range(common_ancestor):
            if i < len(path1) - 1:
                distance += self.get_euclidean_distance(
                    path1[i].index, path1[i + 1].index
                )
        
        for i in range(common_ancestor):
            if i < len(path2) - 1:
                distance += self.get_euclidean_distance(
                    path2[i].index, path2[i + 1].index
                )
        
        return distance
    
    def get_segment_length(self, node_index: int) -> float:
        """Get length of segment from node to its parent."""
        parent = self.get_parent(node_index)
        if parent is None:
            return 0.0
        return self.get_euclidean_distance(node_index, parent.index)
    
    def get_total_length(self, node_type: Optional[int] = None) -> float:
        """Get total length of all segments, optionally filtered by type."""
        total = 0.0
        for node in self.nodes.values():
            if node_type is None or node.type_id == node_type:
                total += self.get_segment_length(node.index)
        return total
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get 3D bounding box as (min_corner, max_corner)."""
        positions = np.array([node.position for node in self.nodes.values()])
        return positions.min(axis=0), positions.max(axis=0)
    
    def get_center_of_mass(self) -> np.ndarray:
        """Get center of mass of all nodes."""
        positions = np.array([node.position for node in self.nodes.values()])
        return positions.mean(axis=0)
    
    def __len__(self) -> int:
        """Return number of nodes in neuron."""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        """String representation of neuron."""
        n_nodes = len(self.nodes)
        n_axons = len(self.axon_nodes)
        n_dendrites = len(self.dendrite_nodes)
        n_terminals = len(self.terminal_nodes)
        return f"Neuron(nodes={n_nodes}, axons={n_axons}, dendrites={n_dendrites}, terminals={n_terminals})"
