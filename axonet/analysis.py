"""
Comprehensive morphological analysis for neuron data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from .core import Neuron, SWCNode


class MorphologyAnalyzer:
    """
    Comprehensive morphological analysis of neuron structures.
    """
    
    def __init__(self, neuron: Neuron):
        self.neuron = neuron
        self._cache = {}
    
    def compute_all_features(self) -> Dict[str, float]:
        """
        Compute all available morphological features.
        
        Returns:
            Dictionary mapping feature names to values
        """
        features = {}
        
        # Basic morphology
        features.update(self.compute_basic_morphology())
        
        # Volumetric and spatial
        features.update(self.compute_volumetric_features())
        
        # Graph-theoretic
        features.update(self.compute_graph_features())
        
        # Fractal and complexity
        features.update(self.compute_complexity_features())
        
        # Neuron-specific
        features.update(self.compute_neuron_specific_features())
        
        # Statistical distributions
        features.update(self.compute_distribution_features())
        
        # Spatial orientation
        features.update(self.compute_orientation_features())
        
        return features
    
    def compute_basic_morphology(self) -> Dict[str, float]:
        """Compute basic morphological measurements."""
        features = {}
        
        # Length metrics
        features['total_dendritic_length'] = self.neuron.get_total_length(3) + self.neuron.get_total_length(4)
        features['total_axonal_length'] = self.neuron.get_total_length(2)
        features['total_neurite_length'] = self.neuron.get_total_length()
        
        # Segment lengths
        segment_lengths = [self.neuron.get_segment_length(node.index) 
                          for node in self.neuron.nodes.values() 
                          if node.parent != -1]
        
        if segment_lengths:
            features['mean_segment_length'] = np.mean(segment_lengths)
            features['median_segment_length'] = np.median(segment_lengths)
            features['std_segment_length'] = np.std(segment_lengths)
            features['max_segment_length'] = np.max(segment_lengths)
            features['min_segment_length'] = np.min(segment_lengths)
        else:
            features.update({
                'mean_segment_length': 0.0,
                'median_segment_length': 0.0,
                'std_segment_length': 0.0,
                'max_segment_length': 0.0,
                'min_segment_length': 0.0
            })
        
        # Branch metrics
        features['n_branch_points'] = len(self.neuron.branch_points)
        features['n_terminals'] = len(self.neuron.terminal_nodes)
        features['n_dendritic_branches'] = len([n for n in self.neuron.branch_points if n.is_dendrite])
        features['n_axonal_branches'] = len([n for n in self.neuron.branch_points if n.is_axon])
        
        # Branch orders
        branch_orders = [self.neuron.get_branch_order(node.index) for node in self.neuron.nodes.values()]
        features['max_branch_order'] = max(branch_orders) if branch_orders else 0
        features['mean_branch_order'] = np.mean(branch_orders) if branch_orders else 0
        
        # Path metrics
        if self.neuron.soma_index is not None:
            soma_to_terminals = []
            for terminal in self.neuron.terminal_nodes:
                path = self.neuron.get_path_to_root(terminal.index)
                path_length = sum(self.neuron.get_euclidean_distance(path[i].index, path[i+1].index) 
                                for i in range(len(path)-1))
                soma_to_terminals.append(path_length)
            
            if soma_to_terminals:
                features['max_path_distance'] = max(soma_to_terminals)
                features['mean_path_distance'] = np.mean(soma_to_terminals)
                features['median_path_distance'] = np.median(soma_to_terminals)
            else:
                features.update({
                    'max_path_distance': 0.0,
                    'mean_path_distance': 0.0,
                    'median_path_distance': 0.0
                })
        else:
            features.update({
                'max_path_distance': 0.0,
                'mean_path_distance': 0.0,
                'median_path_distance': 0.0
            })
        
        return features
    
    def compute_volumetric_features(self) -> Dict[str, float]:
        """Compute volumetric and spatial measurements."""
        features = {}
        
        # Volume and surface area estimates
        total_volume = 0.0
        total_surface_area = 0.0
        
        for node in self.neuron.nodes.values():
            if node.parent != -1 and node.parent in self.neuron.nodes:
                parent = self.neuron.nodes[node.parent]
                segment_length = self.neuron.get_euclidean_distance(node.index, parent.index)
                
                # Volume as truncated cone
                r1, r2 = parent.radius, node.radius
                volume = (np.pi * segment_length / 3) * (r1**2 + r1*r2 + r2**2)
                total_volume += volume
                
                # Surface area as truncated cone
                surface_area = np.pi * (r1 + r2) * np.sqrt((r1 - r2)**2 + segment_length**2)
                total_surface_area += surface_area
        
        features['total_volume'] = total_volume
        features['total_surface_area'] = total_surface_area
        features['surface_to_volume_ratio'] = total_surface_area / total_volume if total_volume > 0 else 0
        
        # Bounding box
        min_corner, max_corner = self.neuron.get_bounding_box()
        bbox_dims = max_corner - min_corner
        features['bbox_volume'] = np.prod(bbox_dims)
        features['bbox_x'] = bbox_dims[0]
        features['bbox_y'] = bbox_dims[1]
        features['bbox_z'] = bbox_dims[2]
        features['space_filling_ratio'] = total_volume / features['bbox_volume'] if features['bbox_volume'] > 0 else 0
        
        # Radius statistics
        radii = [node.radius for node in self.neuron.nodes.values()]
        features['mean_radius'] = np.mean(radii)
        features['std_radius'] = np.std(radii)
        features['max_radius'] = np.max(radii)
        features['min_radius'] = np.min(radii)
        
        # Dendritic vs axonal ratios
        dend_length = self.neuron.get_total_length(3) + self.neuron.get_total_length(4)
        axon_length = self.neuron.get_total_length(2)
        features['axon_dendrite_length_ratio'] = axon_length / dend_length if dend_length > 0 else 0
        
        return features
    
    def compute_graph_features(self) -> Dict[str, float]:
        """Compute graph-theoretic measures."""
        features = {}
        G = self.neuron.graph
        
        # Basic graph properties
        features['n_nodes'] = G.number_of_nodes()
        features['n_edges'] = G.number_of_edges()
        features['n_components'] = nx.number_weakly_connected_components(G)
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        features['mean_degree'] = np.mean(degrees) if degrees else 0
        features['max_degree'] = max(degrees) if degrees else 0
        
        # Tree properties
        if nx.is_tree(G.to_undirected()):
            features['is_tree'] = 1.0
            features['tree_depth'] = max(nx.shortest_path_length(G.to_undirected(), 
                                                               self.neuron.soma_index or 0).values()) if self.neuron.soma_index else 0
        else:
            features['is_tree'] = 0.0
            features['tree_depth'] = 0
        
        # Centrality measures
        if G.number_of_nodes() > 1:
            try:
                betweenness = nx.betweenness_centrality(G)
                features['mean_betweenness'] = np.mean(list(betweenness.values()))
                features['max_betweenness'] = max(betweenness.values())
            except:
                features['mean_betweenness'] = 0.0
                features['max_betweenness'] = 0.0
            
            try:
                closeness = nx.closeness_centrality(G)
                features['mean_closeness'] = np.mean(list(closeness.values()))
                features['max_closeness'] = max(closeness.values())
            except:
                features['mean_closeness'] = 0.0
                features['max_closeness'] = 0.0
        else:
            features.update({
                'mean_betweenness': 0.0,
                'max_betweenness': 0.0,
                'mean_closeness': 0.0,
                'max_closeness': 0.0
            })
        
        return features
    
    def compute_complexity_features(self) -> Dict[str, float]:
        """Compute fractal and complexity measures."""
        features = {}
        
        # Sholl analysis
        if self.neuron.soma_index is not None:
            soma_pos = self.neuron.nodes[self.neuron.soma_index].position
            distances = [np.linalg.norm(node.position - soma_pos) for node in self.neuron.nodes.values()]
            
            if distances:
                max_dist = max(distances)
                n_shells = 20
                shell_radii = np.linspace(0, max_dist, n_shells)
                intersections = []
                
                for radius in shell_radii:
                    count = sum(1 for d in distances if abs(d - radius) < max_dist / n_shells)
                    intersections.append(count)
                
                features['sholl_max_intersections'] = max(intersections)
                features['sholl_mean_intersections'] = np.mean(intersections)
                features['sholl_regression_coeff'] = np.polyfit(shell_radii, intersections, 1)[0] if len(shell_radii) > 1 else 0
            else:
                features.update({
                    'sholl_max_intersections': 0,
                    'sholl_mean_intersections': 0,
                    'sholl_regression_coeff': 0
                })
        else:
            features.update({
                'sholl_max_intersections': 0,
                'sholl_mean_intersections': 0,
                'sholl_regression_coeff': 0
            })
        
        # Fractal dimension (box-counting approximation)
        positions = np.array([node.position for node in self.neuron.nodes.values()])
        if len(positions) > 1:
            # 2D projection for fractal dimension
            positions_2d = positions[:, :2]  # Use X, Y coordinates
            
            # Box counting
            min_coords = positions_2d.min(axis=0)
            max_coords = positions_2d.max(axis=0)
            size = max_coords - min_coords
            
            if np.all(size > 0):
                scales = np.logspace(0, -2, 10)  # Different box sizes
                counts = []
                
                for scale in scales:
                    box_size = size * scale
                    if np.all(box_size > 0):
                        n_boxes = np.ceil(size / box_size).astype(int)
                        grid = np.zeros(n_boxes)
                        
                        for pos in positions_2d:
                            box_idx = ((pos - min_coords) / box_size).astype(int)
                            box_idx = np.clip(box_idx, 0, n_boxes - 1)
                            grid[box_idx[0], box_idx[1]] = 1
                        
                        counts.append(np.sum(grid > 0))
                
                if len(counts) > 1 and counts[0] > 0:
                    # Fit line to log-log plot
                    log_scales = np.log(scales[:len(counts)])
                    log_counts = np.log(counts)
                    if len(log_scales) > 1:
                        slope = np.polyfit(log_scales, log_counts, 1)[0]
                        features['fractal_dimension_2d'] = -slope
                    else:
                        features['fractal_dimension_2d'] = 0
                else:
                    features['fractal_dimension_2d'] = 0
            else:
                features['fractal_dimension_2d'] = 0
        else:
            features['fractal_dimension_2d'] = 0
        
        return features
    
    def compute_neuron_specific_features(self) -> Dict[str, float]:
        """Compute neuron-specific features."""
        features = {}
        
        # Soma characteristics
        soma_nodes = self.neuron.soma_nodes
        if soma_nodes:
            features['n_soma_nodes'] = len(soma_nodes)
            soma_radii = [node.radius for node in soma_nodes]
            features['mean_soma_radius'] = np.mean(soma_radii)
            features['max_soma_radius'] = max(soma_radii)
        else:
            features.update({
                'n_soma_nodes': 0,
                'mean_soma_radius': 0,
                'max_soma_radius': 0
            })
        
        # Primary neurites (direct soma connections)
        if self.neuron.soma_index is not None:
            primary_neurites = [node for node in self.neuron.nodes.values() 
                              if node.parent == self.neuron.soma_index]
            features['n_primary_neurites'] = len(primary_neurites)
        else:
            features['n_primary_neurites'] = 0
        
        # Compartment ratios
        n_axons = len(self.neuron.axon_nodes)
        n_dendrites = len(self.neuron.dendrite_nodes)
        features['axon_dendrite_node_ratio'] = n_axons / n_dendrites if n_dendrites > 0 else 0
        
        n_axon_terminals = len([n for n in self.neuron.terminal_nodes if n.is_axon])
        n_dendrite_terminals = len([n for n in self.neuron.terminal_nodes if n.is_dendrite])
        features['axon_dendrite_terminal_ratio'] = n_axon_terminals / n_dendrite_terminals if n_dendrite_terminals > 0 else 0
        
        return features
    
    def compute_distribution_features(self) -> Dict[str, float]:
        """Compute statistical distribution features."""
        features = {}
        
        # Segment length distribution
        segment_lengths = [self.neuron.get_segment_length(node.index) 
                          for node in self.neuron.nodes.values() 
                          if node.parent != -1]
        
        if segment_lengths:
            features['segment_length_skewness'] = self._skewness(segment_lengths)
            features['segment_length_kurtosis'] = self._kurtosis(segment_lengths)
        else:
            features.update({
                'segment_length_skewness': 0,
                'segment_length_kurtosis': 0
            })
        
        # Radius distribution
        radii = [node.radius for node in self.neuron.nodes.values()]
        features['radius_skewness'] = self._skewness(radii)
        features['radius_kurtosis'] = self._kurtosis(radii)
        
        # Branch order distribution
        branch_orders = [self.neuron.get_branch_order(node.index) for node in self.neuron.nodes.values()]
        features['branch_order_skewness'] = self._skewness(branch_orders)
        features['branch_order_kurtosis'] = self._kurtosis(branch_orders)
        
        return features
    
    def compute_orientation_features(self) -> Dict[str, float]:
        """Compute spatial orientation features."""
        features = {}
        
        positions = np.array([node.position for node in self.neuron.nodes.values()])
        
        if len(positions) > 1:
            # Basic spatial extent and orientation
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            center = positions.mean(axis=0)
            
            # Spatial dimensions
            dimensions = max_pos - min_pos
            features['spatial_extent_x'] = dimensions[0]
            features['spatial_extent_y'] = dimensions[1]
            features['spatial_extent_z'] = dimensions[2]
            
            # Aspect ratios
            if dimensions[2] > 0:
                features['xy_aspect_ratio'] = dimensions[0] / dimensions[1] if dimensions[1] > 0 else 1.0
                features['xz_aspect_ratio'] = dimensions[0] / dimensions[2]
                features['yz_aspect_ratio'] = dimensions[1] / dimensions[2]
            else:
                features.update({
                    'xy_aspect_ratio': 1.0,
                    'xz_aspect_ratio': 1.0,
                    'yz_aspect_ratio': 1.0
                })
            
            # Center of mass relative to bounding box
            bbox_center = (min_pos + max_pos) / 2
            center_offset = np.linalg.norm(center - bbox_center)
            features['center_offset'] = center_offset
            
            # Sphericity (how close to spherical)
            if np.all(dimensions > 0):
                sphericity = (36 * np.pi * (np.prod(dimensions)**2))**(1/3) / (np.sum(dimensions**2))
                features['sphericity'] = sphericity
            else:
                features['sphericity'] = 0.0
        else:
            features.update({
                'spatial_extent_x': 0.0,
                'spatial_extent_y': 0.0,
                'spatial_extent_z': 0.0,
                'xy_aspect_ratio': 1.0,
                'xz_aspect_ratio': 1.0,
                'yz_aspect_ratio': 1.0,
                'center_offset': 0.0,
                'sphericity': 0.0
            })
        
        return features
    
    def _skewness(self, data: List[float]) -> float:
        """Compute skewness of data."""
        if len(data) < 3:
            return 0.0
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: List[float]) -> float:
        """Compute kurtosis of data."""
        if len(data) < 4:
            return 0.0
        data = np.array(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
