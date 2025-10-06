"""
3D mesh rendering for neuron morphology.
"""

import numpy as np
import trimesh
from typing import Optional, Union
from pathlib import Path
from ..core import Neuron


class MeshRenderer:
    """Build 3D mesh representation of a neuron from SWC data."""
    
    def __init__(self, neuron: Neuron):
        self.neuron = neuron
        self.meshes: list[trimesh.Trimesh] = []
    
    def create_cylinder(self, start: np.ndarray, end: np.ndarray, 
                       radius_start: float, radius_end: float,
                       sections: int = 8) -> trimesh.Trimesh:
        """Create a tapered cylinder between two points."""
        direction = end - start
        height = np.linalg.norm(direction)
        
        if height < 1e-6:
            # Degenerate case: return sphere at start point
            return trimesh.creation.icosphere(subdivisions=2, radius=radius_start)
        
        # Create cylinder along z-axis
        cylinder = trimesh.creation.cylinder(
            radius=1.0,
            height=height,
            sections=sections
        )
        
        # Scale radii (tapered cylinder)
        vertices = cylinder.vertices
        z_coords = vertices[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        z_normalized = (z_coords - z_min) / (z_max - z_min)
        
        # Interpolate radius along cylinder
        radii = radius_start + (radius_end - radius_start) * z_normalized
        vertices[:, 0] *= radii
        vertices[:, 1] *= radii
        
        # Align cylinder with direction vector
        direction_norm = direction / height
        z_axis = np.array([0, 0, 1])
        
        # Rotation axis and angle
        rotation_axis = np.cross(z_axis, direction_norm)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm > 1e-6:
            rotation_axis /= rotation_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1.0, 1.0))
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle, rotation_axis
            )
        else:
            # Vectors are parallel or anti-parallel
            if np.dot(z_axis, direction_norm) > 0:
                rotation_matrix = np.eye(4)
            else:
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    np.pi, [1, 0, 0]
                )
        
        cylinder.apply_transform(rotation_matrix)
        
        # Translate to start position
        translation = start + direction / 2
        cylinder.apply_translation(translation)
        
        return cylinder
    
    def create_sphere(self, center: np.ndarray, radius: float) -> trimesh.Trimesh:
        """Create a sphere at the given position."""
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        sphere.apply_translation(center)
        return sphere
    
    def build_mesh(self, add_spheres: bool = True) -> trimesh.Trimesh:
        """Build complete neuron mesh from SWC nodes."""
        meshes = []
        
        for node in self.neuron.nodes.values():
            position = node.position
            
            # Add sphere at node position (optional, helps with visualization)
            if add_spheres:
                sphere = self.create_sphere(position, node.radius)
                meshes.append(sphere)
            
            # Create cylinder to parent node
            if node.parent != -1 and node.parent in self.neuron.nodes:
                parent = self.neuron.nodes[node.parent]
                parent_pos = parent.position
                
                cylinder = self.create_cylinder(
                    parent_pos, position,
                    parent.radius, node.radius
                )
                meshes.append(cylinder)
        
        # Combine all meshes
        if not meshes:
            raise ValueError("No meshes generated from SWC file")
        
        combined = trimesh.util.concatenate(meshes)
        return combined
    
    def render_to_file(self, output_path: Union[str, Path], 
                      add_spheres: bool = True) -> trimesh.Trimesh:
        """
        Render neuron to 3D mesh file.
        
        Args:
            output_path: Path to output mesh file
            add_spheres: Whether to add spheres at node positions
            
        Returns:
            Generated trimesh object
        """
        mesh = self.build_mesh(add_spheres=add_spheres)
        mesh.export(output_path)
        return mesh
