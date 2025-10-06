"""
3D mesh rendering for neuron morphology.
"""

import numpy as np
import trimesh
from typing import Optional, Union
from pathlib import Path
from ..core import Neuron
from ..io import NeuronClass, classify_type_id
import matplotlib.cm as cm
from .sweep import sweep_circle


class MeshRenderer:
    """Build 3D mesh representation of a neuron from SWC data."""
    
    def __init__(self, neuron: Neuron):
        self.neuron = neuron
        self.meshes: list[trimesh.Trimesh] = []

    def _extract_branches(self) -> list[tuple[np.ndarray, np.ndarray]]:
        branches: list[tuple[np.ndarray, np.ndarray]] = []
        node_map = self.neuron.nodes

        candidates: list[int] = []
        for node in node_map.values():
            is_root = node.parent == -1
            parent_branch = False
            if not is_root and node.parent in node_map:
                parent_branch = len(node_map[node.parent].children) > 1
            if is_root or parent_branch:
                candidates.append(node.index)

        for start_idx in candidates:
            start_node = node_map[start_idx]
            if len(start_node.children) == 0:
                continue

            for child_idx in start_node.children:
                path_positions: list[np.ndarray] = [start_node.position]
                path_radii: list[float] = [start_node.radius]

                current_idx = child_idx
                while True:
                    current = node_map[current_idx]
                    path_positions.append(current.position)
                    path_radii.append(current.radius)

                    is_terminal = len(current.children) == 0
                    is_branch = len(current.children) > 1
                    if is_terminal or is_branch:
                        break
                    current_idx = current.children[0]

                path = np.vstack(path_positions)
                radii = np.asarray(path_radii, dtype=np.float64)
                if len(path) >= 2:
                    branches.append((path, radii))

        return branches

    def _junction_node_indices(self) -> list[int]:
        idxs: list[int] = []
        for node in self.neuron.nodes.values():
            is_root = node.parent == -1
            deg = len(node.children)
            if is_root or deg != 1:
                idxs.append(node.index)
        return idxs

    def _create_sphere(self, center: np.ndarray, radius: float, *, subdivisions: int = 2) -> trimesh.Trimesh:
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        sphere.apply_translation(center)
        return sphere

    def _get_soma_node(self):
        soma_idx = self.neuron.soma_index
        if soma_idx is None:
            roots = [n for n in self.neuron.nodes.values() if n.parent == -1]
            if not roots:
                return None
            return roots[0]
        return self.neuron.nodes.get(soma_idx)

    def _extract_branches_from_soma(self) -> list[tuple[np.ndarray, np.ndarray]]:
        soma = self._get_soma_node()
        if soma is None:
            return []

        branches: list[tuple[np.ndarray, np.ndarray]] = []

        def walk_from(start_idx: int):
            start_node = self.neuron.nodes[start_idx]
            for child_idx in start_node.children:
                path_positions: list[np.ndarray] = [start_node.position]
                path_radii: list[float] = [start_node.radius]
                current_idx = child_idx
                while True:
                    current = self.neuron.nodes[current_idx]
                    path_positions.append(current.position)
                    path_radii.append(current.radius)
                    deg = len(current.children)
                    if deg != 1:
                        break
                    current_idx = current.children[0]
                path = np.vstack(path_positions)
                radii = np.asarray(path_radii, dtype=np.float64)
                if len(path) >= 2:
                    branches.append((path, radii))
                if len(self.neuron.nodes[current_idx].children) > 1:
                    walk_from(current_idx)

        walk_from(soma.index)
        return branches

    def _classify(self, type_id: int) -> str:
        if type_id == 1:
            return "soma"
        if type_id == 2:
            return "axon"
        if type_id in (3, 4):
            return "dendrite"
        return "other"

    def _extract_branches_with_type(self) -> list[tuple[np.ndarray, np.ndarray, str]]:
        typed: list[tuple[np.ndarray, np.ndarray, str]] = []
        node_map = self.neuron.nodes
        candidates: list[int] = []
        for node in node_map.values():
            is_root = node.parent == -1
            parent_branch = False
            if not is_root and node.parent in node_map:
                parent_branch = len(node_map[node.parent].children) > 1
            if is_root or parent_branch:
                candidates.append(node.index)

        for start_idx in candidates:
            start_node = node_map[start_idx]
            if len(start_node.children) == 0:
                continue
            for child_idx in start_node.children:
                path_positions: list[np.ndarray] = [start_node.position]
                path_radii: list[float] = [start_node.radius]
                current_idx = child_idx
                while True:
                    current = node_map[current_idx]
                    path_positions.append(current.position)
                    path_radii.append(current.radius)
                    is_terminal = len(current.children) == 0
                    is_branch = len(current.children) > 1
                    if is_terminal or is_branch:
                        break
                    current_idx = current.children[0]
                path = np.vstack(path_positions)
                radii = np.asarray(path_radii, dtype=np.float64)
                if len(path) >= 2:
                    cls = self._classify(start_node.type_id)
                    typed.append((path, radii, cls))
        return typed

    def build_mesh(self, *, segments: int = 24, cap: bool = False, translate_to_origin: bool = True) -> trimesh.Trimesh:
        meshes: list[trimesh.Trimesh] = []

        soma = self._get_soma_node()
        soma_pos = soma.position if soma is not None else np.zeros(3)
        soma_radius = soma.radius if soma is not None else 0.0

        for path, radii in self._extract_branches_from_soma():
            if translate_to_origin:
                path = path - soma_pos
            mesh = sweep_circle(path=path, radii=radii, segments=segments, cap=cap, connect=False, kwargs={"process": False})
            meshes.append(mesh)

        if soma is not None and soma_radius > 0.0:
            center = np.zeros(3) if translate_to_origin else soma_pos
            meshes.append(self._create_sphere(center, soma_radius))

        if not meshes:
            raise ValueError("No meshes generated from SWC data")

        combined = trimesh.util.concatenate(meshes)
        return combined

    def build_mesh_by_type(self, *, segments: int = 24, cap: bool = False, translate_to_origin: bool = True) -> dict[str, trimesh.Trimesh]:
        groups: dict[str, list[trimesh.Trimesh]] = {
            NeuronClass.SOMA.name: [],
            NeuronClass.AXON.name: [],
            NeuronClass.BASAL_DENDRITE.name: [],
            NeuronClass.APICAL_DENDRITE.name: [],
            NeuronClass.OTHER.name: [],
        }
        soma = self._get_soma_node()
        soma_pos = soma.position if soma is not None else np.zeros(3)
        if soma is not None and soma.radius > 0.0:
            center = np.zeros(3) if translate_to_origin else soma_pos
            groups[NeuronClass.SOMA.name].append(self._create_sphere(center, soma.radius))

        for path, radii in self._extract_branches_from_soma():
            if translate_to_origin:
                path = path - soma_pos
            # Determine class from the starting node type
            # Find the starting node by matching position with tolerance
            cls_enum = NeuronClass.OTHER
            def _match_node(pos: np.ndarray):
                world = pos + (soma_pos if translate_to_origin else 0)
                for n in self.neuron.nodes.values():
                    if np.allclose(n.position, world, atol=1e-9):
                        return n
                return None
            start_node = _match_node(path[0])
            if start_node is not None and start_node.type_id == NeuronClass.SOMA:
                ref_idx = 1 if len(path) > 1 else 0
                node_for_class = _match_node(path[ref_idx])
                if node_for_class is not None:
                    cls_enum = classify_type_id(node_for_class.type_id)
            else:
                if start_node is not None:
                    cls_enum = classify_type_id(start_node.type_id)
            cls = cls_enum.name
            mesh = sweep_circle(path=path, radii=radii, segments=segments, cap=cap, connect=False, kwargs={"process": False})
            groups[cls].append(mesh)
        out: dict[str, trimesh.Trimesh] = {}
        for k, lst in groups.items():
            if lst:
                out[k] = trimesh.util.concatenate(lst)
        if not out:
            raise ValueError("No meshes generated from SWC data")
        return out

    def build_scene(self, *, segments: int = 24, cap: bool = False, translate_to_origin: bool = True, glue_union: bool = False, colorize: bool = False, cmap: str = "viridis") -> trimesh.Scene:
        parts = self.build_mesh_by_type(segments=segments, cap=cap, translate_to_origin=translate_to_origin)
        if glue_union:
            # Boolean union per class
            for key, mesh in list(parts.items()):
                try:
                    # union requires watertight meshes; keep fallback
                    parts[key] = mesh.union(mesh.split(only_watertight=False), engine=None)  # try default engine
                except Exception:
                    # leave as-is if union fails
                    pass
        scene = trimesh.Scene()
        color_map = None
        if colorize:
            try:
                color_map = cm.get_cmap(cmap)
            except Exception:
                color_map = cm.get_cmap('viridis')

        def _class_color(name: str) -> Optional[np.ndarray]:
            if color_map is None:
                return None
            order = {
                NeuronClass.SOMA.name: 0.0,
                NeuronClass.AXON.name: 0.5,
                NeuronClass.BASAL_DENDRITE.name: 0.75,
                NeuronClass.APICAL_DENDRITE.name: 1.0,
                NeuronClass.OTHER.name: 0.25,
            }
            t = order.get(name, 0.25)
            rgba = color_map(t)
            return (np.array(rgba[:3]) * 255).astype(np.uint8)

        for name, mesh in parts.items():
            if colorize:
                color = _class_color(name)
                if color is not None:
                    # set a uniform vertex color for the mesh
                    vc = np.tile(color, (len(mesh.vertices), 1))
                    mesh.visual.vertex_colors = vc
            scene.add_geometry(mesh, node_name=name)
        # Ensure origin at soma already handled by translation
        return scene
    
    def render_to_file(self, output_path: Union[str, Path], *, segments: int = 24, cap: bool = False, translate_to_origin: bool = True) -> trimesh.Trimesh:
        mesh = self.build_mesh(segments=segments, cap=cap, translate_to_origin=translate_to_origin)
        mesh.export(output_path)
        return mesh
