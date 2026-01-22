"""
3D mesh rendering for neuron morphology.
"""

import hashlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
import trimesh
import matplotlib.cm as cm

from ..core import Neuron
from ..io import NeuronClass, classify_type_id
from .sweep import sweep_circle


# ========================= Mesh Cache System =========================

MESH_CACHE_VERSION = 1

def mesh_cache_key(swc_path: Path, segments: int, cap: bool, radius_scale: float,
                   radius_adaptive_alpha: float, radius_ref_percentile: float) -> str:
    """Generate cache key from SWC file stats and mesh parameters."""
    stat = swc_path.stat()
    key_parts = [
        f"v{MESH_CACHE_VERSION}",
        str(stat.st_mtime_ns),
        str(stat.st_size),
        str(segments),
        str(int(cap)),
        f"{radius_scale:.6f}",
        f"{radius_adaptive_alpha:.6f}",
        f"{radius_ref_percentile:.6f}",
    ]
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()[:12]


def default_cache_dir(swc_path: Path) -> Path:
    """Default cache directory: mesh_cache/ subdirectory next to SWC file."""
    return swc_path.parent / "mesh_cache"


def cache_path_for_swc(swc_path: Path, cache_key: str, cache_dir: Optional[Path] = None) -> Path:
    """Get cache file path for an SWC file."""
    if cache_dir is None:
        cache_dir = default_cache_dir(swc_path)
    return cache_dir / f"{swc_path.stem}_{cache_key}.npz"


def load_gpu_cache(cache_path: Path) -> Optional[dict[str, np.ndarray]]:
    """Load GPU-ready arrays from npz cache."""
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def save_gpu_cache(cache_path: Path, gpu_arrays: dict[str, np.ndarray]) -> None:
    """Save GPU-ready arrays to npz cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **gpu_arrays)


def clear_cache(directory: Path, *, recursive: bool = False) -> int:
    """Remove mesh cache files from directory. Returns count removed."""
    count = 0
    search_fn = directory.rglob if recursive else directory.glob
    for f in search_fn("*.npz"):
        f.unlink()
        count += 1
    return count


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

    def build_mesh(self, *, segments: int = 32, cap: bool = False, translate_to_origin: bool = True, radius_scale: float = 1.0, radius_adaptive_alpha: float = 0.0, radius_ref_percentile: float = 50.0) -> trimesh.Trimesh:
        meshes: list[trimesh.Trimesh] = []

        soma = self._get_soma_node()
        soma_pos = soma.position if soma is not None else np.zeros(3)
        soma_radius = soma.radius if soma is not None else 0.0

        branches = self._extract_branches_from_soma()

        # Compute reference radius across all neurites (exclude soma)
        ref_radius = None
        if len(branches) > 0:
            all_r = np.concatenate([np.asarray(r, dtype=np.float64) for _, r in branches])
            if all_r.size > 0:
                q = float(np.clip(radius_ref_percentile, 0.0, 100.0))
                ref_radius = float(np.percentile(all_r, q))

        for path, radii in branches:
            if translate_to_origin:
                path = path - soma_pos
            # Scale neurite radii (soma sphere left unscaled). Optionally adapt scale more for thin neurites.
            r = np.asarray(radii, dtype=np.float64)
            if radius_adaptive_alpha > 0.0 and ref_radius is not None and ref_radius > 0.0:
                # Weight increases as radius gets smaller relative to reference.
                # weight = (ref/r)^alpha, clamped to avoid extreme boosts.
                w = (ref_radius / (r + 1e-12)) ** float(radius_adaptive_alpha)
                w = np.clip(w, 0.0, 10.0)
                # Convert weight into scale mixing coefficient in [0,1]
                mix = w / (1.0 + w)
                scale = 1.0 + (float(radius_scale) - 1.0) * mix
                radii_scaled = r * scale
            else:
                radii_scaled = r * float(radius_scale)
            mesh = sweep_circle(path=path, radii=radii_scaled, segments=segments, cap=cap, connect=False, kwargs={"process": False})
            meshes.append(mesh)

        if soma is not None and soma_radius > 0.0:
            center = np.zeros(3) if translate_to_origin else soma_pos
            meshes.append(self._create_sphere(center, soma_radius))

        if not meshes:
            raise ValueError("No meshes generated from SWC data")

        combined = trimesh.util.concatenate(meshes)
        return combined

    def build_mesh_by_type(self, *, segments: int = 32, cap: bool = False, translate_to_origin: bool = True, radius_scale: float = 1.0, radius_adaptive_alpha: float = 0.0, radius_ref_percentile: float = 50.0) -> dict[str, trimesh.Trimesh]:
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

        branches = self._extract_branches_from_soma()

        # Compute reference radius across all neurites
        ref_radius = None
        if len(branches) > 0:
            all_r = np.concatenate([np.asarray(r, dtype=np.float64) for _, r in branches])
            if all_r.size > 0:
                q = float(np.clip(radius_ref_percentile, 0.0, 100.0))
                ref_radius = float(np.percentile(all_r, q))

        for path, radii in branches:
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
            r = np.asarray(radii, dtype=np.float64)
            if radius_adaptive_alpha > 0.0 and ref_radius is not None and ref_radius > 0.0:
                w = (ref_radius / (r + 1e-12)) ** float(radius_adaptive_alpha)
                w = np.clip(w, 0.0, 10.0)
                mix = w / (1.0 + w)
                scale = 1.0 + (float(radius_scale) - 1.0) * mix
                radii_scaled = r * scale
            else:
                radii_scaled = r * float(radius_scale)
            mesh = sweep_circle(path=path, radii=radii_scaled, segments=segments, cap=cap, connect=False, kwargs={"process": False})
            groups[cls].append(mesh)
        out: dict[str, trimesh.Trimesh] = {}
        for k, lst in groups.items():
            if lst:
                out[k] = trimesh.util.concatenate(lst)
        if not out:
            raise ValueError("No meshes generated from SWC data")
        return out

    def build_gpu_arrays(self, *, segments: int = 32, cap: bool = False, translate_to_origin: bool = True,
                         radius_scale: float = 1.0, radius_adaptive_alpha: float = 0.0,
                         radius_ref_percentile: float = 50.0) -> dict[str, np.ndarray]:
        """Build GPU-ready expanded vertex arrays by neuron class.
        
        Returns dict mapping class name to (N, 3) float32 arrays ready for VBO upload.
        """
        meshes = self.build_mesh_by_type(
            segments=segments, cap=cap, translate_to_origin=translate_to_origin,
            radius_scale=radius_scale, radius_adaptive_alpha=radius_adaptive_alpha,
            radius_ref_percentile=radius_ref_percentile
        )
        gpu_arrays = {}
        for k, mesh in meshes.items():
            v = np.asarray(mesh.vertices, dtype=np.float32)
            f = np.asarray(mesh.faces, dtype=np.int32)
            gpu_arrays[k] = v[f.reshape(-1)]
        return gpu_arrays

    def build_gpu_arrays_cached(self, swc_path: Path, *, segments: int = 32, cap: bool = False,
                                 translate_to_origin: bool = True, radius_scale: float = 1.0,
                                 radius_adaptive_alpha: float = 0.0, radius_ref_percentile: float = 50.0,
                                 cache_dir: Optional[Path] = None) -> dict[str, np.ndarray]:
        """Build GPU-ready arrays with disk caching (npz format).
        
        Args:
            swc_path: Path to source SWC file (used for cache key)
            cache_dir: Directory for cache files. If None, uses mesh_cache/ next to SWC.
        """
        cache_key = mesh_cache_key(swc_path, segments, cap, radius_scale,
                                   radius_adaptive_alpha, radius_ref_percentile)
        cache_file = cache_path_for_swc(swc_path, cache_key, cache_dir)

        cached = load_gpu_cache(cache_file)
        if cached is not None:
            return cached

        gpu_arrays = self.build_gpu_arrays(
            segments=segments, cap=cap, translate_to_origin=translate_to_origin,
            radius_scale=radius_scale, radius_adaptive_alpha=radius_adaptive_alpha,
            radius_ref_percentile=radius_ref_percentile
        )
        save_gpu_cache(cache_file, gpu_arrays)
        return gpu_arrays

    def build_scene(self, *, segments: int = 32, cap: bool = False, translate_to_origin: bool = True, glue_union: bool = False, colorize: bool = False, cmap: str = "viridis", radius_scale: float = 1.0, radius_adaptive_alpha: float = 0.0, radius_ref_percentile: float = 50.0) -> trimesh.Scene:
        parts = self.build_mesh_by_type(segments=segments, cap=cap, translate_to_origin=translate_to_origin, radius_scale=radius_scale, radius_adaptive_alpha=radius_adaptive_alpha, radius_ref_percentile=radius_ref_percentile)
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
    
    def render_to_file(self, output_path: Union[str, Path], *, segments: int = 32, cap: bool = False, translate_to_origin: bool = True, radius_scale: float = 1.0, radius_adaptive_alpha: float = 0.0, radius_ref_percentile: float = 50.0) -> trimesh.Trimesh:
        mesh = self.build_mesh(segments=segments, cap=cap, translate_to_origin=translate_to_origin, radius_scale=radius_scale, radius_adaptive_alpha=radius_adaptive_alpha, radius_ref_percentile=radius_ref_percentile)
        mesh.export(output_path)
        return mesh
