"""VAE evaluator for annotating SWC files with embeddings and generating visualizations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import multiprocessing
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from imageio.v2 import imwrite
from PIL import Image

from ..io import load_swc, NeuronClass
from ..models.d3_swc_vae import SegVAE2D, load_model
from ..visualization.render import OffscreenContext, NeuroRenderCore, RenderConfig, RenderMode

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _get_cache_key(swc_path: Path, width: int, height: int, **render_kwargs) -> str:
    """Generate cache key for rendered output.
    
    Args:
        swc_path: Path to SWC file
        width: Render width
        height: Render height
        **render_kwargs: Additional render parameters
    
    Returns:
        Cache key string
    """
    key_parts = [
        str(swc_path),
        str(width),
        str(height),
        str(render_kwargs.get("segments", 18)),
        str(render_kwargs.get("radius_scale", 1.0)),
        str(render_kwargs.get("radius_adaptive_alpha", 0.0)),
        str(render_kwargs.get("radius_ref_percentile", 50.0)),
        str(render_kwargs.get("projection", "ortho")),
        str(render_kwargs.get("fovy", 55.0)),
        str(render_kwargs.get("margin", 0.85)),
        str(render_kwargs.get("supersample_factor", 2)),
    ]
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_path(cache_dir: Path, cache_key: str) -> Path:
    """Get cache file path for a given cache key.
    
    Args:
        cache_dir: Cache directory
        cache_key: Cache key
    
    Returns:
        Path to cache file
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.pkl"


def _load_cache(cache_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
    """Load rendered output from cache.
    
    Args:
        cache_path: Path to cache file
    
    Returns:
        Tuple of (input_img, seg_mask, camera_info) or None if not found
    """
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return None
    return None


def _save_cache(cache_path: Path, input_img: np.ndarray, seg_mask: np.ndarray, camera_info: Dict) -> None:
    """Save rendered output to cache.
    
    Args:
        cache_path: Path to cache file
        input_img: Rendered input image
        seg_mask: Segmentation mask
        camera_info: Camera information
    """
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump((input_img, seg_mask, camera_info), f)
    except Exception as e:
        logger.warning(f"Failed to save cache to {cache_path}: {e}")


def _get_embedding_cache_key(batch_paths: List[Path], checkpoint_path: Optional[Path], reduce: str) -> str:
    """Generate cache key for embedding batch.
    
    Args:
        batch_paths: Sorted list of SWC file paths in the batch
        checkpoint_path: Path to model checkpoint (for model versioning)
        reduce: Embedding reduction method
    
    Returns:
        Cache key string
    """
    sorted_paths = sorted([str(p) for p in batch_paths])
    key_parts = [
        "|".join(sorted_paths),
        str(checkpoint_path) if checkpoint_path else "no_checkpoint",
        reduce,
    ]
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_embedding_cache_path(embedding_cache_dir: Path, cache_key: str) -> Path:
    """Get cache file path for embeddings.
    
    Args:
        embedding_cache_dir: Embedding cache directory
        cache_key: Cache key
    
    Returns:
        Path to cache file
    """
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    return embedding_cache_dir / f"{cache_key}.pkl"


def _load_embedding_cache(cache_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load embeddings from cache.
    
    Args:
        cache_path: Path to cache file
    
    Returns:
        Dict with 'raw' and 'reduced' embeddings or None if not found
    """
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load embedding cache from {cache_path}: {e}")
            return None
    return None


def _save_embedding_cache(cache_path: Path, embeddings_raw: Dict[str, np.ndarray], embeddings_reduced: Dict[str, np.ndarray]) -> None:
    """Save embeddings to cache.
    
    Args:
        cache_path: Path to cache file
        embeddings_raw: Raw embeddings dict (z, mu, logvar)
        embeddings_reduced: Reduced embeddings dict (z, mu, logvar)
    """
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump({"raw": embeddings_raw, "reduced": embeddings_reduced}, f)
    except Exception as e:
        logger.warning(f"Failed to save embedding cache to {cache_path}: {e}")


def _worker_render(args: Tuple[Path, int, int, Dict, Optional[Path]]) -> Tuple[Path, Optional[Tuple[np.ndarray, np.ndarray, Dict]], bool]:
    """Worker function for multiprocessing rendering.
    
    Args:
        args: Tuple of (swc_path, width, height, render_kwargs, cache_dir)
    
    Returns:
        Tuple of (swc_path, (input_img, seg_mask, camera_info), from_cache)
    """
    swc_path, width, height, render_kwargs, cache_dir = args
    
    if cache_dir is not None:
        cache_key = _get_cache_key(swc_path, width, height, **render_kwargs)
        cache_path = _get_cache_path(cache_dir, cache_key)
        
        cached = _load_cache(cache_path)
        if cached is not None:
            return (swc_path, cached, True)
    
    try:
        result = render_swc_to_input(swc_path, width=width, height=height, **render_kwargs)
        if cache_dir is not None:
            _save_cache(cache_path, *result)
        return (swc_path, result, False)
    except Exception as e:
        logger.error(f"Error rendering {swc_path.name} in worker: {e}", exc_info=True)
        return (swc_path, None, False)


def create_false_color_mask(mask: np.ndarray, class_colors: Dict[int, np.ndarray]) -> np.ndarray:
    """Create false color visualization of segmentation mask.
    
    Args:
        mask: (H, W) uint8 array with class IDs
        class_colors: Dict mapping class ID to RGB color [0-255]
    
    Returns:
        (H, W, 3) uint8 RGB image
    """
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        mask_bool = mask == class_id
        rgb[mask_bool] = color
    
    return rgb


def render_swc_to_input(
    swc_path: Path,
    *,
    width: int = 1024,
    height: int = 1024,
    segments: int = 18,
    radius_scale: float = 1.0,
    radius_adaptive_alpha: float = 0.0,
    radius_ref_percentile: float = 50.0,
    projection: str = "ortho",
    fovy: float = 55.0,
    margin: float = 0.85,
    depth_shading: bool = False,
    background: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    supersample_factor: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Render SWC file to input image (mask_bw) and segmentation mask.
    
    Returns:
        (input_image, seg_mask, camera_info)
        - input_image: (H, W) float32 [0-1] grayscale
        - seg_mask: (H, W) uint8 class IDs
        - camera_info: dict with camera parameters
    """
    logger.debug(f"Rendering {swc_path.name} to input (size: {width}x{height}, supersample: {supersample_factor})")
    with OffscreenContext(width, height, visible=False, samples=0) as ctx:
        core = NeuroRenderCore(ctx)
        core.load_swc(
            swc_path,
            segments=segments,
            radius_scale=radius_scale,
            radius_adaptive_alpha=radius_adaptive_alpha,
            radius_ref_percentile=radius_ref_percentile,
        )
        core.set_projection(perspective=(projection == "perspective"), fovy=fovy)
        core.depth_shading = depth_shading
        core.fit_camera(margin=margin)
        
        # Render input (mask_bw)
        input_img = core.render_depth(factor=supersample_factor)
        input_img = input_img.astype(np.float32)
        
        # Normalize to [0, 1]
        if input_img.max() > 0:
            input_img = input_img / input_img.max()
        
        # Render segmentation mask
        class_to_id = {
            NeuronClass.SOMA.name: 1,
            NeuronClass.AXON.name: 2,
            NeuronClass.BASAL_DENDRITE.name: 3,
            NeuronClass.APICAL_DENDRITE.name: 4,
            NeuronClass.OTHER.name: 5,
        }
        seg_mask = core.render_class_id_mask_supersampled(class_to_id, factor=supersample_factor)
        
        # Get camera info
        camera_info = {
            "eye": [float(x) for x in core.camera.eye],
            "target": [float(x) for x in core.camera.target],
            "up": [float(x) for x in core.camera.up],
            "fovy": float(core.camera.fovy),
            "ortho_scale": float(core.camera.ortho_scale),
            "near": float(core.camera.near),
            "far": float(core.camera.far),
            "perspective": core.camera.perspective,
        }
        
        return input_img, seg_mask, camera_info


def extract_embeddings(
    model: SegVAE2D,
    input_batch: torch.Tensor,
    device: str,
    reduce: str = "mean",
) -> Dict[str, np.ndarray]:
    """Extract embeddings from VAE model.
    
    Args:
        model: Trained SegVAE2D model
        input_batch: (B, 1, H, W) float32 tensor [0-1]
        device: Device string
        reduce: How to reduce spatial dimensions ("mean", "max", "flatten", or "none")
    
    Returns:
        Dict with 'z', 'mu', 'logvar' as numpy arrays
        If reduce is not "none", spatial dimensions are reduced to 1D vectors
    """
    batch_size = input_batch.shape[0]
    input_shape = input_batch.shape[2:]
    logger.debug(f"Extracting embeddings: batch_size={batch_size}, input_shape={input_shape}, reduce={reduce}, device={device}")
    model.eval()
    with torch.no_grad():
        input_batch = input_batch.to(device)
        logger.debug(f"Input tensor moved to {device}, shape: {input_batch.shape}")
        z, mu, logvar, e2, e1, e0 = model.encode(input_batch)
        logger.debug(f"Encoded shapes - z: {z.shape}, mu: {mu.shape}, logvar: {logvar.shape}")
        
        if reduce == "mean":
            z = z.mean(dim=(2, 3))
            mu = mu.mean(dim=(2, 3))
            logvar = logvar.mean(dim=(2, 3))
            logger.debug(f"Applied mean pooling: z={z.shape}, mu={mu.shape}, logvar={logvar.shape}")
        elif reduce == "max":
            z = z.max(dim=3)[0].max(dim=2)[0]
            mu = mu.max(dim=3)[0].max(dim=2)[0]
            logvar = logvar.max(dim=3)[0].max(dim=2)[0]
            logger.debug(f"Applied max pooling: z={z.shape}, mu={mu.shape}, logvar={logvar.shape}")
        elif reduce == "flatten":
            z = z.flatten(1)
            mu = mu.flatten(1)
            logvar = logvar.flatten(1)
            logger.debug(f"Applied flatten: z={z.shape}, mu={mu.shape}, logvar={logvar.shape}")
        else:
            logger.debug(f"No reduction applied: z={z.shape}, mu={mu.shape}, logvar={logvar.shape}")
        
        result = {
            "z": z.cpu().numpy(),
            "mu": mu.cpu().numpy(),
            "logvar": logvar.cpu().numpy(),
        }
        logger.debug(f"Returning embeddings: z shape={result['z'].shape}, mu shape={result['mu'].shape}")
        return result


def evaluate_batch(
    swc_paths: List[Path],
    model: SegVAE2D,
    device: str,
    *,
    width: int = 1024,
    height: int = 1024,
    batch_size: int = 8,
    save_visualizations: bool = False,
    output_dir: Optional[Path] = None,
    class_colors: Optional[Dict[int, np.ndarray]] = None,
    cache_dir: Optional[Path] = None,
    num_workers: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    embedding_reduce: str = "mean",
) -> List[Dict]:
    """Evaluate a batch of SWC files and extract embeddings.
    
    Args:
        swc_paths: List of SWC file paths
        model: Trained SegVAE2D model
        device: Device string
        width: Render width
        height: Render height
        batch_size: Batch size for processing
        save_visualizations: Whether to save false color mask visualizations
        output_dir: Directory to save outputs (required if save_visualizations=True)
        class_colors: Dict mapping class ID to RGB color [0-255]
        cache_dir: Directory for caching rendered outputs (optional)
        num_workers: Number of worker processes for rendering (default: half of CPU count)
        checkpoint_path: Path to model checkpoint (for embedding cache key)
        embedding_reduce: Embedding reduction method (for embedding cache key)
    
    Returns:
        List of annotation dicts, one per SWC file
    """
    logger.info(f"Evaluating batch: {len(swc_paths)} files, batch_size={batch_size}, render_size={width}x{height}")
    
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)
    logger.info(f"Using {num_workers} worker processes for rendering")
    
    if cache_dir is not None:
        logger.info(f"Using cache directory: {cache_dir}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        embedding_cache_dir = cache_dir / "embedding_cache"
        embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using embedding cache directory: {embedding_cache_dir}")
    else:
        embedding_cache_dir = None
    
    if class_colors is None:
        class_colors = {
            0: np.array([0, 0, 0], dtype=np.uint8),
            1: np.array([255, 0, 0], dtype=np.uint8),
            2: np.array([0, 255, 0], dtype=np.uint8),
            3: np.array([0, 0, 255], dtype=np.uint8),
            4: np.array([255, 0, 255], dtype=np.uint8),
            5: np.array([255, 165, 0], dtype=np.uint8),
        }
    
    if save_visualizations and output_dir is None:
        raise ValueError("output_dir must be provided if save_visualizations=True")
    
    if save_visualizations:
        output_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    annotations = []
    n_total = len(swc_paths)
    n_batches = (n_total + batch_size - 1) // batch_size
    
    render_kwargs = {
        "segments": 18,
        "radius_scale": 1.0,
        "radius_adaptive_alpha": 0.0,
        "radius_ref_percentile": 50.0,
        "projection": "ortho",
        "fovy": 55.0,
        "margin": 0.85,
        "supersample_factor": 2,
    }
    
    total_cached_count = 0
    total_rendered_count = 0
    
    for batch_idx, batch_start in enumerate(range(0, n_total, batch_size)):
        batch_paths = swc_paths[batch_start:batch_start + batch_size]
        
        logger.info(f"  Batch [{batch_idx + 1}/{n_batches}] Processing {len(batch_paths)} files...")
        
        batch_inputs = []
        batch_seg_masks = []
        batch_camera_infos = []
        batch_cached_count = 0
        batch_rendered_count = 0
        batch_failed = []
        
        render_tasks = [(swc_path, width, height, render_kwargs, cache_dir) for swc_path in batch_paths]
        
        logger.info(f"    Rendering batch with {num_workers} workers...")
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.map(_worker_render, render_tasks)
        
        for swc_path, result, from_cache in results:
            if result is None:
                logger.warning(f"      Failed to render {swc_path.name}")
                batch_failed.append(swc_path)
                continue
            
            input_img, seg_mask, camera_info = result
            batch_inputs.append(input_img)
            batch_seg_masks.append(seg_mask)
            batch_camera_infos.append(camera_info)
            
            if from_cache:
                batch_cached_count += 1
            else:
                batch_rendered_count += 1
        
        total_cached_count += batch_cached_count
        total_rendered_count += batch_rendered_count
        
        logger.info(f"    Batch rendering: {batch_cached_count} from cache, {batch_rendered_count} newly rendered, {len(batch_inputs)}/{len(batch_paths)} successful")
        
        if not batch_inputs:
            logger.warning(f"    Batch {batch_idx + 1} produced no valid inputs, skipping")
            continue
        
        successful_batch_paths = [p for p in batch_paths if p not in batch_failed]
        
        embeddings_raw = None
        embeddings_reduced = None
        from_embedding_cache = False
        
        if embedding_cache_dir is not None:
            embedding_cache_key = _get_embedding_cache_key(successful_batch_paths, checkpoint_path, embedding_reduce)
            embedding_cache_path = _get_embedding_cache_path(embedding_cache_dir, embedding_cache_key)
            
            cached_embeddings = _load_embedding_cache(embedding_cache_path)
            if cached_embeddings is not None:
                logger.info(f"    Loading embeddings from cache...")
                embeddings_raw = cached_embeddings["raw"]
                embeddings_reduced = cached_embeddings["reduced"]
                from_embedding_cache = True
                logger.debug(f"    Cached embeddings: raw z={embeddings_raw['z'].shape}, reduced z={embeddings_reduced['z'].shape}")
        
        if embeddings_raw is None or embeddings_reduced is None:
            logger.info(f"    Extracting embeddings from {len(batch_inputs)} samples...")
            input_tensor = torch.from_numpy(np.stack(batch_inputs)).unsqueeze(1).float()
            logger.debug(f"    Stacked input tensor shape: {input_tensor.shape}")
            
            embeddings_raw = extract_embeddings(model, input_tensor, device, reduce="none")
            logger.debug(f"    Raw embeddings: z={embeddings_raw['z'].shape}, mu={embeddings_raw['mu'].shape}")
            
            embeddings_reduced = extract_embeddings(model, input_tensor, device, reduce=embedding_reduce)
            logger.debug(f"    Reduced embeddings: z={embeddings_reduced['z'].shape}, mu={embeddings_reduced['mu'].shape}")
            
            if embedding_cache_dir is not None:
                logger.info(f"    Saving embeddings to cache...")
                _save_embedding_cache(embedding_cache_path, embeddings_raw, embeddings_reduced)
        
        if from_embedding_cache:
            logger.info(f"    Embeddings loaded from cache")
        
        for i, swc_path in enumerate(successful_batch_paths):
            if i >= len(batch_inputs):
                continue
            
            annotation = {
                "swc_path": str(swc_path),
                "swc_name": swc_path.name,
                "embedding_z_raw": embeddings_raw["z"][i].tolist(),
                "embedding_mu_raw": embeddings_raw["mu"][i].tolist(),
                "embedding_logvar_raw": embeddings_raw["logvar"][i].tolist(),
                "embedding_z": embeddings_reduced["z"][i].tolist(),
                "embedding_mu": embeddings_reduced["mu"][i].tolist(),
                "embedding_logvar": embeddings_reduced["logvar"][i].tolist(),
                "embedding_dim_raw": list(embeddings_raw["z"][i].shape),
                "embedding_dim": int(embeddings_reduced["z"][i].shape[0]),
                "camera": batch_camera_infos[i],
            }
            
            if save_visualizations:
                seg_mask = batch_seg_masks[i]
                false_color = create_false_color_mask(seg_mask, class_colors)
                vis_path = vis_dir / f"{swc_path.stem}_mask_falsecolor.png"
                imwrite(vis_path, false_color)
                annotation["visualization_path"] = str(vis_path.relative_to(output_dir))
            
            annotations.append(annotation)
    
    logger.info(f"  Completed: {len(annotations)}/{n_total} files processed successfully")
    logger.info(f"  Total rendering: {total_cached_count} from cache, {total_rendered_count} newly rendered")
    return annotations


def render_thumbnail(
    swc_path: Path,
    *,
    width: int = 64,
    height: int = 64,
    segments: int = 18,
    radius_scale: float = 1.0,
    radius_adaptive_alpha: float = 0.0,
    radius_ref_percentile: float = 50.0,
    projection: str = "ortho",
    fovy: float = 55.0,
    margin: float = 0.85,
) -> np.ndarray:
    """Render neuron thumbnail with transparent background.
    
    Returns:
        (H, W, 4) uint8 RGBA image with transparent background
    """
    logger.debug(f"Rendering thumbnail for {swc_path.name} (size: {width}x{height})")
    with OffscreenContext(width, height, visible=False, samples=0) as ctx:
        core = NeuroRenderCore(ctx)
        core.load_swc(
            swc_path,
            segments=segments,
            radius_scale=radius_scale,
            radius_adaptive_alpha=radius_adaptive_alpha,
            radius_ref_percentile=radius_ref_percentile,
        )
        core.set_projection(perspective=(projection == "perspective"), fovy=fovy)
        core.fit_camera(margin=margin)
        
        config = RenderConfig(
            mode=RenderMode.COLOR,
            background=(0.0, 0.0, 0.0, 0.0),
            disable_srgb=True,
            disable_blend=True,
            disable_cull=True,
        )
        rgba = core.render(config)
        
        return rgba


def create_sprite_image(thumbnails: List[np.ndarray], thumbnail_size: int = 64) -> np.ndarray:
    """Create sprite image from thumbnails.
    
    Args:
        thumbnails: List of (H, W, 4) uint8 RGBA images
        thumbnail_size: Size of each thumbnail (assumed square)
    
    Returns:
        (sprite_H, sprite_W, 4) uint8 RGBA sprite image
    """
    n = len(thumbnails)
    if n == 0:
        raise ValueError("No thumbnails provided")
    
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    sprite_width = cols * thumbnail_size
    sprite_height = rows * thumbnail_size
    
    logger.debug(f"Creating sprite: {n} thumbnails, grid={rows}x{cols}, size={sprite_width}x{sprite_height}")
    sprite = np.zeros((sprite_height, sprite_width, 4), dtype=np.uint8)
    
    for idx, thumb in enumerate(thumbnails):
        row = idx // cols
        col = idx % cols
        
        if thumb.shape[0] != thumbnail_size or thumb.shape[1] != thumbnail_size:
            thumb_pil = Image.fromarray(thumb, mode='RGBA')
            thumb_pil = thumb_pil.resize((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
            thumb = np.array(thumb_pil)
        
        y_start = row * thumbnail_size
        y_end = y_start + thumbnail_size
        x_start = col * thumbnail_size
        x_end = x_start + thumbnail_size
        
        sprite[y_start:y_end, x_start:x_end] = thumb
    
    return sprite


def load_metadata_json(metadata_path: Path) -> Dict[str, Dict]:
    """Load metadata JSON and index by filename.
    
    Returns:
        Dict mapping filename (e.g., "601506507_transformed.swc") to metadata dict
    """
    logger.debug(f"Loading metadata JSON from {metadata_path}")
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    logger.debug(f"Loaded {len(data)} entries from metadata JSON")
    metadata_dict = {}
    for entry in data:
        filename = entry.get("filename", "")
        metadata_dict[filename] = entry
    
    logger.debug(f"Indexed {len(metadata_dict)} entries by filename")
    return metadata_dict


def create_metadata_tsv(
    annotations: List[Dict],
    metadata_dict: Dict[str, Dict],
    output_path: Path,
    metadata_fields: Optional[List[str]] = None,
) -> None:
    """Create metadata TSV file for TensorBoard projector.
    
    Args:
        annotations: List of annotation dicts from evaluate_batch
        metadata_dict: Dict mapping filename to metadata from JSON
        output_path: Path to save metadata.tsv
        metadata_fields: List of field names to include (if None, includes common fields)
    """
    if metadata_fields is None:
        metadata_fields = [
            "swc_name",
            "n_nodes",
            "n_axons",
            "n_dendrites",
            "total_neurite_length",
            "mean_radius",
            "max_radius",
            "axon_dendrite_length_ratio",
            "max_branch_order",
            "total_volume",
            "surface_to_volume_ratio",
        ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        writer.writerow(metadata_fields)
        
        for ann in annotations:
            swc_name = ann["swc_name"]
            row = []
            
            for field in metadata_fields:
                if field == "swc_name":
                    row.append(swc_name)
                elif field in ann:
                    row.append(ann[field])
                elif swc_name in metadata_dict and field in metadata_dict[swc_name]:
                    row.append(metadata_dict[swc_name][field])
                else:
                    row.append("")
            
            writer.writerow(row)


def create_tensor_tsv(embeddings: np.ndarray, output_path: Path) -> None:
    """Create tensor TSV file for TensorBoard projector.
    
    Args:
        embeddings: (N, D) numpy array of embeddings
        output_path: Path to save tensors.tsv
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for embedding in embeddings:
            writer.writerow(embedding.tolist())


def create_projector_config(
    embeddings: np.ndarray,
    tensor_path: str,
    metadata_path: str,
    sprite_path: str,
    sprite_size: int,
    tensor_name: str = "Neuron Embeddings",
    output_path: Path = None,
) -> None:
    """Create TensorBoard projector config JSON.
    
    Args:
        embeddings: (N, D) numpy array of embeddings
        tensor_path: URL or path to tensors.tsv
        metadata_path: URL or path to metadata.tsv
        sprite_path: URL or path to sprite.png
        sprite_size: Size of each thumbnail in sprite (e.g., 64)
        tensor_name: Name for the tensor in TensorBoard
        output_path: Path to save projector_config.json
    """
    config = {
        "embeddings": [
            {
                "tensorName": tensor_name,
                "tensorShape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
                "tensorPath": tensor_path,
                "metadataPath": metadata_path,
                "sprite": {
                    "imagePath": sprite_path,
                    "singleImageDim": [sprite_size, sprite_size],
                }
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SWC files with VAE and extract embeddings")
    parser.add_argument("--swc-dir", type=Path, required=True, help="Directory containing SWC files")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for annotations")
    
    parser.add_argument("--pattern", type=str, default="*.swc", help="Glob pattern for SWC files")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--width", type=int, default=1024, help="Render width")
    parser.add_argument("--height", type=int, default=1024, help="Render height")
    parser.add_argument("--device", type=str, default=None, help="Device (mps/cuda/cpu, auto-detect if not set)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory for caching rendered outputs")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes for rendering (default: half of CPU count)")
    
    parser.add_argument("--save-visualizations", action="store_true", help="Save false color mask visualizations")
    
    parser.add_argument("--in-channels", type=int, default=1, help="Model input channels")
    parser.add_argument("--base-channels", type=int, default=64, help="Model base channels")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of segmentation classes")
    parser.add_argument("--latent-channels", type=int, default=128, help="Latent dimension")
    parser.add_argument("--use-depth", action="store_true", help="Model uses depth head")
    parser.add_argument("--use-recon", action="store_true", help="Model uses reconstruction head")
    parser.add_argument("--kld-weight", type=float, default=1.0, help="KLD weight")
    
    parser.add_argument("--projector", action="store_true", help="Generate TensorBoard projector dataset")
    parser.add_argument("--metadata-json", type=Path, help="Path to morph_transformed-cats.json for metadata")
    parser.add_argument("--thumbnail-size", type=int, default=64, help="Size of thumbnails in sprite")
    parser.add_argument("--embedding-reduce", choices=["mean", "max", "flatten", "none"], default="mean", help="How to reduce spatial embeddings")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info("="*60)
    logger.info("VAE Evaluator Starting")
    logger.info("="*60)
    logger.info(f"Arguments: swc_dir={args.swc_dir}, checkpoint={args.checkpoint}, output={args.output}")
    logger.info(f"Mode: {'PROJECTOR' if args.projector else 'STANDARD'}")
    
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    logger.info(f"Searching for SWC files in {args.swc_dir} with pattern '{args.pattern}'")
    swc_paths = sorted(args.swc_dir.glob(args.pattern))
    if not swc_paths:
        logger.error(f"No SWC files found in {args.swc_dir} matching pattern {args.pattern}")
        return
    
    logger.info(f"Found {len(swc_paths)} SWC files")
    if len(swc_paths) <= 10:
        logger.debug(f"SWC files: {[p.name for p in swc_paths]}")
    else:
        logger.debug(f"First 5 SWC files: {[p.name for p in swc_paths[:5]]}...")
    
    model = load_model(
        args.checkpoint,
        device,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        latent_channels=args.latent_channels,
        use_depth=args.use_depth,
        use_recon=args.use_recon,
        kld_weight=args.kld_weight,
    )
    logger.info(f"Model loaded. Latent dimension: {args.latent_channels}")
    
    logger.info(f"Creating output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory ready: {args.output}")
    
    if args.projector:
        logger.info("PROJECTOR MODE: Generating TensorBoard projector dataset")
        metadata_dict = {}
        if args.metadata_json and args.metadata_json.exists():
            logger.info(f"Loading metadata from {args.metadata_json}")
            metadata_dict = load_metadata_json(args.metadata_json)
            logger.info(f"Loaded metadata for {len(metadata_dict)} entries")
        else:
            logger.warning("No metadata JSON provided, metadata TSV will have limited fields")
        
        cache_dir = args.cache_dir if args.cache_dir else args.output / "render_cache"
        logger.info(f"Processing {len(swc_paths)} SWC files for projector...")
        annotations = evaluate_batch(
            swc_paths,
            model,
            device,
            width=args.width,
            height=args.height,
            batch_size=args.batch_size,
            save_visualizations=False,
            output_dir=args.output,
            cache_dir=cache_dir,
            num_workers=args.num_workers,
            checkpoint_path=args.checkpoint,
            embedding_reduce=args.embedding_reduce,
        )
        
        if not annotations:
            logger.error("No annotations generated. Cannot create projector dataset.")
            return
        
        successful_swc_paths = [Path(ann["swc_path"]) for ann in annotations]
        n_total = len(successful_swc_paths)
        logger.info(f"Successfully processed {n_total} files for projector dataset")
        
        logger.info(f"Extracting embeddings from annotations (reduce={args.embedding_reduce})...")
        embeddings_reduced = np.array([ann["embedding_mu"] for ann in annotations])
        logger.info(f"  Extracted embeddings shape: {embeddings_reduced.shape}")
        
        logger.info(f"Rendering {n_total} thumbnails (size: {args.thumbnail_size}x{args.thumbnail_size})...")
        thumbnails = []
        for idx, swc_path in enumerate(successful_swc_paths):
            if (idx + 1) % 10 == 0 or idx == 0 or idx == n_total - 1:
                logger.info(f"  [{idx + 1}/{n_total}] {swc_path.name}")
            thumb = render_thumbnail(swc_path, width=args.thumbnail_size, height=args.thumbnail_size)
            thumbnails.append(thumb)
        
        logger.info("Creating sprite image...")
        sprite = create_sprite_image(thumbnails, thumbnail_size=args.thumbnail_size)
        logger.debug(f"Sprite shape: {sprite.shape}")
        sprite_path = args.output / "sprite.png"
        imwrite(sprite_path, sprite)
        logger.info(f"Saved sprite to {sprite_path} ({sprite_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        logger.info("Writing tensor TSV file...")
        tensor_path = args.output / "tensors.tsv"
        create_tensor_tsv(embeddings_reduced, tensor_path)
        logger.info(f"Saved tensors to {tensor_path} ({tensor_path.stat().st_size / 1024 / 1024:.2f} MB)")
        
        logger.info("Writing metadata TSV file...")
        metadata_path = args.output / "metadata.tsv"
        create_metadata_tsv(annotations, metadata_dict, metadata_path)
        logger.info(f"Saved metadata to {metadata_path} ({metadata_path.stat().st_size / 1024:.2f} KB)")
        
        logger.info("Writing projector config JSON...")
        config_path = args.output / "projector_config.json"
        create_projector_config(
            embeddings_reduced,
            tensor_path="tensors.tsv",
            metadata_path="metadata.tsv",
            sprite_path="sprite.png",
            sprite_size=args.thumbnail_size,
            tensor_name="Neuron Embeddings",
            output_path=config_path,
        )
        logger.info(f"Saved projector config to {config_path}")
        
        logger.info("="*60)
        logger.info("Projector dataset ready!")
        logger.info(f"  - Processed {len(annotations)} neurons")
        logger.info(f"  - Embedding shape: {embeddings_reduced.shape} (N={embeddings_reduced.shape[0]}, D={embeddings_reduced.shape[1]})")
        logger.info(f"  - Sprite: {sprite_path.name} ({len(thumbnails)} thumbnails)")
        logger.info(f"  - Files generated:")
        logger.info(f"    * {tensor_path.name}")
        logger.info(f"    * {metadata_path.name}")
        logger.info(f"    * {sprite_path.name}")
        logger.info(f"    * {config_path.name}")
        logger.info(f"\nTo use with TensorBoard, host these files and update paths in {config_path.name}")
        logger.info("="*60)
    else:
        cache_dir = args.cache_dir if args.cache_dir else args.output / "render_cache"
        logger.info(f"Processing {len(swc_paths)} SWC files in standard mode...")
        annotations = evaluate_batch(
            swc_paths,
            model,
            device,
            width=args.width,
            height=args.height,
            batch_size=args.batch_size,
            save_visualizations=args.save_visualizations,
            output_dir=args.output,
            cache_dir=cache_dir,
            num_workers=args.num_workers,
            checkpoint_path=args.checkpoint,
            embedding_reduce=args.embedding_reduce,
        )
        
        logger.info("Writing annotations JSON...")
        annotations_path = args.output / "annotations.json"
        with open(annotations_path, "w") as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Saved {len(annotations)} annotations to {annotations_path} ({annotations_path.stat().st_size / 1024:.2f} KB)")
        
        if args.save_visualizations:
            logger.info(f"Visualizations saved to {args.output / 'visualizations'}")


if __name__ == "__main__":
    main()

