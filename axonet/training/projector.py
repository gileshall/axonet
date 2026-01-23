"""Projector module for generating TensorBoard projector datasets from manifest.jsonl.

Operates on existing rendered images from manifest.jsonl (like trainer.py),
extracts embeddings from VAE model with flexible options for variational skip connections,
and generates projector files for visualization.

Key features:
- Works from manifest.jsonl (no new rendering)
- Flexible embedding construction (bottleneck + skip connections)
- Options for pose sampling (single random, all poses, or aggregated)
- Optional metadata integration
- Optional sprite generation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import torch
from imageio.v2 import imread, imwrite
from PIL import Image

from ..models.d3_swc_vae import SegVAE2D, load_model
from .trainer import NeuronDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def extract_embeddings_flexible(
    model: SegVAE2D,
    input_batch: torch.Tensor,
    device: str,
    *,
    use_bottleneck: bool = True,
    use_skips: Literal["none", "all", "only"] = "all",
    use_mu: bool = True,
    reduce: str = "mean",
) -> Dict[str, np.ndarray]:
    """Extract embeddings with flexible options for variational components.
    
    Args:
        model: Trained SegVAE2D model
        input_batch: (B, 1, H, W) float32 tensor [0-1]
        device: Device string
        use_bottleneck: Include bottleneck mu/logvar/z
        use_skips: "none" (ignore skips), "all" (bottleneck + skips), "only" (only skips)
        use_mu: Use mu (deterministic) instead of z (stochastic) for embeddings
        reduce: How to reduce spatial dimensions ("mean", "max", "flatten", or "none")
    
    Returns:
        Dict with embedding arrays (keys depend on options)
    """
    batch_size = input_batch.shape[0]
    input_shape = input_batch.shape[2:]
    logger.debug(f"Extracting embeddings: batch_size={batch_size}, input_shape={input_shape}, "
                 f"use_bottleneck={use_bottleneck}, use_skips={use_skips}, use_mu={use_mu}, reduce={reduce}")
    
    model.eval()
    with torch.no_grad():
        input_batch = input_batch.to(device)
        
        z, mu, logvar, e2, e1, e0 = model.encode(input_batch)
        shared, slog = model.decode_shared(z, e2, e1, e0)
        
        embeddings = {}
        
        if use_bottleneck and use_skips != "only":
            if use_mu:
                emb = mu
            else:
                emb = z
            embeddings["bottleneck"] = emb
        
        if use_skips in ["all", "only"]:
            skip_embs = []
            skip_names = []
            
            if use_mu:
                if "mu2" in slog:
                    skip_embs.append(slog["mu2"])
                    skip_names.append("skip2")
                if "mu1" in slog:
                    skip_embs.append(slog["mu1"])
                    skip_names.append("skip1")
                if "mu0" in slog:
                    skip_embs.append(slog["mu0"])
                    skip_names.append("skip0")
            else:
                if "logvar2" in slog:
                    mu2 = slog["mu2"]
                    logvar2 = slog["logvar2"]
                    z2 = model.reparameterize(mu2, logvar2)
                    skip_embs.append(z2)
                    skip_names.append("skip2")
                if "logvar1" in slog:
                    mu1 = slog["mu1"]
                    logvar1 = slog["logvar1"]
                    z1 = model.reparameterize(mu1, logvar1)
                    skip_embs.append(z1)
                    skip_names.append("skip1")
                if "logvar0" in slog:
                    mu0 = slog["mu0"]
                    logvar0 = slog["logvar0"]
                    z0 = model.reparameterize(mu0, logvar0)
                    skip_embs.append(z0)
                    skip_names.append("skip0")
            
            if skip_embs:
                if len(skip_embs) == 1:
                    embeddings["skip"] = skip_embs[0]
                else:
                    for name, emb in zip(skip_names, skip_embs):
                        embeddings[name] = emb
        
        if not embeddings:
            raise ValueError("No embeddings selected: use_bottleneck=False and use_skips='none'")
        
        result = {}
        for key, emb in embeddings.items():
            if reduce == "mean":
                reduced = emb.mean(dim=(2, 3))
            elif reduce == "max":
                reduced = emb.max(dim=3)[0].max(dim=2)[0]
            elif reduce == "flatten":
                reduced = emb.flatten(1)
            elif reduce == "none":
                reduced = emb
            else:
                raise ValueError(f"Unknown reduce method: {reduce}")
            
            result[key] = reduced.cpu().numpy()
        
        if len(result) == 1:
            result["embedding"] = list(result.values())[0]
        else:
            combined = np.concatenate([v for v in result.values()], axis=1)
            result["embedding"] = combined
        
        logger.debug(f"Extracted embeddings: {list(result.keys())}, final shape={result['embedding'].shape}")
        return result


def load_manifest(manifest_path: Path) -> List[Dict]:
    """Load manifest.jsonl entries."""
    logger.info(f"Loading manifest from {manifest_path}")
    entries = []
    with open(manifest_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            entries.append(entry)
    logger.info(f"Loaded {len(entries)} manifest entries")
    return entries


def group_manifest_by_swc(entries: List[Dict]) -> Tuple[Dict[str, List[int]], Dict[int, Dict]]:
    """Group manifest entries by SWC filename using indices.
    
    Returns:
        (grouped_dict, idx_to_entry) where:
        - grouped_dict maps swc_name to list of entry indices
        - idx_to_entry maps index to entry dict
    """
    grouped = {}
    idx_to_entry = {}
    for idx, entry in enumerate(entries):
        swc_name = entry.get("swc", "")
        if swc_name:
            if swc_name not in grouped:
                grouped[swc_name] = []
            grouped[swc_name].append(idx)
            idx_to_entry[idx] = entry
    logger.info(f"Grouped into {len(grouped)} unique SWC files")
    return grouped, idx_to_entry


def sample_pose_indices(
    n_entries: int,
    mode: Literal["random", "all", "first"],
    seed: Optional[int] = None,
) -> List[int]:
    """Sample pose indices for a single SWC file.
    
    Args:
        n_entries: Number of entries available
        mode: "random" (single random pose), "all" (all poses), "first" (first pose)
        seed: Random seed for "random" mode
    
    Returns:
        List of selected indices
    """
    if mode == "random":
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        return [rng.randint(0, n_entries - 1)]
    elif mode == "all":
        return list(range(n_entries))
    elif mode == "first":
        return [0]
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")


def aggregate_embeddings(
    embeddings_list: List[np.ndarray],
    method: Literal["mean", "max", "concat"] = "mean",
) -> np.ndarray:
    """Aggregate embeddings from multiple poses.
    
    Args:
        embeddings_list: List of (D,) embedding vectors
        method: "mean" (average), "max" (element-wise max), "concat" (concatenate)
    
    Returns:
        Aggregated embedding vector
    """
    if method == "mean":
        return np.mean(embeddings_list, axis=0)
    elif method == "max":
        return np.max(embeddings_list, axis=0)
    elif method == "concat":
        return np.concatenate(embeddings_list, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def load_metadata_csv(csv_path: Path) -> Dict[str, Dict]:
    """Load metadata CSV and index by cell_specimen_id."""
    logger.info(f"Loading metadata CSV from {csv_path}")
    metadata_dict = {}
    
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cell_specimen_id = row.get("cell_specimen_id", "")
            if cell_specimen_id:
                metadata_dict[cell_specimen_id] = row
    
    logger.info(f"Loaded metadata for {len(metadata_dict)} entries")
    return metadata_dict


def load_metadata_json(json_path: Path) -> Dict[str, Dict]:
    """Load metadata JSON and index by filename."""
    logger.info(f"Loading metadata JSON from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    metadata_dict = {}
    for entry in data:
        filename = entry.get("filename", "")
        if filename:
            metadata_dict[filename] = entry
    
    logger.info(f"Loaded metadata for {len(metadata_dict)} entries")
    return metadata_dict


def extract_cell_specimen_id(swc_name: str) -> Optional[str]:
    """Extract cell_specimen_id from SWC filename."""
    if not swc_name:
        return None
    parts = swc_name.split("_")
    if parts:
        return parts[0]
    return None


def create_metadata_tsv(
    annotations: List[Dict],
    metadata_csv_dict: Dict[str, Dict],
    metadata_json_dict: Dict[str, Dict],
    output_path: Path,
    metadata_fields: Optional[List[str]] = None,
) -> None:
    """Create metadata TSV file for TensorBoard projector."""
    if metadata_fields is None:
        metadata_fields = [
            "swc_name",
            "cell_specimen_id",
            "cell_specimen_name",
            "hemisphere",
            "structure",
            "donor_id",
            "donor_name",
            "biological_sex",
            "age",
            "full_genotype",
            "dendrite_type",
            "apical_dendrite_status",
            "neuron_reconstruction_type",
            "cell_soma_normalized_depth",
            "depth_from_pia_um",
            "T-type Label",
            "MET-type Label",
        ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(metadata_fields)
        
        for ann in annotations:
            swc_name = ann["swc_name"]
            cell_specimen_id = extract_cell_specimen_id(swc_name)
            row = []
            
            for field in metadata_fields:
                if field == "swc_name":
                    row.append(swc_name)
                elif field == "cell_specimen_id":
                    row.append(cell_specimen_id if cell_specimen_id else "")
                elif field in ann:
                    row.append(ann[field])
                elif cell_specimen_id and cell_specimen_id in metadata_csv_dict and field in metadata_csv_dict[cell_specimen_id]:
                    row.append(metadata_csv_dict[cell_specimen_id][field])
                elif swc_name in metadata_json_dict and field in metadata_json_dict[swc_name]:
                    row.append(metadata_json_dict[swc_name][field])
                else:
                    row.append("")
            
            writer.writerow(row)


def create_tensor_tsv(embeddings: np.ndarray, output_path: Path) -> None:
    """Create tensor TSV file for TensorBoard projector."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for embedding in embeddings:
            writer.writerow(embedding.tolist())


def create_projector_config(
    embeddings: np.ndarray,
    tensor_path: str,
    metadata_path: str,
    sprite_path: Optional[str],
    sprite_size: int,
    tensor_name: str = "Neuron Embeddings",
    output_path: Path = None,
) -> None:
    """Create TensorBoard projector config JSON."""
    config = {
        "embeddings": [
            {
                "tensorName": tensor_name,
                "tensorShape": [int(embeddings.shape[0]), int(embeddings.shape[1])],
                "tensorPath": tensor_path,
                "metadataPath": metadata_path,
            }
        ]
    }
    
    if sprite_path:
        config["embeddings"][0]["sprite"] = {
            "imagePath": sprite_path,
            "singleImageDim": [sprite_size, sprite_size],
        }
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def create_sprite_image(thumbnails: List[np.ndarray], thumbnail_size: int = 64) -> np.ndarray:
    """Create sprite image from thumbnails."""
    n = len(thumbnails)
    if n == 0:
        raise ValueError("No thumbnails provided")
    
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    sprite_width = cols * thumbnail_size
    sprite_height = rows * thumbnail_size
    
    sprite = np.zeros((sprite_height, sprite_width, 4), dtype=np.uint8)
    
    for idx, thumb in enumerate(thumbnails):
        row = idx // cols
        col = idx % cols
        
        if thumb.shape[0] != thumbnail_size or thumb.shape[1] != thumbnail_size:
            if thumb.shape[2] == 3:
                thumb_pil = Image.fromarray(thumb, mode='RGB')
            else:
                thumb_pil = Image.fromarray(thumb, mode='RGBA')
            thumb_pil = thumb_pil.resize((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
            thumb = np.array(thumb_pil)
            if thumb.shape[2] == 3:
                thumb_rgba = np.zeros((thumbnail_size, thumbnail_size, 4), dtype=np.uint8)
                thumb_rgba[:, :, :3] = thumb
                thumb_rgba[:, :, 3] = 255
                thumb = thumb_rgba
        
        y_start = row * thumbnail_size
        y_end = y_start + thumbnail_size
        x_start = col * thumbnail_size
        x_end = x_start + thumbnail_size
        
        sprite[y_start:y_end, x_start:x_end] = thumb
    
    return sprite


def process_manifest_for_projector(
    manifest_path: Path,
    data_dir: Path,
    model: SegVAE2D,
    device: str,
    *,
    batch_size: int = 8,
    pose_sampling: Literal["random", "all", "first"] = "random",
    pose_aggregation: Literal["mean", "max", "concat"] = "mean",
    use_bottleneck: bool = True,
    use_skips: Literal["none", "all", "only"] = "all",
    use_mu: bool = True,
    reduce: str = "mean",
    seed: Optional[int] = None,
) -> Tuple[List[Dict], np.ndarray]:
    """Process manifest.jsonl and extract embeddings.
    
    Returns:
        (annotations, embeddings) where annotations is list of dicts and embeddings is (N, D) array
    """
    logger.info(f"Processing manifest: {manifest_path}")
    
    entries = load_manifest(manifest_path)
    grouped, idx_to_entry = group_manifest_by_swc(entries)
    
    dataset = NeuronDataset(manifest_path, data_dir)
    
    annotations = []
    all_embeddings = []
    
    swc_names = sorted(grouped.keys())
    n_total = len(swc_names)
    
    logger.info(f"Processing {n_total} unique SWC files with pose_sampling={pose_sampling}")
    
    for swc_idx, swc_name in enumerate(swc_names):
        if (swc_idx + 1) % 10 == 0 or swc_idx == 0 or swc_idx == n_total - 1:
            logger.info(f"  [{swc_idx + 1}/{n_total}] {swc_name}")
        
        swc_entry_indices = grouped[swc_name]
        selected_local_indices = sample_pose_indices(len(swc_entry_indices), mode=pose_sampling, seed=seed)
        selected_indices = [swc_entry_indices[i] for i in selected_local_indices]
        
        if pose_sampling == "all" and len(selected_indices) > 1:
            embeddings_list = []
            entry_info_list = []
            for entry_idx in selected_indices:
                sample = dataset[entry_idx]
                input_tensor = sample["input"].unsqueeze(0).to(device)
                
                emb_dict = extract_embeddings_flexible(
                    model, input_tensor, device,
                    use_bottleneck=use_bottleneck,
                    use_skips=use_skips,
                    use_mu=use_mu,
                    reduce=reduce,
                )
                embeddings_list.append(emb_dict["embedding"][0])
                entry_info_list.append(idx_to_entry[entry_idx])
            
            if pose_aggregation == "none":
                for pose_idx, (emb, entry_info) in enumerate(zip(embeddings_list, entry_info_list)):
                    all_embeddings.append(emb)
                    annotation = {
                        "swc_name": swc_name,
                        "pose_idx": entry_info.get("idx", pose_idx),
                        "n_poses": len(selected_indices),
                        "pose_sampling": pose_sampling,
                        "pose_aggregation": pose_aggregation,
                    }
                    annotations.append(annotation)
            else:
                if pose_aggregation == "mean":
                    aggregated = np.mean(embeddings_list, axis=0)
                elif pose_aggregation == "max":
                    aggregated = np.max(embeddings_list, axis=0)
                elif pose_aggregation == "concat":
                    aggregated = np.concatenate(embeddings_list, axis=0)
                else:
                    raise ValueError(f"Unknown pose aggregation method: {pose_aggregation}")
                all_embeddings.append(aggregated)
                
                annotation = {
                    "swc_name": swc_name,
                    "n_poses": len(selected_indices),
                    "pose_sampling": pose_sampling,
                    "pose_aggregation": pose_aggregation,
                }
                annotations.append(annotation)
        else:
            entry_idx = selected_indices[0]
            sample = dataset[entry_idx]
            input_tensor = sample["input"].unsqueeze(0).to(device)
            
            emb_dict = extract_embeddings_flexible(
                model, input_tensor, device,
                use_bottleneck=use_bottleneck,
                use_skips=use_skips,
                use_mu=use_mu,
                reduce=reduce,
            )
            all_embeddings.append(emb_dict["embedding"][0])
            
            annotation = {
                "swc_name": swc_name,
                "n_poses": 1,
                "pose_sampling": pose_sampling,
            }
            annotations.append(annotation)
    
    embeddings_array = np.array(all_embeddings)
    logger.info(f"Extracted embeddings shape: {embeddings_array.shape}")
    
    return annotations, embeddings_array


def create_sprite_from_manifest(
    annotations: List[Dict],
    manifest_path: Path,
    data_dir: Path,
    thumbnail_size: int = 64,
) -> np.ndarray:
    """Create sprite from mask_color images in manifest."""
    logger.info(f"Creating sprite from manifest entries...")
    
    entries = load_manifest(manifest_path)
    grouped, idx_to_entry = group_manifest_by_swc(entries)
    
    thumbnails = []
    for ann in annotations:
        swc_name = ann["swc_name"]
        if swc_name not in grouped:
            thumbnails.append(np.zeros((thumbnail_size, thumbnail_size, 4), dtype=np.uint8))
            continue
        
        swc_entry_indices = grouped[swc_name]
        
        if "pose_idx" in ann:
            pose_idx = ann["pose_idx"]
            matching_entry = None
            for entry_idx in swc_entry_indices:
                entry = idx_to_entry[entry_idx]
                if entry.get("idx") == pose_idx:
                    matching_entry = entry
                    break
            if matching_entry:
                mask_color_path = matching_entry.get("mask_color", "")
            else:
                random_idx = random.choice(swc_entry_indices)
                random_entry = idx_to_entry[random_idx]
                mask_color_path = random_entry.get("mask_color", "")
        else:
            random_idx = random.choice(swc_entry_indices)
            random_entry = idx_to_entry[random_idx]
            mask_color_path = random_entry.get("mask_color", "")
        
        if not mask_color_path:
            thumbnails.append(np.zeros((thumbnail_size, thumbnail_size, 4), dtype=np.uint8))
            continue
        
        image_path = data_dir / mask_color_path
        if not image_path.exists():
            thumbnails.append(np.zeros((thumbnail_size, thumbnail_size, 4), dtype=np.uint8))
            continue
        
        try:
            img = imread(image_path)
            if img.shape[2] == 3:
                img_rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
                img_rgba[:, :, :3] = img
                img_rgba[:, :, 3] = 255
                img = img_rgba
            thumbnails.append(img)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            thumbnails.append(np.zeros((thumbnail_size, thumbnail_size, 4), dtype=np.uint8))
    
    sprite = create_sprite_image(thumbnails, thumbnail_size=thumbnail_size)
    return sprite


def main():
    parser = argparse.ArgumentParser(description="Generate TensorBoard projector dataset from manifest.jsonl")
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest JSONL file")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset directory (root for manifest paths)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for projector files")
    
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--device", type=str, default=None, help="Device (mps/cuda/cpu, auto-detect if not set)")
    
    parser.add_argument("--pose-sampling", choices=["random", "all", "first"], default="random",
                       help="How to sample poses: random (single random), all (all poses), first (first pose)")
    parser.add_argument("--pose-aggregation", choices=["mean", "max", "concat", "none"], default="none",
                       help="How to aggregate embeddings when pose_sampling=all")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for pose sampling)")
    
    parser.add_argument("--use-bottleneck", action="store_true", default=True,
                       help="Include bottleneck embeddings (default: True)")
    parser.add_argument("--no-bottleneck", dest="use_bottleneck", action="store_false",
                       help="Exclude bottleneck embeddings")
    parser.add_argument("--use-skips", choices=["none", "all", "only"], default="all",
                       help="Skip connection usage: none (ignore), all (bottleneck+skips), only (only skips)")
    parser.add_argument("--use-mu", action="store_true", default=True,
                       help="Use mu (deterministic) instead of z (stochastic) for embeddings (default: True)")
    parser.add_argument("--use-z", dest="use_mu", action="store_false",
                       help="Use z (stochastic) instead of mu (deterministic)")
    parser.add_argument("--reduce", choices=["mean", "max", "flatten", "none"], default="mean",
                       help="How to reduce spatial dimensions")
    
    parser.add_argument("--metadata-csv", type=Path, default=None,
                       help="Path to PatchSeq_metadata.csv (optional)")
    parser.add_argument("--metadata-json", type=Path, default=None,
                       help="Path to morph_transformed-cats.json (optional)")
    
    parser.add_argument("--sprite", action="store_true", help="Generate sprite image (default: False)")
    parser.add_argument("--thumbnail-size", type=int, default=64, help="Size of thumbnails in sprite")
    
    parser.add_argument("--in-channels", type=int, default=1, help="Model input channels")
    parser.add_argument("--base-channels", type=int, default=64, help="Model base channels")
    parser.add_argument("--num-classes", type=int, default=4, help="Number of segmentation classes")
    parser.add_argument("--latent-channels", type=int, default=128, help="Latent dimension")
    parser.add_argument("--use-depth", action="store_true", help="Model uses depth head")
    parser.add_argument("--use-recon", action="store_true", help="Model uses reconstruction head")
    parser.add_argument("--kld-weight", type=float, default=1.0, help="KLD weight")
    
    parser.add_argument("--embedding-only", action="store_true", default=True,
                       help="Skip loading decoder head weights for optimization (default: True)")
    parser.add_argument("--no-embedding-only", dest="embedding_only", action="store_false",
                       help="Load all model weights including decoder heads")
    
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("="*60)
    logger.info("Projector Generator from Manifest")
    logger.info("="*60)
    
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
    
    model = load_model(
        args.checkpoint,
        device,
        embedding_only=args.embedding_only,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        num_classes=args.num_classes,
        latent_channels=args.latent_channels,
        use_depth=args.use_depth,
        use_recon=args.use_recon,
        kld_weight=args.kld_weight,
    )
    
    logger.info(f"Creating output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger.info("Processing manifest and extracting embeddings...")
    annotations, embeddings = process_manifest_for_projector(
        args.manifest,
        args.data_dir,
        model,
        device,
        batch_size=args.batch_size,
        pose_sampling=args.pose_sampling,
        pose_aggregation=args.pose_aggregation,
        use_bottleneck=args.use_bottleneck,
        use_skips=args.use_skips,
        use_mu=args.use_mu,
        reduce=args.reduce,
        seed=args.seed,
    )
    
    logger.info("Writing tensor TSV file...")
    tensor_path = args.output / "tensors.tsv"
    create_tensor_tsv(embeddings, tensor_path)
    logger.info(f"Saved tensors to {tensor_path} ({tensor_path.stat().st_size / 1024 / 1024:.2f} MB)")
    
    metadata_csv_dict = {}
    if args.metadata_csv and args.metadata_csv.exists():
        logger.info(f"Loading metadata CSV from {args.metadata_csv}")
        metadata_csv_dict = load_metadata_csv(args.metadata_csv)
    
    metadata_json_dict = {}
    if args.metadata_json and args.metadata_json.exists():
        logger.info(f"Loading metadata JSON from {args.metadata_json}")
        metadata_json_dict = load_metadata_json(args.metadata_json)
    
    logger.info("Writing metadata TSV file...")
    metadata_path = args.output / "metadata.tsv"
    create_metadata_tsv(annotations, metadata_csv_dict, metadata_json_dict, metadata_path)
    logger.info(f"Saved metadata to {metadata_path} ({metadata_path.stat().st_size / 1024:.2f} KB)")
    
    sprite_path = None
    if args.sprite:
        logger.info("Creating sprite image...")
        sprite = create_sprite_from_manifest(annotations, args.manifest, args.data_dir, args.thumbnail_size)
        sprite_path_file = args.output / "sprite.png"
        imwrite(sprite_path_file, sprite)
        sprite_path = "sprite.png"
        logger.info(f"Saved sprite to {sprite_path_file} ({sprite_path_file.stat().st_size / 1024 / 1024:.2f} MB)")
    
    logger.info("Writing projector config JSON...")
    config_path = args.output / "projector_config.json"
    create_projector_config(
        embeddings,
        tensor_path="tensors.tsv",
        metadata_path="metadata.tsv",
        sprite_path=sprite_path,
        sprite_size=args.thumbnail_size,
        tensor_name="Neuron Embeddings",
        output_path=config_path,
    )
    logger.info(f"Saved projector config to {config_path}")
    
    logger.info("="*60)
    logger.info("Projector dataset ready!")
    logger.info(f"  - Processed {len(annotations)} neurons")
    logger.info(f"  - Embedding shape: {embeddings.shape} (N={embeddings.shape[0]}, D={embeddings.shape[1]})")
    if args.sprite:
        logger.info(f"  - Sprite: sprite.png")
    logger.info(f"  - Files generated:")
    logger.info(f"    * {tensor_path.name}")
    logger.info(f"    * {metadata_path.name}")
    logger.info(f"    * {config_path.name}")
    if args.sprite:
        logger.info(f"    * sprite.png")
    logger.info("="*60)


if __name__ == "__main__":
    main()

