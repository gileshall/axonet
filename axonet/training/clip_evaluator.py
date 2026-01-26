"""Evaluation script for trained CLIP models.

Handles the fact that multiple poses (images) represent the same neuron
by aggregating per-image embeddings into per-neuron embeddings before
evaluation. Supports mean pooling and max-similarity strategies.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.metadata_adapters import get_adapter
from .clip_dataset import NeuronCLIPDataset, NeuronTextGenerator
from .clip_trainer import CLIPLightning


# ---------------------------------------------------------------------------
# Model loading & embedding computation
# ---------------------------------------------------------------------------

def load_clip_model(checkpoint_path: Path, device: str = "cpu") -> CLIPLightning:
    """Load trained CLIP model from checkpoint."""
    model = CLIPLightning.load_from_checkpoint(str(checkpoint_path), map_location=device)
    model.eval()
    model.to(device)
    return model


def compute_embeddings(
    model: CLIPLightning,
    dataloader: DataLoader,
    device: str = "cpu",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute per-image embeddings for the dataset."""
    image_embeds = []
    text_embeds = []
    texts = []
    neuron_ids = []

    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            images = batch["input"].to(device)
            batch_texts = batch["text"]
            batch_ids = batch["neuron_id"]

            img_emb = model.image_encoder.encode_for_clip(images)
            image_embeds.append(img_emb.cpu())

            txt_emb = model.text_encoder(batch_texts)
            text_embeds.append(txt_emb.cpu())

            texts.extend(batch_texts)
            neuron_ids.extend(batch_ids)

            total += len(batch_texts)
            if max_samples and total >= max_samples:
                break

    return {
        "image_embeds": torch.cat(image_embeds, dim=0),
        "text_embeds": torch.cat(text_embeds, dim=0),
        "texts": texts,
        "neuron_ids": neuron_ids,
    }


# ---------------------------------------------------------------------------
# Neuron-level aggregation
# ---------------------------------------------------------------------------

def aggregate_by_neuron(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    texts: List[str],
    neuron_ids: List[str],
    strategy: str = "mean",
) -> Dict[str, Any]:
    """Aggregate per-image embeddings to per-neuron embeddings.

    Args:
        strategy: "mean" - average all pose embeddings for each neuron.
                  "best" - for each neuron, keep the pose whose embedding
                           is closest to the mean (most representative view).

    Returns dict with aggregated embeddings, one entry per unique neuron.
    """
    # Group indices by neuron ID
    neuron_groups: Dict[str, List[int]] = defaultdict(list)
    for i, nid in enumerate(neuron_ids):
        neuron_groups[str(nid)].append(i)

    unique_ids = []
    agg_image_embeds = []
    agg_text_embeds = []
    agg_texts = []
    pose_counts = []

    for nid, indices in neuron_groups.items():
        unique_ids.append(nid)
        pose_counts.append(len(indices))

        img_group = image_embeds[indices]  # (K, D)
        # Text is identical across poses for same neuron; take first
        agg_texts.append(texts[indices[0]])
        agg_text_embeds.append(text_embeds[indices[0]])

        if strategy == "mean":
            agg_image_embeds.append(img_group.mean(dim=0))
        elif strategy == "best":
            # Pick the pose closest to the centroid
            centroid = img_group.mean(dim=0, keepdim=True)
            centroid = F.normalize(centroid, p=2, dim=-1)
            normed = F.normalize(img_group, p=2, dim=-1)
            sims = (normed @ centroid.T).squeeze()
            best_idx = sims.argmax().item()
            agg_image_embeds.append(img_group[best_idx])
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    return {
        "image_embeds": torch.stack(agg_image_embeds),
        "text_embeds": torch.stack(agg_text_embeds),
        "texts": agg_texts,
        "neuron_ids": unique_ids,
        "pose_counts": pose_counts,
    }


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    texts: List[str],
    ks: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute neuron-level retrieval metrics.

    Since many neurons share identical text descriptions (e.g. all hippocampal
    pyramidal cells map to the same string), we use "soft" matching: a retrieval
    is correct if the returned text matches the ground-truth text string.
    """
    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    sim_matrix = image_embeds @ text_embeds.T  # (N, N)
    n = sim_matrix.shape[0]

    metrics = {}

    # Build set of indices that share each text string
    text_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(texts):
        text_to_indices[t].append(i)

    # Image → Text: for each image, find rank of first correct text
    i2t_ranks = []
    for i in range(n):
        correct_indices = set(text_to_indices[texts[i]])
        sorted_indices = torch.argsort(sim_matrix[i], descending=True).tolist()
        for rank, idx in enumerate(sorted_indices, 1):
            if idx in correct_indices:
                i2t_ranks.append(rank)
                break

    i2t_ranks = np.array(i2t_ranks)
    for k in ks:
        metrics[f"i2t_R@{k}"] = (i2t_ranks <= k).mean() * 100
    metrics["i2t_MRR"] = (1.0 / i2t_ranks).mean() * 100
    metrics["i2t_median_rank"] = float(np.median(i2t_ranks))

    # Text → Image: for each text, find rank of first correct image
    t2i_ranks = []
    for i in range(n):
        correct_indices = set(text_to_indices[texts[i]])
        sorted_indices = torch.argsort(sim_matrix[:, i], descending=True).tolist()
        for rank, idx in enumerate(sorted_indices, 1):
            if idx in correct_indices:
                t2i_ranks.append(rank)
                break

    t2i_ranks = np.array(t2i_ranks)
    for k in ks:
        metrics[f"t2i_R@{k}"] = (t2i_ranks <= k).mean() * 100
    metrics["t2i_MRR"] = (1.0 / t2i_ranks).mean() * 100
    metrics["t2i_median_rank"] = float(np.median(t2i_ranks))

    return metrics


# ---------------------------------------------------------------------------
# Label extraction helpers
# ---------------------------------------------------------------------------

def extract_cell_type_from_text(text: str) -> str:
    """Extract primary cell type from generated text description."""
    text_lower = text.lower()
    cell_types = [
        ("basket", "basket"),
        ("chandelier", "chandelier"),
        ("martinotti", "martinotti"),
        ("stellate", "stellate"),
        ("granule", "granule"),
        ("purkinje", "purkinje"),
        ("mitral", "mitral"),
        ("pyramidal", "pyramidal"),
        ("interneuron", "interneuron"),
        ("principal", "principal"),
    ]
    for pattern, label in cell_types:
        if pattern in text_lower:
            return label
    return "other"


def extract_region_from_text(text: str) -> str:
    """Extract brain region from generated text description."""
    text_lower = text.lower()
    regions = [
        ("neocortex", "neocortex"),
        ("hippocampus", "hippocampus"),
        ("cerebellum", "cerebellum"),
        ("thalamus", "thalamus"),
        ("striatum", "striatum"),
        ("amygdala", "amygdala"),
        ("olfactory", "olfactory"),
        ("retina", "retina"),
    ]
    for pattern, label in regions:
        if pattern in text_lower:
            return label
    return "other"


# ---------------------------------------------------------------------------
# Zero-shot classification
# ---------------------------------------------------------------------------

def zero_shot_classification(
    model: CLIPLightning,
    image_embeds: torch.Tensor,
    class_names: List[str],
    prompt_template: str = "a {} neuron",
    device: str = "cpu",
) -> torch.Tensor:
    """Classify neuron embeddings using text prompts."""
    prompts = [prompt_template.format(c) for c in class_names]

    with torch.no_grad():
        text_embeds = model.text_encoder(prompts)
        text_embeds = text_embeds.to(device)

    image_embeds = F.normalize(image_embeds.to(device), p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    sims = image_embeds @ text_embeds.T
    return sims.argmax(dim=-1).cpu()


def run_zero_shot_eval(
    model: CLIPLightning,
    image_embeds: torch.Tensor,
    texts: List[str],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run zero-shot classification on neuron-level embeddings."""
    results = {}

    gt_cell_types = [extract_cell_type_from_text(t) for t in texts]
    gt_regions = [extract_region_from_text(t) for t in texts]

    # -- Cell type --
    cell_type_classes = ["pyramidal", "interneuron", "stellate", "granule", "basket", "principal"]
    cell_type_preds = zero_shot_classification(
        model, image_embeds, cell_type_classes,
        prompt_template="a {} neuron", device=device,
    )
    pred_cell_types = [cell_type_classes[i] for i in cell_type_preds]

    valid_mask = [gt in cell_type_classes for gt in gt_cell_types]
    gt_f = [gt for gt, v in zip(gt_cell_types, valid_mask) if v]
    pr_f = [p for p, v in zip(pred_cell_types, valid_mask) if v]

    if gt_f:
        results["cell_type_accuracy"] = accuracy_score(gt_f, pr_f) * 100
        results["cell_type_report"] = classification_report(gt_f, pr_f, zero_division=0)
        results["cell_type_confusion"] = confusion_matrix(gt_f, pr_f, labels=cell_type_classes)
        results["cell_type_classes"] = cell_type_classes
        results["cell_type_n"] = len(gt_f)

    # -- Brain region --
    region_classes = ["neocortex", "hippocampus", "cerebellum", "olfactory", "retina"]
    region_preds = zero_shot_classification(
        model, image_embeds, region_classes,
        prompt_template="a neuron from {}", device=device,
    )
    pred_regions = [region_classes[i] for i in region_preds]

    valid_mask = [gt in region_classes for gt in gt_regions]
    gt_f = [gt for gt, v in zip(gt_regions, valid_mask) if v]
    pr_f = [p for p, v in zip(pred_regions, valid_mask) if v]

    if gt_f:
        results["region_accuracy"] = accuracy_score(gt_f, pr_f) * 100
        results["region_report"] = classification_report(gt_f, pr_f, zero_division=0)
        results["region_classes"] = region_classes
        results["region_n"] = len(gt_f)

    return results


# ---------------------------------------------------------------------------
# t-SNE visualization
# ---------------------------------------------------------------------------

def create_tsne_visualization(
    embeddings: torch.Tensor,
    labels: List[str],
    title: str,
    output_path: Path,
    max_samples: int = 2000,
    perplexity: int = 30,
):
    """Create t-SNE visualization of neuron embeddings."""
    n = len(labels)
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = [labels[i] for i in indices]

    print(f"Running t-SNE for {title} ({len(labels)} neurons)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings.numpy())

    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
    label_to_color = {l: cmap(i % 20) for i, l in enumerate(unique_labels)}

    for label in unique_labels:
        mask = [l == label for l in labels]
        coords_subset = coords[mask]
        count = coords_subset.shape[0]
        ax.scatter(
            coords_subset[:, 0], coords_subset[:, 1],
            c=[label_to_color[label]],
            label=f"{label} ({count})",
            alpha=0.6, s=20,
        )

    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Novel query retrieval
# ---------------------------------------------------------------------------

def run_novel_query_test(
    model: CLIPLightning,
    image_embeds: torch.Tensor,
    texts: List[str],
    neuron_ids: List[str],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Test retrieval with novel text queries (neuron-level)."""
    results = {}

    queries = [
        ("a pyramidal neuron from hippocampus", {"cell": "pyramidal", "region": "hippocampus"}),
        ("an interneuron from neocortex layer 5", {"cell": "interneuron", "region": "neocortex"}),
        ("a stellate neuron from neocortex layer 4", {"cell": "stellate", "region": "neocortex"}),
        ("a granule neuron from cerebellum", {"cell": "granule", "region": "cerebellum"}),
        ("a basket neuron from neocortex", {"cell": "basket", "region": "neocortex"}),
    ]

    image_embeds = F.normalize(image_embeds.to(device), p=2, dim=-1)

    for query, expected in queries:
        with torch.no_grad():
            query_embed = model.text_encoder([query])
            query_embed = F.normalize(query_embed.to(device), p=2, dim=-1)

        sims = (image_embeds @ query_embed.T).squeeze()
        top_k_indices = torch.argsort(sims, descending=True)[:10].cpu().numpy()

        top_texts = [texts[i] for i in top_k_indices]
        top_ids = [neuron_ids[i] for i in top_k_indices]
        top_sims = [sims[i].item() for i in top_k_indices]

        matches = sum(
            1 for t in top_texts
            if expected["cell"] in t.lower() and expected["region"] in t.lower()
        )

        results[query] = {
            "top_10_matches": matches,
            "top_10_precision": matches / 10 * 100,
            "top_results": [
                {"text": t, "neuron_id": nid, "similarity": f"{s:.3f}"}
                for t, nid, s in zip(top_texts[:5], top_ids[:5], top_sims[:5])
            ],
        }

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    retrieval_metrics: Dict[str, float],
    zero_shot_results: Dict[str, Any],
    novel_query_results: Dict[str, Any],
    n_images: int,
    n_neurons: int,
    pose_stats: Dict[str, float],
    pooling: str,
    output_path: Optional[Path] = None,
):
    """Print evaluation report."""
    lines = []

    lines.append("\n" + "=" * 70)
    lines.append("CLIP Model Evaluation Report")
    lines.append("=" * 70)

    lines.append("\n[Dataset]")
    lines.append(f"  Total images:      {n_images:,}")
    lines.append(f"  Unique neurons:    {n_neurons:,}")
    lines.append(f"  Poses per neuron:  {pose_stats['mean']:.1f} mean, "
                 f"{pose_stats['median']:.0f} median, "
                 f"{pose_stats['min']:.0f}-{pose_stats['max']:.0f} range")
    lines.append(f"  Pooling strategy:  {pooling}")

    # Retrieval metrics
    lines.append("\n[Retrieval Metrics (neuron-level, soft matching)]")
    lines.append("-" * 40)
    lines.append("Image → Text:")
    for k in [1, 5, 10]:
        lines.append(f"  R@{k}:  {retrieval_metrics[f'i2t_R@{k}']:6.1f}%")
    lines.append(f"  MRR:  {retrieval_metrics['i2t_MRR']:6.1f}%")
    lines.append(f"  Median Rank: {retrieval_metrics['i2t_median_rank']:.0f}")

    lines.append("\nText → Image:")
    for k in [1, 5, 10]:
        lines.append(f"  R@{k}:  {retrieval_metrics[f't2i_R@{k}']:6.1f}%")
    lines.append(f"  MRR:  {retrieval_metrics['t2i_MRR']:6.1f}%")
    lines.append(f"  Median Rank: {retrieval_metrics['t2i_median_rank']:.0f}")

    # Zero-shot classification
    lines.append("\n[Zero-Shot Classification (neuron-level)]")
    lines.append("-" * 40)

    if "cell_type_accuracy" in zero_shot_results:
        lines.append(f"Cell Type Accuracy: {zero_shot_results['cell_type_accuracy']:.1f}%"
                     f"  (n={zero_shot_results['cell_type_n']})")
        lines.append("\n" + zero_shot_results["cell_type_report"])

    if "region_accuracy" in zero_shot_results:
        lines.append(f"Brain Region Accuracy: {zero_shot_results['region_accuracy']:.1f}%"
                     f"  (n={zero_shot_results['region_n']})")

    # Novel query tests
    lines.append("\n[Novel Query Retrieval (neuron-level)]")
    lines.append("-" * 40)
    for query, result in novel_query_results.items():
        lines.append(f"\nQuery: \"{query}\"")
        lines.append(f"  Top-10 Precision: {result['top_10_precision']:.0f}%")
        lines.append("  Top results:")
        for i, r in enumerate(result["top_results"], 1):
            lines.append(f"    {i}. [{r['similarity']}] {r['text']}  (id={r['neuron_id']})")

    lines.append("\n" + "=" * 70)

    report = "\n".join(lines)
    print(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained CLIP model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to CLIP checkpoint")
    parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata file")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest file (auto-detects val/test)")
    parser.add_argument("--source", default="neuromorpho", help="Data source adapter")
    parser.add_argument("--id-column", default="neuron_id", help="ID column in metadata")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max images to process")
    parser.add_argument("--image-size", type=int, default=512, help="Image size")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"), help="Output dir")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE visualization")
    parser.add_argument(
        "--pooling", type=str, default="mean",
        choices=["mean", "best"],
        help="Multi-pose aggregation: 'mean' averages all poses, "
             "'best' picks the most representative pose"
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Find manifest
    if args.manifest is None:
        for name in ["manifest_val.jsonl", "manifest_test.jsonl", "manifest.jsonl"]:
            candidate = args.data_dir / name
            if candidate.exists():
                args.manifest = candidate
                break
        if args.manifest is None:
            raise ValueError(f"No manifest found in {args.data_dir}")
    print(f"Manifest: {args.manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_clip_model(args.checkpoint, device=device)

    # Create dataset
    adapter = get_adapter(args.source)
    text_gen = NeuronTextGenerator(adapter, augment=False)
    dataset = NeuronCLIPDataset(
        manifest_path=args.manifest,
        data_root=args.data_dir,
        metadata_path=args.metadata,
        adapter=adapter,
        id_column=args.id_column,
        text_generator=text_gen,
        image_size=args.image_size,
    )

    n_images = len(dataset)
    print(f"Dataset: {n_images:,} images")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Compute per-image embeddings
    print("\nComputing per-image embeddings...")
    raw = compute_embeddings(model, dataloader, device=device, max_samples=args.max_samples)

    n_raw = len(raw["texts"])
    print(f"Computed {n_raw:,} image embeddings")

    # Aggregate to neuron level
    print(f"\nAggregating to neuron level (strategy={args.pooling})...")
    agg = aggregate_by_neuron(
        raw["image_embeds"], raw["text_embeds"],
        raw["texts"], raw["neuron_ids"],
        strategy=args.pooling,
    )

    n_neurons = len(agg["neuron_ids"])
    pose_counts = np.array(agg["pose_counts"])
    pose_stats = {
        "mean": float(pose_counts.mean()),
        "median": float(np.median(pose_counts)),
        "min": float(pose_counts.min()),
        "max": float(pose_counts.max()),
    }
    print(f"Aggregated: {n_raw:,} images -> {n_neurons:,} neurons "
          f"({pose_stats['mean']:.1f} poses/neuron)")

    # All downstream evaluation uses neuron-level data
    image_embeds = agg["image_embeds"]
    text_embeds = agg["text_embeds"]
    texts = agg["texts"]
    neuron_ids = agg["neuron_ids"]

    # Retrieval
    print("\nComputing retrieval metrics...")
    retrieval_metrics = compute_retrieval_metrics(image_embeds, text_embeds, texts)

    # Zero-shot
    print("Running zero-shot classification...")
    zero_shot_results = run_zero_shot_eval(model, image_embeds, texts, device=device)

    # Novel queries
    print("Testing novel queries...")
    novel_query_results = run_novel_query_test(
        model, image_embeds, texts, neuron_ids, device=device
    )

    # t-SNE
    if not args.skip_tsne:
        print("\nCreating visualizations...")
        cell_types = [extract_cell_type_from_text(t) for t in texts]
        create_tsne_visualization(
            image_embeds, cell_types,
            f"Neuron Embeddings by Cell Type ({args.pooling} pooling)",
            args.output_dir / "tsne_cell_type.png",
        )
        regions = [extract_region_from_text(t) for t in texts]
        create_tsne_visualization(
            image_embeds, regions,
            f"Neuron Embeddings by Brain Region ({args.pooling} pooling)",
            args.output_dir / "tsne_region.png",
        )

    # Report
    print_report(
        retrieval_metrics, zero_shot_results, novel_query_results,
        n_images=n_raw, n_neurons=n_neurons,
        pose_stats=pose_stats, pooling=args.pooling,
        output_path=args.output_dir / "eval_report.txt",
    )

    # Save machine-readable metrics
    metrics_json = {
        "dataset": {
            "n_images": n_raw,
            "n_neurons": n_neurons,
            "pose_stats": pose_stats,
            "pooling": args.pooling,
        },
        "retrieval": retrieval_metrics,
        "zero_shot": {
            k: v for k, v in zero_shot_results.items()
            if not isinstance(v, np.ndarray) and "report" not in k
        },
        "novel_queries": {
            q: {"top_10_precision": r["top_10_precision"]}
            for q, r in novel_query_results.items()
        },
    }
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
