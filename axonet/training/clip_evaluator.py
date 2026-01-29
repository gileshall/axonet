"""Evaluation script for trained CLIP models.

Handles the fact that multiple poses (images) represent the same neuron
by aggregating per-image embeddings into per-neuron embeddings before
evaluation. Supports mean pooling and max-similarity strategies.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import matplotlib.pyplot as plt
import numpy as np
import torch

# Suppress DDP stream mismatch warning (benign)
if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
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

def load_clip_model(
    checkpoint_path: Path,
    device: str = "cpu",
    stage1_checkpoint: Optional[Path] = None,
) -> CLIPLightning:
    """Load trained CLIP model from checkpoint.

    Handles checkpoints saved with torch.compile() by stripping '_orig_mod.' prefix.
    Allows overriding the stage1_checkpoint path stored in the checkpoint.
    """
    # Load checkpoint to check for compiled model state
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if state dict has _orig_mod. prefix (from torch.compile)
    needs_rewrite = False
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if any(k.startswith("image_encoder._orig_mod.") for k in state_dict.keys()):
            print("Detected torch.compile checkpoint, stripping '_orig_mod.' prefix...")
            new_state_dict = {}
            for k, v in state_dict.items():
                if "._orig_mod." in k:
                    new_k = k.replace("._orig_mod.", ".")
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            checkpoint["state_dict"] = new_state_dict
            needs_rewrite = True

    # Override stage1_checkpoint in hyperparameters if provided
    load_kwargs = {"map_location": device}
    if stage1_checkpoint is not None:
        load_kwargs["stage1_checkpoint"] = str(stage1_checkpoint)

    if needs_rewrite:
        with tempfile.NamedTemporaryFile(suffix=".ckpt", delete=False) as f:
            torch.save(checkpoint, f.name)
            model = CLIPLightning.load_from_checkpoint(f.name, **load_kwargs)
            os.unlink(f.name)
    else:
        model = CLIPLightning.load_from_checkpoint(str(checkpoint_path), **load_kwargs)

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


# Species with enough neurons to meaningfully evaluate
EVAL_SPECIES = [
    "mouse", "rat", "human", "chimpanzee", "giraffe", "monkey",
    "leopard", "cheetah", "C. elegans", "Lion",
]


def extract_species_from_metadata(metadata: Dict[str, Any]) -> str:
    """Extract species directly from metadata entry."""
    species = metadata.get("species", "unknown")
    if isinstance(species, str) and species.lower() not in ("", "unknown", "not reported"):
        return species
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
# Species zero-shot classification
# ---------------------------------------------------------------------------

def run_species_zero_shot(
    model: CLIPLightning,
    image_embeds: torch.Tensor,
    metadata_list: List[Dict[str, Any]],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Zero-shot species classification using neuron-level embeddings."""
    gt_species = [extract_species_from_metadata(m) for m in metadata_list]

    # Determine which species are actually present
    species_counts = {}
    for s in gt_species:
        species_counts[s] = species_counts.get(s, 0) + 1

    # Only evaluate species with >= 5 examples
    eval_species = [s for s in EVAL_SPECIES if species_counts.get(s, 0) >= 5]
    if not eval_species:
        return {"species_accuracy": None, "species_n": 0, "note": "no eval species with >= 5 samples"}

    preds = zero_shot_classification(
        model, image_embeds, eval_species,
        prompt_template="a {} neuron", device=device,
    )
    pred_species = [eval_species[i] for i in preds]

    valid_mask = [gt in eval_species for gt in gt_species]
    gt_f = [gt for gt, v in zip(gt_species, valid_mask) if v]
    pr_f = [p for p, v in zip(pred_species, valid_mask) if v]

    results: Dict[str, Any] = {}
    if gt_f:
        results["species_accuracy"] = accuracy_score(gt_f, pr_f) * 100
        results["species_report"] = classification_report(gt_f, pr_f, zero_division=0)
        results["species_classes"] = eval_species
        results["species_n"] = len(gt_f)

    return results


# ---------------------------------------------------------------------------
# Morphometric retrieval
# ---------------------------------------------------------------------------

def run_morphometric_retrieval(
    model: CLIPLightning,
    image_embeds: torch.Tensor,
    metadata_list: List[Dict[str, Any]],
    device: str = "cpu",
    k: int = 50,
) -> Dict[str, Any]:
    """Test whether morphometric text queries retrieve neurons with matching properties.

    For example: does "a large neuron" retrieve neurons with high total length?
    We measure this by checking whether the retrieved set has a significantly
    higher mean value than the population for the relevant morphometric.
    """
    queries = [
        {
            "text": "a large neuron",
            "morph_key": "length",
            "direction": "high",  # expect high values
        },
        {
            "text": "a small, compact neuron",
            "morph_key": "length",
            "direction": "low",
        },
        {
            "text": "a densely branched neuron",
            "morph_key": "n_bifs",
            "direction": "high",
        },
        {
            "text": "a sparsely branched neuron",
            "morph_key": "n_bifs",
            "direction": "low",
        },
        {
            "text": "a sprawling neuron with long dendrites",
            "morph_key": "length",
            "direction": "high",
        },
    ]

    image_embeds_norm = F.normalize(image_embeds.to(device), p=2, dim=-1)
    results = {}

    for q in queries:
        with torch.no_grad():
            query_embed = model.text_encoder([q["text"]])
            query_embed = F.normalize(query_embed.to(device), p=2, dim=-1)

        sims = (image_embeds_norm @ query_embed.T).squeeze()
        top_indices = torch.argsort(sims, descending=True)[:k].cpu().numpy()

        # Get morphometric values for top-k and population
        morph_key = q["morph_key"]
        top_values = []
        for idx in top_indices:
            morph = metadata_list[idx].get("morphometry", {})
            if morph and isinstance(morph, dict):
                val = morph.get(morph_key)
                if val is not None:
                    top_values.append(float(val))

        pop_values = []
        for m in metadata_list:
            morph = m.get("morphometry", {})
            if morph and isinstance(morph, dict):
                val = morph.get(morph_key)
                if val is not None:
                    pop_values.append(float(val))

        if not top_values or not pop_values:
            continue

        top_mean = np.mean(top_values)
        pop_mean = np.mean(pop_values)
        pop_std = np.std(pop_values)

        # Effect size (Cohen's d)
        effect_size = (top_mean - pop_mean) / pop_std if pop_std > 0 else 0.0

        # For "low" direction queries, we expect negative effect size
        directional_correct = (
            (q["direction"] == "high" and effect_size > 0) or
            (q["direction"] == "low" and effect_size < 0)
        )

        results[q["text"]] = {
            "morph_key": morph_key,
            "direction": q["direction"],
            "top_k_mean": float(top_mean),
            "population_mean": float(pop_mean),
            "effect_size": float(effect_size),
            "directional_correct": directional_correct,
            "n_top_with_data": len(top_values),
        }

    return results


# ---------------------------------------------------------------------------
# Compositional query evaluation
# ---------------------------------------------------------------------------

def run_compositional_queries(
    model: CLIPLightning,
    image_embeds: torch.Tensor,
    texts: List[str],
    metadata_list: List[Dict[str, Any]],
    device: str = "cpu",
    k: int = 10,
) -> Dict[str, Any]:
    """Test retrieval with compositional queries that combine multiple attributes.

    These queries combine species + cell type + region in ways that may not
    have been seen as exact training strings, testing compositional understanding.
    """
    queries = [
        {
            "text": "a mouse pyramidal neuron from hippocampus",
            "expected": {"species": "mouse", "cell_type": "pyramidal", "region": "hippocampus"},
        },
        {
            "text": "a rat interneuron from neocortex",
            "expected": {"species": "rat", "cell_type": "interneuron", "region": "neocortex"},
        },
        {
            "text": "a human pyramidal neuron from neocortex",
            "expected": {"species": "human", "cell_type": "pyramidal", "region": "neocortex"},
        },
        {
            "text": "a mouse granule neuron from cerebellum",
            "expected": {"species": "mouse", "cell_type": "granule", "region": "cerebellum"},
        },
        {
            "text": "a rat pyramidal neuron from hippocampus",
            "expected": {"species": "rat", "cell_type": "pyramidal", "region": "hippocampus"},
        },
        {
            "text": "a mouse stellate neuron from neocortex",
            "expected": {"species": "mouse", "cell_type": "stellate", "region": "neocortex"},
        },
        {
            "text": "a large mouse neuron from neocortex",
            "expected": {"species": "mouse", "region": "neocortex"},
        },
        {
            "text": "a chimpanzee pyramidal neuron from neocortex",
            "expected": {"species": "chimpanzee", "cell_type": "pyramidal", "region": "neocortex"},
        },
    ]

    image_embeds_norm = F.normalize(image_embeds.to(device), p=2, dim=-1)
    results = {}

    for q in queries:
        with torch.no_grad():
            query_embed = model.text_encoder([q["text"]])
            query_embed = F.normalize(query_embed.to(device), p=2, dim=-1)

        sims = (image_embeds_norm @ query_embed.T).squeeze()
        top_indices = torch.argsort(sims, descending=True)[:k].cpu().numpy()
        top_sims = [sims[i].item() for i in top_indices]

        # Check each retrieved result against expected attributes
        match_counts = {attr: 0 for attr in q["expected"]}
        full_matches = 0

        for idx in top_indices:
            meta = metadata_list[idx]
            text = texts[idx]

            attr_match = {}
            expected = q["expected"]

            if "species" in expected:
                sp = extract_species_from_metadata(meta)
                attr_match["species"] = (sp.lower() == expected["species"].lower())

            if "cell_type" in expected:
                attr_match["cell_type"] = (expected["cell_type"].lower() in text.lower())

            if "region" in expected:
                attr_match["region"] = (expected["region"].lower() in text.lower())

            for attr, matched in attr_match.items():
                if matched:
                    match_counts[attr] += 1

            if all(attr_match.values()):
                full_matches += 1

        results[q["text"]] = {
            "full_match_precision": full_matches / k * 100,
            "per_attribute_precision": {
                attr: count / k * 100 for attr, count in match_counts.items()
            },
            "top_similarity": float(top_sims[0]) if top_sims else 0.0,
        }

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
    """Create t-SNE visualization of neuron embeddings using colored shapes."""
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

    # High-contrast colors (colorblind-friendly)
    colors = [
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#a65628",  # brown
        "#f781bf",  # pink
        "#999999",  # gray
        "#00ced1",  # dark cyan
        "#ffd700",  # gold
    ]

    # Distinct marker shapes
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "<", ">"]

    for i, label in enumerate(unique_labels):
        mask = [l == label for l in labels]
        coords_subset = coords[mask]
        count = coords_subset.shape[0]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.scatter(
            coords_subset[:, 0], coords_subset[:, 1],
            c=color,
            marker=marker,
            label=f"{label} ({count})",
            alpha=0.7, s=30,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
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
    species_results: Optional[Dict[str, Any]] = None,
    morphometric_results: Optional[Dict[str, Any]] = None,
    compositional_results: Optional[Dict[str, Any]] = None,
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

    # Species zero-shot
    if species_results and "species_accuracy" in species_results:
        lines.append(f"\n[Species Zero-Shot Classification (neuron-level)]")
        lines.append("-" * 40)
        lines.append(f"Species Accuracy: {species_results['species_accuracy']:.1f}%"
                     f"  (n={species_results['species_n']})")
        if "species_report" in species_results:
            lines.append("\n" + species_results["species_report"])

    # Morphometric retrieval
    if morphometric_results:
        lines.append(f"\n[Morphometric Retrieval (top-50)]")
        lines.append("-" * 40)
        for query_text, r in morphometric_results.items():
            direction_mark = "OK" if r["directional_correct"] else "WRONG"
            lines.append(f"\nQuery: \"{query_text}\"")
            lines.append(f"  Key: {r['morph_key']}, expected: {r['direction']}")
            lines.append(f"  Top-50 mean: {r['top_k_mean']:.1f}, population mean: {r['population_mean']:.1f}")
            lines.append(f"  Effect size (Cohen's d): {r['effect_size']:.2f}  [{direction_mark}]")

    # Compositional queries
    if compositional_results:
        lines.append(f"\n[Compositional Query Retrieval (top-10)]")
        lines.append("-" * 40)
        for query_text, r in compositional_results.items():
            lines.append(f"\nQuery: \"{query_text}\"")
            lines.append(f"  Full-match precision: {r['full_match_precision']:.0f}%")
            for attr, prec in r["per_attribute_precision"].items():
                lines.append(f"    {attr}: {prec:.0f}%")

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

def _load_metadata_by_id(metadata_path: Path, id_column: str) -> Dict[str, Dict[str, Any]]:
    """Load metadata indexed by neuron ID."""
    metadata = {}
    path = Path(metadata_path)
    if path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    key = str(e.get(id_column, ""))
                    if key:
                        metadata[key] = e
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for e in data:
                key = str(e.get(id_column, ""))
                if key:
                    metadata[key] = e
        else:
            metadata = data
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained CLIP model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to CLIP checkpoint")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None,
                        help="Path to Stage 1 checkpoint (overrides path saved in CLIP checkpoint)")
    parser.add_argument("--data-dir", type=Path, required=True, help="Data directory")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata file")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Manifest file (auto-detects val/test)")
    parser.add_argument("--source", default="neuromorpho", help="Data source adapter")
    parser.add_argument("--id-column", default="neuron_name",
                        help="ID column in metadata (use 'neuron_name' for NeuroMorpho, 'cell_specimen_id' for Allen)")
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
    model = load_clip_model(
        args.checkpoint,
        device=device,
        stage1_checkpoint=args.stage1_checkpoint,
    )

    # Load metadata for species/morphometric evaluation
    print(f"Loading metadata from {args.metadata}...")
    metadata_by_id = _load_metadata_by_id(args.metadata, args.id_column)
    print(f"  {len(metadata_by_id):,} metadata entries")

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

    # Build neuron-level metadata list (aligned with aggregated embeddings)
    metadata_list = []
    for nid in neuron_ids:
        metadata_list.append(metadata_by_id.get(str(nid), {}))

    # Retrieval
    print("\nComputing retrieval metrics...")
    retrieval_metrics = compute_retrieval_metrics(image_embeds, text_embeds, texts)

    # Zero-shot cell type & region
    print("Running zero-shot classification...")
    zero_shot_results = run_zero_shot_eval(model, image_embeds, texts, device=device)

    # Species zero-shot
    print("Running species zero-shot classification...")
    species_results = run_species_zero_shot(model, image_embeds, metadata_list, device=device)

    # Morphometric retrieval
    print("Testing morphometric retrieval...")
    morphometric_results = run_morphometric_retrieval(
        model, image_embeds, metadata_list, device=device
    )

    # Compositional queries
    print("Testing compositional queries...")
    compositional_results = run_compositional_queries(
        model, image_embeds, texts, metadata_list, device=device
    )

    # Novel queries (original)
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
        # Species t-SNE
        species_labels = [extract_species_from_metadata(m) for m in metadata_list]
        create_tsne_visualization(
            image_embeds, species_labels,
            f"Neuron Embeddings by Species ({args.pooling} pooling)",
            args.output_dir / "tsne_species.png",
            max_samples=2000,
        )

    # Report
    print_report(
        retrieval_metrics, zero_shot_results, novel_query_results,
        n_images=n_raw, n_neurons=n_neurons,
        pose_stats=pose_stats, pooling=args.pooling,
        output_path=args.output_dir / "eval_report.txt",
        species_results=species_results,
        morphometric_results=morphometric_results,
        compositional_results=compositional_results,
    )

    # Helper to convert numpy types to native Python for JSON serialization
    def to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

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
        "species_zero_shot": {
            k: v for k, v in species_results.items()
            if not isinstance(v, np.ndarray) and "report" not in k
        } if species_results else {},
        "morphometric_retrieval": to_json_serializable(morphometric_results) if morphometric_results else {},
        "compositional_queries": to_json_serializable(compositional_results) if compositional_results else {},
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
