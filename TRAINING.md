# Training Guide

This document covers the two-stage training pipeline, from data acquisition through model evaluation.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STAGE 1: Pre-training                             │
│                                                                             │
│   NeuroMorpho.org Data          SegVAE2D Model                              │
│   (~200k neurons)         ┌──────────────────────┐                          │
│         │                 │  U-Net Encoder       │                          │
│         ▼                 │  (learns morphology) │                          │
│   PCA-guided 2D     ───►  │         │            │   ───►  Segmentation     │
│   projections             │  Variational Skips   │         + Depth          │
│   (24 views/neuron)       │         │            │         Reconstruction   │
│                           │  U-Net Decoder       │                          │
│                           └──────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Transfer encoder weights
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STAGE 2: Fine-tuning                              │
│                                                                             │
│   Labeled Data                CLIP Contrastive Learning                     │
│   (with T-types, regions)                                                   │
│         │                 ┌──────────────────────┐                          │
│         ▼                 │  Frozen/Fine-tuned   │                          │
│   2D Projections    ───►  │  VAE Encoder         │   ───►  Image Embedding  │
│         +                 │         │            │              │           │
│   Auto-generated          │  Projection Head     │              ▼           │
│   Text Descriptions       └──────────────────────┘        Contrastive       │
│         │                                                     Loss          │
│         │                 ┌──────────────────────┐              ▲           │
│         └───────────────► │  Text Encoder        │   ───►  Text Embedding   │
│                           └──────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Step 1: Download Source Data

### NeuroMorpho.org

The `neuromorpho_bulk.py` tool queries the NeuroMorpho.org API and downloads standardized SWC files.

**Simple query (single field):**
```bash
python -m axonet.utils.neuromorpho_bulk \
    --query 'species:mouse' \
    --out data/neuromorpho \
    --fetch-morphometry \
    --find \
    --insecure
```

**Complex query with multiple filters:**
```bash
python -m axonet.utils.neuromorpho_bulk \
    --filters '{"domain": ["Dendrites, Soma, Axon"], "attributes": ["Diameter, 3D, Angles"], "species": ["mouse"]}' \
    --out data/neuromorpho \
    --fetch-morphometry \
    --find \
    --insecure
```

**Key options:**
| Flag | Description |
|------|-------------|
| `--query` | Simple Lucene-style query (e.g., `species:mouse`) |
| `--filters` | JSON object for complex multi-field queries |
| `--out` | Output directory |
| `--fetch-morphometry` | Also download morphometric measurements |
| `--find` | Use robust per-neuron scraping to find exact download URLs |
| `--insecure` | Disable SSL verification (needed due to cert issues) |
| `--count-only` | Just print the count, don't download |
| `--max-pages` | Limit number of pages (for testing) |
| `--page-size` | Results per page (max 500) |

**Output structure:**
```
data/neuromorpho/
├── swc/                    # Standardized SWC files
│   ├── neuron_001.CNG.swc
│   ├── neuron_002.CNG.swc
│   └── ...
├── metadata.jsonl          # Per-neuron metadata (cell type, region, etc.)
└── morphometry.jsonl       # Morphometric measurements (if --fetch-morphometry)
```

### Allen Brain Institute

For patch-seq data with transcriptomic labels:

```bash
python -m axonet.utils.allen_bulk \
    --out data/allen \
    --include-metadata
```

## Step 2: Generate Training Dataset

The `dataset_generator.py` renders 2D projections from 3D SWC files using headless GPU rendering.

```bash
python -m axonet.training.dataset_generator \
    --swc-dir data/neuromorpho/swc \
    --out data/rendered \
    --sampling pca \
    --canonical-views 6 \
    --biased-views 12 \
    --random-views 6 \
    --width 512 --height 512 \
    --val-ratio 0.15 \
    --test-ratio 0.0 \
    --supersample-factor 2 \
    --adaptive-framing \
    --margin 0.40 \
    -j 8
```

### Camera Sampling Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `pca` | PCA-guided: canonical + biased + random views | **Recommended** - best coverage |
| `fibonacci` | Uniform sphere coverage via golden angle | Good alternative |
| `random` | Random directions on unit sphere | Maximum variety |

**PCA sampling breakdown:**
- **Canonical views** (default 6): Camera positioned along +/- PC1, PC2, PC3 axes
- **Biased views** (default 12): Concentrated near the PC1-PC2 plane where projected area is largest
- **Random views** (default 6): Uniform random for diversity

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--swc-dir` | required | Directory containing SWC files |
| `--out` | required | Output dataset directory |
| `--sampling` | `pca` | Camera sampling strategy |
| `--width/--height` | 512 | Image dimensions |
| `--val-ratio` | 0.0 | Fraction for validation split |
| `--test-ratio` | 0.0 | Fraction for test split |
| `--supersample-factor` | 4 | Supersampling for anti-aliasing (2-4 recommended) |
| `--adaptive-framing` | off | Adjust zoom per-view to maximize neuron visibility |
| `--margin` | 0.40 | Padding around neuron bounding box |
| `--min-qc` | 0.7 | Minimum fraction of neuron visible in frame |
| `--projection` | `ortho` | `ortho` or `perspective` |
| `-j/--jobs` | 1 | Parallel workers |
| `--no-cache` | off | Disable mesh caching |

### Output Structure

```
data/rendered/
├── manifest_train.jsonl    # Training set index
├── manifest_val.jsonl      # Validation set index (if val-ratio > 0)
├── manifest_test.jsonl     # Test set index (if test-ratio > 0)
└── images/
    ├── neuron_001/
    │   ├── view_00_mask_bw.png     # Binary silhouette (model input)
    │   ├── view_00_mask.png        # Segmentation: 0=bg, 1=soma, 2=axon, 3=basal, 4=apical, 5=other
    │   ├── view_00_mask_color.png  # Colorized segmentation (for visualization)
    │   ├── view_00_depth.png       # Depth map
    │   └── ...
    └── neuron_002/
        └── ...
```

**Manifest format (JSONL):**
```json
{"swc": "neuron_001.swc", "mask_bw": "images/neuron_001/view_00_mask_bw.png", "mask": "images/neuron_001/view_00_mask.png", "depth": "images/neuron_001/view_00_depth.png", "direction": [0.57, 0.57, 0.57], "qc": 0.95}
```

## Step 3: Train Stage 1 (Segmentation VAE)

The SegVAE2D model learns to reconstruct segmentation masks and depth maps from binary silhouettes.

```bash
python -m axonet.training.trainer \
    --data-dir data/rendered \
    --manifest-train data/rendered/manifest_train.jsonl \
    --manifest-val data/rendered/manifest_val.jsonl \
    --batch-size 8 \
    --lr 1e-4 \
    --max-epochs 100 \
    --kld-weight 0.1 \
    --skip-mode variational \
    --base-channels 64 \
    --latent-channels 128 \
    --num-classes 6 \
    --lambda-seg 1.0 \
    --lambda-depth 1.0 \
    --save-dir checkpoints/stage1 \
    --log-dir logs/stage1 \
    --early-stopping
```

### Model Architecture Options

| Flag | Default | Description |
|------|---------|-------------|
| `--base-channels` | 64 | Base channel width (doubles each level) |
| `--latent-channels` | 128 | Bottleneck latent dimension |
| `--num-classes` | 6 | Segmentation classes (bg + 5 compartments) |
| `--skip-mode` | `variational` | `variational` (recommended) or `deterministic` |

### Loss Options

| Flag | Default | Description |
|------|---------|-------------|
| `--kld-weight` | 0.1 | KL divergence weight (β in β-VAE) |
| `--free-nats` | 0.0 | Free-bits to prevent KL collapse |
| `--beta` | 1.0 | Additional KL multiplier |
| `--lambda-seg` | 1.0 | Segmentation loss weight |
| `--lambda-depth` | 1.0 | Depth loss weight |

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-epochs` | 100 | Maximum training epochs |
| `--max-steps` | None | Max steps (overrides epochs) |
| `--lr` | 1e-4 | Learning rate |
| `--lr-scheduler` | `cosine` | `cosine`, `step`, or `constant` |
| `--lr-warmup-steps` | 0 | Warmup steps |
| `--gradient-clip-val` | 0 | Gradient clipping (0 = disabled) |
| `--early-stopping` | off | Enable early stopping |
| `--resume` | None | Resume from checkpoint |

### Monitoring Training

```bash
tensorboard --logdir logs/stage1
```

Key metrics to watch:
- `train/loss`, `val/loss`: Total loss
- `train/seg_loss`: Segmentation reconstruction quality
- `train/depth_loss`: Depth prediction quality
- `train/kld`: KL divergence (should stabilize, not collapse to 0)

The `ValidationImageLogger` callback logs sample predictions: Input | Predicted Seg | GT Seg | Predicted Depth | GT Depth.

## Step 4: Train Stage 2 (CLIP Fine-tuning)

CLIP-style contrastive learning aligns image embeddings with auto-generated text descriptions.

```bash
python -m axonet.training.clip_trainer \
    --stage1-checkpoint checkpoints/stage1/best.ckpt \
    --data-dir data/rendered \
    --manifest-train data/rendered/manifest_train.jsonl \
    --manifest-val data/rendered/manifest_val.jsonl \
    --metadata data/neuromorpho/metadata.jsonl \
    --source neuromorpho \
    --batch-size 64 \
    --clip-embed-dim 512 \
    --temperature 0.07 \
    --learnable-temperature \
    --text-encoder distilbert-base-uncased \
    --freeze-encoder \
    --max-epochs 50 \
    --lr 1e-4 \
    --save-dir checkpoints/clip \
    --log-dir logs/clip
```

### Text Generation

The `NeuronTextGenerator` auto-generates descriptions at multiple levels:

| Level | Example | Weight |
|-------|---------|--------|
| Broad | "a mouse neuron" | 10% |
| Standard | "a mouse pyramidal neuron from visual cortex" | 30% |
| Detailed | "a mouse VISp layer 5 pyramidal neuron with extensive apical dendrite" | 40% |
| Morphometric | "...with 127 branch points and total length 4.2mm" | 20% |

This curriculum helps the model learn hierarchical semantic relationships.

### CLIP Options

| Flag | Default | Description |
|------|---------|-------------|
| `--stage1-checkpoint` | required | Path to Stage 1 checkpoint |
| `--clip-embed-dim` | 512 | CLIP embedding dimension |
| `--temperature` | 0.07 | InfoNCE temperature |
| `--learnable-temperature` | off | Make temperature learnable |
| `--text-encoder` | `distilbert-base-uncased` | Text encoder model |
| `--freeze-encoder` | off | Freeze VAE encoder (train projection only) |
| `--encoder-lr-mult` | 0.1 | LR multiplier for encoder (if not frozen) |
| `--lambda-clip` | 1.0 | Contrastive loss weight |
| `--lambda-kld` | 0.0 | KL regularization (prevent drift from Stage 1) |

### Data Source Adapters

The `--source` flag selects a metadata adapter:

| Source | Description |
|--------|-------------|
| `neuromorpho` | NeuroMorpho.org metadata format |
| `allen` | Allen Brain Institute patch-seq |

## Step 5: Evaluate

```bash
python -m axonet.training.clip_evaluator \
    --checkpoint checkpoints/clip/best.ckpt \
    --data-dir data/rendered \
    --manifest data/rendered/manifest_val.jsonl \
    --metadata data/neuromorpho/metadata.jsonl \
    --source neuromorpho \
    --output-dir eval_results \
    --pooling mean
```

### Evaluation Outputs

```
eval_results/
├── metrics.json            # Quantitative metrics
├── eval_report.txt         # Human-readable summary
├── tsne_cell_type.png      # t-SNE colored by cell type
├── tsne_region.png         # t-SNE colored by brain region
└── confusion_matrix.png    # Zero-shot classification
```

### Metrics

- **Retrieval R@k**: How often correct match is in top-k results
- **Zero-shot classification**: Using text prompts like "a pyramidal neuron"
- **Multi-pose aggregation**: `--pooling mean` averages all poses per neuron

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch-size 4

# Reduce image size
--width 256 --height 256

# Use gradient accumulation
--gradient-accumulation-steps 4
```

### KL Collapse (KL divergence → 0)

The latent space isn't being used. Try:
```bash
# Increase free-nats
--free-nats 0.1

# Reduce KL weight
--kld-weight 0.01

# Use KL warmup (gradually increase KL over N steps)
--beta 0.0  # Start with 0, manually increase
```

### Poor Segmentation Quality

- Verify masks are correct: check a few samples visually
- Increase supersampling: `--supersample-factor 4`
- Check class balance in training data
- Try focal loss for class imbalance (modify trainer)

### Slow Rendering

```bash
# More workers
-j 16

# Reduce supersampling (may affect quality)
--supersample-factor 2

# Disable mesh caching if disk is slow
--no-cache
```

### CLIP Training Unstable

- Start with frozen encoder: `--freeze-encoder`
- Use lower learning rate: `--lr 1e-5`
- Increase batch size for better contrastive learning: `--batch-size 128`

## Cloud Training

### Google Cloud Batch

```bash
# Submit dataset generation job
axonet-cloud generate-dataset \
    --manifest gs://bucket/manifest.jsonl \
    --output gs://bucket/rendered/ \
    --machine-type n1-standard-8 \
    --gpu-type nvidia-tesla-t4

# Submit training job
axonet-cloud train \
    --data gs://bucket/rendered/ \
    --output gs://bucket/checkpoints/ \
    --machine-type n1-highmem-8 \
    --gpu-type nvidia-tesla-v100
```

### Docker

```bash
# Build images
docker build --target dataset -t axonet:dataset .
docker build --target train -t axonet:train .
```

## Reference Configurations

Example configurations are in `configs/`:
- `stage1_neuromorpho.yaml` - Stage 1 settings
- `stage2_allen_clip.yaml` - Stage 2 settings

These are documentation/reference only; the actual training is CLI-driven.

## References

- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
