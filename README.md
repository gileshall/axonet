# Axonet

A machine learning platform for neuron morphology analysis, embedding, and classification using 2D projections of 3D neuronal reconstructions.

## Motivation

Neuron morphology is a critical phenotype for understanding brain cell types, but obtaining labeled morphological data is challenging. Patch-seq experiments that combine morphology with transcriptomic labels are time-consuming and produce limited datasets.

**Axonet addresses data scarcity through a two-stage training approach:**

1. **Pre-train** a segmentation VAE on abundant unlabeled data from [NeuroMorpho.org](https://neuromorpho.org) (~200,000 neurons)
2. **Fine-tune** with CLIP-style contrastive learning on smaller labeled datasets (Allen Brain patch-seq data) to learn semantic embeddings linking morphology with cell type, brain region, and other metadata

The core insight is using **2D projections as data augmentation**: by rendering each 3D neuron from multiple camera angles using PCA-guided sampling, the model learns view-invariant representations. This leverages proven architectures (U-Net encoder with variational skip connections, CLIP contrastive learning) while being robust to limited training data.

## Key Features

### Interactive SWC Viewer

A GPU-accelerated 3D viewer for exploring neuron morphologies with Phong shading:

```bash
python -m axonet.visualization.pyglet_swc_viewer path/to/neuron.swc
```

**Controls:**
| Key | Action |
|-----|--------|
| Mouse drag | Orbit (trackball rotation) |
| Shift+drag / MMB | Pan |
| Scroll wheel | Zoom |
| R | Reset view |
| W | Toggle wireframe |
| O | Cycle depth visualization |
| C | Toggle compartment coloring |
| P | Toggle perspective/orthographic |
| F | Open file browser |
| S | Save screenshot |

### Morphological Feature Analysis

Extract 100+ quantitative features from SWC files for clustering, PCA, and statistical analysis:

```python
import axonet

neuron = axonet.load_swc("neuron.swc")
analyzer = axonet.MorphologyAnalyzer(neuron)
features = analyzer.compute_all_features()
```

Features span 7 categories: basic morphology, volumetric/spatial, graph-theoretic, fractal/complexity, neuron-specific, statistical distributions, and spatial orientation. See [tarpit/analysis.md](tarpit/analysis.md) for the complete feature catalog.

### High-Performance Rendering Pipeline

Generate training datasets with 2D projections using headless GPU rendering (ModernGL):

```bash
python -m axonet.training.dataset_generator \
    --swc-dir data/neurons \
    --out data/rendered \
    --sampling pca \
    --canonical-views 6 \
    --biased-views 12 \
    --random-views 6 \
    --width 512 --height 512 \
    --supersample-factor 2 \
    --adaptive-framing \
    -j 8
```

The PCA-guided camera sampling provides:
- **6 canonical views**: +/- along each principal component axis
- **12 biased views**: concentrated near the PC1-PC2 plane (maximum projected area)
- **6 random views**: uniform sampling for diversity

Each view generates a binary mask (model input), segmentation mask (soma/axon/dendrite labels), and depth map.

### 3D Mesh Export

Convert neuron morphologies to standard mesh formats:

```python
from axonet.visualization.mesh import MeshRenderer
from axonet.io import load_swc

neuron = load_swc("neuron.swc")
renderer = MeshRenderer(neuron)
scene = renderer.build_scene(segments=32, colorize=True)
scene.export("neuron.glb")  # Also supports .ply, .stl, .obj
```

### Colab Notebooks

Interactive notebooks for exploring the library:

- **[axonet_colab.ipynb](notebooks/axonet_colab.ipynb)** - Interactive 3D viewer in browser
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gileshall/axonet/blob/main/notebooks/axonet_colab.ipynb)

- **[axonet_training_tutorial.ipynb](notebooks/axonet_training_tutorial.ipynb)** - End-to-end training pipeline
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gileshall/axonet/blob/main/notebooks/axonet_training_tutorial.ipynb)

## Installation

```bash
# Basic installation
pip install -e .

# With visualization tools (pyglet viewer, plotly)
pip install -e ".[viz]"

# With CLIP text encoders (sentence-transformers)
pip install -e ".[clip]"

# With cloud training support (Google Cloud)
pip install -e ".[cloud]"

# Full installation
pip install -e ".[dev,viz,cloud,clip]"
```

**System requirements:**
- Python 3.10+
- OpenGL 3.3+ capable GPU (for rendering)
- For headless rendering in Colab/cloud: `apt-get install libegl1-mesa-dev libgles2-mesa-dev`

## Quick Start

```python
import axonet
from pathlib import Path

# Load a neuron from SWC file
neuron = axonet.load_swc("path/to/neuron.swc")

# Explore the tree structure
print(f"Nodes: {len(neuron.nodes)}")
print(f"Soma position: {neuron.soma.x}, {neuron.soma.y}, {neuron.soma.z}")

# Compute morphological features
analyzer = axonet.MorphologyAnalyzer(neuron)
features = analyzer.compute_all_features()
print(f"Total dendritic length: {features['total_dendritic_length']:.1f} µm")
print(f"Number of branch points: {features['n_branch_points']}")

# Export as mesh
from axonet.visualization.mesh import MeshRenderer
mesh = MeshRenderer(neuron)
mesh.render_to_file("neuron.ply")
```

## Training Pipeline

> **Note:** Pre-trained weights are not yet available. The pipeline is fully functional for training on your own data.

### 1. Download Neurons from NeuroMorpho.org

```bash
python -m axonet.utils.neuromorpho_bulk \
    --query 'species:mouse' \
    --out data/neuromorpho \
    --fetch-morphometry \
    --find \
    --insecure
```

Or with complex filters:

```bash
python -m axonet.utils.neuromorpho_bulk \
    --filters '{"domain": ["Dendrites, Soma, Axon"], "species": ["mouse"]}' \
    --out data/neuromorpho \
    --fetch-morphometry \
    --find \
    --insecure
```

### 2. Generate Training Dataset

```bash
python -m axonet.training.dataset_generator \
    --swc-dir data/neuromorpho/swc \
    --out data/rendered \
    --sampling pca \
    --adaptive-framing \
    --width 512 --height 512 \
    --val-ratio 0.15 \
    --supersample-factor 4 \
    -j 8
```

### 3. Train Stage 1: Segmentation VAE

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
    --save-dir checkpoints/stage1 \
    --log-dir logs/stage1
```

### 4. Train Stage 2: CLIP Fine-tuning

```bash
python -m axonet.training.clip_trainer \
    --stage1-checkpoint checkpoints/stage1/best.ckpt \
    --data-dir data/rendered \
    --metadata data/neuromorpho/metadata.jsonl \
    --source neuromorpho \
    --batch-size 64 \
    --clip-embed-dim 512 \
    --temperature 0.07 \
    --learnable-temperature \
    --text-encoder distilbert-base-uncased \
    --max-epochs 50 \
    --save-dir checkpoints/clip \
    --log-dir logs/clip
```

### 5. Evaluate

```bash
python -m axonet.training.clip_evaluator \
    --checkpoint checkpoints/clip/best.ckpt \
    --data-dir data/rendered \
    --metadata data/neuromorpho/metadata.jsonl \
    --source neuromorpho \
    --output-dir eval_results \
    --pooling mean
```

See [TRAINING.md](TRAINING.md) for detailed documentation.

## Architecture Overview

```
axonet/
├── core.py                 # Neuron and SWCNode data structures
├── io.py                   # SWC file parsing (with NeuronClass enum)
├── analysis.py             # 100+ morphological features
├── models/
│   ├── d3_swc_vae.py       # SegVAE2D: U-Net + variational skip connections
│   ├── clip_modules.py     # CLIP projection heads
│   └── text_encoders.py    # Hash and sentence-transformer encoders
├── training/
│   ├── trainer.py          # Stage 1: VAE training (PyTorch Lightning)
│   ├── clip_trainer.py     # Stage 2: CLIP fine-tuning
│   ├── clip_evaluator.py   # Retrieval metrics, t-SNE, zero-shot eval
│   ├── dataset_generator.py # Render SWC → 2D projections
│   └── sampling.py         # Camera sampling (fibonacci, PCA-guided)
├── visualization/
│   ├── render.py           # Headless ModernGL renderer
│   ├── pyglet_swc_viewer.py # Interactive 3D viewer
│   ├── mesh.py             # Trimesh-based mesh generation
│   └── sweep.py            # Tube mesh sweep algorithm
├── utils/
│   ├── neuromorpho_bulk.py # NeuroMorpho.org bulk downloader
│   └── allen_bulk.py       # Allen Brain Institute data
└── cloud/                  # Google Cloud Batch integration
```

## Documentation

- **[TRAINING.md](TRAINING.md)** - Training pipeline, CLI reference, troubleshooting
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Architecture deep-dive, rendering pipeline, extending the library
- **[swc_spec.md](swc_spec.md)** - SWC file format specification

## Data Sources

### NeuroMorpho.org
The world's largest collection of publicly available 3D neuronal reconstructions (~200,000+ neurons from 800+ archives). Used for Stage 1 pre-training to learn general morphological representations.

### Allen Brain Institute
Patch-seq data combining single-cell morphology with transcriptomic (T-type) labels. Used for Stage 2 CLIP fine-tuning to learn semantic embeddings that link morphology with cell type annotations.

## Roadmap

- [ ] Release pre-trained model weights
- [ ] Bring back interactive 3D viewer notebook
- [ ] Web-based viewer
- [ ] Additional data source adapters (MouseLight, etc.)
- [ ] Inference API for embedding extraction

## License

MIT License

## Contributing

Contributions welcome! See [DEVELOPMENT.md](DEVELOPMENT.md) for architecture details and development setup.

---

*Developed at the [Broad Institute](https://www.broadinstitute.org/) by Giles Hall*
