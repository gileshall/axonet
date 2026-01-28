# Development Guide

This document provides a deep-dive into axonet's architecture for developers who want to understand, extend, or contribute to the codebase.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              AXONET                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   │
│  │   Data I/O   │    │ Visualization│    │        Training          │   │
│  ├──────────────┤    ├──────────────┤    ├──────────────────────────┤   │
│  │ core.py      │    │ render.py    │    │ trainer.py (Stage 1)     │   │
│  │ io.py        │───▶│ mesh.py      │───▶│ clip_trainer.py (Stage 2)│   │
│  │ analysis.py  │    │ sweep.py     │    │ dataset_generator.py     │   │
│  └──────────────┘    │ pyglet_*.py  │    │ clip_evaluator.py        │   │
│                      └──────────────┘    └──────────────────────────┘   │
│                             │                        │                  │
│                             ▼                        ▼                  │
│                      ┌──────────────┐    ┌──────────────────────────┐   │
│                      │   Models     │    │        Utilities         │   │
│                      ├──────────────┤    ├──────────────────────────┤   │
│                      │ d3_swc_vae.py│    │ neuromorpho_bulk.py      │   │
│                      │ clip_modules │    │ allen_bulk.py            │   │
│                      │ text_encoders│    │ metadata_adapters.py     │   │
│                      └──────────────┘    └──────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### Neuron and SWCNode (`core.py`)

The foundation of axonet is a clean representation of neuronal morphology:

```python
@dataclass
class SWCNode:
    """Single node in the neuron tree."""
    index: int           # Unique identifier (1-indexed in SWC spec)
    type_id: int         # 1=soma, 2=axon, 3=basal, 4=apical, 5+=custom
    x: float             # Position in micrometers
    y: float
    z: float
    radius: float        # Half the node thickness
    parent: int          # Parent index (-1 for root)
    children: List[int]  # Filled during tree construction

    @property
    def is_soma(self) -> bool: return self.type_id == 1
    @property
    def is_axon(self) -> bool: return self.type_id == 2
    @property
    def is_terminal(self) -> bool: return len(self.children) == 0
    @property
    def is_branch_point(self) -> bool: return len(self.children) > 1
```

```python
class Neuron:
    """Complete neuron as a tree of SWCNodes."""
    nodes: Dict[int, SWCNode]  # index -> node
    soma: SWCNode              # Convenience reference to soma node

    def get_children(self, idx: int) -> List[int]
    def get_parent(self, idx: int) -> int
    def get_path_to_root(self, idx: int) -> List[int]
    def get_subtree(self, idx: int) -> List[int]
    def get_branch_order(self, idx: int) -> int
    def get_euclidean_distance(self, idx1: int, idx2: int) -> float
    def get_path_distance(self, idx1: int, idx2: int) -> float
    def to_networkx(self) -> nx.DiGraph
```

The `NeuronClass` enum in `io.py` provides semantic labels:
```python
class NeuronClass(Enum):
    SOMA = 1
    AXON = 2
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4
    OTHER = 5
```

## Rendering Pipeline

The rendering system has two implementations:
1. **`render.py`**: Headless ModernGL renderer for batch dataset generation
2. **`pyglet_swc_viewer.py`**: Interactive windowed viewer with Phong shading

### Headless Renderer (`render.py`)

Uses ModernGL for clean GPU resource management without a display.

**Key classes:**

```python
class OffscreenContext:
    """ModernGL standalone context wrapper."""
    ctx: moderngl.Context

class Camera:
    """Manages view and projection matrices."""
    eye: np.ndarray      # Camera position
    target: np.ndarray   # Look-at point
    up: np.ndarray       # Up vector
    perspective: bool    # Perspective vs orthographic
    fov_y: float         # Vertical FOV (perspective)
    ortho_scale: float   # Scale (orthographic)
    near: float
    far: float

class NeuroRenderCore:
    """Main renderer: loads SWC, builds meshes, renders images."""
    def load_swc(self, path: Path, ...)
    def render_class_id_mask(self) -> np.ndarray  # (H, W) uint8
    def render_depth(self) -> np.ndarray          # (H, W) float32
    def qc_fraction_in_frame(self) -> float       # Quality metric
```

### The Supersampling Challenge

A critical engineering challenge was avoiding pixel dropouts in segmentation masks. Standard rasterization misses thin neurites at grazing angles because fragments are only generated when triangles cover pixel centers.

**The solution: Supersampling with majority pooling**

```python
def render_class_id_mask_supersampled(self, factor: int = 2) -> np.ndarray:
    """Render at factor×resolution, then majority-pool down."""
    # 1. Render at 2x or 4x resolution
    mask_hi = self._render_mask_highres(self.width * factor, self.height * factor)

    # 2. Majority pooling: for each factor×factor block, take most common non-zero value
    return majority_pool_uint8(mask_hi, factor, prefer_nonzero=True)
```

The `prefer_nonzero=True` ensures that if ANY subpixel hits a neurite, that class wins—preserving connectivity even at grazing angles.

**Why not MSAA?** MSAA averages values during resolve, which destroys integer class IDs (average of class 2 and 3 is not a valid class). Integer textures can't use MSAA directly.

### Depth pooling

```python
def average_pool_depth(depth_hi: np.ndarray, factor: int, prefer_valid: bool = True) -> np.ndarray:
    """Downsample depth by averaging, ignoring background pixels."""
    # Background pixels have depth ~1.0 (far plane)
    # Only average valid (foreground) depths to avoid edge blurring
    valid = blocks < 0.999
    result = np.nanmean(np.where(valid, blocks, np.nan), axis=2)
```

### Mesh Generation (`mesh.py`, `sweep.py`)

Neurons are rendered as tube meshes generated by sweeping circles along paths:

```python
class MeshRenderer:
    """Converts Neuron to triangle meshes via tube sweeping."""

    def build_mesh_by_type(self) -> Dict[NeuronClass, trimesh.Trimesh]:
        """Build separate meshes for each compartment type."""
        for cls in [SOMA, AXON, BASAL_DENDRITE, APICAL_DENDRITE]:
            edges = self._get_edges_of_type(cls)
            mesh = sweep_tubes(edges, segments=32)
            meshes[cls] = mesh
        return meshes
```

The `sweep.py` module implements the tube sweep algorithm:
1. For each edge (parent→child), compute a circular cross-section at each endpoint
2. Connect the circles with triangle strips
3. Handle varying radii (taper) along the neurite
4. Optionally cap tube ends

### Camera Sampling (`sampling.py`)

PCA-guided sampling maximizes the information in each view:

```python
def pca_guided_sampling(positions: np.ndarray, n_canonical=6, n_biased=12, n_random=6):
    """
    Three tiers of camera directions:
    1. Canonical: +/- PC1, PC2, PC3 (guaranteed broadside views)
    2. Biased: concentrated near PC1-PC2 plane (maximum projected area)
    3. Random: uniform on S² for diversity
    """
    eigenvectors, eigenvalues, center = compute_neuron_pca(positions)

    # Canonical: the 6 principal directions
    canonical = np.vstack([eigenvectors.T, -eigenvectors.T])

    # Biased: sample near the PC1-PC2 plane using rejection sampling
    # or by perturbing vectors in the PC1-PC2 plane

    # Random: uniform on sphere
    random_dirs = random_sphere(n_random)

    return np.vstack([canonical, biased, random_dirs])
```

## Model Architecture

### SegVAE2D (`models/d3_swc_vae.py`)

A variational U-Net with a novel **variational skip connections** mechanism:

```
Input (1, H, W)
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ ENCODER                                                       │
│                                                               │
│  Conv Block (64)  ──────────────────────────────▶ Skip e0     │
│       │                                              │        │
│       ▼                                              ▼        │
│  Conv Block (128) ──────────────────────────────▶ Skip e1     │
│       │                                              │        │
│       ▼                                              ▼        │
│  Conv Block (256) ──────────────────────────────▶ Skip e2     │
│       │                                              │        │
│       ▼                                                       │
│  Conv Block (512) ─────▶ μ, σ ─────▶ z (reparameterize)       │
│                              ▲                                │
│                              │                                │
│                         KL divergence                         │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ VARIATIONAL SKIPS (key innovation)                            │
│                                                               │
│  Each skip connection e0, e1, e2 passes through its own       │
│  variational layer: μ_skip, σ_skip → z_skip                   │
│                                                               │
│  This prevents the decoder from bypassing the stochastic      │
│  bottleneck by "cheating" through deterministic skips.        │
│                                                               │
│  Total KL = KL_bottleneck + KL_skip0 + KL_skip1 + KL_skip2    │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ DECODER                                                       │
│                                                               │
│  Upsample + Concat z_skip2 ──▶ Conv Block                     │
│       │                                                       │
│       ▼                                                       │
│  Upsample + Concat z_skip1 ──▶ Conv Block                     │
│       │                                                       │
│       ▼                                                       │
│  Upsample + Concat z_skip0 ──▶ Conv Block                     │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────────────────────────────┐                  │
│  │ Segmentation Head (6 classes)           │──▶ (6, H, W)     │
│  │ Depth Head (1 channel)                  │──▶ (1, H, W)     │
│  └─────────────────────────────────────────┘                  │
└───────────────────────────────────────────────────────────────┘
```

**Why variational skips?**

In a standard VAE, skip connections let the decoder access high-resolution features directly from the encoder, bypassing the stochastic bottleneck. This makes the latent code z less useful—the model can reconstruct without it.

Variational skips force each skip connection through its own μ/σ parameterization, adding stochasticity at multiple scales. The model must encode information in the latent distribution at every level.

### Loss Function

```python
loss = λ_seg * CrossEntropy(pred_seg, gt_seg)
     + λ_depth * L1(pred_depth, gt_depth)
     + β * (KL_bottleneck + KL_skip0 + KL_skip1 + KL_skip2 - free_nats).clamp(min=0)
```

The `free_nats` term implements "free bits" to prevent KL collapse—the first N nats of KL are "free" (not penalized), ensuring the latent space remains useful.

### CLIP Extension (`models/clip_modules.py`)

Stage 2 adds projection heads to map both image and text embeddings to a shared space:

```python
class CLIPProjectionHead(nn.Module):
    """Maps VAE bottleneck to CLIP embedding space."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.gelu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return F.normalize(x, p=2, dim=-1)  # L2 normalize
```

### Text Encoders (`models/text_encoders.py`)

Two options:

1. **HashTextEncoder**: Lightweight, no external dependencies
   - N-gram feature hashing (1, 2, 3-grams)
   - Deterministic, fast
   - Good for debugging

2. **SentenceTransformerEncoder**: Semantic understanding
   - Uses `sentence-transformers` library
   - Pre-trained on large text corpora
   - Better text-image alignment

### InfoNCE Loss (`training/losses/infonce.py`)

```python
def forward(self, image_embeds, text_embeds, temperature):
    # Cosine similarity matrix
    logits = image_embeds @ text_embeds.T / temperature

    # Symmetric cross-entropy
    labels = torch.arange(len(logits), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2
```

## Data Adapters

The `MetadataAdapter` pattern normalizes heterogeneous data sources:

```python
class MetadataAdapter(ABC):
    @abstractmethod
    def get_neuron_id(self, record: Dict) -> str: ...
    @abstractmethod
    def get_cell_type(self, record: Dict) -> str: ...
    @abstractmethod
    def get_brain_region(self, record: Dict) -> str: ...
    @abstractmethod
    def get_species(self, record: Dict) -> str: ...
    @abstractmethod
    def to_text_description(self, record: Dict, level: str) -> str: ...

class NeuromorphoAdapter(MetadataAdapter):
    """Adapter for NeuroMorpho.org metadata."""

class AllenAdapter(MetadataAdapter):
    """Adapter for Allen Brain Institute patch-seq data."""
```

## PyTorch Lightning Integration

Training uses PyTorch Lightning for:
- Automatic GPU/multi-GPU handling
- Gradient accumulation
- Mixed precision
- Checkpointing
- Logging (TensorBoard, W&B)

```python
class SegVAE2DLightning(LightningModule):
    def training_step(self, batch, batch_idx):
        out = self.model(batch["input"])
        loss = self.loss_fn(out, batch)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [sched]
```

## Cloud Infrastructure

### Provider Abstraction (`cloud/`)

```python
class CloudProvider(ABC):
    @abstractmethod
    def storage(self) -> StorageBackend: ...
    @abstractmethod
    def compute(self) -> ComputeBackend: ...
    @abstractmethod
    def batch(self) -> BatchBackend: ...

class GoogleCloudProvider(CloudProvider):
    """Google Cloud implementation."""

class LocalProvider(CloudProvider):
    """Local subprocess fallback for development."""
```

### Job Configuration

```python
@dataclass
class JobConfig:
    command: List[str]
    image: str
    cpu: int
    memory_gb: int
    gpu_type: Optional[str]
    gpu_count: int
    disk_gb: int
    env: Dict[str, str]
    input_paths: List[str]
    output_path: str
    retries: int
```

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/gileshall/axonet.git
cd axonet
pip install -e ".[dev,viz,clip]"

# Run tests
pytest

# Format code
black axonet/
isort axonet/

# Type checking
mypy axonet/
```

## Directory Structure

```
axonet/
├── __init__.py             # Public API exports
├── core.py                 # Neuron, SWCNode (257 lines)
├── io.py                   # SWC parsing, NeuronClass enum
├── analysis.py             # MorphologyAnalyzer (446 lines)
├── models/
│   ├── __init__.py
│   ├── d3_swc_vae.py       # SegVAE2D model (604 lines)
│   ├── clip_modules.py     # CLIP projection heads
│   └── text_encoders.py    # Hash and transformer encoders
├── training/
│   ├── __init__.py
│   ├── trainer.py          # Stage 1 VAE training (894 lines)
│   ├── clip_trainer.py     # Stage 2 CLIP training (827 lines)
│   ├── clip_dataset.py     # CLIP dataset + text generation (488 lines)
│   ├── clip_evaluator.py   # Evaluation suite (1057 lines)
│   ├── dataset_generator.py # Rendering pipeline (495 lines)
│   ├── sampling.py         # Camera sampling strategies
│   ├── callbacks.py        # Lightning callbacks
│   └── losses/
│       └── infonce.py      # Contrastive loss
├── visualization/
│   ├── __init__.py
│   ├── render.py           # ModernGL headless renderer (928 lines)
│   ├── pyglet_swc_viewer.py # Interactive viewer (785 lines)
│   ├── mesh.py             # Trimesh generation (425 lines)
│   └── sweep.py            # Tube sweep algorithm
├── data/
│   └── metadata_adapters.py # Data source adapters
├── utils/
│   ├── neuromorpho_bulk.py # NeuroMorpho.org downloader (440 lines)
│   └── allen_bulk.py       # Allen Brain data
└── cloud/
    ├── __init__.py
    ├── cli.py              # Cloud CLI (404 lines)
    ├── provider.py         # Provider abstraction
    ├── local.py            # Local fallback
    └── google/
        ├── storage.py      # GCS
        ├── batch.py        # Cloud Batch
        └── compute.py      # GCE
```

## Key Design Decisions

1. **2D projections over 3D voxels**: Voxelizing neurons is expensive and loses fine detail. 2D projections are cheap to render and the model learns view-invariant features.

2. **Variational skip connections**: Prevents the VAE from ignoring the latent space by forcing stochasticity at all scales.

3. **PCA-guided camera sampling**: Ensures informative views by aligning cameras with the neuron's natural axes.

4. **Supersampling for masks**: OpenGL rasterization misses thin structures. Supersampling with majority pooling preserves connectivity.

5. **CLI-driven training**: Configuration files can become stale. CLI arguments are explicit, composable, and self-documenting via `--help`.

6. **Adapter pattern for data sources**: Different databases have different schemas. Adapters normalize to a common interface.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run `black`, `isort`, `mypy`
5. Submit a pull request

## References

- [ModernGL documentation](https://moderngl.readthedocs.io/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Trimesh](https://trimsh.org/)
- [SWC file format specification](swc_spec.md)
