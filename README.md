# Axonet

A comprehensive library for neuron morphology analysis from SWC files.

## Features

- **SWC File Loading**: Robust parsing of SWC neuronal morphology files
- **Graph Representation**: NetworkX-based graph structures for analysis
- **Comprehensive Analysis**: 100+ morphological features including:
  - Basic morphology (length, branch, path metrics)
  - Volumetric and spatial measurements
  - Graph-theoretic properties
  - Fractal and complexity measures
  - Statistical distributions
  - Spatial orientation features
- **Visualization**: Multiple rendering options:
  - 3D mesh generation (PLY, STL, OBJ)
  - ANSI art terminal visualization
- **Batch Processing**: Analyze multiple neurons efficiently

## Installation

```bash
pip install -e .
```

For development with additional visualization tools:
```bash
pip install -e ".[dev,viz]"
```

## Quick Start

```python
import axonet

# Load a neuron
neuron = axonet.load_swc("path/to/neuron.swc")

# Analyze morphology
analyzer = axonet.MorphologyAnalyzer(neuron)
features = analyzer.compute_all_features()

# Visualize
renderer = axonet.ANSIRenderer(neuron)
print(renderer.visualize())

# Export 3D mesh
mesh_renderer = axonet.MeshRenderer(neuron)
mesh_renderer.render_to_file("neuron.ply")
```

## API Reference

### Core Classes

- `Neuron`: Main neuron representation with graph structure
- `SWCNode`: Individual node representation
- `MorphologyAnalyzer`: Comprehensive feature extraction
- `MeshRenderer`: 3D mesh generation
- `ANSIRenderer`: Terminal visualization

### Key Functions

- `load_swc()`: Load neuron from SWC file
- `save_swc()`: Save neuron to SWC file
- `load_multiple_swc()`: Batch loading

## Analysis Features

The library computes over 100 morphological features across 7 categories:

1. **Basic Morphology** (~20-30 features)
   - Length metrics, branch counts, path distances
2. **Volumetric/Spatial** (~15-20 features)
   - Volume, surface area, bounding box, radius statistics
3. **Graph-Theoretic** (~15-20 features)
   - Connectivity, centrality measures, tree properties
4. **Fractal/Complexity** (~10-15 features)
   - Sholl analysis, fractal dimension, complexity indices
5. **Neuron-Specific** (~10-15 features)
   - Soma characteristics, compartment ratios
6. **Statistical Distributions** (~15-20 features)
   - Skewness, kurtosis of various measurements
7. **Spatial Orientation** (~8-10 features)
   - Principal components, planarity, directionality

## Examples

See the `examples/` directory for detailed usage examples.

## License

MIT License

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
