#!/usr/bin/env python3
"""
Basic analysis example using Axonet library.
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import axonet


def main():
    """Run basic analysis example."""
    # Example SWC file path (you can change this)
    swc_file = "morph/601506507_raw.swc"
    
    if not Path(swc_file).exists():
        print(f"SWC file not found: {swc_file}")
        print("Please provide a valid SWC file path")
        return 1
    
    print(f"Loading neuron from {swc_file}")
    
    # Load neuron
    neuron = axonet.load_swc(swc_file)
    print(f"Loaded neuron: {neuron}")
    
    # Basic information
    print(f"Number of nodes: {len(neuron)}")
    print(f"Soma nodes: {len(neuron.soma_nodes)}")
    print(f"Axon nodes: {len(neuron.axon_nodes)}")
    print(f"Dendrite nodes: {len(neuron.dendrite_nodes)}")
    print(f"Terminal nodes: {len(neuron.terminal_nodes)}")
    print(f"Branch points: {len(neuron.branch_points)}")
    
    # Analyze morphology
    print("\nComputing morphological features...")
    analyzer = axonet.MorphologyAnalyzer(neuron)
    features = analyzer.compute_all_features()
    
    print(f"Computed {len(features)} features")
    
    # Show some key features
    key_features = [
        'total_neurite_length',
        'total_dendritic_length', 
        'total_axonal_length',
        'n_branch_points',
        'n_terminals',
        'max_branch_order',
        'total_volume',
        'fractal_dimension_2d'
    ]
    
    print("\nKey morphological features:")
    for feature in key_features:
        if feature in features:
            print(f"  {feature}: {features[feature]:.4f}")
    
    # Visualize in terminal
    print("\nTerminal visualization:")
    renderer = axonet.ANSIRenderer(neuron)
    print(renderer.visualize(size=64))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
