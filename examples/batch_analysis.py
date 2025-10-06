#!/usr/bin/env python3
"""
Batch analysis example for multiple neurons.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import axonet
import numpy as np


def main():
    """Run batch analysis example."""
    morph_dir = Path("morph")
    
    if not morph_dir.exists():
        print(f"Directory not found: {morph_dir}")
        return 1
    
    # Find all SWC files
    swc_files = list(morph_dir.glob("*.swc"))
    print(f"Found {len(swc_files)} SWC files")
    
    if not swc_files:
        print("No SWC files found")
        return 1
    
    # Analyze first 5 files as example
    swc_files = swc_files[:5]
    print(f"Analyzing {len(swc_files)} files...")
    
    all_features = []
    failed_files = []
    
    for i, swc_file in enumerate(swc_files):
        print(f"Processing {i+1}/{len(swc_files)}: {swc_file.name}")
        
        try:
            # Load neuron
            neuron = axonet.load_swc(swc_file)
            
            # Analyze
            analyzer = axonet.MorphologyAnalyzer(neuron)
            features = analyzer.compute_all_features()
            features['filename'] = swc_file.name
            features['n_nodes'] = len(neuron)
            
            all_features.append(features)
            print(f"  ✓ Computed {len(features)} features")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_files.append(swc_file.name)
            continue
    
    print(f"\nSuccessfully analyzed {len(all_features)} neurons")
    if failed_files:
        print(f"Failed to analyze {len(failed_files)} files: {failed_files}")
    
    if not all_features:
        print("No neurons successfully analyzed")
        return 1
    
    # Save results
    output_file = "batch_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_features, f, indent=2)
    print(f"Saved results to {output_file}")
    
    # Show summary statistics
    print("\nSummary statistics:")
    
    # Convert to array for easier analysis
    feature_names = list(all_features[0].keys())
    feature_names = [f for f in feature_names if f != 'filename' and isinstance(all_features[0][f], (int, float))]
    
    print(f"Analyzed {len(all_features)} neurons with {len(feature_names)} features each")
    
    # Show some key statistics
    key_features = ['total_neurite_length', 'n_branch_points', 'n_terminals', 'total_volume']
    
    for feature in key_features:
        if feature in feature_names:
            values = [f[feature] for f in all_features if feature in f]
            if values:
                print(f"  {feature}:")
                print(f"    Mean: {np.mean(values):.4f}")
                print(f"    Std:  {np.std(values):.4f}")
                print(f"    Min:  {np.min(values):.4f}")
                print(f"    Max:  {np.max(values):.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
