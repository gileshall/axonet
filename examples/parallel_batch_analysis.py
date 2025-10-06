#!/usr/bin/env python3
"""
Parallel batch analysis example using the enhanced AxonNet library.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from axonet.batch_analysis import analyze_batch_parallel, find_swc_files, save_results, print_summary


def main():
    """Run parallel batch analysis example."""
    morph_dir = Path("morph")
    
    if not morph_dir.exists():
        print(f"Directory not found: {morph_dir}")
        return 1
    
    # Find all SWC files
    print("Finding SWC files...")
    swc_files = find_swc_files([morph_dir], recursive=False)
    print(f"Found {len(swc_files)} SWC files")
    
    if not swc_files:
        print("No SWC files found")
        return 1
    
    # Test with first 10 files for demonstration
    test_files = swc_files[:10]
    print(f"Testing with {len(test_files)} files")
    
    # Compare sequential vs parallel performance
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    # Sequential analysis (using original method)
    print("\nSequential Analysis:")
    start_time = time.time()
    
    from axonet.io import load_swc
    from axonet.analysis import MorphologyAnalyzer
    
    sequential_results = []
    for i, swc_file in enumerate(test_files):
        try:
            neuron = load_swc(swc_file)
            analyzer = MorphologyAnalyzer(neuron)
            features = analyzer.compute_all_features()
            features['filename'] = swc_file.name
            features['n_nodes'] = len(neuron)
            sequential_results.append(features)
            print(f"  [{i+1}/{len(test_files)}] {swc_file.name}")
        except Exception as e:
            print(f"  [{i+1}/{len(test_files)}] {swc_file.name} - Error: {e}")
    
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.2f} seconds")
    
    # Parallel analysis
    print(f"\nParallel Analysis (4 workers):")
    start_time = time.time()
    
    parallel_results = analyze_batch_parallel(
        test_files, 
        n_workers=4,
        include_metadata=False
    )
    
    parallel_time = time.time() - start_time
    print(f"Parallel time: {parallel_time:.2f} seconds")
    
    # Performance comparison
    speedup = sequential_time / parallel_time if parallel_time > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Time saved: {sequential_time - parallel_time:.2f} seconds")
    
    # Save results
    print(f"\nSaving results...")
    save_results(parallel_results, "parallel_analysis_results.json", "json")
    
    # Print summary
    print_summary(parallel_results)
    
    # Demonstrate different command line options
    print("\n" + "="*60)
    print("COMMAND LINE EXAMPLES")
    print("="*60)
    print("1. Basic parallel analysis:")
    print("   python -m axonet.cli batch-parallel morph/ -o results.json")
    print()
    print("2. With custom worker count and CSV output:")
    print("   python -m axonet.cli batch-parallel morph/ -o results.csv -f csv -w 8")
    print()
    print("3. Recursive search with progress updates:")
    print("   python -m axonet.cli batch-parallel data/ -o results.json -r --progress")
    print()
    print("4. Analyze specific files:")
    print("   python -m axonet.cli batch-parallel file1.swc file2.swc -o results.json")
    print()
    print("5. Limited analysis for testing:")
    print("   python -m axonet.cli batch-parallel morph/ -o results.json --max-files 5")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
