"""
Enhanced batch analysis with multiprocessing support.
"""

import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from .io import load_swc
from .analysis import MorphologyAnalyzer


def analyze_single_neuron(filepath: Union[str, Path], 
                         include_metadata: bool = True) -> Dict[str, Any]:
    """
    Analyze a single neuron file.
    
    Args:
        filepath: Path to SWC file
        include_metadata: Whether to include file metadata
        
    Returns:
        Dictionary with analysis results and metadata
    """
    try:
        # Load neuron
        neuron = load_swc(filepath)
        
        # Analyze morphology
        analyzer = MorphologyAnalyzer(neuron)
        features = analyzer.compute_all_features()
        
        # Add metadata
        result = {
            'filename': Path(filepath).name,
            'filepath': str(filepath),
            'n_nodes': len(neuron),
            'n_axons': len(neuron.axon_nodes),
            'n_dendrites': len(neuron.dendrite_nodes),
            'n_terminals': len(neuron.terminal_nodes),
            'n_branch_points': len(neuron.branch_points),
            'analysis_successful': True,
            'error_message': None
        }
        
        if include_metadata:
            result['metadata'] = neuron.metadata
        
        # Add all computed features
        result.update(features)
        
        return result
        
    except Exception as e:
        return {
            'filename': Path(filepath).name,
            'filepath': str(filepath),
            'analysis_successful': False,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }


def analyze_batch_parallel(filepaths: List[Union[str, Path]], 
                          n_workers: Optional[int] = None,
                          include_metadata: bool = True,
                          progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
    """
    Analyze multiple neurons in parallel using multiprocessing.
    
    Args:
        filepaths: List of SWC file paths
        n_workers: Number of worker processes (default: CPU count)
        include_metadata: Whether to include file metadata
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of analysis results
    """
    if n_workers is None:
        n_workers = min(mp.cpu_count(), len(filepaths))
    
    print(f"Analyzing {len(filepaths)} neurons using {n_workers} workers...")
    
    results = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(analyze_single_neuron, path, include_metadata): path 
            for path in filepaths
        }
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_path), 1):
            try:
                result = future.result()
                results.append(result)
                
                if result['analysis_successful']:
                    successful += 1
                    print(f"✓ [{i}/{len(filepaths)}] {result['filename']} - {result['n_nodes']} nodes")
                else:
                    failed += 1
                    print(f"✗ [{i}/{len(filepaths)}] {result['filename']} - {result['error_message']}")
                
                # Progress callback
                if progress_callback:
                    progress_callback(i, len(filepaths), successful, failed)
                    
            except Exception as e:
                failed += 1
                error_result = {
                    'filename': Path(future_to_path[future]).name,
                    'filepath': str(future_to_path[future]),
                    'analysis_successful': False,
                    'error_message': f"Unexpected error: {str(e)}",
                    'traceback': traceback.format_exc()
                }
                results.append(error_result)
                print(f"✗ [{i}/{len(filepaths)}] {error_result['filename']} - {error_result['error_message']}")
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Successfully analyzed: {successful}")
    print(f"Failed: {failed}")
    print(f"Average time per neuron: {elapsed_time/len(filepaths):.2f}s")
    
    return results


def find_swc_files(input_paths: List[Union[str, Path]], 
                   recursive: bool = False) -> List[Path]:
    """
    Find all SWC files in given paths.
    
    Args:
        input_paths: List of files or directories to search
        recursive: Whether to search recursively in directories
        
    Returns:
        List of SWC file paths
    """
    swc_files = []
    
    for input_path in input_paths:
        path = Path(input_path)
        
        if path.is_file() and path.suffix.lower() == '.swc':
            swc_files.append(path)
        elif path.is_dir():
            if recursive:
                swc_files.extend(path.rglob("*.swc"))
            else:
                swc_files.extend(path.glob("*.swc"))
        else:
            print(f"Warning: {path} is not a valid file or directory")
    
    return sorted(swc_files)


def save_results(results: List[Dict[str, Any]], 
                output_path: Union[str, Path],
                format: str = "json",
                include_failed: bool = True) -> None:
    """
    Save analysis results to file.
    
    Args:
        results: List of analysis results
        output_path: Output file path
        format: Output format ('json' or 'csv')
        include_failed: Whether to include failed analyses
    """
    output_path = Path(output_path)
    
    # Filter results if needed
    if not include_failed:
        results = [r for r in results if r.get('analysis_successful', False)]
    
    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format.lower() == 'csv':
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Saved {len(results)} results to {output_path}")


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics of analysis results."""
    successful_results = [r for r in results if r.get('analysis_successful', False)]
    
    if not successful_results:
        print("No successful analyses to summarize")
        return
    
    print(f"\nSummary Statistics ({len(successful_results)} successful analyses):")
    
    # Key features to summarize
    key_features = [
        'total_neurite_length',
        'n_branch_points', 
        'n_terminals',
        'total_volume',
        'n_nodes'
    ]
    
    for feature in key_features:
        values = [r.get(feature) for r in successful_results if feature in r and r[feature] is not None]
        if values:
            import numpy as np
            print(f"  {feature}:")
            print(f"    Mean: {np.mean(values):.4f}")
            print(f"    Std:  {np.std(values):.4f}")
            print(f"    Min:  {np.min(values):.4f}")
            print(f"    Max:  {np.max(values):.4f}")


def main():
    """Main CLI entry point for batch analysis."""
    parser = argparse.ArgumentParser(
        description="Batch analysis of neuron morphology files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all SWC files in a directory
  python -m axonet.batch_analysis morph/ -o results.json
  
  # Analyze specific files with 8 workers
  python -m axonet.batch_analysis file1.swc file2.swc -o results.csv -f csv -w 8
  
  # Recursive search with progress updates
  python -m axonet.batch_analysis data/ -o results.json -r --progress
        """
    )
    
    parser.add_argument('inputs', nargs='+', 
                       help='SWC files or directories to analyze')
    parser.add_argument('-o', '--output', required=True,
                       help='Output file path')
    parser.add_argument('-f', '--format', choices=['json', 'csv'], 
                       default='json', help='Output format')
    parser.add_argument('-w', '--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count)')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Search directories recursively')
    parser.add_argument('--no-metadata', action='store_true',
                       help='Exclude file metadata from results')
    parser.add_argument('--exclude-failed', action='store_true',
                       help='Exclude failed analyses from output')
    parser.add_argument('--progress', action='store_true',
                       help='Show detailed progress updates')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    try:
        # Find SWC files
        print("Finding SWC files...")
        swc_files = find_swc_files(args.inputs, recursive=args.recursive)
        
        if not swc_files:
            print("No SWC files found")
            return 1
        
        print(f"Found {len(swc_files)} SWC files")
        
        # Limit files if requested
        if args.max_files and len(swc_files) > args.max_files:
            swc_files = swc_files[:args.max_files]
            print(f"Limited to first {len(swc_files)} files")
        
        # Progress callback
        def progress_callback(completed, total, successful, failed):
            if args.progress:
                print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) - "
                      f"✓ {successful} ✗ {failed}")
        
        # Run analysis
        results = analyze_batch_parallel(
            swc_files,
            n_workers=args.workers,
            include_metadata=not args.no_metadata,
            progress_callback=progress_callback if args.progress else None
        )
        
        # Save results
        save_results(results, args.output, args.format, not args.exclude_failed)
        
        # Print summary
        print_summary(results)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
