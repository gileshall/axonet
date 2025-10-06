"""
Command-line interface for AxonNet.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .io import load_swc, load_multiple_swc
from .analysis import MorphologyAnalyzer
from .visualization import ANSIRenderer, MeshRenderer


def analyze_neuron(filepath: str, output: str = None, format: str = "json"):
    """Analyze a single neuron and output features."""
    try:
        neuron = load_swc(filepath)
        analyzer = MorphologyAnalyzer(neuron)
        features = analyzer.compute_all_features()
        
        if output:
            output_path = Path(output)
            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(features, f, indent=2)
            elif format == "csv":
                import pandas as pd
                df = pd.DataFrame([features])
                df.to_csv(output_path, index=False)
            else:
                print(f"Unknown format: {format}")
                return 1
        else:
            print(json.dumps(features, indent=2))
        
        print(f"Analyzed {len(neuron)} nodes, computed {len(features)} features")
        return 0
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}", file=sys.stderr)
        return 1


def analyze_batch(input_dir: str, output: str, format: str = "json"):
    """Analyze multiple neurons in a directory."""
    input_path = Path(input_dir)
    swc_files = list(input_path.glob("*.swc"))
    
    if not swc_files:
        print(f"No SWC files found in {input_dir}", file=sys.stderr)
        return 1
    
    print(f"Found {len(swc_files)} SWC files")
    
    all_features = []
    for swc_file in swc_files:
        try:
            neuron = load_swc(swc_file)
            analyzer = MorphologyAnalyzer(neuron)
            features = analyzer.compute_all_features()
            features['filename'] = swc_file.name
            all_features.append(features)
            print(f"Analyzed {swc_file.name}")
        except Exception as e:
            print(f"Error analyzing {swc_file.name}: {e}", file=sys.stderr)
            continue
    
    if not all_features:
        print("No neurons successfully analyzed", file=sys.stderr)
        return 1
    
    output_path = Path(output)
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(all_features, f, indent=2)
    elif format == "csv":
        import pandas as pd
        df = pd.DataFrame(all_features)
        df.to_csv(output_path, index=False)
    else:
        print(f"Unknown format: {format}")
        return 1
    
    print(f"Saved analysis of {len(all_features)} neurons to {output_path}")
    return 0


def visualize_neuron(filepath: str, size: int = 128, no_color: bool = False):
    """Visualize a neuron in the terminal."""
    try:
        neuron = load_swc(filepath)
        renderer = ANSIRenderer(neuron, size=size, use_color=not no_color)
        output = renderer.visualize(size=size, use_color=not no_color)
        print(output)
        
        # Print legend
        if not no_color:
            from .visualization.ansi import ANSI_COLORS, SWC_TYPE_LABELS, RESET
            print("\nLegend:")
            type_counts = {}
            for node in neuron.nodes.values():
                type_counts[node.type_id] = type_counts.get(node.type_id, 0) + 1
            
            for type_id in sorted(type_counts.keys()):
                color = ANSI_COLORS.get(type_id, '')
                label = SWC_TYPE_LABELS.get(type_id, str(type_id))
                count = type_counts[type_id]
                print(f"{color}■{RESET} {label}: {count} nodes")
        
        return 0
        
    except Exception as e:
        print(f"Error visualizing {filepath}: {e}", file=sys.stderr)
        return 1


def export_mesh(filepath: str, output: str, segments: int = 24, no_cap: bool = False):
    """Export neuron as 3D mesh."""
    try:
        neuron = load_swc(filepath)
        renderer = MeshRenderer(neuron)
        mesh = renderer.render_to_file(output, segments=segments, cap=(not no_cap))
        print(f"Exported mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces to {output}")
        return 0
        
    except Exception as e:
        print(f"Error exporting mesh: {e}", file=sys.stderr)
        return 1


def view_3d(filepath: str, segments: int = 18):
    """Launch interactive 3D viewer."""
    from .visualization.pyglet_swc_viewer import SWCViewer, pyglet
    _ = SWCViewer(Path(filepath), segments=segments)
    pyglet.app.run()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="AxonNet: Neuron morphology analysis")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze neuron morphology')
    analyze_parser.add_argument('input', help='Input SWC file')
    analyze_parser.add_argument('-o', '--output', help='Output file (optional)')
    analyze_parser.add_argument('-f', '--format', choices=['json', 'csv'], default='json', help='Output format')
    
    # Batch analyze command (legacy)
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple neurons (legacy)')
    batch_parser.add_argument('input_dir', help='Directory containing SWC files')
    batch_parser.add_argument('output', help='Output file')
    batch_parser.add_argument('-f', '--format', choices=['json', 'csv'], default='json', help='Output format')
    
    # New parallel batch analyze command
    parallel_parser = subparsers.add_parser('batch-parallel', help='Analyze multiple neurons in parallel')
    parallel_parser.add_argument('inputs', nargs='+', help='SWC files or directories to analyze')
    parallel_parser.add_argument('-o', '--output', required=True, help='Output file path')
    parallel_parser.add_argument('-f', '--format', choices=['json', 'csv'], default='json', help='Output format')
    parallel_parser.add_argument('-w', '--workers', type=int, default=None, help='Number of worker processes')
    parallel_parser.add_argument('-r', '--recursive', action='store_true', help='Search directories recursively')
    parallel_parser.add_argument('--max-files', type=int, default=None, help='Maximum number of files to process')
    parallel_parser.add_argument('--progress', action='store_true', help='Show detailed progress updates')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize neuron in terminal')
    viz_parser.add_argument('input', help='Input SWC file')
    viz_parser.add_argument('-s', '--size', type=int, default=128, help='Grid size')
    viz_parser.add_argument('--no-color', action='store_true', help='Disable colors')
    
    # Export mesh command
    mesh_parser = subparsers.add_parser('mesh', help='Export as 3D mesh')
    mesh_parser.add_argument('input', help='Input SWC file')
    mesh_parser.add_argument('output', help='Output mesh file (e.g. .ply, .obj)')
    mesh_parser.add_argument('--segments', type=int, default=24, help='Segments around branch circumference')
    mesh_parser.add_argument('--no-cap', action='store_true', help='Do not cap branch ends')

    # 3D viewer command
    viewer_parser = subparsers.add_parser('viewer', help='Launch interactive 3D viewer')
    viewer_parser.add_argument('input', help='Input SWC file')
    viewer_parser.add_argument('--segments', type=int, default=18, help='Segments around branch circumference')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        return analyze_neuron(args.input, args.output, args.format)
    elif args.command == 'batch':
        return analyze_batch(args.input_dir, args.output, args.format)
    elif args.command == 'batch-parallel':
        from .batch_analysis import main as batch_main
        # Convert argparse namespace to sys.argv for batch_analysis.main()
        sys.argv = ['batch_analysis'] + args.inputs + ['-o', args.output, '-f', args.format]
        if args.workers:
            sys.argv.extend(['-w', str(args.workers)])
        if args.recursive:
            sys.argv.append('-r')
        if args.max_files:
            sys.argv.extend(['--max-files', str(args.max_files)])
        if args.progress:
            sys.argv.append('--progress')
        return batch_main()
    elif args.command == 'visualize':
        return visualize_neuron(args.input, args.size, args.no_color)
    elif args.command == 'mesh':
        return export_mesh(args.input, args.output, args.segments, args.no_cap)
    elif args.command == 'viewer':
        return view_3d(args.input, args.segments)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
