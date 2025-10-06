#!/usr/bin/env python3
"""
Example of SVG visualization for neuron morphology.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from axonet.core import Neuron
from axonet.io import load_swc
from axonet.visualization import SVGNeuronRenderer, ViewPose


def main():
    """Demonstrate SVG visualization capabilities."""
    parser = argparse.ArgumentParser(
        description="Generate SVG visualizations of neuron morphology from SWC files"
    )
    parser.add_argument(
        "swc_file",
        type=str,
        help="Path to SWC file containing neuron morphology"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for generated SVG files (default: current directory)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[800, 800],
        metavar=("WIDTH", "HEIGHT"),
        help="SVG resolution in pixels (default: 800 800)"
    )
    parser.add_argument(
        "--poses",
        nargs="+",
        choices=[pose.value for pose in ViewPose],
        default=[pose.value for pose in ViewPose],
        help="Camera poses to generate (default: all poses)"
    )
    parser.add_argument(
        "--panel",
        action="store_true",
        help="Generate multi-pose panel SVG"
    )
    parser.add_argument(
        "--rasterize",
        action="store_true",
        help="Generate rasterized PNG images"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for rasterization (default: 300)"
    )
    parser.add_argument(
        "--background",
        type=str,
        default="black",
        help="Background color (default: black)"
    )
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="Color paths by neuron type using matplotlib colormap"
    )
    parser.add_argument(
        "--pen-scale",
        type=float,
        default=1.0,
        help="Scale factor for pen size without affecting layout (default: 1.0)"
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name for neuron type coloring (default: viridis)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    swc_path = Path(args.swc_file)
    if not swc_path.exists():
        print(f"Error: SWC file '{swc_path}' not found")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load neuron from SWC file
    print(f"Loading neuron from {swc_path}...")
    try:
        neuron = load_swc(swc_path)
        print(f"Loaded neuron with {len(neuron.nodes)} nodes")
    except Exception as e:
        print(f"Error loading SWC file: {e}")
        sys.exit(1)
    
    # Create SVG renderer
    renderer = SVGNeuronRenderer(
        neuron, 
        resolution=tuple(args.resolution),
        background_color=args.background,
        colorize=args.colorize,
        pen_scale=args.pen_scale,
        cmap=args.cmap
    )
    
    # Convert pose strings to ViewPose enums
    poses = [ViewPose(pose) for pose in args.poses]
    
    # Generate individual SVG files for each pose
    for pose in poses:
        output_file = output_dir / f"neuron_{pose.value}.svg"
        print(f"Generating {output_file}...")
        renderer.render_to_file(output_file, pose=pose)
    
    # Create multi-pose panel if requested
    if args.panel:
        panel_file = output_dir / "neuron_panel.svg"
        print(f"Generating multi-pose panel: {panel_file}")
        renderer.create_multi_pose_panel(
            poses=poses,
            output_path=panel_file,
            panel_size=(1600, 1200)
        )
    
    # Generate rasterized images if requested
    if args.rasterize:
        print("Generating rasterized images...")
        try:
            for pose in poses:
                bitmap = renderer.rasterize(pose=pose, dpi=args.dpi)
                png_file = output_dir / f"neuron_{pose.value}.png"
                
                from PIL import Image
                Image.fromarray(bitmap).save(png_file)
                print(f"Saved rasterized image: {png_file}")
            
            # Also rasterize the panel if it was generated
            if args.panel:
                panel_bitmap = renderer.rasterize_panel(poses, dpi=args.dpi)
                panel_png_file = output_dir / "neuron_panel.png"
                
                from PIL import Image
                Image.fromarray(panel_bitmap).save(panel_png_file)
                print(f"Saved rasterized panel: {panel_png_file}")
                
        except ImportError as e:
            print(f"Rasterization not available: {e}")
            print("Install required packages: pip install cairosvg pillow")
    
    print("SVG visualization complete!")


if __name__ == "__main__":
    main()
