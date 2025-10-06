"""
2D SVG rendering for neuron morphology projections.
"""

import numpy as np
import trimesh as tm
from trimesh.scene.cameras import Camera
import svgwrite
from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from ..core import Neuron


class ViewPose(Enum):
    """Pre-defined camera poses for neuron visualization."""
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"
    ISOMETRIC = "isometric"


class SVGNeuronRenderer:
    """Generate 2D SVG projections of 3D neuron morphology."""
    
    def __init__(self, neuron: Neuron, resolution: Tuple[int, int] = (800, 800),
                 background_color: str = "black", colorize: bool = False, 
                 pen_scale: float = 1.0, cmap: str = "viridis"):
        self.neuron = neuron
        self.resolution = resolution
        self.background_color = background_color
        self.colorize = colorize
        self.pen_scale = pen_scale
        self.cmap = cmap
        self.paths_3d: List[tm.path.Path3D] = []
        self.paths_2d: List[tm.path.Path2D] = []
        
    def _get_path_color(self, metadata: Dict) -> str:
        """Get color for path based on neuron type and colorize setting."""
        if not self.colorize:
            return "white"
        
        # Get colormap
        try:
            colormap = cm.get_cmap(self.cmap)
        except ValueError:
            # Fallback to viridis if cmap is invalid
            colormap = cm.get_cmap('viridis')
        
        # Define class indices for consistent coloring
        if metadata.get('is_soma', False):
            class_idx = 0  # Soma
        elif metadata.get('is_axon', False):
            class_idx = 1  # Axon
        elif metadata.get('is_dendrite', False):
            class_idx = 2  # Dendrite
        else:
            class_idx = 3  # Other/Unknown
        
        # Map class index to colormap (0-1 range)
        # Use 3 classes: soma, axon, dendrite (0, 0.5, 1.0)
        if class_idx < 3:
            color_value = class_idx * 0.5  # 0, 0.5, 1.0
        else:
            color_value = 0.25  # Middle for unknown types
        
        # Get color from colormap
        color_rgba = colormap(color_value)
        
        # Convert to hex color
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(color_rgba[0] * 255),
            int(color_rgba[1] * 255),
            int(color_rgba[2] * 255)
        )
        
        return color_hex
        
    def _create_3d_paths(self) -> List[tm.path.Path3D]:
        """Convert neuron morphology to 3D paths."""
        paths = []
        
        # Group nodes by type for different styling
        soma_nodes = []
        axon_nodes = []
        dendrite_nodes = []
        
        for node in self.neuron.nodes.values():
            if node.type_id == 1:  # soma
                soma_nodes.append(node)
            elif node.type_id == 2:  # axon
                axon_nodes.append(node)
            else:  # dendrite
                dendrite_nodes.append(node)
        
        # Create paths for each connection
        for node in self.neuron.nodes.values():
            if node.parent != -1 and node.parent in self.neuron.nodes:
                parent = self.neuron.nodes[node.parent]
                
                # Create line segment between parent and child
                vertices = np.array([parent.position, node.position])
                entities = [tm.path.entities.Line([0, 1])]
                
                path_3d = tm.path.Path3D(entities=entities, vertices=vertices)
                path_3d.metadata = {
                    'node_type': node.type_id,
                    'radius_start': parent.radius,
                    'radius_end': node.radius,
                    'is_soma': node.type_id == 1,
                    'is_axon': node.type_id == 2,
                    'is_dendrite': node.type_id not in [1, 2]
                }
                paths.append(path_3d)
        
        return paths
    
    def _get_camera_pose(self, pose: ViewPose, soma_center: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """Get camera transformation matrix for given pose, centered on soma."""
        # Use soma center as the target
        target = soma_center
        size = np.max(bounds[1] - bounds[0])
        distance = size * 1.5
        
        # Define camera positions and orientations directly
        if pose == ViewPose.TOP:
            # Looking down from above
            eye = target + np.array([0, 0, distance])
            # Camera coordinate system: X=right, Y=up, Z=into screen
            transform = np.array([
                [1, 0, 0, -eye[0]],
                [0, 1, 0, -eye[1]], 
                [0, 0, 1, -eye[2]],
                [0, 0, 0, 1]
            ])
        elif pose == ViewPose.BOTTOM:
            # Looking up from below
            eye = target + np.array([0, 0, -distance])
            # Flip Y and Z for bottom view
            transform = np.array([
                [1, 0, 0, -eye[0]],
                [0, -1, 0, eye[1]], 
                [0, 0, -1, eye[2]],
                [0, 0, 0, 1]
            ])
        elif pose == ViewPose.LEFT:
            # Looking right from left side
            eye = target + np.array([-distance, 0, 0])
            # Rotate coordinate system: X=forward, Y=up, Z=right
            transform = np.array([
                [0, 0, -1, eye[0]],
                [0, 1, 0, -eye[1]], 
                [1, 0, 0, -eye[2]],
                [0, 0, 0, 1]
            ])
        elif pose == ViewPose.RIGHT:
            # Looking left from right side
            eye = target + np.array([distance, 0, 0])
            # Rotate coordinate system: X=backward, Y=up, Z=left
            transform = np.array([
                [0, 0, 1, -eye[0]],
                [0, 1, 0, -eye[1]], 
                [-1, 0, 0, eye[2]],
                [0, 0, 0, 1]
            ])
        elif pose == ViewPose.FRONT:
            # Looking back from front
            eye = target + np.array([0, distance, 0])
            # Rotate coordinate system: X=right, Y=backward, Z=up
            transform = np.array([
                [1, 0, 0, -eye[0]],
                [0, 0, -1, eye[1]], 
                [0, 1, 0, -eye[2]],
                [0, 0, 0, 1]
            ])
        elif pose == ViewPose.BACK:
            # Looking front from back
            eye = target + np.array([0, -distance, 0])
            # Rotate coordinate system: X=left, Y=forward, Z=up
            transform = np.array([
                [-1, 0, 0, eye[0]],
                [0, 0, 1, -eye[1]], 
                [0, 1, 0, -eye[2]],
                [0, 0, 0, 1]
            ])
        elif pose == ViewPose.ISOMETRIC:
            # Isometric view
            eye = target + np.array([distance, distance, distance])
            # Isometric projection matrix
            transform = np.array([
                [0.707, -0.408, 0.577, -eye[0] * 0.707 + eye[1] * 0.408 - eye[2] * 0.577],
                [0, 0.816, 0.577, -eye[1] * 0.816 - eye[2] * 0.577],
                [-0.707, -0.408, 0.577, eye[0] * 0.707 + eye[1] * 0.408 - eye[2] * 0.577],
                [0, 0, 0, 1]
            ])
        else:
            raise ValueError(f"Unknown pose: {pose}")
        
        return transform
    
    def _project_to_2d(self, pose: ViewPose = ViewPose.ISOMETRIC) -> List[tm.path.Path2D]:
        """Project 3D paths to 2D using specified camera pose."""
        if not self.paths_3d:
            self.paths_3d = self._create_3d_paths()
        
        # Get bounding box for camera positioning
        all_vertices = np.vstack([path.vertices for path in self.paths_3d])
        bounds = np.array([all_vertices.min(axis=0), all_vertices.max(axis=0)])
        
        # Find soma center (there should be only one soma node)
        soma_nodes = [node for node in self.neuron.nodes.values() if node.type_id == 1]
        if not soma_nodes:
            # Fallback to bounds center if no soma found
            soma_center = np.mean(bounds, axis=0)
        else:
            soma_center = soma_nodes[0].position
        
        # Get camera transformation
        cam_tf = self._get_camera_pose(pose, soma_center, bounds)
        
        # Simple orthographic projection with depth sorting
        paths_with_depth = []
        
        for path_3d in self.paths_3d:
            # Transform vertices to camera frame using pure numpy
            V_world = path_3d.vertices
            V_homogeneous = np.column_stack([V_world, np.ones(len(V_world))])
            V_cam = (cam_tf @ V_homogeneous.T).T[:, :3]
            
            # For orthographic projection, just use X and Y coordinates
            # Skip Z coordinate (depth) for 2D projection
            verts_2d = V_cam[:, :2]
            
            # Check for degenerate projections (all points at same location)
            if len(verts_2d) > 1:
                coord_range = np.max(verts_2d, axis=0) - np.min(verts_2d, axis=0)
                if np.max(coord_range) < 1e-6:  # All points collapsed to same location
                    continue  # Skip this path
            
            # Calculate average depth (Z coordinate) for this path
            # Use negative Z because we want farther objects to have larger depth values
            avg_depth = -np.mean(V_cam[:, 2])
            
            # Create 2D path preserving entities
            path_2d = tm.path.Path2D(entities=path_3d.entities, vertices=verts_2d)
            path_2d.metadata = path_3d.metadata.copy()
            
            # Store path with its depth for sorting
            paths_with_depth.append((avg_depth, path_2d))
        
        # Sort paths by depth (farthest first, so they're drawn first)
        paths_with_depth.sort(key=lambda x: x[0], reverse=True)
        
        # Extract sorted paths
        paths_2d = [path for _, path in paths_with_depth]
        
        # Fit the 2D projection to the canvas
        paths_2d = self._fit_to_canvas(paths_2d)
        
        return paths_2d
    
    def _fit_to_canvas(self, paths_2d: List[tm.path.Path2D]) -> List[tm.path.Path2D]:
        """Scale and center 2D paths to fit the canvas with minimal whitespace."""
        if not paths_2d:
            return paths_2d
        
        # Get all 2D vertices
        all_vertices = np.vstack([path.vertices for path in paths_2d])
        
        # Calculate bounding box of 2D projection
        min_coords = all_vertices.min(axis=0)
        max_coords = all_vertices.max(axis=0)
        current_size = max_coords - min_coords
        current_center = (min_coords + max_coords) / 2
        
        # Calculate scaling to fit canvas with some padding
        canvas_size = np.array(self.resolution)
        padding = 0.1  # 10% padding
        scale_factor = min(canvas_size * (1 - 2 * padding) / current_size)
        
        # Scale and center
        scaled_vertices = (all_vertices - current_center) * scale_factor + canvas_size / 2
        
        # Update paths with scaled vertices
        vertex_idx = 0
        for path_2d in paths_2d:
            num_vertices = len(path_2d.vertices)
            path_2d.vertices = scaled_vertices[vertex_idx:vertex_idx + num_vertices]
            vertex_idx += num_vertices
        
        return paths_2d
    
    def _get_path_style(self, metadata: Dict) -> Dict[str, str]:
        """Get SVG styling for path based on metadata."""
        # Get color based on colorize setting
        stroke_color = self._get_path_color(metadata)
        
        # Base stroke widths (will be scaled by pen_scale)
        if metadata.get('is_soma', False):
            base_width = 3
        elif metadata.get('is_axon', False):
            base_width = 2
        else:  # dendrite
            base_width = 1.5
            
        return {
            'stroke': stroke_color,
            'stroke-width': str(base_width * self.pen_scale),
            'fill': 'none',
            'stroke-linecap': 'round',
            'stroke-linejoin': 'round'
        }
    
    def _adjust_stroke_width(self, base_width: float, radius_start: float, 
                           radius_end: float, view_scale: float = 1.0) -> float:
        """Adjust stroke width based on radius and projection scale."""
        avg_radius = (radius_start + radius_end) / 2
        return max(0.5, base_width * avg_radius * view_scale * self.pen_scale)
    
    def generate_svg(self, pose: ViewPose = ViewPose.ISOMETRIC, 
                    output_path: Optional[Union[str, Path]] = None) -> svgwrite.Drawing:
        """Generate SVG representation of neuron."""
        # Project to 2D
        paths_2d = self._project_to_2d(pose)
        
        if not paths_2d:
            raise ValueError("No valid 2D paths generated")
        
        # Create SVG drawing
        dwg = svgwrite.Drawing(size=self.resolution)
        
        # Add background
        dwg.add(dwg.rect(insert=(0, 0), size=self.resolution, 
                        fill=self.background_color, stroke='none'))
        
        # Add CSS styles for different neuron types using colormap
        # Get colors from colormap
        soma_color = self._get_path_color({'is_soma': True})
        axon_color = self._get_path_color({'is_axon': True})
        dendrite_color = self._get_path_color({'is_dendrite': True})
        
        # Add style definitions
        dwg.add(dwg.style(f"""
            .soma {{ fill: none; stroke: {soma_color}; stroke-linecap: round; stroke-linejoin: round; }}
            .axon {{ fill: none; stroke: {axon_color}; stroke-linecap: round; stroke-linejoin: round; }}
            .dendrite {{ fill: none; stroke: {dendrite_color}; stroke-linecap: round; stroke-linejoin: round; }}
        """))
        
        # Calculate view scale for stroke width adjustment
        all_vertices = np.vstack([path.vertices for path in paths_2d])
        view_scale = min(self.resolution) / (np.max(all_vertices) - np.min(all_vertices))
        
        # Draw paths in depth order (farthest first)
        for path_2d in paths_2d:
            metadata = path_2d.metadata
            
            # Determine CSS class based on neuron type
            if metadata.get('is_soma', False):
                css_class = 'soma'
            elif metadata.get('is_axon', False):
                css_class = 'axon'
            else:
                css_class = 'dendrite'
            
            # Get base style
            style = self._get_path_style(metadata)
            
            # Adjust stroke width based on radius
            if 'radius_start' in metadata and 'radius_end' in metadata:
                base_width = float(style['stroke-width'])
                adjusted_width = self._adjust_stroke_width(
                    base_width, 
                    metadata['radius_start'], 
                    metadata['radius_end'],
                    view_scale
                )
                style['stroke-width'] = str(adjusted_width)
            
            # Convert trimesh path to SVG path
            svg_path = self._trimesh_to_svg_path(path_2d)
            
            # Add path with CSS class (remove individual style attributes)
            path_attrs = {'d': svg_path, 'class': css_class}
            if 'stroke-width' in style:
                path_attrs['stroke-width'] = style['stroke-width']
            
            dwg.add(dwg.path(**path_attrs))
        
        # Save if output path provided
        if output_path:
            dwg.saveas(str(output_path))
        
        return dwg
    
    def _trimesh_to_svg_path(self, path_2d: tm.path.Path2D) -> str:
        """Convert trimesh Path2D to SVG path string."""
        svg_commands = []
        
        for entity in path_2d.entities:
            if isinstance(entity, tm.path.entities.Line):
                start_idx, end_idx = entity.points
                start = path_2d.vertices[start_idx]
                end = path_2d.vertices[end_idx]
                
                if not svg_commands:
                    svg_commands.append(f"M {start[0]:.2f},{start[1]:.2f}")
                svg_commands.append(f"L {end[0]:.2f},{end[1]:.2f}")
        
        return " ".join(svg_commands)
    
    def render_to_file(self, output_path: Union[str, Path], 
                      pose: ViewPose = ViewPose.ISOMETRIC) -> svgwrite.Drawing:
        """Render neuron to SVG file."""
        return self.generate_svg(pose=pose, output_path=output_path)
    
    def rasterize(self, pose: ViewPose = ViewPose.ISOMETRIC, 
                 dpi: int = 300) -> np.ndarray:
        """Rasterize SVG to bitmap using sphinx."""
        try:
            import cairosvg
            from PIL import Image
            import io
            
            # Generate SVG
            svg_drawing = self.generate_svg(pose=pose)
            svg_string = svg_drawing.tostring()
            
            # Convert SVG to PNG
            png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), 
                                       dpi=dpi)
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(png_data))
            return np.array(image)
            
        except ImportError:
            raise ImportError("cairosvg and PIL required for rasterization. "
                            "Install with: pip install cairosvg pillow")
    
    def rasterize_panel(self, poses: List[ViewPose], 
                       dpi: int = 300) -> np.ndarray:
        """Rasterize multi-pose panel SVG to bitmap."""
        try:
            import cairosvg
            from PIL import Image
            import io
            
            # Generate panel SVG
            panel_svg = self.create_multi_pose_panel(poses, "temp_panel.svg")
            svg_string = panel_svg.tostring()
            
            # Convert SVG to PNG
            png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), 
                                       dpi=dpi)
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(png_data))
            return np.array(image)
            
        except ImportError:
            raise ImportError("cairosvg and PIL required for rasterization. "
                            "Install with: pip install cairosvg pillow")
    
    def create_multi_pose_panel(self, poses: List[ViewPose], 
                               output_path: Union[str, Path],
                               panel_size: Tuple[int, int] = (1600, 1200)) -> svgwrite.Drawing:
        """Create a panel with multiple poses."""
        if len(poses) > 12:
            raise ValueError("Maximum 12 poses supported for panel")
        
        # Calculate grid layout
        if len(poses) <= 4:
            cols = 2
        elif len(poses) <= 6:
            cols = 3
        elif len(poses) <= 9:
            cols = 3
        else:
            cols = 4
        rows = (len(poses) + cols - 1) // cols
        
        cell_width = panel_size[0] // cols
        cell_height = panel_size[1] // rows
        
        # Create main drawing
        dwg = svgwrite.Drawing(size=panel_size)
        dwg.add(dwg.rect(insert=(0, 0), size=panel_size, 
                        fill=self.background_color, stroke='none'))
        
        # Add CSS styles for different neuron types using colormap
        # Get colors from colormap
        soma_color = self._get_path_color({'is_soma': True})
        axon_color = self._get_path_color({'is_axon': True})
        dendrite_color = self._get_path_color({'is_dendrite': True})
        
        # Add style definitions
        dwg.add(dwg.style(f"""
            .soma {{ fill: none; stroke: {soma_color}; stroke-linecap: round; stroke-linejoin: round; }}
            .axon {{ fill: none; stroke: {axon_color}; stroke-linecap: round; stroke-linejoin: round; }}
            .dendrite {{ fill: none; stroke: {dendrite_color}; stroke-linecap: round; stroke-linejoin: round; }}
        """))
        
        for i, pose in enumerate(poses):
            row = i // cols
            col = i % cols
            
            x = col * cell_width
            y = row * cell_height
            
            # Generate SVG for this pose
            pose_svg = self.generate_svg(pose=pose)
            
            # Scale to fit cell
            scale = min(cell_width / self.resolution[0], 
                       cell_height / self.resolution[1]) * 0.9
            
            # Add pose label
            label = dwg.text(pose.value.title(), 
                           insert=(x + 10, y + 20),
                           font_size=16, 
                           fill='black')
            dwg.add(label)
            
            # Add scaled SVG content
            group = dwg.g(transform=f"translate({x},{y}) scale({scale})")
            
            # Get the 2D paths for this pose (already depth-sorted)
            paths_2d = self._project_to_2d(pose)
            
            # Calculate view scale for stroke width adjustment
            if paths_2d:
                all_vertices = np.vstack([path.vertices for path in paths_2d])
                view_scale = min(self.resolution) / (np.max(all_vertices) - np.min(all_vertices))
                
                # Draw paths in depth order (farthest first)
                for path_2d in paths_2d:
                    metadata = path_2d.metadata
                    
                    # Determine CSS class based on neuron type
                    if metadata.get('is_soma', False):
                        css_class = 'soma'
                    elif metadata.get('is_axon', False):
                        css_class = 'axon'
                    else:
                        css_class = 'dendrite'
                    
                    # Get base style
                    style = self._get_path_style(metadata)
                    
                    # Adjust stroke width based on radius
                    if 'radius_start' in metadata and 'radius_end' in metadata:
                        base_width = float(style['stroke-width'])
                        adjusted_width = self._adjust_stroke_width(
                            base_width, 
                            metadata['radius_start'], 
                            metadata['radius_end'],
                            view_scale
                        )
                        style['stroke-width'] = str(adjusted_width)
                    
                    # Convert trimesh path to SVG path
                    svg_path = self._trimesh_to_svg_path(path_2d)
                    
                    # Add path with CSS class
                    path_attrs = {'d': svg_path, 'class': css_class}
                    if 'stroke-width' in style:
                        path_attrs['stroke-width'] = style['stroke-width']
                    
                    group.add(dwg.path(**path_attrs))
            
            dwg.add(group)
        
        # Save panel
        dwg.saveas(str(output_path))
        return dwg
    
    def _parse_path_attributes(self, path_str: str) -> Dict[str, str]:
        """Parse SVG path attributes from string."""
        import re
        attrs = {}
        
        # Extract common attributes
        for attr in ['fill', 'stroke', 'stroke-width', 'stroke-linecap', 'stroke-linejoin']:
            match = re.search(f'{attr}="([^"]*)"', path_str)
            if match:
                attrs[attr] = match.group(1)
        
        return attrs
