"""
ANSI art rendering for neuron morphology.
"""

import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from ..core import Neuron, SWCNode

# SWC type definitions
SWC_TYPE_LABELS = {
    0: 'undefined', 1: 'soma', 2: 'axon', 3: 'basal dendrite',
    4: 'apical dendrite', 5: 'custom', 6: 'unspecified neurite', 7: 'glia processes'
}

# ANSI color codes for each type
ANSI_COLORS = {
    0: '\033[90m',  # gray
    1: '\033[91m',  # red (soma)
    2: '\033[94m',  # blue (axon)
    3: '\033[96m',  # cyan (basal dendrite)
    4: '\033[95m',  # magenta (apical dendrite)
    5: '\033[92m',  # green (custom)
    6: '\033[93m',  # yellow (unspecified)
    7: '\033[33m',  # orange-ish (glia)
}
RESET = '\033[0m'


class RadialProjector:
    """Convert 3D neuron to 2D radial representation"""
    
    def __init__(self, neuron: Neuron, size: int = 128):
        self.neuron = neuron
        self.size = size
        self.center = size // 2
        self.soma_idx = self._find_soma()
        
    def _find_soma(self) -> int:
        """Find soma node (type 1) or root node"""
        for idx, node in self.neuron.nodes.items():
            if node.is_soma:
                return idx
        # Return root if no soma
        for idx, node in self.neuron.nodes.items():
            if node.parent == -1:
                return idx
        return list(self.neuron.nodes.keys())[0]
    
    def _compute_radial_coords(self) -> Dict[int, Tuple[float, float]]:
        """Convert 3D coords to 2D radial (distance, angle)"""
        soma = self.neuron.nodes[self.soma_idx]
        soma_pos = soma.position
        
        radial_coords = {}
        for idx, node in self.neuron.nodes.items():
            pos = node.position
            delta = pos - soma_pos
            
            # Distance from soma (3D Euclidean)
            dist = np.linalg.norm(delta)
            
            # Angle in XY plane
            angle = np.arctan2(delta[1], delta[0])
            
            radial_coords[idx] = (dist, angle)
        
        return radial_coords
    
    def project_to_grid(self) -> Dict[int, Tuple[int, int]]:
        """Convert radial coordinates to grid positions"""
        radial = self._compute_radial_coords()
        
        # Find max distance for scaling
        max_dist = max(d for d, _ in radial.values()) if radial else 1
        scale = (self.size // 2 - 5) / max_dist if max_dist > 0 else 1
        
        grid_pos = {}
        for idx, (dist, angle) in radial.items():
            # Scale distance
            r = dist * scale
            
            # Convert polar to Cartesian
            x = self.center + int(r * np.cos(angle))
            y = self.center + int(r * np.sin(angle))
            
            # Clamp to grid
            x = max(0, min(self.size - 1, x))
            y = max(0, min(self.size - 1, y))
            
            grid_pos[idx] = (x, y)
        
        return grid_pos


class ANSIRenderer:
    """Render neuron structure as ANSI art with contiguous lines"""
    
    def __init__(self, neuron: Neuron, size: int = 128, use_color: bool = True):
        self.neuron = neuron
        self.size = size
        self.use_color = use_color
        self.grid = [[' ' for _ in range(size)] for _ in range(size)]
        self.color_grid = [[0 for _ in range(size)] for _ in range(size)]
        self.connections = [[set() for _ in range(size)] for _ in range(size)]  # Track connections at each cell
        
    def draw_line(self, x0: int, y0: int, x1: int, y1: int, type_id: int, thickness: float):
        """Draw line using Bresenham's algorithm with proper connection tracking"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        char_set = 'heavy' if thickness > 2 else 'light'
        
        points = []
        x, y = x0, y0
        
        while True:
            if 0 <= x < self.size and 0 <= y < self.size:
                points.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        # Now draw the points with proper direction tracking
        for i, (x, y) in enumerate(points):
            # Determine connections from this point
            dirs = set()
            
            if i > 0:
                px, py = points[i-1]
                # Add direction FROM previous point TO current point
                if px < x:
                    dirs.add('W')  # Coming from West
                elif px > x:
                    dirs.add('E')  # Coming from East
                if py < y:
                    dirs.add('N')  # Coming from North
                elif py > y:
                    dirs.add('S')  # Coming from South
            
            if i < len(points) - 1:
                nx, ny = points[i+1]
                # Add direction FROM current point TO next point
                if nx < x:
                    dirs.add('W')  # Going to West
                elif nx > x:
                    dirs.add('E')  # Going to East
                if ny < y:
                    dirs.add('N')  # Going to North
                elif ny > y:
                    dirs.add('S')  # Going to South
            
            # Merge connections at this cell (important for branch points!)
            self.connections[y][x].update(dirs)
            # Keep the most prominent type at each location
            if self.color_grid[y][x] == 0 or type_id == 1:  # Prefer soma
                self.color_grid[y][x] = type_id
    
    def _get_box_char(self, connections: Set[str], char_set: str = 'light') -> str:
        """Convert connection set to appropriate box-drawing character"""
        if not connections:
            return ' '
        
        # Normalize to sorted string
        conn = ''.join(sorted(connections))
        
        if char_set == 'light':
            # Horizontal lines
            if conn in ['E', 'W', 'EW']:
                return '─'
            # Vertical lines
            elif conn in ['N', 'S', 'NS']:
                return '│'
            # Corners
            elif conn == 'ES':
                return '┌'
            elif conn == 'SW':
                return '┐'
            elif conn == 'NW':
                return '└'
            elif conn == 'NE':
                return '┘'
            # T-junctions
            elif conn == 'ENS':
                return '├'
            elif conn == 'NSW':
                return '┤'
            elif conn == 'ESW':
                return '┬'
            elif conn == 'NEW':
                return '┴'
            # 4-way junction
            elif conn == 'ENSW':
                return '┼'
            # 3-way that doesn't match above patterns
            elif len(connections) >= 3:
                return '┼'
            # 2-way that doesn't match above patterns  
            elif len(connections) == 2:
                # Fallback for unusual 2-way connections
                if 'N' in connections or 'S' in connections:
                    return '│'
                else:
                    return '─'
            else:
                return '·'
        else:  # heavy
            # Horizontal lines
            if conn in ['E', 'W', 'EW']:
                return '━'
            # Vertical lines
            elif conn in ['N', 'S', 'NS']:
                return '┃'
            # Corners
            elif conn == 'ES':
                return '┏'
            elif conn == 'SW':
                return '┓'
            elif conn == 'NW':
                return '┗'
            elif conn == 'NE':
                return '┛'
            # T-junctions
            elif conn == 'ENS':
                return '┣'
            elif conn == 'NSW':
                return '┫'
            elif conn == 'ESW':
                return '┳'
            elif conn == 'NEW':
                return '┻'
            # 4-way junction
            elif conn == 'ENSW':
                return '╋'
            # Fallbacks
            elif len(connections) >= 3:
                return '╋'
            elif len(connections) == 2:
                if 'N' in connections or 'S' in connections:
                    return '┃'
                else:
                    return '━'
            else:
                return '·'
    
    def finalize_grid(self):
        """Convert connections to actual box-drawing characters"""
        for y in range(self.size):
            for x in range(self.size):
                if self.connections[y][x]:
                    # Determine thickness based on surrounding nodes
                    char_set = 'light'  # Could make this smarter
                    self.grid[y][x] = self._get_box_char(self.connections[y][x], char_set)
    
    def mark_special_nodes(self, grid_pos: Dict[int, Tuple[int, int]], soma_idx: int):
        """Mark soma and branch points"""
        for idx, node in self.neuron.nodes.items():
            x, y = grid_pos[idx]
            if not (0 <= x < self.size and 0 <= y < self.size):
                continue
            
            # Mark soma prominently
            if idx == soma_idx:
                self.grid[y][x] = '●'
                self.color_grid[y][x] = node.type_id
            # Mark branch points (nodes with multiple children) 
            elif len(node.children) > 1:
                # Only mark if not already part of path
                if len(self.connections[y][x]) == 0:
                    self.grid[y][x] = '+'
            # Mark terminals (leaf nodes) only if isolated
            elif len(node.children) == 0 and node.parent != -1:
                # Only mark terminals that don't have connections (isolated points)
                if len(self.connections[y][x]) == 0:
                    self.grid[y][x] = '·'
    
    def render(self, grid_pos: Dict[int, Tuple[int, int]], soma_idx: int) -> str:
        """Render the complete neuron"""
        # Draw all edges first
        for idx, node in self.neuron.nodes.items():
            if node.parent != -1 and node.parent in self.neuron.nodes:
                parent = self.neuron.nodes[node.parent]
                x0, y0 = grid_pos[node.parent]
                x1, y1 = grid_pos[idx]
                
                # Skip if parent and child are at same cell (causes dangles)
                if x0 == x1 and y0 == y1:
                    continue
                
                thickness = (node.radius + parent.radius) / 2
                self.draw_line(x0, y0, x1, y1, node.type_id, thickness)
        
        # Convert connections to characters
        self.finalize_grid()
        
        # Mark special nodes
        self.mark_special_nodes(grid_pos, soma_idx)
        
        # Convert to string with colors
        return self._grid_to_string()
    
    def _grid_to_string(self) -> str:
        """Convert grid to colored string, trimming empty rows"""
        # Find first and last non-empty rows
        first_row = 0
        last_row = self.size - 1
        
        for y in range(self.size):
            if any(self.grid[y][x] != ' ' for x in range(self.size)):
                first_row = y
                break
        
        for y in range(self.size - 1, -1, -1):
            if any(self.grid[y][x] != ' ' for x in range(self.size)):
                last_row = y
                break
        
        # Build output with only non-empty rows
        lines = []
        for y in range(first_row, last_row + 1):
            line_chars = []
            current_color = None
            
            for x in range(self.size):
                char = self.grid[y][x]
                type_id = self.color_grid[y][x]
                
                if self.use_color and char != ' ' and type_id != current_color:
                    if current_color is not None:
                        line_chars.append(RESET)
                    line_chars.append(ANSI_COLORS.get(type_id, ''))
                    current_color = type_id
                
                line_chars.append(char)
            
            if self.use_color and current_color is not None:
                line_chars.append(RESET)
            
            lines.append(''.join(line_chars))
        
        return '\n'.join(lines)
    
    def visualize(self, size: int = 128, use_color: bool = True) -> str:
        """Main function to visualize neuron"""
        if not self.neuron.nodes:
            return "No valid nodes found in neuron"
        
        # Update size and color settings
        self.size = size
        self.use_color = use_color
        self.grid = [[' ' for _ in range(size)] for _ in range(size)]
        self.color_grid = [[0 for _ in range(size)] for _ in range(size)]
        self.connections = [[set() for _ in range(size)] for _ in range(size)]
        
        # Project to radial grid
        projector = RadialProjector(self.neuron, size)
        grid_pos = projector.project_to_grid()
        soma_idx = projector.soma_idx
        
        # Render
        output = self.render(grid_pos, soma_idx)
        
        return output
