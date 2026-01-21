"""
Neuro Render Core — ModernGL-based renderer for SWC neurons

This module provides UI-agnostic OpenGL rendering using ModernGL for cleaner
resource management and reduced boilerplate.

Key concepts
------------
• OffscreenContext: ModernGL standalone context wrapper
• NeuroRenderCore: loads SWC, builds triangle meshes, manages camera,
  draws with orthographic/perspective projection, and computes QC metrics.

Dependencies
------------
- numpy, moderngl, and project-local modules:
    from .mesh import MeshRenderer
    from ..io import load_swc, NeuronClass
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import moderngl

from .mesh import MeshRenderer
from ..io import load_swc, NeuronClass

COLORS = {
    'red': np.array([1.0, 0.0, 0.0], dtype=np.float32),
    'green': np.array([0.0, 1.0, 0.0], dtype=np.float32),
    'blue': np.array([0.0, 0.0, 1.0], dtype=np.float32),
    'yellow': np.array([1.0, 1.0, 0.0], dtype=np.float32),
    'purple': np.array([1.0, 0.0, 1.0], dtype=np.float32),
    'orange': np.array([1.0, 0.5, 0.0], dtype=np.float32),
    'brown': np.array([0.5, 0.25, 0.0], dtype=np.float32),
    'gray': np.array([0.5, 0.5, 0.5], dtype=np.float32),
    'black': np.array([0.0, 0.0, 0.0], dtype=np.float32),
    'white': np.array([1.0, 1.0, 1.0], dtype=np.float32),
}

# ============================ Math helpers ============================

def majority_pool_uint8(mask_hi: np.ndarray, factor: int, *, prefer_nonzero: bool = True) -> np.ndarray:
    """Downsample (H*factor, W*factor) -> (H, W) by majority voting within each factor×factor block.
    Vectorized implementation.
    """
    Hh, Wh = mask_hi.shape
    assert Hh % factor == 0 and Wh % factor == 0, f"Supersample factor {factor} must divide image size ({Hh}, {Wh})"
    H, W = Hh // factor, Wh // factor
    blocks = mask_hi.reshape(H, factor, W, factor).transpose(0, 2, 1, 3).reshape(H, W, -1)

    if prefer_nonzero:
        return blocks.max(axis=2).astype(np.uint8)
    else:
        from scipy.stats import mode
        result, _ = mode(blocks, axis=2, keepdims=False)
        return result.astype(np.uint8)


def average_pool_depth(depth_hi: np.ndarray, factor: int, *, prefer_valid: bool = True, background_threshold: float = 0.999) -> np.ndarray:
    """Downsample (H*factor, W*factor) -> (H, W) by averaging depth. Vectorized implementation."""
    Hh, Wh = depth_hi.shape
    assert Hh % factor == 0 and Wh % factor == 0, f"Supersample factor {factor} must divide image size ({Hh}, {Wh})"
    H, W = Hh // factor, Wh // factor
    blocks = depth_hi.reshape(H, factor, W, factor).transpose(0, 2, 1, 3).reshape(H, W, -1)

    if prefer_valid:
        valid = blocks < background_threshold
        masked = np.where(valid, blocks, np.nan)
        result = np.nanmean(masked, axis=2)
        all_bg = ~valid.any(axis=2)
        result[all_bg] = blocks[all_bg].mean(axis=1)
    else:
        result = blocks.mean(axis=2)

    return result.astype(np.float32)


def perspective(fovy_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(1e-6, aspect)
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def orthographic(left: float, right: float, bottom: float, top: float, znear: float, zfar: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2.0 / (right - left)
    m[1, 1] = 2.0 / (top - bottom)
    m[2, 2] = -2.0 / (zfar - znear)
    m[0, 3] = -(right + left) / (right - left)
    m[1, 3] = -(top + bottom) / (top - bottom)
    m[2, 3] = -(zfar + znear) / (zfar - znear)
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-12)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye[:3]
    return m


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = x*x*C + c;   m[0, 1] = x*y*C - z*s; m[0, 2] = x*z*C + y*s
    m[1, 0] = y*x*C + z*s; m[1, 1] = y*y*C + c;   m[1, 2] = y*z*C - x*s
    m[2, 0] = z*x*C - y*s; m[2, 1] = z*y*C + x*s; m[2, 2] = z*z*C + c
    return m


def _aabb_corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    xs = [bmin[0], bmax[0]]
    ys = [bmin[1], bmax[1]]
    zs = [bmin[2], bmax[2]]
    corners = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corners.append([xs[i], ys[j], zs[k]])
    return np.array(corners, dtype=np.float32)

# ========================= Shader sources =========================

VERT_SRC = """
#version 330 core
in vec3 in_position;

uniform mat4 u_mvp;
uniform mat4 u_mv;

flat out float v_viewz;

void main(){
    vec4 viewpos = u_mv * vec4(in_position, 1.0);
    v_viewz = viewpos.z;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

FRAG_SRC_RGBA = """
#version 330 core
flat in float v_viewz;

uniform vec3 u_color;
uniform float u_depth_near;
uniform float u_depth_far;
uniform int u_render_mode;

out vec4 FragColor;

void main(){
    if (u_render_mode == 1){
        float d = clamp(((-v_viewz) - u_depth_near) / max(1e-6, (u_depth_far - u_depth_near)), 0.0, 1.0);
        FragColor = vec4(d, d, d, 1.0);
        return;
    }
    FragColor = vec4(u_color, 1.0);
}
"""

FRAG_SRC_DEPTH = """
#version 330 core
flat in float v_viewz;

uniform float u_depth_near;
uniform float u_depth_far;

out float FragDepth;

void main(){
    float d = clamp(((-v_viewz) - u_depth_near) / max(1e-6, (u_depth_far - u_depth_near)), 0.0, 1.0);
    FragDepth = d;
}
"""

FRAG_SRC_INTEGER = """
#version 330 core

uniform int u_class_id;

out uint FragColor;

void main(){
    FragColor = uint(u_class_id);
}
"""

# ============================ Core types ============================

class RenderMode(Enum):
    COLOR = 0
    DEPTH_TO_COLOR = 1
    CLASS_ID_INTEGER = 3


@dataclass
class RenderConfig:
    mode: RenderMode
    background: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    class_to_id: Optional[Dict[str, int]] = None
    disable_srgb: bool = False
    disable_blend: bool = False
    disable_cull: bool = False


@dataclass
class Camera:
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fovy: float = 55.0
    ortho_scale: float = 1.0
    perspective: bool = True
    near: float = 0.02
    far: float = 100.0

    def view(self) -> np.ndarray:
        return look_at(self.eye, self.target, self.up)


@dataclass
class DrawRange:
    start: int
    count: int
    color: np.ndarray
    cls: str


class OffscreenContext:
    """ModernGL standalone context wrapper."""

    def __init__(self, width: int, height: int, *, visible: bool = False, samples: int = 0):
        self.width = int(width)
        self.height = int(height)
        self.visible = bool(visible)
        self.samples = int(samples)
        self.ctx = moderngl.create_standalone_context(require=330)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def close(self):
        if self.ctx:
            self.ctx.release()
            self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class NeuroRenderCore:
    """ModernGL-based renderer for SWC neurons."""

    def __init__(self, ctx: OffscreenContext):
        self.ctx = ctx
        self.gl = ctx.ctx

        self.prog_rgba = self.gl.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC_RGBA)
        self.prog_depth = self.gl.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC_DEPTH)
        self.prog_integer = self.gl.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC_INTEGER)

        self.vbo: Optional[moderngl.Buffer] = None
        self.vao_rgba: Optional[moderngl.VertexArray] = None
        self.vao_depth: Optional[moderngl.VertexArray] = None
        self.vao_integer: Optional[moderngl.VertexArray] = None

        self.neuron = None
        self.draw_ranges: list[DrawRange] = []
        self.vertex_count = 0
        self.bounds_min = np.array([0, 0, 0], dtype=np.float32)
        self.bounds_max = np.array([1, 1, 1], dtype=np.float32)
        self.center = np.zeros(3, dtype=np.float32)
        self.diag = 1.0

        self.camera = Camera(
            eye=np.array([0.0, 0.0, 3.0], dtype=np.float32),
            target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.aspect = max(1e-6, self.ctx.width / float(self.ctx.height))

        self.depth_shading = False
        self.color_mode = 0
        self.layer_visible: Dict[str, bool] = {
            NeuronClass.SOMA.name: True,
            NeuronClass.AXON.name: True,
            NeuronClass.BASAL_DENDRITE.name: True,
            NeuronClass.APICAL_DENDRITE.name: True,
            NeuronClass.OTHER.name: True,
        }

        self._cls_colors: Dict[str, np.ndarray] = {
            NeuronClass.SOMA.name: COLORS['red'],
            NeuronClass.AXON.name: COLORS['green'],
            NeuronClass.BASAL_DENDRITE.name: COLORS['blue'],
            NeuronClass.APICAL_DENDRITE.name: COLORS['purple'],
            NeuronClass.OTHER.name: COLORS['orange'],
        }

        self._pos_cpu: Optional[np.ndarray] = None
        
        self._fbo_rgba: Optional[moderngl.Framebuffer] = None
        self._fbo_depth: Optional[moderngl.Framebuffer] = None
        self._fbo_integer: Optional[moderngl.Framebuffer] = None

    def _ensure_fbo_rgba(self, width: int, height: int) -> moderngl.Framebuffer:
        if self._fbo_rgba is None or self._fbo_rgba.size != (width, height):
            if self._fbo_rgba:
                self._fbo_rgba.release()
            color = self.gl.texture((width, height), 4, dtype='f1')
            depth = self.gl.depth_renderbuffer((width, height))
            self._fbo_rgba = self.gl.framebuffer(color_attachments=[color], depth_attachment=depth)
        return self._fbo_rgba

    def _ensure_fbo_depth(self, width: int, height: int) -> moderngl.Framebuffer:
        if self._fbo_depth is None or self._fbo_depth.size != (width, height):
            if self._fbo_depth:
                self._fbo_depth.release()
            color = self.gl.texture((width, height), 1, dtype='f4')
            depth = self.gl.depth_renderbuffer((width, height))
            self._fbo_depth = self.gl.framebuffer(color_attachments=[color], depth_attachment=depth)
        return self._fbo_depth

    def _ensure_fbo_integer(self, width: int, height: int) -> moderngl.Framebuffer:
        if self._fbo_integer is None or self._fbo_integer.size != (width, height):
            if self._fbo_integer:
                self._fbo_integer.release()
            color = self.gl.texture((width, height), 1, dtype='u1')
            depth = self.gl.depth_renderbuffer((width, height))
            self._fbo_integer = self.gl.framebuffer(color_attachments=[color], depth_attachment=depth)
        return self._fbo_integer

    # ------------------------- Data / geometry -------------------------
    def load_swc(self, path: Path, *, segments: int = 18, radius_scale: float = 1.0, 
                 radius_adaptive_alpha: float = 0.0, radius_ref_percentile: float = 50.0) -> None:
        neuron = load_swc(Path(path), validate=True)
        self.set_neuron(neuron, segments=segments, radius_scale=radius_scale, 
                       radius_adaptive_alpha=radius_adaptive_alpha, radius_ref_percentile=radius_ref_percentile)

    def set_neuron(self, neuron, *, segments: int = 18, radius_scale: float = 1.0,
                   radius_adaptive_alpha: float = 0.0, radius_ref_percentile: float = 50.0) -> None:
        self.neuron = neuron
        renderer = MeshRenderer(neuron)
        by_type = renderer.build_mesh_by_type(segments=segments, cap=True, radius_scale=radius_scale,
                                              radius_adaptive_alpha=radius_adaptive_alpha, 
                                              radius_ref_percentile=radius_ref_percentile)

        order = [
            NeuronClass.SOMA.name,
            NeuronClass.AXON.name,
            NeuronClass.BASAL_DENDRITE.name,
            NeuronClass.APICAL_DENDRITE.name,
            NeuronClass.OTHER.name,
        ]

        self.draw_ranges.clear()
        positions = []
        bmin = None
        bmax = None
        start = 0
        
        for cls in order:
            m = by_type.get(cls)
            if m is None:
                continue
            v = np.asarray(m.vertices, dtype=np.float32)
            f = np.asarray(m.faces, dtype=np.int32)
            tri_count = f.shape[0]
            pos = v[f.reshape(-1)]
            positions.append(pos)
            count = int(tri_count * 3)
            self.draw_ranges.append(DrawRange(start, count, self._cls_colors[cls], cls))
            start += count
            bm = m.bounds[0].astype(np.float32)
            bM = m.bounds[1].astype(np.float32)
            bmin = bm if bmin is None else np.minimum(bmin, bm)
            bmax = bM if bmax is None else np.maximum(bmax, bM)

        if not positions:
            raise RuntimeError("No mesh data generated from neuron!")

        pos_all = np.concatenate(positions, axis=0).astype(np.float32)
        self.vertex_count = int(pos_all.shape[0])
        self.bounds_min = bmin
        self.bounds_max = bmax
        self.center = 0.5 * (bmin + bmax)
        self.diag = float(np.linalg.norm(bmax - bmin))

        if self.vbo:
            self.vbo.release()
        self.vbo = self.gl.buffer(pos_all.tobytes())

        if self.vao_rgba:
            self.vao_rgba.release()
        if self.vao_depth:
            self.vao_depth.release()
        if self.vao_integer:
            self.vao_integer.release()

        self.vao_rgba = self.gl.vertex_array(self.prog_rgba, [(self.vbo, '3f', 'in_position')])
        self.vao_depth = self.gl.vertex_array(self.prog_depth, [(self.vbo, '3f', 'in_position')])
        self.vao_integer = self.gl.vertex_array(self.prog_integer, [(self.vbo, '3f', 'in_position')])

        self._pos_cpu = pos_all.copy()

    # ----------------------------- Camera -----------------------------
    def fit_camera(self, *, margin: float = 0.85) -> None:
        radius = max(1e-6, 0.5 * self.diag)
        self.camera.target = self.center.copy()
        if self.camera.perspective:
            half_h = margin * radius
            dist = half_h / math.tan(math.radians(self.camera.fovy) * 0.5)
            self.camera.eye = self.camera.target + np.array([0.0, 0.0, dist], dtype=np.float32)
        else:
            self.camera.ortho_scale = margin * radius
            self.camera.eye = self.camera.target + np.array([0.0, 0.0, 3.0 * radius + 1.0], dtype=np.float32)

    def set_projection(self, *, perspective: Optional[bool] = None, fovy: Optional[float] = None, 
                       ortho_scale: Optional[float] = None):
        if perspective is not None:
            self.camera.perspective = bool(perspective)
        if fovy is not None:
            self.camera.fovy = float(fovy)
        if ortho_scale is not None:
            self.camera.ortho_scale = float(ortho_scale)

    def orbit(self, x0: float, y0: float, x1: float, y1: float, width: int, height: int):
        v0 = self._arcball_vec(x0, y0, width, height)
        v1 = self._arcball_vec(x1, y1, width, height)
        axis = np.cross(v0, v1)
        dot = max(-1.0, min(1.0, float(np.dot(v0, v1))))
        angle = math.acos(dot)
        if np.linalg.norm(axis) < 1e-6 or angle == 0.0:
            return
        axis = axis / np.linalg.norm(axis)
        R = _axis_angle_matrix(axis, angle)
        eye_dir = self.camera.eye - self.camera.target
        up_dir = self.camera.up
        eye_dir = (R @ np.append(eye_dir, 1.0))[:3]
        up_dir = (R @ np.append(up_dir, 1.0))[:3]
        self.camera.eye = self.camera.target + eye_dir
        self.camera.up = up_dir / (np.linalg.norm(up_dir) + 1e-12)

    def pan(self, dx: float, dy: float):
        V = self.camera.view()
        right = V[:3, 0]
        upv = V[:3, 1]
        dist = float(np.linalg.norm(self.camera.eye - self.camera.target))
        if self.camera.perspective:
            scale = 2.0 * dist * math.tan(math.radians(self.camera.fovy) * 0.5)
        else:
            scale = self.camera.ortho_scale * 2.0
        sx = dx * scale * self.aspect
        sy = dy * scale
        delta = -right * sx + upv * sy
        self.camera.eye += delta
        self.camera.target += delta

    def dolly(self, factor: float):
        view_dir = self.camera.target - self.camera.eye
        dist = np.linalg.norm(view_dir)
        if self.camera.perspective:
            new_dist = max(0.05, min(500.0, dist * (1.0 / factor)))
            self.camera.eye = self.camera.target - (view_dir / (dist + 1e-9)) * new_dist
        else:
            self.camera.ortho_scale *= factor
            self.camera.ortho_scale = max(self.diag * 0.02, min(self.diag * 20.0, self.camera.ortho_scale))

    def _arcball_vec(self, x: float, y: float, w: int, h: int) -> np.ndarray:
        if w <= 0 or h <= 0:
            return np.array([0, 0, 1], dtype=np.float32)
        nx = (2.0 * x - w) / max(1.0, w)
        ny = (h - 2.0 * y) / max(1.0, h)
        v = np.array([nx, ny, 0.0], dtype=np.float32)
        d = float(np.dot(v, v))
        if d <= 1.0:
            v[2] = math.sqrt(1.0 - d)
        else:
            v /= math.sqrt(d)
        return v

    # --------------------------- Projection ---------------------------
    def _compute_projection(self, aspect: Optional[float] = None) -> np.ndarray:
        aspect = aspect or max(1e-6, self.aspect)
        if self.camera.perspective:
            self.camera.near = 0.02
            dist = float(np.linalg.norm(self.camera.eye - self.camera.target))
            self.camera.far = max(self.camera.near + 1.0, dist + self.diag * 3.0)
            return perspective(self.camera.fovy, aspect, self.camera.near, self.camera.far)
        else:
            half_h = self.camera.ortho_scale
            half_w = half_h * aspect
            self.camera.near = 0.02
            dist = float(np.linalg.norm(self.camera.eye - self.camera.target))
            self.camera.far = max(self.camera.near + 1.0, dist + self.diag * 3.0)
            return orthographic(-half_w, half_w, -half_h, half_h, self.camera.near, self.camera.far)

    def _compute_matrices(self, aspect: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        V = self.camera.view()
        P = self._compute_projection(aspect)
        M = np.eye(4, dtype=np.float32)
        MVP = P @ V @ M
        MV = V @ M
        return MVP, MV

    # ------------------------------ Draw ------------------------------
    def render(self, config: RenderConfig) -> np.ndarray:
        if config.mode == RenderMode.CLASS_ID_INTEGER:
            return self._render_integer_ids(config)
        elif config.mode == RenderMode.DEPTH_TO_COLOR:
            return self._render_depth_to_color(config)
        else:
            return self._render_color(config)

    def _render_color(self, config: RenderConfig) -> np.ndarray:
        fbo = self._ensure_fbo_rgba(self.ctx.width, self.ctx.height)
        fbo.use()
        self.gl.clear(*config.background)

        if config.disable_cull:
            self.gl.disable(moderngl.CULL_FACE)

        MVP, MV = self._compute_matrices()
        self.prog_rgba['u_mvp'].write(MVP.T.astype('f4').tobytes())
        self.prog_rgba['u_mv'].write(MV.T.astype('f4').tobytes())
        self.prog_rgba['u_depth_near'].value = float(self.camera.near)
        self.prog_rgba['u_depth_far'].value = float(self.camera.far)
        self.prog_rgba['u_render_mode'].value = 0

        if self.color_mode == 1:
            for dr in self.draw_ranges:
                if not self.layer_visible.get(dr.cls, True):
                    continue
                self.prog_rgba['u_color'].value = tuple(dr.color)
                self.vao_rgba.render(moderngl.TRIANGLES, vertices=dr.count, first=dr.start)
        else:
            self.prog_rgba['u_color'].value = (0.82, 0.82, 0.90)
            for dr in self.draw_ranges:
                if not self.layer_visible.get(dr.cls, True):
                    continue
                self.vao_rgba.render(moderngl.TRIANGLES, vertices=dr.count, first=dr.start)

        if config.disable_cull:
            self.gl.enable(moderngl.CULL_FACE)

        data = fbo.color_attachments[0].read()
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.ctx.height, self.ctx.width, 4)
        return np.flipud(img).copy()

    def _render_depth_to_color(self, config: RenderConfig) -> np.ndarray:
        fbo = self._ensure_fbo_rgba(self.ctx.width, self.ctx.height)
        fbo.use()
        self.gl.clear(*config.background)

        if config.disable_cull:
            self.gl.disable(moderngl.CULL_FACE)

        MVP, MV = self._compute_matrices()
        self.prog_rgba['u_mvp'].write(MVP.T.astype('f4').tobytes())
        self.prog_rgba['u_mv'].write(MV.T.astype('f4').tobytes())
        self.prog_rgba['u_depth_near'].value = float(self.camera.near)
        self.prog_rgba['u_depth_far'].value = float(self.camera.far)
        self.prog_rgba['u_render_mode'].value = 1

        for dr in self.draw_ranges:
            if not self.layer_visible.get(dr.cls, True):
                continue
            self.vao_rgba.render(moderngl.TRIANGLES, vertices=dr.count, first=dr.start)

        if config.disable_cull:
            self.gl.enable(moderngl.CULL_FACE)

        data = fbo.color_attachments[0].read()
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.ctx.height, self.ctx.width, 4)
        img = np.flipud(img)
        return img[..., 0].astype(np.float32) / 255.0

    def _render_integer_ids(self, config: RenderConfig) -> np.ndarray:
        fbo = self._ensure_fbo_integer(self.ctx.width, self.ctx.height)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.gl.disable(moderngl.CULL_FACE)

        MVP, MV = self._compute_matrices()
        self.prog_integer['u_mvp'].write(MVP.T.astype('f4').tobytes())
        self.prog_integer['u_mv'].write(MV.T.astype('f4').tobytes())

        for dr in self.draw_ranges:
            if not self.layer_visible.get(dr.cls, True):
                continue
            cid = int(config.class_to_id.get(dr.cls, 0))
            cid = max(0, min(255, cid))
            self.prog_integer['u_class_id'].value = cid
            self.vao_integer.render(moderngl.TRIANGLES, vertices=dr.count, first=dr.start)

        self.gl.enable(moderngl.CULL_FACE)

        data = fbo.color_attachments[0].read()
        mask = np.frombuffer(data, dtype=np.uint8).reshape(self.ctx.height, self.ctx.width)
        return np.flipud(mask).copy()

    def render_rgba(self, *, background=(0.0, 0.0, 0.0, 0.0), also_return_depth=False, 
                    depth_factor: int = 2) -> Union[np.ndarray, tuple]:
        """Render current scene to RGBA8 numpy array [H, W, 4]."""
        config = RenderConfig(mode=RenderMode.COLOR, background=background)
        rgba = self.render(config)

        if also_return_depth:
            depth = self.render_depth(factor=depth_factor)
            return rgba, depth
        return rgba

    def render_depth(self, *, factor: int = 2) -> np.ndarray:
        """Render depth map to numpy array [H, W] in [0, 1] with supersampling."""
        factor = int(max(1, factor))
        return self._render_depth_supersampled(factor)

    def _render_depth_supersampled(self, factor: int) -> np.ndarray:
        W, H = self.ctx.width, self.ctx.height
        Whi, Hhi = W * factor, H * factor

        fbo = self._ensure_fbo_depth(Whi, Hhi)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.gl.disable(moderngl.CULL_FACE)

        prev_aspect = self.aspect
        self.aspect = max(1e-6, Whi / float(Hhi))
        MVP, MV = self._compute_matrices()

        self.prog_depth['u_mvp'].write(MVP.T.astype('f4').tobytes())
        self.prog_depth['u_mv'].write(MV.T.astype('f4').tobytes())
        self.prog_depth['u_depth_near'].value = float(self.camera.near)
        self.prog_depth['u_depth_far'].value = float(self.camera.far)

        for dr in self.draw_ranges:
            if not self.layer_visible.get(dr.cls, True):
                continue
            self.vao_depth.render(moderngl.TRIANGLES, vertices=dr.count, first=dr.start)

        self.gl.enable(moderngl.CULL_FACE)
        self.aspect = prev_aspect

        data = fbo.color_attachments[0].read()
        depth_hi = np.frombuffer(data, dtype=np.float32).reshape(Hhi, Whi)
        depth_hi = np.flipud(depth_hi)

        if factor == 1:
            return depth_hi.copy()
        return average_pool_depth(depth_hi, factor, prefer_valid=True)

    def render_class_id_mask_msaa_safe(self, class_to_id: Dict[str, int]) -> np.ndarray:
        """Render per-pixel class id mask."""
        config = RenderConfig(mode=RenderMode.CLASS_ID_INTEGER, background=(0.0, 0.0, 0.0, 0.0),
                              class_to_id=class_to_id)
        return self.render(config)

    def _render_integer_ids_supersampled(self, class_to_id: Dict[str, int], factor: int) -> np.ndarray:
        W, H = self.ctx.width, self.ctx.height
        Whi, Hhi = W * factor, H * factor

        fbo = self._ensure_fbo_integer(Whi, Hhi)
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.gl.disable(moderngl.CULL_FACE)

        prev_aspect = self.aspect
        self.aspect = max(1e-6, Whi / float(Hhi))
        MVP, MV = self._compute_matrices()

        self.prog_integer['u_mvp'].write(MVP.T.astype('f4').tobytes())
        self.prog_integer['u_mv'].write(MV.T.astype('f4').tobytes())

        for dr in self.draw_ranges:
            if not self.layer_visible.get(dr.cls, True):
                continue
            cid = int(class_to_id.get(dr.cls, 0))
            cid = max(0, min(255, cid))
            self.prog_integer['u_class_id'].value = cid
            self.vao_integer.render(moderngl.TRIANGLES, vertices=dr.count, first=dr.start)

        self.gl.enable(moderngl.CULL_FACE)
        self.aspect = prev_aspect

        data = fbo.color_attachments[0].read()
        mask_hi = np.frombuffer(data, dtype=np.uint8).reshape(Hhi, Whi)
        mask_hi = np.flipud(mask_hi)

        return majority_pool_uint8(mask_hi, factor, prefer_nonzero=True)

    def render_class_id_mask_supersampled(self, class_to_id: Dict[str, int], factor: int = 2) -> np.ndarray:
        """Render class IDs with supersampling and majority-pool downsampling."""
        factor = int(max(1, factor))
        if factor == 1:
            return self.render_class_id_mask_msaa_safe(class_to_id)
        return self._render_integer_ids_supersampled(class_to_id, factor)

    # ------------------------------- QC -------------------------------
    def qc_fraction_in_frame(self) -> float:
        """Fast QC estimate: fraction of AABB-derived points inside clip volume."""
        if self.vertex_count == 0:
            return 0.0
        corners = _aabb_corners(self.bounds_min, self.bounds_max)
        extras = [
            self.center,
            np.array([self.bounds_min[0], self.center[1], self.center[2]]),
            np.array([self.bounds_max[0], self.center[1], self.center[2]]),
            np.array([self.center[0], self.bounds_min[1], self.center[2]]),
            np.array([self.center[0], self.bounds_max[1], self.center[2]]),
            np.array([self.center[0], self.center[1], self.bounds_min[2]]),
            np.array([self.center[0], self.center[1], self.bounds_max[2]]),
            0.5 * (self.bounds_min + self.bounds_max),
        ]
        pts = np.vstack([corners, np.stack(extras, axis=0)]).astype(np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts4 = np.hstack([pts, ones])
        V = self.camera.view()
        P = self._compute_projection()
        MVP = P @ V
        clip = (MVP @ pts4.T).T
        ndc = clip[:, :3] / np.maximum(1e-6, clip[:, 3:4])
        inside = np.logical_and.reduce((
            ndc[:, 0] >= -1, ndc[:, 0] <= 1,
            ndc[:, 1] >= -1, ndc[:, 1] <= 1,
            ndc[:, 2] >= -1, ndc[:, 2] <= 1,
        ))
        return float(np.count_nonzero(inside)) / float(ndc.shape[0])

# ============================= Convenience =============================

def render_swc_to_image(
    swc_path: Path | str,
    *,
    width: int = 1024,
    height: int = 768,
    segments: int = 32,
    perspective: bool = False,
    fovy: float = 55.0,
    ortho_margin: float = 0.85,
    depth_shading: bool = False,
    color_by_type: bool = True,
    background: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> Tuple[np.ndarray, float]:
    """One-shot helper for headless pipelines. Returns (rgba_uint8, qc_fraction)."""
    with OffscreenContext(width, height, visible=False) as ctx:
        core = NeuroRenderCore(ctx)
        core.load_swc(Path(swc_path), segments=segments)
        core.set_projection(perspective=perspective, fovy=fovy)
        core.depth_shading = depth_shading
        core.color_mode = 1 if color_by_type else 0
        core.fit_camera(margin=ortho_margin)
        img = core.render_rgba(background=background)
        qc = core.qc_fraction_in_frame()
        return img, qc


if __name__ == "__main__":
    import argparse
    from imageio.v2 import imwrite

    parser = argparse.ArgumentParser()
    parser.add_argument("swc", type=str)
    parser.add_argument("--out", type=str, default="render.png")
    parser.add_argument("--w", type=int, default=1280)
    parser.add_argument("--h", type=int, default=900)
    parser.add_argument("--persp", action="store_true")
    parser.add_argument("--fov", type=float, default=55.0)
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.85)
    parser.add_argument("--depth", action="store_true")
    args = parser.parse_args()

    rgba, qc = render_swc_to_image(
        args.swc,
        width=args.w,
        height=args.h,
        segments=args.segments,
        perspective=args.persp,
        fovy=args.fov,
        ortho_margin=args.margin,
        depth_shading=args.depth,
        color_by_type=True,
        background=(0.0, 0.0, 0.0, 0.0),
    )
    imwrite(args.out, rgba)
    print(f"Saved {args.out} | QC={qc*100:.1f}%")
