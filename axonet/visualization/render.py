"""
Neuro Render Core — UI-agnostic OpenGL renderer for SWC neurons

This module cleanly separates **rendering/scene math** from any UI code.
It can be used by:
  1) an interactive UI (e.g., pyglet window w/ events), and
  2) a headless dataset generator (offscreen renders to numpy arrays / PNGs).

Key concepts
------------
• OffscreenContext: minimal GL context manager (visible window or headless).
• NeuroRenderCore: loads SWC, builds triangle meshes, manages camera, 
  draws with orthographic/perspective projection, and computes QC metrics.
• No input handling, no buttons, no window events — pure engine.

Dependencies
------------
- numpy, pyglet, and your project-local modules:
    from .mesh import MeshRenderer
    from ..io import load_swc, NeuronClass

Notes
-----
- For headless: set pyglet.options["headless"] = True **before** importing pyglet.gl
  (You can toggle this via OffscreenContext(headless=True)).
- Color management: outputs linear to sRGB framebuffer if available; offscreen
  FBO path converts to 8-bit RGBA.
"""

from __future__ import annotations

import math
import ctypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pyglet
from pyglet import gl

# ---- Project-local imports ----
from .mesh import MeshRenderer
from ..io import load_swc, NeuronClass

COLORS = {
    'red': np.array([1.0, 0.0, 0.0]).astype(np.float32),
    'green': np.array([0.0, 1.0, 0.0]).astype(np.float32),
    'blue': np.array([0.0, 0.0, 1.0]).astype(np.float32),
    'yellow': np.array([1.0, 1.0, 0.0]).astype(np.float32),
    'purple': np.array([1.0, 0.0, 1.0]).astype(np.float32),
    'orange': np.array([1.0, 0.5, 0.0]).astype(np.float32),
    'brown': np.array([0.5, 0.25, 0.0]).astype(np.float32),
    'gray': np.array([0.5, 0.5, 0.5]).astype(np.float32),
    'black': np.array([0.0, 0.0, 0.0]).astype(np.float32),
    'white': np.array([1.0, 1.0, 1.0]).astype(np.float32),
}

# ============================ Math helpers ============================

def majority_pool_uint8(mask_hi: np.ndarray, factor: int, *, prefer_nonzero: bool = True) -> np.ndarray:
    """Downsample (H*factor, W*factor) -> (H, W) by majority voting within each factor×factor block.
    
    Args:
        mask_hi: uint8 array of shape (H*factor, W*factor)
        factor: Downsampling factor (must divide image dimensions)
        prefer_nonzero: If True, prefer nonzero labels over background (0) in ties
    
    Returns:
        uint8 array of shape (H, W) with majority vote per block
    """
    Hh, Wh = mask_hi.shape
    assert Hh % factor == 0 and Wh % factor == 0, f"Supersample factor {factor} must divide image size ({Hh}, {Wh})"
    H, W = Hh // factor, Wh // factor

    blocks = mask_hi.reshape(H, factor, W, factor)
    blocks = np.transpose(blocks, (0, 2, 1, 3)).reshape(H, W, factor * factor)

    out = np.zeros((H, W), dtype=np.uint8)

    for i in range(H):
        row = blocks[i]
        for j in range(W):
            vals = row[j]
            if prefer_nonzero:
                nz = vals[vals != 0]
                if nz.size:
                    uniq, counts = np.unique(nz, return_counts=True)
                    winner = uniq[np.argmax(counts)]
                    out[i, j] = winner
                    continue
                out[i, j] = 0
            else:
                uniq, counts = np.unique(vals, return_counts=True)
                arg = np.argmax(counts)
                maxc = counts[arg]
                winner = np.min(uniq[counts == maxc])
                out[i, j] = winner

    return out

def average_pool_depth(depth_hi: np.ndarray, factor: int, *, prefer_valid: bool = True, background_threshold: float = 0.999) -> np.ndarray:
    """Downsample (H*factor, W*factor) -> (H, W) by averaging depth within each factor×factor block.
    
    Args:
        depth_hi: float32 array of shape (H*factor, W*factor) with depth values in [0, 1]
        factor: Downsampling factor (must divide image dimensions)
        prefer_valid: If True, prefer valid depth values over background in averaging
        background_threshold: Values >= this threshold are considered background
    
    Returns:
        float32 array of shape (H, W) with averaged depth per block
    """
    Hh, Wh = depth_hi.shape
    assert Hh % factor == 0 and Wh % factor == 0, f"Supersample factor {factor} must divide image size ({Hh}, {Wh})"
    H, W = Hh // factor, Wh // factor

    blocks = depth_hi.reshape(H, factor, W, factor)
    blocks = np.transpose(blocks, (0, 2, 1, 3)).reshape(H, W, factor * factor)

    out = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        row = blocks[i]
        for j in range(W):
            vals = row[j]
            if prefer_valid:
                valid = vals[vals < background_threshold]
                if valid.size > 0:
                    out[i, j] = np.mean(valid)
                else:
                    out[i, j] = np.mean(vals)
            else:
                out[i, j] = np.mean(vals)

    return out

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

VERT_SRC = b"""
#version 330 core
layout(location=0) in vec3 position;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

flat out float v_viewz;

void main(){
    vec4 world = u_model * vec4(position, 1.0);
    vec4 viewpos = u_view * world;
    v_viewz = viewpos.z;
    gl_Position = u_proj * viewpos;
}
"""

FRAG_SRC_RGBA = b"""
#version 330 core
flat in float v_viewz;

uniform vec3 u_color;
uniform float u_depth_near;
uniform float u_depth_far;
uniform int  u_render_mode; // 0=color, 1=depth-to-color

out vec4 FragColor;

void main(){
    if (u_render_mode == 1){
        float d = clamp(((-v_viewz) - u_depth_near) / max(1e-6, (u_depth_far - u_depth_near)), 0.0, 1.0);
        FragColor = vec4(d, d, d, 1.0);
        return;
    }

    vec3 color = u_color;
    FragColor = vec4(color, 1.0);
}
"""

FRAG_SRC_INTEGER = b"""
#version 330 core

uniform int u_class_id;  // class id for mask mode (0-255)

out uint FragColor;  // Single uint output for integer IDs

void main(){
    FragColor = uint(u_class_id);
}
"""

# ======================== GL utility helpers ========================

def _as_glmat(m: np.ndarray):
    m = np.asarray(m, dtype=np.float32)
    return (gl.GLfloat * 16)(*m.flatten())


def _compile(source_bytes: bytes, shader_type):
    shader = gl.glCreateShader(shader_type)
    src = ctypes.create_string_buffer(source_bytes)
    src_ptr = ctypes.cast(ctypes.pointer(ctypes.pointer(src)), ctypes.POINTER(ctypes.POINTER(gl.GLchar)))
    gl.glShaderSource(shader, 1, src_ptr, None)
    gl.glCompileShader(shader)

    ok = gl.GLint()
    gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, ctypes.byref(ok))
    if not ok.value:
        log_len = gl.GLint()
        gl.glGetShaderiv(shader, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        log = ctypes.create_string_buffer(max(1, log_len.value))
        gl.glGetShaderInfoLog(shader, log_len, None, log)
        raise RuntimeError(f"Shader compile error ({shader_type}):\n{log.value.decode()}")
    return shader


def _make_program_rgba():
    """Create shader program for RGBA rendering."""
    vs = _compile(VERT_SRC, gl.GL_VERTEX_SHADER)
    fs = _compile(FRAG_SRC_RGBA, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)

    ok = gl.GLint()
    gl.glGetProgramiv(prog, gl.GL_LINK_STATUS, ctypes.byref(ok))
    if not ok.value:
        log_len = gl.GLint()
        gl.glGetProgramiv(prog, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        log = ctypes.create_string_buffer(max(1, log_len.value))
        gl.glGetProgramInfoLog(prog, log_len, None, log)
        raise RuntimeError("Program link error:\n" + log.value.decode())

    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return prog

def _make_program_integer():
    """Create shader program for integer ID rendering."""
    vs = _compile(VERT_SRC, gl.GL_VERTEX_SHADER)
    fs = _compile(FRAG_SRC_INTEGER, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)

    ok = gl.GLint()
    gl.glGetProgramiv(prog, gl.GL_LINK_STATUS, ctypes.byref(ok))
    if not ok.value:
        log_len = gl.GLint()
        gl.glGetProgramiv(prog, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        log = ctypes.create_string_buffer(max(1, log_len.value))
        gl.glGetProgramInfoLog(prog, log_len, None, log)
        raise RuntimeError("Program link error:\n" + log.value.decode())

    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return prog

# ============================ Core types ============================

class RenderMode(Enum):
    """Render output modes."""
    COLOR = 0
    DEPTH_TO_COLOR = 1
    CLASS_ID_INTEGER = 3

@dataclass
class RenderConfig:
    """Configuration for rendering."""
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
    """A tiny GL context manager. Use visible=False for headless rendering."""

    def __init__(self, width: int, height: int, *, visible: bool = False, samples: int = 16):
        self.width = int(width)
        self.height = int(height)
        self.visible = bool(visible)
        self.samples = int(samples)
        # sRGB + MSAA config if available
        cfg = gl.Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=samples,
                        major_version=3, minor_version=3, forward_compatible=True)
        self.window = pyglet.window.Window(self.width, self.height, "NeuroRenderCore", visible=self.visible, config=cfg)
        gl.glEnable(gl.GL_FRAMEBUFFER_SRGB)
        gl.glViewport(0, 0, self.width, self.height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        
        # Enable antialiasing
        if samples > 1:
            print("MSAA is configured and enabled")
            gl.glEnable(gl.GL_MULTISAMPLE)
        
        # Enable blending for smooth edges (required for polygon/line smoothing)
        # gl.glEnable(gl.GL_BLEND)
        # gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Prefer MSAA over polygon/line smoothing to avoid gaps on thin geometry
        # gl.glEnable(gl.GL_POLYGON_SMOOTH)
        # gl.glEnable(gl.GL_LINE_SMOOTH)
        # gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
        # gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

    def make_current(self):
        # pyglet makes the context current on creation; no-op kept for symmetry
        pass

    def close(self):
        if self.window:
            self.window.close()
            self.window = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def create_integer_fbo(self) -> Tuple[int, int, int]:
        """Create integer FBO for MSAA-safe ID rendering.
        Returns (fbo_id, color_tex_id, depth_rbo_id).
        """
        fbo = gl.GLuint()
        gl.glGenFramebuffers(1, ctypes.byref(fbo))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        
        # Create integer color texture
        color_tex = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(color_tex))
        # For integer IDs, disable MSAA to avoid format compatibility issues
        # Use single-sample rendering for pixel-perfect integer IDs
        gl.glBindTexture(gl.GL_TEXTURE_2D, color_tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R8UI, self.width, self.height, 0, 
                      gl.GL_RED_INTEGER, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, 
                                gl.GL_TEXTURE_2D, color_tex, 0)
        
        # Create depth renderbuffer
        depth_rbo = gl.GLuint()
        gl.glGenRenderbuffers(1, ctypes.byref(depth_rbo))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rbo)
        # Use single-sample depth buffer for integer FBOs
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, 
                               self.width, self.height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, 
                                   gl.GL_RENDERBUFFER, depth_rbo)
        
        # Check completeness
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Integer FBO incomplete: {status}")
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return int(fbo.value), int(color_tex.value), int(depth_rbo.value)

class NeuroRenderCore:
    """UI-agnostic renderer for SWC neurons.

    Typical use:
        with OffscreenContext(W, H, visible=False) as ctx:
            core = NeuroRenderCore(ctx)
            core.load_swc(path)
            core.fit_camera(margin=1.15)
            rgba = core.render_rgba()   # np.uint8 [H, W, 4]
            qc = core.qc_fraction_in_frame()
    """

    def __init__(self, ctx: OffscreenContext):
        self.ctx = ctx
        self.program_rgba = _make_program_rgba()
        self.program_integer = _make_program_integer()

        # RGBA program uniforms
        self.uni_model_rgba = gl.glGetUniformLocation(self.program_rgba, b"u_model")
        self.uni_view_rgba = gl.glGetUniformLocation(self.program_rgba, b"u_view")
        self.uni_proj_rgba = gl.glGetUniformLocation(self.program_rgba, b"u_proj")
        self.uni_color_rgba = gl.glGetUniformLocation(self.program_rgba, b"u_color")
        self.uni_depth_near_rgba = gl.glGetUniformLocation(self.program_rgba, b"u_depth_near")
        self.uni_depth_far_rgba = gl.glGetUniformLocation(self.program_rgba, b"u_depth_far")
        self.uni_render_mode_rgba = gl.glGetUniformLocation(self.program_rgba, b"u_render_mode")

        # Integer program uniforms
        self.uni_model_int = gl.glGetUniformLocation(self.program_integer, b"u_model")
        self.uni_view_int = gl.glGetUniformLocation(self.program_integer, b"u_view")
        self.uni_proj_int = gl.glGetUniformLocation(self.program_integer, b"u_proj")
        self.uni_class_id_int = gl.glGetUniformLocation(self.program_integer, b"u_class_id")

        # buffers
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(self.vao))
        gl.glBindVertexArray(self.vao)
        self.vbo_pos = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(self.vbo_pos))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # scene data
        self.neuron = None
        self.draw_ranges: list[DrawRange] = []
        self.vertex_count = 0
        self.bounds_min = np.array([0, 0, 0], dtype=np.float32)
        self.bounds_max = np.array([1, 1, 1], dtype=np.float32)
        self.center = np.zeros(3, dtype=np.float32)
        self.diag = 1.0

        # camera
        self.camera = Camera(
            eye=np.array([0.0, 0.0, 3.0], dtype=np.float32),
            target=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            up=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.aspect = max(1e-6, self.ctx.width / float(self.ctx.height))

        # style options
        self.depth_shading = False
        self.color_mode = 0  # 0 single, 1 by type
        self.layer_visible: Dict[str, bool] = {
            NeuronClass.SOMA.name: True,
            NeuronClass.AXON.name: True,
            NeuronClass.BASAL_DENDRITE.name: True,
            NeuronClass.APICAL_DENDRITE.name: True,
            NeuronClass.OTHER.name: True,
        }

        # colors
        self._cls_colors: Dict[str, np.ndarray] = {
            NeuronClass.SOMA.name: COLORS['red'],
            NeuronClass.AXON.name: COLORS['green'],
            NeuronClass.BASAL_DENDRITE.name: COLORS['blue'],
            NeuronClass.APICAL_DENDRITE.name: COLORS['purple'],
            NeuronClass.OTHER.name: COLORS['orange'],
        }

        # cached CPU arrays for QC if needed
        self._pos_cpu: Optional[np.ndarray] = None
        
        # FBO for integer rendering
        self._integer_fbo: Optional[Tuple[int, int, int]] = None

    # ------------------------- Data / geometry -------------------------
    def load_swc(self, path: Path, *, segments: int = 18, radius_scale: float = 1.0, radius_adaptive_alpha: float = 0.0, radius_ref_percentile: float = 50.0) -> None:
        neuron = load_swc(Path(path), validate=True)
        self.set_neuron(neuron, segments=segments, radius_scale=radius_scale, radius_adaptive_alpha=radius_adaptive_alpha, radius_ref_percentile=radius_ref_percentile)

    def set_neuron(self, neuron, *, segments: int = 18, radius_scale: float = 1.0, radius_adaptive_alpha: float = 0.0, radius_ref_percentile: float = 50.0) -> None:
        self.neuron = neuron
        renderer = MeshRenderer(neuron)
        by_type = renderer.build_mesh_by_type(segments=segments, cap=True, radius_scale=radius_scale, radius_adaptive_alpha=radius_adaptive_alpha, radius_ref_percentile=radius_ref_percentile)

        order = [
            NeuronClass.SOMA.name,
            NeuronClass.AXON.name,
            NeuronClass.BASAL_DENDRITE.name,
            NeuronClass.APICAL_DENDRITE.name,
            NeuronClass.OTHER.name,
        ]

        self.draw_ranges.clear()
        positions = []
        bmin = None; bmax = None
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

        pos_all = np.concatenate(positions, axis=0)
        self.vertex_count = int(pos_all.shape[0])
        self.bounds_min = bmin
        self.bounds_max = bmax
        self.center = 0.5 * (bmin + bmax)
        self.diag = float(np.linalg.norm(bmax - bmin))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        buf_t = (gl.GLfloat * pos_all.size)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, pos_all.nbytes, buf_t.from_buffer(pos_all), gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # cache for QC if you prefer sampling vertices instead of AABB corners
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

    def set_projection(self, *, perspective: Optional[bool] = None, fovy: Optional[float] = None, ortho_scale: Optional[float] = None):
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
        # factor < 1 zoom in, >1 zoom out
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
    def _compute_projection(self) -> np.ndarray:
        aspect = max(1e-6, self.aspect)
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

    # ------------------------------ Draw ------------------------------
    def _setup_render_state(self, config: RenderConfig) -> Tuple[bool, bool, bool]:
        """Setup GL state for rendering. Returns (srgb_was_enabled, blend_was_enabled, cull_was_enabled)."""
        srgb_enabled = gl.glIsEnabled(gl.GL_FRAMEBUFFER_SRGB)
        blend_enabled = gl.glIsEnabled(gl.GL_BLEND)
        cull_enabled = gl.glIsEnabled(gl.GL_CULL_FACE)
        
        if config.disable_srgb and srgb_enabled:
            gl.glDisable(gl.GL_FRAMEBUFFER_SRGB)
        if config.disable_blend and blend_enabled:
            gl.glDisable(gl.GL_BLEND)
        if config.disable_cull and cull_enabled:
            gl.glDisable(gl.GL_CULL_FACE)
            
        return srgb_enabled, blend_enabled, cull_enabled
    
    def _restore_render_state(self, srgb_enabled: bool, blend_enabled: bool, cull_enabled: bool):
        """Restore GL state after rendering."""
        if srgb_enabled:
            gl.glEnable(gl.GL_FRAMEBUFFER_SRGB)
        if blend_enabled:
            gl.glEnable(gl.GL_BLEND)
        if cull_enabled:
            gl.glEnable(gl.GL_CULL_FACE)
    
    def _setup_uniforms_rgba(self, config: RenderConfig):
        """Setup RGBA shader uniforms for rendering."""
        V = self.camera.view()
        P = self._compute_projection()
        M = np.eye(4, dtype=np.float32)

        gl.glUniformMatrix4fv(self.uni_model_rgba, 1, gl.GL_TRUE, _as_glmat(M))
        gl.glUniformMatrix4fv(self.uni_view_rgba, 1, gl.GL_TRUE, _as_glmat(V))
        gl.glUniformMatrix4fv(self.uni_proj_rgba, 1, gl.GL_TRUE, _as_glmat(P))
        gl.glUniform1f(self.uni_depth_near_rgba, float(self.camera.near))
        gl.glUniform1f(self.uni_depth_far_rgba, float(self.camera.far))
        gl.glUniform1i(self.uni_render_mode_rgba, config.mode.value)

    def _setup_uniforms_integer(self, config: RenderConfig):
        """Setup integer shader uniforms for rendering."""
        V = self.camera.view()
        P = self._compute_projection()
        M = np.eye(4, dtype=np.float32)

        gl.glUniformMatrix4fv(self.uni_model_int, 1, gl.GL_TRUE, _as_glmat(M))
        gl.glUniformMatrix4fv(self.uni_view_int, 1, gl.GL_TRUE, _as_glmat(V))
        gl.glUniformMatrix4fv(self.uni_proj_int, 1, gl.GL_TRUE, _as_glmat(P))

    def render(self, config: RenderConfig) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Unified render method with different strategies."""
        if config.mode == RenderMode.CLASS_ID_INTEGER:
            return self._render_integer_ids(config)
        elif config.mode == RenderMode.DEPTH_TO_COLOR:
            return self._render_depth_to_color(config)
        else:  # COLOR
            return self._render_color(config)

    def _render_color(self, config: RenderConfig) -> np.ndarray:
        """Render color image."""
        gl.glViewport(0, 0, self.ctx.width, self.ctx.height)
        gl.glClearColor(*config.background)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        srgb_enabled, blend_enabled, cull_enabled = self._setup_render_state(config)
        
        gl.glBindVertexArray(self.vao)
        gl.glUseProgram(self.program_rgba)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        
        self._setup_uniforms_rgba(config)
        
        if self.color_mode == 1:
            for dr in self.draw_ranges:
                if not self.layer_visible.get(dr.cls, True):
                    continue
                gl.glUniform3f(self.uni_color_rgba, float(dr.color[0]), float(dr.color[1]), float(dr.color[2]))
                gl.glDrawArrays(gl.GL_TRIANGLES, dr.start, dr.count)
        else:
            gl.glUniform3f(self.uni_color_rgba, 0.82, 0.82, 0.90)
            for dr in self.draw_ranges:
                if not self.layer_visible.get(dr.cls, True):
                    continue
                gl.glDrawArrays(gl.GL_TRIANGLES, dr.start, dr.count)

        gl.glFinish()
        
        buffer = (gl.GLubyte * (self.ctx.width * self.ctx.height * 4))()
        gl.glReadPixels(0, 0, self.ctx.width, self.ctx.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, buffer)
        img = np.frombuffer(buffer, dtype=np.uint8).reshape(self.ctx.height, self.ctx.width, 4)
        img = np.flipud(img)
        
        self._restore_render_state(srgb_enabled, blend_enabled, cull_enabled)
        return img

    def _render_depth_to_color(self, config: RenderConfig) -> np.ndarray:
        """Render depth as color."""
        gl.glViewport(0, 0, self.ctx.width, self.ctx.height)
        gl.glClearColor(*config.background)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        srgb_enabled, blend_enabled, cull_enabled = self._setup_render_state(config)
        
        gl.glBindVertexArray(self.vao)
        gl.glUseProgram(self.program_rgba)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        
        self._setup_uniforms_rgba(config)

        for dr in self.draw_ranges:
            if not self.layer_visible.get(dr.cls, True):
                continue
            gl.glUniform3f(self.uni_color_rgba, 1.0, 1.0, 1.0)
            gl.glDrawArrays(gl.GL_TRIANGLES, dr.start, dr.count)

        gl.glFinish()

        buffer = (gl.GLubyte * (self.ctx.width * self.ctx.height * 4))()
        gl.glReadPixels(0, 0, self.ctx.width, self.ctx.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, buffer)
        img = np.frombuffer(buffer, dtype=np.uint8).reshape(self.ctx.height, self.ctx.width, 4)
        img = np.flipud(img)
        depth01 = img[..., 0].astype(np.float32) / 255.0
        
        self._restore_render_state(srgb_enabled, blend_enabled, cull_enabled)
        return depth01

    def _render_integer_ids(self, config: RenderConfig) -> np.ndarray:
        """Render class IDs to integer FBO (single-sample)."""
        if self._integer_fbo is None:
            self._integer_fbo = self.ctx.create_integer_fbo()
        
        msaa_fbo, msaa_tex, msaa_depth = self._integer_fbo
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, msaa_fbo)
        gl.glViewport(0, 0, self.ctx.width, self.ctx.height)
        
        # Clear integer color buffer properly
        clear_value = (gl.GLuint * 4)(0, 0, 0, 0)
        gl.glClearBufferuiv(gl.GL_COLOR, 0, clear_value)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        srgb_enabled, blend_enabled, cull_enabled = self._setup_render_state(config)
        
        gl.glBindVertexArray(self.vao)
        gl.glUseProgram(self.program_integer)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        
        self._setup_uniforms_integer(config)

        for dr in self.draw_ranges:
            if not self.layer_visible.get(dr.cls, True):
                continue
            cid = int(config.class_to_id.get(dr.cls, 0))
            cid = max(0, min(255, cid))
            gl.glUniform1i(self.uni_class_id_int, cid)
            gl.glDrawArrays(gl.GL_TRIANGLES, dr.start, dr.count)

        gl.glFinish()
        
        # Integer FBOs use single-sample for pixel-perfect IDs
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, msaa_fbo)
        
        # Read back integer IDs directly
        buffer = (gl.GLubyte * (self.ctx.width * self.ctx.height))()
        gl.glReadPixels(0, 0, self.ctx.width, self.ctx.height, gl.GL_RED_INTEGER, gl.GL_UNSIGNED_BYTE, buffer)
        mask = np.frombuffer(buffer, dtype=np.uint8).reshape(self.ctx.height, self.ctx.width)
        mask = np.flipud(mask)
        
        # Restore default framebuffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        
        self._restore_render_state(srgb_enabled, blend_enabled, cull_enabled)
        return mask

    def render_rgba(self, *, background=(0.0, 0.0, 0.0, 0.0), also_return_depth=False, depth_factor: int = 2) -> Union[np.ndarray, tuple]:
        """Render current scene to RGBA8 numpy array [H, W, 4].
        If also_return_depth=True, returns (rgba, depth) as tuple.
        
        Args:
            background: Background color as (R, G, B, A) tuple
            also_return_depth: If True, also return depth map
            depth_factor: Supersampling factor for depth (2, 3, or 4 recommended)
        """
        config = RenderConfig(mode=RenderMode.COLOR, background=background)
        rgba = self.render(config)
        
        if also_return_depth:
            depth = self.render_depth(factor=depth_factor)
            return rgba, depth
        return rgba
    
    def render_depth(self, *, factor: int = 2) -> np.ndarray:
        """Render depth map to numpy array [H, W] in [0, 1] with supersampling.
        
        Args:
            factor: Supersampling factor (2, 3, or 4 recommended). If factor=1, uses single-sample.
        
        Returns:
            float32 array of shape (H, W) with depth values in [0, 1]
        """
        factor = int(max(1, factor))
        if factor == 1:
            config = RenderConfig(mode=RenderMode.DEPTH_TO_COLOR, background=(0.0, 0.0, 0.0, 0.0),
                                disable_srgb=True, disable_blend=True, disable_cull=True)
            return self.render(config)
        return self._render_depth_supersampled(factor)

    def _render_depth_supersampled(self, factor: int) -> np.ndarray:
        """Render depth at (W*factor, H*factor) then average-pool down to (H, W)."""
        W, H = self.ctx.width, self.ctx.height
        Whi, Hhi = W * factor, H * factor

        fbo = gl.GLuint()
        gl.glGenFramebuffers(1, ctypes.byref(fbo))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        color_tex = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(color_tex))
        gl.glBindTexture(gl.GL_TEXTURE_2D, color_tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, Whi, Hhi, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, color_tex, 0)

        depth_rbo = gl.GLuint()
        gl.glGenRenderbuffers(1, ctypes.byref(depth_rbo))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rbo)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, Whi, Hhi)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rbo)

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Supersampled depth FBO incomplete: {status}")

        gl.glViewport(0, 0, Whi, Hhi)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        prev_aspect = self.aspect
        self.aspect = max(1e-6, Whi / float(Hhi))

        config = RenderConfig(mode=RenderMode.DEPTH_TO_COLOR, background=(0.0, 0.0, 0.0, 0.0),
                            disable_srgb=True, disable_blend=True, disable_cull=True)
        srgb_enabled, blend_enabled, cull_enabled = self._setup_render_state(config)

        gl.glBindVertexArray(self.vao)
        gl.glUseProgram(self.program_rgba)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        self._setup_uniforms_rgba(config)

        for dr in self.draw_ranges:
            if not self.layer_visible.get(dr.cls, True):
                continue
            gl.glUniform3f(self.uni_color_rgba, 1.0, 1.0, 1.0)
            gl.glDrawArrays(gl.GL_TRIANGLES, dr.start, dr.count)

        gl.glFinish()

        buf = (gl.GLubyte * (Whi * Hhi * 4))()
        gl.glReadPixels(0, 0, Whi, Hhi, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, buf)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(Hhi, Whi, 4)
        img = np.flipud(img)
        depth_hi = img[..., 0].astype(np.float32) / 255.0

        gl.glViewport(0, 0, self.ctx.width, self.ctx.height)
        self._restore_render_state(srgb_enabled, blend_enabled, cull_enabled)
        self.aspect = prev_aspect

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glDeleteFramebuffers(1, ctypes.byref(fbo))
        gl.glDeleteTextures(1, ctypes.byref(color_tex))
        gl.glDeleteRenderbuffers(1, ctypes.byref(depth_rbo))

        return average_pool_depth(depth_hi, factor, prefer_valid=True)

    def render_class_id_mask_msaa_safe(self, class_to_id: Dict[str, int]) -> np.ndarray:
        """Render per-pixel class id mask using integer FBO with MSAA-safe resolve."""
        config = RenderConfig(mode=RenderMode.CLASS_ID_INTEGER, background=(0.0, 0.0, 0.0, 0.0),
                            class_to_id=class_to_id, disable_srgb=True, disable_blend=True, disable_cull=True)
        return self.render(config)

    def _render_integer_ids_supersampled(self, class_to_id: Dict[str, int], factor: int) -> np.ndarray:
        """Render integer classes at (W*factor, H*factor) then majority-pool down to (H, W)."""
        W, H = self.ctx.width, self.ctx.height
        Whi, Hhi = W * factor, H * factor

        fbo = gl.GLuint()
        gl.glGenFramebuffers(1, ctypes.byref(fbo))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)

        color_tex = gl.GLuint()
        gl.glGenTextures(1, ctypes.byref(color_tex))
        gl.glBindTexture(gl.GL_TEXTURE_2D, color_tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_R8UI, Whi, Hhi, 0, gl.GL_RED_INTEGER, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, color_tex, 0)

        depth_rbo = gl.GLuint()
        gl.glGenRenderbuffers(1, ctypes.byref(depth_rbo))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rbo)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, Whi, Hhi)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rbo)

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Supersampled integer FBO incomplete: {status}")

        gl.glViewport(0, 0, Whi, Hhi)

        clear_value = (gl.GLuint * 4)(0, 0, 0, 0)
        gl.glClearBufferuiv(gl.GL_COLOR, 0, clear_value)
        gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

        prev_aspect = self.aspect
        self.aspect = max(1e-6, Whi / float(Hhi))

        config = RenderConfig(mode=RenderMode.CLASS_ID_INTEGER, background=(0.0, 0.0, 0.0, 0.0),
                            class_to_id=class_to_id, disable_srgb=True, disable_blend=True, disable_cull=True)
        srgb_enabled, blend_enabled, cull_enabled = self._setup_render_state(config)

        gl.glBindVertexArray(self.vao)
        gl.glUseProgram(self.program_integer)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        self._setup_uniforms_integer(config)

        for dr in self.draw_ranges:
            if not self.layer_visible.get(dr.cls, True):
                continue
            cid = int(class_to_id.get(dr.cls, 0))
            cid = max(0, min(255, cid))
            gl.glUniform1i(self.uni_class_id_int, cid)
            gl.glDrawArrays(gl.GL_TRIANGLES, dr.start, dr.count)

        gl.glFinish()

        buf = (gl.GLubyte * (Whi * Hhi))()
        gl.glReadPixels(0, 0, Whi, Hhi, gl.GL_RED_INTEGER, gl.GL_UNSIGNED_BYTE, buf)
        mask_hi = np.frombuffer(buf, dtype=np.uint8).reshape(Hhi, Whi)
        mask_hi = np.flipud(mask_hi)

        gl.glViewport(0, 0, self.ctx.width, self.ctx.height)
        self._restore_render_state(srgb_enabled, blend_enabled, cull_enabled)
        self.aspect = prev_aspect

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glDeleteFramebuffers(1, ctypes.byref(fbo))
        gl.glDeleteTextures(1, ctypes.byref(color_tex))
        gl.glDeleteRenderbuffers(1, ctypes.byref(depth_rbo))

        return majority_pool_uint8(mask_hi, factor, prefer_nonzero=True)

    def render_class_id_mask_supersampled(self, class_to_id: Dict[str, int], factor: int = 2) -> np.ndarray:
        """Render class IDs with supersampling and majority-pool downsampling.
        
        Args:
            class_to_id: Dictionary mapping class names to integer IDs
            factor: Supersampling factor (2, 3, or 4 recommended). Must result in dimensions
                   divisible by the factor. If factor=1, falls back to single-sample rendering.
        
        Returns:
            uint8 array of shape (H, W) with class IDs
        """
        factor = int(max(1, factor))
        if factor == 1:
            return self.render_class_id_mask_msaa_safe(class_to_id)
        return self._render_integer_ids_supersampled(class_to_id, factor)

    # ------------------------------- QC -------------------------------
    def qc_fraction_in_frame(self) -> float:
        """Fast QC estimate: fraction of AABB-derived points inside clip volume.
        Uses 8 AABB corners + 8 axial midpoints. Optionally, you can use
        self._pos_cpu for random vertex sampling if you prefer.
        """
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
    """One-shot helper for headless pipelines.

    Returns (rgba_uint8, qc_fraction).
    """
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

# ================================ Demo =================================
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
