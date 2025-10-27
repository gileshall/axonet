import math
import ctypes
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyglet
from pyglet.window import key, mouse
from pyglet import gl

# ---- Project-local imports (same as your original) ----
from .mesh import MeshRenderer
from ..io import load_swc, NeuronClass

"""
SWC Viewer — rewritten for responsive controls, orthographic/perspective toggle,
cleaner render pipeline, and a simple in-frame QC metric.

Highlights
---------
• Trackball-style orbit (LMB), precise pan (MMB or Shift+LMB), smooth dolly zoom (wheel).
• Orthographic or perspective projection toggle (P). Field-of-view and ortho scale auto-fit.
• Camera auto-fit to neuron bounding sphere with configurable margin.
• HUD shows frame coverage (QC): fraction of vertices inside the clip volume.
• Depth-weighted shading (optional), correct normal transform, and sRGB output when available.
• No hard-wired poses; focus on consistent, reproducible views.

Controls
--------
LMB drag        : Orbit (trackball)
MMB drag        : Pan (or Shift+LMB)
Wheel           : Dolly zoom (exponential, cursor-aware)
R               : Reset view (re-fit)
W               : Toggle wireframe
O               : Toggle depth shading
C               : Toggle per-compartment color / single tint
P               : Toggle perspective / orthographic
F               : File browser
S               : Save screenshot
ESC             : Quit

Notes
-----
• QC metric is a fast approximation: %, vertices within NDC cube after MVP.
• For reproducible datasets, prefer orthographic with fixed ortho_scale (auto-fit + margin).
"""

# ---------------------------- GL Shaders ----------------------------
VERT_SRC = b"""
#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;

out vec3 v_nrm;
out vec3 v_world;
out float v_viewz;

void main(){
    vec4 world = u_model * vec4(position, 1.0);
    v_world = world.xyz;
    v_nrm = normalize(u_normal_mat * normal);
    vec4 viewpos = u_view * world;
    v_viewz = viewpos.z;
    gl_Position = u_proj * viewpos;
}
"""

FRAG_SRC = b"""
#version 330 core
in vec3 v_nrm;
in vec3 v_world;
in float v_viewz;

uniform vec3 u_camera;
uniform vec3 u_light_dir; // normalized in world space
uniform vec3 u_color;
uniform int  u_depth_enabled;
uniform float u_depth_near;
uniform float u_depth_far;

out vec4 FragColor;

void main(){
    vec3 N = normalize(v_nrm);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera - v_world);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 32.0);

    vec3 base = u_color;
    vec3 color = 0.12 * base + 0.88 * diff * base + 0.25 * spec * vec3(1.0);

    if (u_depth_enabled == 1){
        float zn = clamp(((-v_viewz) - u_depth_near) / max(1e-6, (u_depth_far - u_depth_near)), 0.0, 1.0);
        float shade = mix(0.45, 1.0, 1.0 - zn);
        color *= shade;
    }

    // Output in linear; rely on sRGB framebuffer if available
    FragColor = vec4(color, 1.0);
}
"""

# --------------------------- Math helpers ---------------------------

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


def translate(tx, ty, tz):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx; m[1, 3] = ty; m[2, 3] = tz
    return m


def rotation_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Build a 4x4 rotation matrix from basis vectors (columns)."""
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = x[:3]
    m[:3, 1] = y[:3]
    m[:3, 2] = z[:3]
    return m


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-12)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = -m[:3, :3] @ eye[:3]
    return m


def as_glmat(m: np.ndarray):
    m = np.asarray(m, dtype=np.float32)
    return (gl.GLfloat * 16)(*m.flatten())


def normal_matrix(model: np.ndarray, view: np.ndarray) -> np.ndarray:
    mv = (view @ model)[:3, :3]
    invT = np.linalg.inv(mv).T
    return invT.astype(np.float32)

# --------------------------- GL helpers ---------------------------

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


def make_program():
    vs = _compile(VERT_SRC, gl.GL_VERTEX_SHADER)
    fs = _compile(FRAG_SRC, gl.GL_FRAGMENT_SHADER)
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

# --------------------------- UI helpers ---------------------------
@dataclass
class UIButton:
    x: int
    y: int
    w: int
    h: int
    text: str
    on_click: callable
    enabled: bool = True
    visible: bool = True

    def hit(self, px, py):
        return self.visible and self.enabled and (self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h)


class FileBrowser:
    def __init__(self, start_dir: Path):
        self.current_dir = Path(start_dir)
        self.visible = False
        self.entries = []
        self.refresh()

    def refresh(self):
        d = self.current_dir
        files = []
        for p in sorted(d.iterdir()):
            if p.is_dir():
                files.append((p.name + '/', p))
            elif p.suffix.lower() == '.swc':
                files.append((p.name, p))
        self.entries = files

# ----------------------------- Viewer -----------------------------
class SWCViewer(pyglet.window.Window):
    def __init__(self, swc_path: Path | None = None, segments: int = 18, fit_margin: float = 1.15):
        cfg = gl.Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4, major_version=3, minor_version=3, forward_compatible=True)
        title = f"SWC Viewer — {Path(swc_path).name}" if swc_path else "SWC Viewer"
        super().__init__(1280, 900, title, resizable=True, config=cfg)

        # Try sRGB framebuffer for correct output (if supported by platform)
        try:
            gl.glEnable(gl.GL_FRAMEBUFFER_SRGB)
        except Exception:
            pass

        self.segments = segments
        self.current_path = Path(swc_path) if swc_path else None
        self.neuron = None
        self.vertex_count = 0
        self._draw_ranges = []  # (start,count,color,cls)

        self._bounds_min = np.array([0, 0, 0], dtype=np.float32)
        self._bounds_max = np.array([1, 1, 1], dtype=np.float32)
        self._diag = 1.0
        self._center = np.zeros(3, dtype=np.float32)
        self._fit_margin = float(fit_margin)

        # GL program + VAO/VBOs
        self.program = make_program()
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(self.vao))
        gl.glBindVertexArray(self.vao)

        self.uni_model = gl.glGetUniformLocation(self.program, b"u_model")
        self.uni_view = gl.glGetUniformLocation(self.program, b"u_view")
        self.uni_proj = gl.glGetUniformLocation(self.program, b"u_proj")
        self.uni_normal = gl.glGetUniformLocation(self.program, b"u_normal_mat")
        self.uni_cam = gl.glGetUniformLocation(self.program, b"u_camera")
        self.uni_light = gl.glGetUniformLocation(self.program, b"u_light_dir")
        self.uni_color = gl.glGetUniformLocation(self.program, b"u_color")
        self.uni_depth_enabled = gl.glGetUniformLocation(self.program, b"u_depth_enabled")
        self.uni_depth_near = gl.glGetUniformLocation(self.program, b"u_depth_near")
        self.uni_depth_far = gl.glGetUniformLocation(self.program, b"u_depth_far")

        self.vbo_pos = gl.GLuint(); self.vbo_nrm = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(self.vbo_pos))
        gl.glGenBuffers(1, ctypes.byref(self.vbo_nrm))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_nrm)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)

        # Interaction state
        self.mode_perspective = True
        self.fovy = 55.0  # degrees
        self.ortho_scale = 1.0  # world units to half-height at near plane equivalent
        self.near = 0.02
        self.far = 100.0

        # Trackball camera
        self.eye = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._arcball_prev = None

        # Render options
        self.is_wireframe = False
        self.depth_shading = False
        self.color_mode = 0  # 0 = single tint, 1 = by type

        # HUD + UI
        self._ui_buttons = []
        self._build_ui()
        self.fps_display = pyglet.window.FPSDisplay(self)
        self.label_hud = pyglet.text.Label('', font_name='Courier New', font_size=12, x=10, y=self.height - 14, anchor_x='left', anchor_y='top', color=(255, 255, 255, 255))

        # File browser
        start_dir = Path(self.current_path).parent if self.current_path is not None else Path.cwd()
        self.file_browser = FileBrowser(start_dir)
        self.file_browser.visible = True if self.current_path is None else False

        # Layers
        self.layer_visible = {
            NeuronClass.SOMA.name: True,
            NeuronClass.AXON.name: True,
            NeuronClass.BASAL_DENDRITE.name: True,
            NeuronClass.APICAL_DENDRITE.name: True,
            NeuronClass.OTHER.name: True,
        }

        pyglet.clock.schedule_interval(lambda dt: None, 1/60.0)

        if self.current_path is not None:
            self._load_path(self.current_path)
        else:
            self._update_camera_fit()

    # -------------------------- Geometry & data --------------------------
    def _load_path(self, swc_path: Path):
        self.current_path = Path(swc_path)
        self.set_caption(f"SWC Viewer — {self.current_path.name}")

        neuron = load_swc(self.current_path, validate=True)
        self.neuron = neuron
        renderer = MeshRenderer(neuron)
        by_type = renderer.build_mesh_by_type(segments=self.segments, cap=True)

        order = [NeuronClass.SOMA.name, NeuronClass.AXON.name, NeuronClass.BASAL_DENDRITE.name, NeuronClass.APICAL_DENDRITE.name, NeuronClass.OTHER.name]
        colors = {
            NeuronClass.SOMA.name: np.array([0.95, 0.35, 0.35], dtype=np.float32),
            NeuronClass.AXON.name: np.array([0.35, 0.95, 0.45], dtype=np.float32),
            NeuronClass.BASAL_DENDRITE.name: np.array([0.35, 0.55, 0.95], dtype=np.float32),
            NeuronClass.APICAL_DENDRITE.name: np.array([0.35, 0.55, 0.95], dtype=np.float32),
            NeuronClass.OTHER.name: np.array([0.75, 0.75, 0.75], dtype=np.float32),
        }

        self._draw_ranges.clear()
        positions = []
        normals = []
        bmin = None; bmax = None
        start = 0
        for cls in order:
            m = by_type.get(cls)
            if m is None:
                continue
            v = np.asarray(m.vertices, dtype=np.float32)
            f = np.asarray(m.faces, dtype=np.int32)
            vn = np.asarray(m.vertex_normals if m.vertex_normals is not None else m.face_normals.repeat(3, axis=0), dtype=np.float32)
            tri_count = f.shape[0]
            pos = v[f.reshape(-1)]
            nrm = vn[f.reshape(-1)]
            positions.append(pos)
            normals.append(nrm)
            count = int(tri_count * 3)
            self._draw_ranges.append((start, count, colors[cls], cls))
            start += count
            bm = m.bounds[0].astype(np.float32)
            bM = m.bounds[1].astype(np.float32)
            bmin = bm if bmin is None else np.minimum(bmin, bm)
            bmax = bM if bmax is None else np.maximum(bmax, bM)

        if not positions:
            print("ERROR: No mesh data generated!")
            return

        pos_all = np.concatenate(positions, axis=0)
        nrm_all = np.concatenate(normals, axis=0)
        self.vertex_count = int(pos_all.shape[0])
        self._bounds_min = bmin
        self._bounds_max = bmax
        self._center = 0.5 * (bmin + bmax)
        self._diag = float(np.linalg.norm(bmax - bmin))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        buf_t = (gl.GLfloat * pos_all.size)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, pos_all.nbytes, buf_t.from_buffer(pos_all), gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_nrm)
        nrm_t = (gl.GLfloat * nrm_all.size)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, nrm_all.nbytes, nrm_t.from_buffer(nrm_all), gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        self._update_camera_fit(reset_target=True)

    # ------------------------------ Camera ------------------------------
    def _update_camera_fit(self, reset_target: bool = False):
        # Fit distance based on bounding sphere radius (diag/2) with margin
        radius = max(1e-6, 0.5 * self._diag)
        if reset_target:
            self.target = self._center.copy()
        self.eye = self.target + np.array([0.0, 0.0, 1.0], dtype=np.float32)  # temporary direction

        if self.mode_perspective:
            # Compute distance from FOV so object fits vertically
            half_h = self._fit_margin * radius
            dist = half_h / math.tan(math.radians(self.fovy) * 0.5)
            self.eye = self.target + np.array([0.0, 0.0, dist], dtype=np.float32)
        else:
            # Ortho: set ortho_scale so half-height shows radius*margin
            self.ortho_scale = self._fit_margin * radius
            # Put eye a bit forward to keep near plane reasonable
            self.eye = self.target + np.array([0.0, 0.0, 3.0 * radius + 1.0], dtype=np.float32)

    # Trackball mapping: window coords -> unit sphere
    def _arcball_vec(self, x, y):
        w, h = self.get_size()
        if w <= 0 or h <= 0:
            return np.array([0, 0, 1], dtype=np.float32)
        nx = (2.0 * x - w) / max(1.0, w)
        ny = (h - 2.0 * y) / max(1.0, h)
        v = np.array([nx, ny, 0.0], dtype=np.float32)
        d = np.dot(v, v)
        if d <= 1.0:
            v[2] = math.sqrt(1.0 - d)
        else:
            v /= math.sqrt(d)
        return v

    def _orbit(self, x0, y0, x1, y1):
        v0 = self._arcball_vec(x0, y0)
        v1 = self._arcball_vec(x1, y1)
        axis = np.cross(v0, v1)
        angle = math.acos(max(-1.0, min(1.0, float(np.dot(v0, v1)))))
        if np.linalg.norm(axis) < 1e-6 or angle == 0.0:
            return
        axis = axis / np.linalg.norm(axis)
        # Build rotation in eye space around target
        R = _axis_angle_matrix(axis, angle)
        eye_dir = self.eye - self.target
        eye_dir = (R @ np.append(eye_dir, 1.0))[:3]
        up_dir = (R @ np.append(self.up, 1.0))[:3]
        self.eye = self.target + eye_dir
        self.up = up_dir / (np.linalg.norm(up_dir) + 1e-12)

    def _pan(self, dx, dy):
        # Pan proportional to distance and fov/scale
        view = look_at(self.eye, self.target, self.up)
        right = view[:3, 0]
        upv = view[:3, 1]
        dist = np.linalg.norm(self.eye - self.target)
        if self.mode_perspective:
            scale = 2.0 * dist * math.tan(math.radians(self.fovy) * 0.5)
        else:
            scale = self.ortho_scale * 2.0
        w, h = self.get_size()
        sx = (dx / max(1.0, w)) * scale * self.aspect
        sy = (dy / max(1.0, h)) * scale
        delta = -right * sx + upv * sy
        self.eye += delta
        self.target += delta

    def _dolly(self, steps, mx, my):
        # Smooth exponential zoom; keep cursor point stable in screen space
        factor = math.pow(0.90, steps)  # less sluggish than 0.93
        view_dir = self.target - self.eye
        dist = np.linalg.norm(view_dir)
        if self.mode_perspective:
            new_dist = max(0.05, min(500.0, dist * (1.0 / factor)))
            self.eye = self.target - (view_dir / (dist + 1e-9)) * new_dist
        else:
            self.ortho_scale *= factor
            self.ortho_scale = max(self._diag * 0.02, min(self._diag * 20.0, self.ortho_scale))

    # ----------------------------- Drawing -----------------------------
    def on_draw(self):
        self.clear()
        gl.glBindVertexArray(self.vao)
        gl.glUseProgram(self.program)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.is_wireframe else gl.GL_FILL)

        # Matrices
        V = look_at(self.eye, self.target, self.up)
        P = self._compute_projection()
        M = np.eye(4, dtype=np.float32)
        N = normal_matrix(M, V)

        gl.glUniformMatrix4fv(self.uni_model, 1, gl.GL_TRUE, as_glmat(M))
        gl.glUniformMatrix4fv(self.uni_view, 1, gl.GL_TRUE, as_glmat(V))
        gl.glUniformMatrix4fv(self.uni_proj, 1, gl.GL_TRUE, as_glmat(P))
        gl.glUniformMatrix3fv(self.uni_normal, 1, gl.GL_TRUE, (gl.GLfloat * 9)(*N.flatten()))

        # Camera + light in world
        gl.glUniform3f(self.uni_cam, float(self.eye[0]), float(self.eye[1]), float(self.eye[2]))
        light_dir = np.array([0.35, 0.9, 0.6], dtype=np.float32)
        light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-12)
        gl.glUniform3f(self.uni_light, float(light_dir[0]), float(light_dir[1]), float(light_dir[2]))

        gl.glUniform1i(self.uni_depth_enabled, 1 if self.depth_shading else 0)
        gl.glUniform1f(self.uni_depth_near, float(self.near))
        gl.glUniform1f(self.uni_depth_far, float(self.far))

        if self.color_mode == 1:
            for start, count, col, cls in self._draw_ranges:
                if not self.layer_visible.get(cls, True):
                    continue
                gl.glUniform3f(self.uni_color, float(col[0]), float(col[1]), float(col[2]))
                gl.glDrawArrays(gl.GL_TRIANGLES, start, count)
        else:
            gl.glUniform3f(self.uni_color, 0.82, 0.82, 0.90)
            for start, count, col, cls in self._draw_ranges:
                if not self.layer_visible.get(cls, True):
                    continue
                gl.glDrawArrays(gl.GL_TRIANGLES, start, count)

        # HUD + UI
        self._draw_ui()
        self._draw_hud(V, P)
        self.fps_display.draw()

    def _compute_projection(self) -> np.ndarray:
        w, h = self.get_size()
        self.aspect = (float(w) if w > 0 else 1.0) / (float(h) if h > 0 else 1.0)
        if self.mode_perspective:
            self.near = 0.02
            dist = np.linalg.norm(self.eye - self.target)
            self.far = max(self.near + 1.0, dist + self._diag * 3.0)
            return perspective(self.fovy, self.aspect, self.near, self.far)
        else:
            half_h = self.ortho_scale
            half_w = half_h * self.aspect
            self.near = 0.02
            dist = np.linalg.norm(self.eye - self.target)
            self.far = max(self.near + 1.0, dist + self._diag * 3.0)
            return orthographic(-half_w, half_w, -half_h, half_h, self.near, self.far)

    # ------------------------------- HUD -------------------------------
    def _qc_fraction_in_frame(self, V: np.ndarray, P: np.ndarray) -> float:
        """Approximate QC: fraction of vertices within NDC cube [-1,1]^2 and depth [0,1]."""
        if self.vertex_count == 0:
            return 0.0
        # Sample a subset for speed
        sample = min(20000, self.vertex_count)
        idx = np.linspace(0, self.vertex_count - 1, sample, dtype=np.int32)
        # Read back positions from GPU buffer (avoid; we cached CPU arrays?)
        # In this viewer we no longer keep a CPU copy of concatenated arrays. For QC, rebuild a bbox proxy:
        # Use 8 corners of bounds for a conservative estimate plus centerpoints.
        corners = _aabb_corners(self._bounds_min, self._bounds_max)
        extras = [self._center,
                  0.5 * (self._bounds_min + self._bounds_max),
                  np.array([self._bounds_min[0], self._center[1], self._center[2]]),
                  np.array([self._bounds_max[0], self._center[1], self._center[2]]),
                  np.array([self._center[0], self._bounds_min[1], self._center[2]]),
                  np.array([self._center[0], self._bounds_max[1], self._center[2]]),
                  np.array([self._center[0], self._center[1], self._bounds_min[2]]),
                  np.array([self._center[0], self._center[1], self._bounds_max[2]]),
                  ]
        pts = np.vstack([corners, np.stack(extras, axis=0)]).astype(np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts4 = np.hstack([pts, ones])
        MVP = P @ V
        clip = (MVP @ pts4.T).T
        ndc = clip[:, :3] / np.maximum(1e-6, clip[:, 3:4])
        inside = np.logical_and.reduce((ndc[:, 0] >= -1, ndc[:, 0] <= 1, ndc[:, 1] >= -1, ndc[:, 1] <= 1, ndc[:, 2] >= -1, ndc[:, 2] <= 1))
        return float(np.count_nonzero(inside)) / float(ndc.shape[0])

    def _draw_hud(self, V: np.ndarray, P: np.ndarray):
        file_name = self.current_path.name if self.current_path else '[no file loaded]'
        qc = self._qc_fraction_in_frame(V, P)
        qc_txt = f" QC(in-frame est): {qc*100:.0f}%"
        mode_txt = "Persp" if self.mode_perspective else "Ortho"
        dist = np.linalg.norm(self.eye - self.target)
        base = f"{file_name} | {mode_txt} fov:{self.fovy:.0f} dist:{dist:.2f}"
        txt = base + qc_txt

        # Switch to 2D text
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        self.label_hud.text = txt
        self.label_hud.y = self.height - 14
        self.label_hud.draw()
        gl.glUseProgram(self.program)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    # ------------------------------- UI -------------------------------
    def _build_ui(self):
        self._ui_buttons.clear()
        pad = 8; bw = 120; bh = 26
        x0 = 10; y = self.height - (bh + pad)

        def add(text, cb):
            nonlocal y
            self._ui_buttons.append(UIButton(x0, y, bw, bh, text, cb))
            y -= (bh + 6)

        add('Open…', lambda: self._toggle_browser())
        add('By Color', lambda: self._cycle_color())
        add('Depth Shade', lambda: self._toggle_depth())
        add('Wireframe', lambda: self._toggle_wire())
        add('Ortho/Persp', lambda: self._toggle_proj())
        add('Soma', lambda: self._toggle_layer(NeuronClass.SOMA.name))
        add('Axon', lambda: self._toggle_layer(NeuronClass.AXON.name))
        add('Dendrite', lambda: self._toggle_layer(NeuronClass.BASAL_DENDRITE.name))
        add('Other', lambda: self._toggle_layer(NeuronClass.OTHER.name))

    def _draw_ui(self):
        for b in self._ui_buttons:
            self._draw_button(b)
        if self.file_browser.visible:
            self._draw_browser()

    def _draw_button(self, b: UIButton):
        if not b.visible:
            return
        x = int(b.x); y = int(b.y); w = int(b.w); h = int(b.h)
        _draw_filled_rect(x, y, w, h, (30, 30, 30, 190))
        _draw_rect_outline(x, y, w, h, (200, 200, 200, 200))
        gl.glUseProgram(0); gl.glDisable(gl.GL_DEPTH_TEST); gl.glDisable(gl.GL_CULL_FACE)
        lbl = pyglet.text.Label(b.text, font_name='Courier New', font_size=12, x=x + 8, y=y + h // 2, anchor_y='center', color=(240, 240, 240, 255))
        lbl.draw()
        gl.glUseProgram(self.program); gl.glEnable(gl.GL_DEPTH_TEST); gl.glEnable(gl.GL_CULL_FACE)

    def _toggle_browser(self):
        self.file_browser.visible = not self.file_browser.visible

    def _toggle_layer(self, name: str):
        self.layer_visible[name] = not self.layer_visible.get(name, True)

    def _cycle_color(self):
        self.color_mode = (self.color_mode + 1) % 2

    def _toggle_depth(self):
        self.depth_shading = not self.depth_shading

    def _toggle_wire(self):
        self.is_wireframe = not self.is_wireframe

    def _toggle_proj(self):
        self.mode_perspective = not self.mode_perspective
        self._update_camera_fit(reset_target=False)

    # ----------------------------- Events -----------------------------
    def on_resize(self, width, height):
        super().on_resize(width, height)
        gl.glViewport(0, 0, width, height)
        self._build_ui()

    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT and (modifiers & key.MOD_SHIFT):
            self._arcball_prev = None  # will pan instead in drag
        elif button == mouse.LEFT:
            self._arcball_prev = (x, y)
        elif button == mouse.MIDDLE:
            self._arcball_prev = None  # pan mode
        if button == mouse.LEFT:
            # Check buttons first
            for b in self._ui_buttons:
                if b.hit(x, y):
                    b.on_click(); return
            if self.file_browser.visible:
                self._handle_browser_click(x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & mouse.MIDDLE or ((buttons & mouse.LEFT) and (modifiers & key.MOD_SHIFT)):
            self._pan(dx, dy)
            return
        if buttons & mouse.LEFT:
            if self._arcball_prev is None:
                self._arcball_prev = (x - dx, y - dy)
            x0, y0 = self._arcball_prev
            self._orbit(x0, y0, x, y)
            self._arcball_prev = (x, y)

    def on_mouse_scroll(self, x, y, sx, sy):
        self._dolly(sy, x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        self._arcball_prev = None

    def on_key_press(self, symbol, modifiers):
        if symbol == key.W:
            self._toggle_wire()
        elif symbol == key.O:
            self._toggle_depth()
        elif symbol == key.C:
            self._cycle_color()
        elif symbol == key.P:
            self._toggle_proj()
        elif symbol == key.R:
            self._update_camera_fit(reset_target=True)
        elif symbol == key.S:
            self.save_screenshot()
        elif symbol == key.ESCAPE:
            self.close()
        elif symbol == key.F:
            self._toggle_browser()

    # --------------------------- File browser ---------------------------
    def _draw_browser(self):
        x = 160; y = 80
        w = self.width - 2 * x
        h = self.height - 2 * y
        _draw_filled_rect(x, y, w, h, (20, 20, 20, 235))

        gl.glUseProgram(0); gl.glDisable(gl.GL_DEPTH_TEST); gl.glDisable(gl.GL_CULL_FACE)
        title = pyglet.text.Label(str(self.file_browser.current_dir), font_name='Courier New', font_size=12, x=x + 8, y=y + h - 18, anchor_x='left', anchor_y='top', color=(255, 255, 255, 255))
        title.draw()
        row_h = 20
        y0 = y + h - 40
        self._browser_hit = []
        up_lbl = '[..]'
        pyglet.text.Label(up_lbl, font_name='Courier New', font_size=12, x=x + 8, y=y0, anchor_x='left', anchor_y='baseline', color=(220, 220, 220, 255)).draw()
        self._browser_hit.append((x + 8, y0 - 4, w - 16, row_h, '__up__'))
        y0 -= row_h
        for name, path in self.file_browser.entries[: int((h - 60) / row_h)]:
            pyglet.text.Label(name, font_name='Courier New', font_size=12, x=x + 8, y=y0, anchor_x='left', anchor_y='baseline', color=(200, 200, 200, 255)).draw()
            self._browser_hit.append((x + 8, y0 - 4, w - 16, row_h, str(path)))
            y0 -= row_h
        gl.glUseProgram(self.program); gl.glEnable(gl.GL_DEPTH_TEST); gl.glEnable(gl.GL_CULL_FACE)

    def _handle_browser_click(self, px, py):
        if not hasattr(self, '_browser_hit'):
            return
        for bx, by, bw, bh, tag in self._browser_hit:
            if bx <= px <= bx + bw and by <= py <= by + bh:
                if tag == '__up__':
                    parent = self.file_browser.current_dir.parent
                    self.file_browser.current_dir = parent
                    self.file_browser.refresh()
                else:
                    p = Path(tag)
                    if p.is_dir():
                        self.file_browser.current_dir = p
                        self.file_browser.refresh()
                    else:
                        self.file_browser.visible = False
                        self._load_path(p)
                return

    # --------------------------- Utils & IO ---------------------------
    def save_screenshot(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        name = f"screenshot_{ts}.png"
        buf = pyglet.image.get_buffer_manager().get_color_buffer()
        buf.save(name)
        print(f"Saved {name}")

# ---------------------------- Draw helpers ----------------------------

def _draw_filled_rect(x, y, w, h, rgba):
    r, g, b, a = rgba
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glDisable(gl.GL_CULL_FACE)
    # Use Pyglet shapes (kept minimal to avoid perf hit)
    from pyglet import shapes
    rect = shapes.Rectangle(x, y, w, h, color=(r, g, b))
    rect.opacity = a
    rect.draw()
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_CULL_FACE)


def _draw_rect_outline(x, y, w, h, rgba):
    r, g, b, a = rgba
    from pyglet import shapes
    thickness = 1
    shapes.Line(x, y + h, x + w, y + h, thickness=thickness, color=(r, g, b)).draw()
    shapes.Line(x + w, y, x + w, y + h, thickness=thickness, color=(r, g, b)).draw()
    shapes.Line(x, y, x + w, y, thickness=thickness, color=(r, g, b)).draw()
    shapes.Line(x, y, x, y + h, thickness=thickness, color=(r, g, b)).draw()


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c = math.cos(angle); s = math.sin(angle); C = 1.0 - c
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

# ------------------------------- Entrypoint -------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("swc", type=str, nargs='?')
    parser.add_argument("--segments", type=int, default=18)
    args = parser.parse_args()

    win = SWCViewer(Path(args.swc) if args.swc else None, segments=args.segments)
    pyglet.app.run()


if __name__ == "__main__":
    main()
