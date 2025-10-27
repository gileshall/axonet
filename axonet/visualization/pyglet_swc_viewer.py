import math
import ctypes
import argparse
import time
import numpy as np
import pyglet
from pyglet.window import key, mouse
from pyglet import shapes
from pyglet import gl
from pathlib import Path

from .mesh import MeshRenderer
from ..io import load_swc, NeuronClass


VERT_SRC = b"""
#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_normal;
out vec3 v_worldpos;
out float v_viewz;

void main() {
    vec4 world = u_model * vec4(position, 1.0);
    v_worldpos = world.xyz;
    v_normal = mat3(u_model) * normal;
    vec4 viewpos = u_view * world;
    v_viewz = viewpos.z;
    gl_Position = u_proj * viewpos;
}
"""


FRAG_SRC = b"""
#version 330 core
in vec3 v_normal;
in vec3 v_worldpos;
in float v_viewz;

uniform vec3 u_camera;
uniform vec3 u_light_dir;
uniform vec3 u_color;
uniform int u_depth_enabled;
uniform float u_depth_near;
uniform float u_depth_far;

out vec4 FragColor;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera - v_worldpos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 24.0);

    vec3 ambient = 0.15 * u_color;
    vec3 diffuse = 0.85 * diff * u_color;
    vec3 specular = 0.20 * spec * vec3(1.0);

    vec3 color = ambient + diffuse + specular;
    if (u_depth_enabled == 1) {
        float zn = clamp(((-v_viewz) - u_depth_near) / max(1e-6, (u_depth_far - u_depth_near)), 0.0, 1.0);
        float shade = mix(0.35, 1.0, 1.0 - zn);
        color *= shade;
    }
    FragColor = vec4(color, 1.0);
}
"""


def _compile_shader(source_bytes, shader_type):
    shader = gl.glCreateShader(shader_type)
    src_buf = ctypes.create_string_buffer(source_bytes)
    src_ptr = ctypes.cast(ctypes.pointer(ctypes.pointer(src_buf)), ctypes.POINTER(ctypes.POINTER(gl.GLchar)))
    gl.glShaderSource(shader, 1, src_ptr, None)
    gl.glCompileShader(shader)

    status = gl.GLint()
    gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS, ctypes.byref(status))
    if not status.value:
        log_len = gl.GLint()
        gl.glGetShaderiv(shader, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        log = ctypes.create_string_buffer(max(1, log_len.value))
        gl.glGetShaderInfoLog(shader, log_len, None, log)
        raise RuntimeError(f"Shader compile error ({shader_type}):\n{log.value.decode()}")
    return shader


def make_program():
    vert = _compile_shader(VERT_SRC, gl.GL_VERTEX_SHADER)
    frag = _compile_shader(FRAG_SRC, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vert)
    gl.glAttachShader(prog, frag)
    gl.glLinkProgram(prog)

    status = gl.GLint()
    gl.glGetProgramiv(prog, gl.GL_LINK_STATUS, ctypes.byref(status))
    if not status.value:
        log_len = gl.GLint()
        gl.glGetProgramiv(prog, gl.GL_INFO_LOG_LENGTH, ctypes.byref(log_len))
        log = ctypes.create_string_buffer(max(1, log_len.value))
        gl.glGetProgramInfoLog(prog, log_len, None, log)
        raise RuntimeError("Program link error:\n" + log.value.decode())

    gl.glDeleteShader(vert)
    gl.glDeleteShader(frag)
    return prog


def perspective(fovy_deg, aspect, znear, zfar):
    f = 1.0 / math.tan(math.radians(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(1e-6, aspect)
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def translate(tx, ty, tz):
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def rotate_y(angle_deg):
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def rotate_x(angle_deg):
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def matmul(*mats):
    out = np.eye(4, dtype=np.float32)
    for m in mats:
        out = m @ out
    return out


def as_glmat(m):
    m = np.asarray(m, dtype=np.float32)
    return (gl.GLfloat * 16)(*m.flatten())


def mesh_to_arrays(mesh):
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int32)
    vn = np.asarray(mesh.vertex_normals if mesh.vertex_normals is not None else mesh.face_normals.repeat(3, axis=0), dtype=np.float32)
    tri_count = f.shape[0]
    positions = v[f.reshape(-1)]
    normals = vn[f.reshape(-1)]
    return positions.astype(np.float32), normals.astype(np.float32), tri_count * 3


class UIButton:
    def __init__(self, x, y, w, h, text, on_click):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.on_click = on_click
        self.enabled = True
        self.visible = True

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


class SWCViewer(pyglet.window.Window):
    def __init__(self, swc_path: Path | None = None, segments: int = 18):
        config = gl.Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4, major_version=3, minor_version=3, forward_compatible=True)
        title = "SWC Viewer"
        if swc_path is not None:
            try:
                title = f"SWC Viewer - {Path(swc_path).name}"
            except Exception:
                title = "SWC Viewer"
        super().__init__(1280, 900, title, resizable=True, config=config)
        self.segments = segments
        self.current_path = Path(swc_path) if swc_path is not None else None
        self.neuron = None
        self._draw_ranges = []
        self.vertex_count = 0
        self._bounds_min = np.array([0, 0, 0], dtype=np.float32)
        self._bounds_max = np.array([1, 1, 1], dtype=np.float32)
        self._mesh_bounds_diag = 1.0

        self.program = make_program()
        self.vao = gl.GLuint()
        gl.glGenVertexArrays(1, ctypes.byref(self.vao))
        gl.glBindVertexArray(self.vao)

        self.uni_model = gl.glGetUniformLocation(self.program, b"u_model")
        self.uni_view = gl.glGetUniformLocation(self.program, b"u_view")
        self.uni_proj = gl.glGetUniformLocation(self.program, b"u_proj")
        self.uni_cam = gl.glGetUniformLocation(self.program, b"u_camera")
        self.uni_light = gl.glGetUniformLocation(self.program, b"u_light_dir")
        self.uni_color = gl.glGetUniformLocation(self.program, b"u_color")
        self.uni_depth_enabled = gl.glGetUniformLocation(self.program, b"u_depth_enabled")
        self.uni_depth_near = gl.glGetUniformLocation(self.program, b"u_depth_near")
        self.uni_depth_far = gl.glGetUniformLocation(self.program, b"u_depth_far")

        self.vbo_pos = gl.GLuint()
        self.vbo_nrm = gl.GLuint()
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

        self.is_wireframe = False
        self.auto_orbit = False
        self.orbit_speed = 10.0
        self.color_mode = 0
        self.depth_shading = False
        self.pan_active = False
        self.layer_visible = {
            NeuronClass.SOMA.name: True,
            NeuronClass.AXON.name: True,
            NeuronClass.BASAL_DENDRITE.name: True,
            NeuronClass.APICAL_DENDRITE.name: True,
            NeuronClass.OTHER.name: True,
        }
        self.reset_view()
        pyglet.clock.schedule_interval(lambda dt: None, 1 / 60.0)
        self.fps_display = pyglet.window.FPSDisplay(self)

        self._ui_buttons = []
        self._build_ui()
        start_dir = Path(self.current_path).parent if self.current_path is not None else Path.cwd()
        self.file_browser = FileBrowser(start_dir)
        self.file_browser.visible = True if self.current_path is None else False

        self.label_hud = pyglet.text.Label('', font_name='Courier New', font_size=12, x=10, y=self.height - 14, anchor_x='left', anchor_y='top', color=(255, 255, 255, 255))

        self._build_axes_vbo()
        if self.current_path is not None:
            self._load_path(self.current_path)

    def _build_axes_vbo(self):
        axes = np.array([
            0, 0, 0,  1, 0, 0,
            0, 0, 0,  0, 1, 0,
            0, 0, 0,  0, 0, 1,
        ], dtype=np.float32)
        self.axes_count = 6
        self.vbo_axes = gl.GLuint()
        gl.glGenBuffers(1, ctypes.byref(self.vbo_axes))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_axes)
        axes_type = (gl.GLfloat * axes.size)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, axes.nbytes, axes_type.from_buffer(axes), gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    def _build_ui(self):
        pad = 8
        bw = 120
        bh = 26
        x0 = 10
        y = self.height - (bh + pad)

        def add(text, cb):
            nonlocal y
            self._ui_buttons.append(UIButton(x0, y, bw, bh, text, cb))
            y -= (bh + 6)

        add('Open...', lambda: self._toggle_browser())
        add('Axes', lambda: self._toggle_axes())
        add('Auto Orbit', lambda: self._toggle_orbit())
        add('Depth Shade', lambda: self._toggle_depth())
        add('Color Mode', lambda: self._cycle_color())
        add('Top', lambda: self._pose('top'))
        add('Bottom', lambda: self._pose('bottom'))
        add('Left', lambda: self._pose('left'))
        add('Right', lambda: self._pose('right'))
        add('Front', lambda: self._pose('front'))
        add('Back', lambda: self._pose('back'))
        add('Soma', lambda: self._toggle_layer(NeuronClass.SOMA.name))
        add('Axon', lambda: self._toggle_layer(NeuronClass.AXON.name))
        add('Dendrite', lambda: self._toggle_layer(NeuronClass.BASAL_DENDRITE.name))
        add('Other', lambda: self._toggle_layer(NeuronClass.OTHER.name))

    def _toggle_browser(self):
        self.file_browser.visible = not self.file_browser.visible

    def _toggle_axes(self):
        self.show_axes = not getattr(self, 'show_axes', False)

    def _toggle_orbit(self):
        self.auto_orbit = not self.auto_orbit

    def _toggle_depth(self):
        self.depth_shading = not self.depth_shading

    def _cycle_color(self):
        self.color_mode = (self.color_mode + 1) % 2

    def _pose(self, which: str):
        self.pan_x = 0.0
        self.pan_y = 0.0
        if which == 'top':
            self.yaw = 0.0
            self.pitch = -90.0
        elif which == 'bottom':
            self.yaw = 0.0
            self.pitch = 90.0
        elif which == 'left':
            self.yaw = -90.0
            self.pitch = 0.0
        elif which == 'right':
            self.yaw = 90.0
            self.pitch = 0.0
        elif which == 'front':
            self.yaw = 0.0
            self.pitch = 0.0
        elif which == 'back':
            self.yaw = 180.0
            self.pitch = 0.0

    def _toggle_layer(self, name: str):
        self.layer_visible[name] = not self.layer_visible.get(name, True)

    def _load_path(self, swc_path: Path):
        self.current_path = Path(swc_path)
        self.set_caption(f"SWC Viewer - {self.current_path.name}")
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
        pos_arrays = []
        nrm_arrays = []
        draw_ranges = []
        current_start = 0
        bounds_min = None
        bounds_max = None
        for cls in order:
            m = by_type.get(cls)
            if m is None:
                continue
            p, n, cnt = mesh_to_arrays(m)
            pos_arrays.append(p)
            nrm_arrays.append(n)
            draw_ranges.append((current_start, int(cnt), colors[cls], cls))
            current_start += int(cnt)
            bm = m.bounds[0].astype(np.float32)
            bM = m.bounds[1].astype(np.float32)
            bounds_min = bm if bounds_min is None else np.minimum(bounds_min, bm)
            bounds_max = bM if bounds_max is None else np.maximum(bounds_max, bM)
        if not pos_arrays:
            print("ERROR: No mesh data generated!")
            return
        pos = np.concatenate(pos_arrays, axis=0)
        nrm = np.concatenate(nrm_arrays, axis=0)
        self.vertex_count = int(pos.shape[0])
        self._bounds_min = bounds_min
        self._bounds_max = bounds_max
        self._draw_ranges = draw_ranges
        self._mesh_bounds_diag = float(np.linalg.norm(self._bounds_max - self._bounds_min))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_pos)
        pos_type = (gl.GLfloat * pos.size)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, pos.nbytes, pos_type.from_buffer(pos), gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_nrm)
        nrm_type = (gl.GLfloat * nrm.size)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, nrm.nbytes, nrm_type.from_buffer(nrm), gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        self.reset_view()

    def reset_view(self):
        center = 0.5 * (self._bounds_min + self._bounds_max)
        diag = float(np.linalg.norm(self._bounds_max - self._bounds_min))
        self.model_center = center
        self.model_scale = 2.0 / max(1e-6, diag)
        self.yaw = 30.0
        self.pitch = -20.0
        self.distance = 2.2
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.fovy = 60.0

    def on_draw(self):
        self.clear()
        gl.glBindVertexArray(self.vao)
        gl.glUseProgram(self.program)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.is_wireframe else gl.GL_FILL)

        if self.auto_orbit:
            self.yaw += self.orbit_speed * (1.0 / 60.0)
        rot = matmul(rotate_y(self.yaw), rotate_x(self.pitch))
        eye = np.array([0.0, 0.0, self.distance, 1.0], dtype=np.float32)

        pan_scale = math.tan(math.radians(self.fovy * 0.5)) * self.distance
        pan_world = np.array([self.pan_x * pan_scale * self.aspect, -self.pan_y * pan_scale, 0.0, 1.0], dtype=np.float32)

        Tpan = translate(-pan_world[0], -pan_world[1], -eye[2])
        V = rot @ Tpan
        P = perspective(self.fovy, self.aspect, 0.05, 100.0)

        S = np.eye(4, dtype=np.float32)
        S[0, 0] = self.model_scale
        S[1, 1] = self.model_scale
        S[2, 2] = self.model_scale
        Tc = translate(-self.model_center[0], -self.model_center[1], -self.model_center[2])
        M = S @ Tc

        gl.glUniformMatrix4fv(self.uni_model, 1, gl.GL_TRUE, as_glmat(M))
        gl.glUniformMatrix4fv(self.uni_view, 1, gl.GL_TRUE, as_glmat(V))
        gl.glUniformMatrix4fv(self.uni_proj, 1, gl.GL_TRUE, as_glmat(P))

        rotT = rot.T
        cam_world = rotT @ np.array([pan_world[0], pan_world[1], eye[2], 1.0], dtype=np.float32)
        gl.glUniform3f(self.uni_cam, float(cam_world[0]), float(cam_world[1]), float(cam_world[2]))

        light_dir = np.array([0.5, 1.0, 0.8], dtype=np.float32)
        light_dir = light_dir / np.linalg.norm(light_dir)
        gl.glUniform3f(self.uni_light, float(light_dir[0]), float(light_dir[1]), float(light_dir[2]))

        depth_enabled = 1 if self.depth_shading else 0
        gl.glUniform1i(self.uni_depth_enabled, depth_enabled)
        near = 0.1
        far = self.distance + 5.0
        gl.glUniform1f(self.uni_depth_near, float(near))
        gl.glUniform1f(self.uni_depth_far, float(far))

        if self.color_mode == 1:
            for start, count, col, cls in self._draw_ranges:
                if not self.layer_visible.get(cls, True):
                    continue
                gl.glUniform3f(self.uni_color, float(col[0]), float(col[1]), float(col[2]))
                gl.glDrawArrays(gl.GL_TRIANGLES, start, count)
        else:
            gl.glUniform3f(self.uni_color, 0.8, 0.8, 0.9)
            total_drawn = 0
            for start, count, col, cls in self._draw_ranges:
                if not self.layer_visible.get(cls, True):
                    continue
                gl.glDrawArrays(gl.GL_TRIANGLES, start, count)
                total_drawn += count

        if getattr(self, 'show_axes', False):
            gl.glUseProgram(self.program)
            # Model matrix for small axes at origin scaled relative to model scale
            scale = 0.2 / max(1e-6, self.model_scale)
            S = np.eye(4, dtype=np.float32)
            S[0, 0] = scale
            S[1, 1] = scale
            S[2, 2] = scale
            M_axes = S
            gl.glUniformMatrix4fv(self.uni_model, 1, gl.GL_TRUE, as_glmat(M_axes))
            gl.glUniform3f(self.uni_color, 1.0, 0.0, 0.0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_axes)
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))
            gl.glDisableVertexAttribArray(1)
            gl.glLineWidth(1.0)
            gl.glDrawArrays(gl.GL_LINES, 0, 2)
            gl.glUniform3f(self.uni_color, 0.0, 1.0, 0.0)
            gl.glDrawArrays(gl.GL_LINES, 2, 2)
            gl.glUniform3f(self.uni_color, 0.0, 0.6, 1.0)
            gl.glDrawArrays(gl.GL_LINES, 4, 2)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glEnableVertexAttribArray(1)
            gl.glUseProgram(0)

        self._draw_ui()
        self._draw_hud()
        self.fps_display.draw()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        gl.glViewport(0, 0, width, height)
        self.aspect = (float(width) if width > 0 else 1.0) / (float(height) if height > 0 else 1.0)
        self._ui_buttons = []
        self._build_ui()
        self.label_hud.y = self.height - 14

    def on_mouse_press(self, x, y, button, modifiers):
        if button in (mouse.RIGHT, mouse.MIDDLE):
            self.pan_active = True
            return
        if button == mouse.LEFT:
            for b in self._ui_buttons:
                if b.hit(x, y):
                    b.on_click()
                    return
            if self.file_browser.visible:
                self._handle_browser_click(x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        self.pan_active = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.pan_active:
            w, h = self.get_size()
            nx = dx / max(1.0, float(w))
            ny = dy / max(1.0, float(h))
            sensitivity = 1.2
            self.pan_x += nx * sensitivity
            self.pan_y += ny * sensitivity
        else:
            self.yaw += dx * 0.3
            self.pitch += dy * 0.3
            self.pitch = max(-89.9, min(89.9, self.pitch))

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        factor = math.pow(0.93, scroll_y)
        self.distance = max(0.2, min(200.0, self.distance * factor))

    def on_key_press(self, symbol, modifiers):
        if symbol == key.W:
            self.is_wireframe = not self.is_wireframe
        elif symbol == key.R:
            self.reset_view()
        elif symbol == key.O:
            self.auto_orbit = not self.auto_orbit
        elif symbol == key.C:
            self.color_mode = (self.color_mode + 1) % 2
        elif symbol == key.D:
            self.depth_shading = not self.depth_shading
        elif symbol == key.S:
            self.save_screenshot()
        elif symbol == key.ESCAPE:
            self.close()
        elif symbol == key.F:
            self._toggle_browser()

    def save_screenshot(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        name = f"screenshot_{ts}.png"
        buf = pyglet.image.get_buffer_manager().get_color_buffer()
        buf.save(name)
        print(f"Saved {name}")

    def _draw_button(self, b: UIButton):
        if not b.visible:
            return
        x = int(b.x)
        y = int(b.y)
        w = int(b.w)
        h = int(b.h)
        self._draw_filled_rect(x, y, w, h, (30, 30, 30, 180))
        self._draw_rect_outline(x, y, w, h, (200, 200, 200, 200))
        
        # Switch to 2D rendering mode for text
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        
        lbl = pyglet.text.Label(b.text, font_name='Courier New', font_size=12, x=x + 8, y=y + h // 2, anchor_y='center', color=(240, 240, 240, 255))
        lbl.draw()
        
        # Restore 3D rendering state
        gl.glUseProgram(self.program)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    def _draw_filled_rect(self, x, y, w, h, color_rgba):
        r, g, b, a = color_rgba
        rect = shapes.Rectangle(x, y, w, h, color=(r, g, b))
        rect.opacity = a
        rect.draw()

    def _draw_rect_outline(self, x, y, w, h, color_rgba):
        r, g, b, a = color_rgba
        thickness = 1
        top = shapes.Line(x, y + h, x + w, y + h, thickness=thickness, color=(r, g, b))
        right = shapes.Line(x + w, y, x + w, y + h, thickness=thickness, color=(r, g, b))
        bottom = shapes.Line(x, y, x + w, y, thickness=thickness, color=(r, g, b))
        left = shapes.Line(x, y, x, y + h, thickness=thickness, color=(r, g, b))
        top.opacity = right.opacity = bottom.opacity = left.opacity = a
        top.draw(); right.draw(); bottom.draw(); left.draw()

    def _draw_ui(self):
        for b in self._ui_buttons:
            self._draw_button(b)
        if self.file_browser.visible:
            self._draw_browser()

    def _draw_hud(self):
        file_name = self.current_path.name if self.current_path else '[no file loaded]'
        nodes = len(self.neuron) if self.neuron is not None else 0
        base = f"{file_name}  nodes:{nodes}  yaw:{self.yaw:.1f} pitch:{self.pitch:.1f} zoom:{self.distance:.2f}"
        if self.current_path is None:
            hint = "  Press F or click Open… to load an SWC"
            txt = base + hint
        else:
            txt = base
        self.label_hud.text = txt
        
        # Switch to 2D rendering mode for HUD text
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        
        self.label_hud.draw()
        
        # Restore 3D rendering state
        gl.glUseProgram(self.program)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    def _draw_browser(self):
        x = 160
        y = 80
        w = self.width - 2 * x
        h = self.height - 2 * y
        self._draw_filled_rect(x, y, w, h, (20, 20, 20, 230))
        
        # Switch to 2D rendering mode for browser text
        gl.glUseProgram(0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        
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
        
        # Restore 3D rendering state
        gl.glUseProgram(self.program)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

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

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("swc", type=str)
    parser.add_argument("--segments", type=int, default=18)
    args = parser.parse_args()
    win = SWCViewer(Path(args.swc), segments=args.segments)
    pyglet.app.run()


if __name__ == "__main__":
    main()


