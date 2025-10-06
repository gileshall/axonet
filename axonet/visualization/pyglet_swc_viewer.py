import math
import ctypes
import argparse
import time
import numpy as np
import pyglet
from pyglet.window import key, mouse
from pyglet import gl
from pathlib import Path

from .mesh import MeshRenderer
from ..io import load_swc


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


class SWCViewer(pyglet.window.Window):
    def __init__(self, swc_path: Path, segments: int = 18):
        config = gl.Config(double_buffer=True, depth_size=24, sample_buffers=1, samples=4, major_version=3, minor_version=3, forward_compatible=True)
        super().__init__(1280, 900, f"SWC Viewer - {swc_path.name}", resizable=True, config=config)

        neuron = load_swc(swc_path, validate=True)
        renderer = MeshRenderer(neuron)
        by_type = renderer.build_mesh_by_type(segments=segments, cap=True)
        order = ["soma", "axon", "dendrite", "other"]
        colors = {
            "soma": np.array([0.95, 0.35, 0.35], dtype=np.float32),
            "axon": np.array([0.35, 0.95, 0.45], dtype=np.float32),
            "dendrite": np.array([0.35, 0.55, 0.95], dtype=np.float32),
            "other": np.array([0.75, 0.75, 0.75], dtype=np.float32),
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
            draw_ranges.append((current_start, int(cnt), colors[cls]))
            current_start += int(cnt)
            bm = m.bounds[0].astype(np.float32)
            bM = m.bounds[1].astype(np.float32)
            bounds_min = bm if bounds_min is None else np.minimum(bounds_min, bm)
            bounds_max = bM if bounds_max is None else np.maximum(bounds_max, bM)
        if not pos_arrays:
            raise RuntimeError("Empty mesh")
        pos = np.concatenate(pos_arrays, axis=0)
        nrm = np.concatenate(nrm_arrays, axis=0)
        self.vertex_count = int(pos.shape[0])
        self._bounds_min = bounds_min
        self._bounds_max = bounds_max
        self._draw_ranges = draw_ranges
        self._mesh_bounds_diag = float(np.linalg.norm(self._bounds_max - self._bounds_min))
        self._mesh = None

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
        pos_type = (gl.GLfloat * pos.size)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, pos.nbytes, pos_type.from_buffer(pos), gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, ctypes.c_void_p(0))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_nrm)
        nrm_type = (gl.GLfloat * nrm.size)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, nrm.nbytes, nrm_type.from_buffer(nrm), gl.GL_STATIC_DRAW)
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
        self.reset_view()
        pyglet.clock.schedule_interval(lambda dt: None, 1 / 60.0)
        self.fps_display = pyglet.window.FPSDisplay(self)

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
            for start, count, col in self._draw_ranges:
                gl.glUniform3f(self.uni_color, float(col[0]), float(col[1]), float(col[2]))
                gl.glDrawArrays(gl.GL_TRIANGLES, start, count)
        else:
            gl.glUniform3f(self.uni_color, 0.8, 0.8, 0.9)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertex_count)

        gl.glUseProgram(0)
        self.fps_display.draw()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        gl.glViewport(0, 0, width, height)
        self.aspect = (float(width) if width > 0 else 1.0) / (float(height) if height > 0 else 1.0)

    def on_mouse_press(self, x, y, button, modifiers):
        self.pan_active = (button in (mouse.RIGHT, mouse.MIDDLE))

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

    def save_screenshot(self):
        ts = time.strftime("%Y%m%d-%H%M%S")
        name = f"screenshot_{ts}.png"
        buf = pyglet.image.get_buffer_manager().get_color_buffer()
        buf.save(name)
        print(f"Saved {name}")

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("swc", type=str)
    parser.add_argument("--segments", type=int, default=18)
    args = parser.parse_args()
    win = SWCViewer(Path(args.swc), segments=args.segments)
    pyglet.app.run()


if __name__ == "__main__":
    main()


