from __future__ import annotations

import ctypes
import math
import time
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_MULTISAMPLE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    GL_EXTENSIONS,
    glBindBuffer,
    glBindTexture,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glClear,
    glClearColor,
    glDeleteBuffers,
    glDeleteProgram,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDepthMask,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenerateMipmap,
    glGetFloatv,
    glGetString,
    glGenBuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetUniformLocation,
    glActiveTexture,
    glTexImage2D,
    glTexParameteri,
    glTexParameterf,
    glUniform1f,
    glUniform1i,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
    glPixelStorei,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..types import NormalizedPose
from .camera import VirtualCamera
from .scene import DemoScene

try:
    from OpenGL.GL.EXT.texture_filter_anisotropic import (
        GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT,
        GL_TEXTURE_MAX_ANISOTROPY_EXT,
    )
except Exception:
    GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT = None
    GL_TEXTURE_MAX_ANISOTROPY_EXT = None


VERTICES = np.array(
    [
        -1.0, -1.0, 0.0, 0.0,
        1.0, -1.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        -1.0, -1.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        -1.0, 1.0, 0.0, 1.0,
    ],
    dtype=np.float32,
)


class ParallaxGLWidget(QOpenGLWidget):
    def __init__(self, assets_dir: Path, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.camera = VirtualCamera()
        self.scene = DemoScene(assets_dir)
        self.pose = NormalizedPose(valid=False)
        self.use_mouse_mock = False
        self.show_debug = True
        self.depth_debug_mode = False
        self.cinematic_drift_enabled = True
        self.drift_intensity = 0.45
        self._start_time = time.time()

        self.program = None
        self.vao = None
        self.vbo = None
        self.u_mvp_loc = -1
        self.u_alpha_loc = -1
        self.u_neutral_mix_loc = -1
        self.textures = []
        self.gl_failed = False
        self.gl_error = ""
        self.neutral_tone_enabled = True
        self.max_anisotropy = 1.0

        self.last_frame_time = time.time()
        self.render_fps = 0.0
        self.target_fps = 60.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.setTimerType(Qt.PreciseTimer)
        self._apply_render_timer()

    def set_pose(self, pose: NormalizedPose) -> None:
        self.pose = pose

    def set_debug_overlay(self, enabled: bool) -> None:
        self.show_debug = enabled

    def set_parallax_strength(self, x: float, y: float, z: float) -> None:
        self.camera.parallax_strength_x = x
        self.camera.parallax_strength_y = y
        self.camera.parallax_strength_z = z

    def set_depth_spread(self, spread: float) -> None:
        self.scene.set_depth_spread(spread)
        # Normal mode geometry depth changes do not require texture rebuild.
        if self.program and self.depth_debug_mode:
            self._load_textures()

    def set_fov(self, fov_deg: float) -> None:
        self.camera.fov_y_deg = max(38.0, min(95.0, float(fov_deg)))

    def set_render_distance(self, z_far: float) -> None:
        self.camera.z_far = max(self.camera.z_near + 1.0, min(800.0, float(z_far)))

    def set_target_fps(self, fps: float) -> None:
        self.target_fps = max(20.0, min(240.0, float(fps)))
        self._apply_render_timer()

    def _apply_render_timer(self) -> None:
        interval_ms = max(4, int(round(1000.0 / self.target_fps)))
        self.timer.start(interval_ms)

    def set_neutral_tone(self, enabled: bool) -> None:
        self.neutral_tone_enabled = bool(enabled)

    def set_depth_debug_mode(self, enabled: bool) -> None:
        self.depth_debug_mode = enabled
        self.camera.depth_debug_mode = enabled
        if self.program:
            self._load_textures()

    def set_cinematic_drift(self, enabled: bool) -> None:
        self.cinematic_drift_enabled = bool(enabled)

    def set_drift_intensity(self, v: float) -> None:
        self.drift_intensity = max(0.0, min(1.0, float(v)))

    def _effective_pose(self, pose: NormalizedPose, now: float) -> NormalizedPose:
        if not self.cinematic_drift_enabled or self.drift_intensity <= 1e-3:
            return pose

        t = now - self._start_time
        amp = self.drift_intensity
        drift_x = math.sin(t * 0.40) * 0.22 * amp
        drift_y = math.cos(t * 0.33) * 0.16 * amp
        drift_z = math.sin(t * 0.27 + 0.8) * 0.14 * amp

        if not pose.valid:
            return NormalizedPose(x=drift_x, y=drift_y, z=drift_z, valid=True, timestamp=now)

        blend = 0.28 * amp
        return NormalizedPose(
            x=float(np.clip(pose.x + drift_x * blend, -1.0, 1.0)),
            y=float(np.clip(pose.y + drift_y * blend, -1.0, 1.0)),
            z=float(np.clip(pose.z + drift_z * blend, -1.0, 1.0)),
            yaw=pose.yaw,
            pitch=pose.pitch,
            roll=pose.roll,
            valid=True,
            timestamp=pose.timestamp,
        )

    def _layer_half_extents(self, idx: int, layer) -> tuple[float, float]:
        """How big the quad is in world units. Far layers are a bit larger so
        they fully cover the window; near layers are smaller for cut-out feel.
        """
        z = abs(layer.z)
        scale = 0.55 + 0.18 * z
        return self.camera.window_half_w * scale, self.camera.window_half_h * scale

    def initializeGL(self) -> None:
        try:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glEnable(GL_MULTISAMPLE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

            shader_dir = Path(__file__).parent / "shaders"
            vert = (shader_dir / "layer.vert").read_text(encoding="utf-8")
            frag = (shader_dir / "layer.frag").read_text(encoding="utf-8")
            self.program = compileProgram(
                compileShader(vert, GL_VERTEX_SHADER),
                compileShader(frag, GL_FRAGMENT_SHADER),
            )

            self.vao = glGenVertexArrays(1)
            self.vbo = glGenBuffers(1)
            glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, VERTICES.nbytes, VERTICES, GL_STATIC_DRAW)

            stride = 4 * 4
            # Shader uses explicit layout qualifiers: a_pos=0, a_uv=1.
            pos_loc = 0
            uv_loc = 1
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, stride, None)
            glEnableVertexAttribArray(uv_loc)
            glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))

            glBindVertexArray(0)
            self.max_anisotropy = self._query_anisotropy_limit()
            self._load_textures()
            glUseProgram(self.program)
            glUniform1i(glGetUniformLocation(self.program, "u_tex"), 0)
            self.u_mvp_loc = glGetUniformLocation(self.program, "u_mvp")
            self.u_alpha_loc = glGetUniformLocation(self.program, "u_alpha")
            self.u_neutral_mix_loc = glGetUniformLocation(self.program, "u_neutral_mix")
            if self.u_mvp_loc < 0 or self.u_alpha_loc < 0 or self.u_neutral_mix_loc < 0:
                raise RuntimeError("Required shader uniforms not found")
            self.gl_failed = False
        except Exception as exc:
            self.gl_failed = True
            self.gl_error = str(exc)

    def _query_anisotropy_limit(self) -> float:
        if GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT is None:
            return 1.0
        try:
            ext_raw = glGetString(GL_EXTENSIONS)
            ext = ext_raw.decode("ascii", errors="ignore") if ext_raw else ""
            if "GL_EXT_texture_filter_anisotropic" not in ext:
                return 1.0
            return max(1.0, float(glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)))
        except Exception:
            return 1.0

    def _load_textures(self) -> None:
        images = self.scene.load_qimages(self.depth_debug_mode)
        if self.textures:
            glDeleteTextures(len(self.textures), self.textures)
            self.textures = []
        self.textures = glGenTextures(len(images))
        if isinstance(self.textures, int):
            self.textures = [self.textures]

        for tex, img in zip(self.textures, images):
            rgba = img.convertToFormat(QImage.Format.Format_RGBA8888)
            w, h = rgba.width(), rgba.height()
            bpl = rgba.bytesPerLine()
            # PySide6's QImage.bits() returns a sized memoryview; no setsize needed.
            buf = bytes(rgba.constBits())
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, bpl // 4, 4))[:, :w, :]
            # QImage data is top-left origin; OpenGL UVs are bottom-left origin.
            arr = np.ascontiguousarray(np.flipud(arr))
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            if GL_TEXTURE_MAX_ANISOTROPY_EXT is not None and self.max_anisotropy > 1.0:
                glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, min(4.0, self.max_anisotropy))
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                w,
                h,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                arr,
            )
            glGenerateMipmap(GL_TEXTURE_2D)

    def resizeGL(self, w: int, h: int) -> None:
        glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        now = time.time()
        dt = max(1e-6, now - self.last_frame_time)
        self.last_frame_time = now
        self.render_fps = 1.0 / dt

        glClearColor(0.08, 0.1, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.gl_failed:
            painter = QPainter(self)
            painter.setPen(Qt.white)
            painter.drawText(10, 24, "OpenGL init failed")
            painter.drawText(10, 44, self.gl_error[:120])
            painter.end()
            return

        pose = self._effective_pose(self.pose, now)
        self.camera.update_from_head_pose(pose)

        w = max(1, self.width())
        h = max(1, self.height())
        aspect = float(w) / float(h)
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix(aspect)
        vp = proj @ view

        glUseProgram(self.program)
        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(self.vao)
        glUniform1f(self.u_neutral_mix_loc, 1.0 if self.neutral_tone_enabled else 0.0)

        # Draw farthest first so depth-test + alpha discard composite cleanly.
        live_layers = self.scene.get_layers(self.depth_debug_mode)
        layers = list(enumerate(live_layers))
        layers.sort(key=lambda it: it[1].z)  # most negative (far) first

        for idx, layer in layers:
            # Keep painter-style compositing stable for semi-transparent layers.
            glDepthMask(GL_FALSE)
            half_w, half_h = self._layer_half_extents(idx, layer)
            model = self.camera.model_matrix_for_layer(layer.z, half_w, half_h)
            mvp = vp @ model
            glUniformMatrix4fv(self.u_mvp_loc, 1, GL_TRUE, mvp.astype(np.float32))
            glUniform1f(self.u_alpha_loc, layer.alpha)
            if idx < len(self.textures):
                glBindTexture(GL_TEXTURE_2D, self.textures[idx])
            glDrawArrays(GL_TRIANGLES, 0, 6)
        glDepthMask(GL_TRUE)

        glBindVertexArray(0)

        if self.show_debug:
            painter = QPainter(self)
            painter.setPen(Qt.white)
            painter.drawText(10, 20, f"Render FPS: {self.render_fps:.1f}")
            painter.drawText(10, 40, f"Pose x/y/z: {pose.x:.2f} {pose.y:.2f} {pose.z:.2f}")
            painter.drawText(
                10,
                60,
                f"Eye xyz: {self.camera.eye_x:.2f} {self.camera.eye_y:.2f} {self.camera.eye_z:.2f} | fov={self.camera.fov_y_deg:.1f}",
            )
            painter.drawText(
                10,
                80,
                (
                    f"Strength xyz: {self.camera.parallax_strength_x:.2f} "
                    f"{self.camera.parallax_strength_y:.2f} {self.camera.parallax_strength_z:.2f} "
                    f"| spread={self.scene.depth_spread:.2f} | z_far={self.camera.z_far:.0f} "
                    f"| target_fps={self.target_fps:.0f} "
                    f"| neutral={'on' if self.neutral_tone_enabled else 'off'} "
                    f"| depth_debug={'on' if self.depth_debug_mode else 'off'} "
                    f"| drift={'on' if self.cinematic_drift_enabled else 'off'}:{self.drift_intensity:.2f}"
                ),
            )
            painter.end()

    def mouseMoveEvent(self, event):
        if not self.use_mouse_mock:
            return
        x = (event.position().x() / max(1.0, self.width())) * 2.0 - 1.0
        y = (event.position().y() / max(1.0, self.height())) * 2.0 - 1.0
        self.pose = NormalizedPose(x=float(x), y=float(y), z=self.pose.z, valid=True)

    def wheelEvent(self, event):
        if not self.use_mouse_mock:
            return
        dz = event.angleDelta().y() / 1200.0
        self.pose = NormalizedPose(x=self.pose.x, y=self.pose.y, z=float(np.clip(self.pose.z + dz, -1.0, 1.0)), valid=True)

    def cleanup(self) -> None:
        # Ensure a current context for GL resource deletion on strict drivers.
        if self.context() is not None:
            self.makeCurrent()
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.program:
            glDeleteProgram(self.program)
        if self.textures:
            glDeleteTextures(len(self.textures), self.textures)
        if self.context() is not None:
            self.doneCurrent()

    def closeEvent(self, event):
        try:
            self.cleanup()
        except Exception:
            pass
        return super().closeEvent(event)
