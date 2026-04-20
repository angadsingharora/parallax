from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
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
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetAttribLocation,
    glGetUniformLocation,
    glTexImage2D,
    glTexParameteri,
    glUniform1f,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)
from OpenGL.GL.shaders import compileProgram, compileShader
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from ..types import NormalizedPose
from .camera import VirtualCamera
from .scene import DemoScene


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

        self.program = None
        self.vao = None
        self.vbo = None
        self.textures = []
        self.gl_failed = False
        self.gl_error = ""

        self.last_frame_time = time.time()
        self.render_fps = 0.0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

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
        if self.program:
            self._load_textures()

    def set_fov(self, fov_deg: float) -> None:
        self.camera.fov_y_deg = max(38.0, min(95.0, float(fov_deg)))

    def set_depth_debug_mode(self, enabled: bool) -> None:
        self.depth_debug_mode = enabled
        self.camera.depth_debug_mode = enabled
        if self.program:
            self._load_textures()

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
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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
            pos_loc = glGetAttribLocation(self.program, "a_pos")
            uv_loc = glGetAttribLocation(self.program, "a_uv")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 2, GL_FLOAT, GL_FALSE, stride, None)
            glEnableVertexAttribArray(uv_loc)
            glVertexAttribPointer(uv_loc, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))

            glBindVertexArray(0)
            self._load_textures()
            self.gl_failed = False
        except Exception as exc:
            self.gl_failed = True
            self.gl_error = str(exc)

    def _load_textures(self) -> None:
        images = self.scene.load_qimages(self.depth_debug_mode)
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
            arr = np.ascontiguousarray(arr)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
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

        pose = self.pose
        self.camera.update_from_head_pose(pose)

        w = max(1, self.width())
        h = max(1, self.height())
        aspect = float(w) / float(h)
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix(aspect)
        vp = proj @ view

        glUseProgram(self.program)
        glBindVertexArray(self.vao)

        u_mvp = glGetUniformLocation(self.program, "u_mvp")
        u_alpha = glGetUniformLocation(self.program, "u_alpha")

        # Draw farthest first so depth-test + alpha discard composite cleanly.
        live_layers = self.scene.get_layers(self.depth_debug_mode)
        layers = list(enumerate(live_layers))
        layers.sort(key=lambda it: it[1].z)  # most negative (far) first

        for idx, layer in layers:
            half_w, half_h = self._layer_half_extents(idx, layer)
            model = self.camera.model_matrix_for_layer(layer.z, half_w, half_h)
            mvp = vp @ model
            glUniformMatrix4fv(u_mvp, 1, GL_TRUE, mvp.astype(np.float32))
            glUniform1f(u_alpha, layer.alpha)
            if idx < len(self.textures):
                glBindTexture(GL_TEXTURE_2D, self.textures[idx])
            glDrawArrays(GL_TRIANGLES, 0, 6)

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
                    f"| spread={self.scene.depth_spread:.2f} | depth_debug={'on' if self.depth_debug_mode else 'off'}"
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
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.program:
            glDeleteProgram(self.program)
        if self.textures:
            glDeleteTextures(len(self.textures), self.textures)

    def closeEvent(self, event):
        try:
            self.cleanup()
        except Exception:
            pass
        return super().closeEvent(event)
