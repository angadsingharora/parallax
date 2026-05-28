from pathlib import Path
from types import SimpleNamespace

import pytest

from app.render import gl_widget as gw


def _bare_widget():
    # Avoid QOpenGLWidget construction for headless unit tests.
    w = gw.ParallaxGLWidget.__new__(gw.ParallaxGLWidget)
    w.program = None
    w.vao = None
    w.vbo = None
    w.u_mvp_loc = -1
    w.u_alpha_loc = -1
    w.u_neutral_mix_loc = -1
    w.textures = []
    w.gl_failed = False
    w.gl_error = ""
    w.max_anisotropy = 1.0
    return w


def test_initialize_gl_queries_anisotropy_before_texture_load(monkeypatch):
    w = _bare_widget()
    call_order = []

    monkeypatch.setattr(gw, "glEnable", lambda *_: None)
    monkeypatch.setattr(gw, "glBlendFunc", lambda *_: None)
    monkeypatch.setattr(gw, "glPixelStorei", lambda *_: None)
    monkeypatch.setattr(gw, "compileShader", lambda src, shader_type: f"shader-{shader_type}")
    monkeypatch.setattr(gw, "compileProgram", lambda *args: 101)
    monkeypatch.setattr(gw, "glGenVertexArrays", lambda n: 201)
    monkeypatch.setattr(gw, "glGenBuffers", lambda n: 301)
    monkeypatch.setattr(gw, "glBindVertexArray", lambda *_: None)
    monkeypatch.setattr(gw, "glBindBuffer", lambda *_: None)
    monkeypatch.setattr(gw, "glBufferData", lambda *_: None)
    monkeypatch.setattr(gw, "glEnableVertexAttribArray", lambda *_: None)
    monkeypatch.setattr(gw, "glVertexAttribPointer", lambda *_: None)
    monkeypatch.setattr(gw, "glUseProgram", lambda *_: None)
    monkeypatch.setattr(gw, "glUniform1i", lambda *_: None)
    monkeypatch.setattr(gw, "glGetUniformLocation", lambda *_: 1)

    def fake_query():
        call_order.append("query")
        return 3.5

    def fake_load():
        call_order.append("load")
        assert w.max_anisotropy == 3.5

    w._query_anisotropy_limit = fake_query
    w._load_textures = fake_load

    gw.ParallaxGLWidget.initializeGL(w)

    assert call_order == ["query", "load"]
    assert w.gl_failed is False
    assert w.program == 101


def test_cleanup_makes_context_current_before_gl_deletes(monkeypatch):
    w = _bare_widget()
    events = []
    w.vbo = 11
    w.vao = 22
    w.program = 33
    w.textures = [44, 55]
    w.context = lambda: object()
    w.makeCurrent = lambda: events.append("makeCurrent")
    w.doneCurrent = lambda: events.append("doneCurrent")

    monkeypatch.setattr(gw, "glDeleteBuffers", lambda n, vals: events.append(("del_vbo", n, tuple(vals))))
    monkeypatch.setattr(gw, "glDeleteVertexArrays", lambda n, vals: events.append(("del_vao", n, tuple(vals))))
    monkeypatch.setattr(gw, "glDeleteProgram", lambda prog: events.append(("del_program", prog)))
    monkeypatch.setattr(gw, "glDeleteTextures", lambda n, vals: events.append(("del_tex", n, tuple(vals))))

    gw.ParallaxGLWidget.cleanup(w)

    assert events[0] == "makeCurrent"
    assert events[-1] == "doneCurrent"
    assert ("del_vbo", 1, (11,)) in events
    assert ("del_vao", 1, (22,)) in events
    assert ("del_program", 33) in events
    assert ("del_tex", 2, (44, 55)) in events


def test_apply_interaction_overrides_updates_camera_state():
    w = _bare_widget()
    w.pan_x = 0.3
    w.pan_y = -0.2
    w.zoom_offset = -0.4
    w.orbit_yaw_deg = 3.5
    w.orbit_pitch_deg = -2.0
    w.camera = SimpleNamespace(
        eye_x=1.0,
        eye_y=2.0,
        eye_z=3.0,
        z_near=0.1,
        head_yaw=0.1,
        head_pitch=-0.2,
        max_yaw_tilt_deg=7.0,
        max_pitch_tilt_deg=5.0,
    )

    gw.ParallaxGLWidget._apply_interaction_overrides(w)

    assert w.camera.eye_x == pytest.approx(1.3)
    assert w.camera.eye_y == pytest.approx(1.8)
    assert w.camera.eye_z == pytest.approx(2.6)
    assert w.camera.head_yaw == pytest.approx(0.6)
    assert w.camera.head_pitch == pytest.approx(-0.6)


def test_apply_interaction_overrides_clamps_zoom_and_head():
    w = _bare_widget()
    w.pan_x = 0.0
    w.pan_y = 0.0
    w.zoom_offset = -10.0
    w.orbit_yaw_deg = 30.0
    w.orbit_pitch_deg = -30.0
    w.camera = SimpleNamespace(
        eye_x=0.0,
        eye_y=0.0,
        eye_z=0.2,
        z_near=0.1,
        head_yaw=0.9,
        head_pitch=-0.9,
        max_yaw_tilt_deg=7.0,
        max_pitch_tilt_deg=5.0,
    )

    gw.ParallaxGLWidget._apply_interaction_overrides(w)

    assert w.camera.eye_z == pytest.approx(0.15)
    assert w.camera.head_yaw == 1.0
    assert w.camera.head_pitch == -1.0
