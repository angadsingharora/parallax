import os

import main


def test_resolve_qt_opengl_auto_windows():
    assert main._resolve_qt_opengl("auto", "win32") == "angle"


def test_resolve_qt_opengl_auto_non_windows():
    assert main._resolve_qt_opengl("auto", "linux") == "desktop"
    assert main._resolve_qt_opengl("auto", "darwin") == "desktop"


def test_resolve_qt_opengl_named_values():
    assert main._resolve_qt_opengl("desktop", "win32") == "desktop"
    assert main._resolve_qt_opengl("angle", "win32") == "angle"
    assert main._resolve_qt_opengl("software", "linux") == "software"


def test_resolve_qt_opengl_invalid_and_blank():
    assert main._resolve_qt_opengl("", "linux") is None
    assert main._resolve_qt_opengl("invalid", "linux") is None


def test_apply_rendering_platform_sets_env_for_auto(monkeypatch):
    monkeypatch.setenv("PARALLAX_RENDERING_PLATFORM", "auto")
    monkeypatch.setattr(main.sys, "platform", "win32")
    main._apply_rendering_platform()
    assert os.environ["QT_OPENGL"] == "angle"


def test_apply_rendering_platform_ignores_invalid(monkeypatch, capsys):
    monkeypatch.setenv("PARALLAX_RENDERING_PLATFORM", "badvalue")
    monkeypatch.setenv("QT_OPENGL", "desktop")
    main._apply_rendering_platform()
    assert os.environ["QT_OPENGL"] == "desktop"
    err = capsys.readouterr().err
    assert "Expected one of: auto, desktop, angle, software." in err
