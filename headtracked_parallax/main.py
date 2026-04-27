import os
import sys
import warnings

# Silence noisy startup logs from TensorFlow Lite / MediaPipe / absl / glog
# before any of those libraries get imported transitively.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("ABSL_LOG_LEVEL", "3")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow


def _apply_rendering_platform() -> None:
    """
    Configure Qt rendering backend from PARALLAX_RENDERING_PLATFORM.

    Supported values:
    - desktop: native desktop OpenGL
    - angle: ANGLE backend on Windows
    - software: software rasterization
    """
    requested = os.environ.get("PARALLAX_RENDERING_PLATFORM", "").strip().lower()
    if not requested:
        return

    qt_opengl_by_platform = {
        "desktop": "desktop",
        "angle": "angle",
        "software": "software",
    }
    qt_opengl = qt_opengl_by_platform.get(requested)
    if qt_opengl is None:
        print(
            "Ignoring unsupported PARALLAX_RENDERING_PLATFORM value "
            f"'{requested}'. Expected one of: desktop, angle, software.",
            file=sys.stderr,
        )
        return

    os.environ["QT_OPENGL"] = qt_opengl


def main() -> int:
    _apply_rendering_platform()
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
