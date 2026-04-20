import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@contextmanager
def _silence_fd_stderr():
    """Redirect OS-level stderr (fd 2) to the null device.

    MediaPipe / TFLite write some startup logs from C++ directly to fd 2,
    bypassing Python's ``sys.stderr``. Suppressing them requires temporarily
    redirecting the file descriptor itself.
    """
    try:
        sys.stderr.flush()
    except Exception:
        pass
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


@dataclass(slots=True)
class FaceLandmarkResult:
    face_present: bool
    confidence: float
    landmarks_norm: Optional[np.ndarray]
    landmarks_px: Optional[np.ndarray]
    iris_px: Optional[np.ndarray]


class FaceLandmarkTracker:
    def __init__(self, max_faces: int = 1):
        self._mesh = None
        self._backend = None

        try:
            with _silence_fd_stderr():
                import mediapipe as mp
        except Exception as exc:
            raise RuntimeError("MediaPipe import failed. Install a full mediapipe wheel with Face Mesh support.") from exc

        # Common API path on full mediapipe installs.
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            with _silence_fd_stderr():
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=max_faces,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                # Warm up the graph on a dummy frame so TFLite / absl emit
                # their one-shot init logs (XNNPACK delegate, feedback
                # manager warnings, etc.) while stderr is still muted.
                warmup = np.zeros((96, 96, 3), dtype=np.uint8)
                try:
                    self._mesh.process(warmup)
                except Exception:
                    pass
            self._backend = "solutions"
            return

        # Some wheels expose only mediapipe.tasks and do not include Face Mesh models.
        raise RuntimeError(
            "Installed mediapipe package does not expose Face Mesh. "
            "Install a full build, e.g. `pip install --upgrade --force-reinstall mediapipe==0.10.14`."
        )

    def process(self, frame_bgr: np.ndarray) -> FaceLandmarkResult:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self._mesh.process(frame_rgb)
        if not out.multi_face_landmarks:
            return FaceLandmarkResult(False, 0.0, None, None, None)

        lms = out.multi_face_landmarks[0].landmark
        arr_norm = np.array([(lm.x, lm.y, lm.z) for lm in lms], dtype=np.float32)
        arr_px = np.zeros((arr_norm.shape[0], 2), dtype=np.float32)
        arr_px[:, 0] = arr_norm[:, 0] * w
        arr_px[:, 1] = arr_norm[:, 1] * h

        iris_points = None
        iris_ids = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]
        if arr_px.shape[0] > 477:
            iris_points = arr_px[iris_ids]

        return FaceLandmarkResult(True, 1.0, arr_norm, arr_px, iris_points)

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
