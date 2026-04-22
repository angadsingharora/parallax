from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from ..calibration import CalibrationModel
from ..config import AppConfig, DEFAULT_CONFIG
from ..smoothing import PoseSmoother
from ..types import HeadPose, NormalizedPose, TrackedState
from ..tracker.camera import CameraStream
from ..tracker.face_landmarks import FaceLandmarkTracker
from ..tracker.gaze_estimator import estimate_eye_offset, fuse_head_and_eye
from ..tracker.head_pose import LANDMARK_IDS, draw_pose_axes, estimate_from_landmarks
from ..render.gl_widget import ParallaxGLWidget
from .controls_panel import ControlsPanel


class TrackingWorker:
    def __init__(self, config: AppConfig):
        self.config = config
        self.camera = CameraStream(config.camera_index, config.camera_width, config.camera_height)
        self.face = None
        self.init_error: str | None = None
        try:
            self.face = FaceLandmarkTracker(max_faces=1)
        except Exception as exc:
            self.init_error = str(exc)
        self.calibration = CalibrationModel(
            range_tx=config.range_tx_mm,
            range_ty=config.range_ty_mm,
            range_tz=config.range_tz_mm,
            deadzone=config.deadzone,
        )
        self.smoother = PoseSmoother(config.alpha_xy, config.alpha_z, config.alpha_angle)

        self.lock = threading.Lock()
        self.state = TrackedState()
        self.running = False
        self.thread: threading.Thread | None = None
        self.last_valid_norm = NormalizedPose(valid=False)
        self.last_face_ts = 0.0
        self.use_eye_refine = False

    def start(self) -> bool:
        if self.face is None:
            return False
        if not self.camera.start():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.face is not None:
            self.face.close()
        self.camera.stop()

    def capture_neutral(self) -> None:
        with self.lock:
            self.calibration.capture_neutral(self.state.raw_pose)
            self.state.calibration_ready = self.calibration.ready

    def set_smoothing_alpha(self, a: float) -> None:
        self.smoother = PoseSmoother(alpha_xy=a, alpha_z=max(0.05, a - 0.08), alpha_angle=max(0.05, a - 0.05))

    def set_deadzone(self, v: float) -> None:
        self.calibration.deadzone = v

    def get_state(self) -> TrackedState:
        with self.lock:
            return replace(self.state)

    def _loop(self):
        last_ts = time.time()
        frame_count = 0
        fps_start = time.time()

        while self.running:
            ok, frame, ts = self.camera.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            if self.face is None:
                time.sleep(0.01)
                continue
            res = self.face.process(frame)
            now = time.time()
            pose = HeadPose(valid=False, timestamp=now)
            norm = NormalizedPose(valid=False, timestamp=now)
            debug_frame = frame.copy()

            if res.face_present and res.landmarks_px is not None:
                pose = estimate_from_landmarks(res.landmarks_px, frame.shape[:2])
                if pose.valid:
                    norm = self.calibration.apply(pose)
                    if self.use_eye_refine:
                        ex, ey, conf = estimate_eye_offset(res.landmarks_px)
                        fx, fy = fuse_head_and_eye((norm.x, norm.y), (ex, ey), conf)
                        norm.x, norm.y = fx, fy
                    norm = self.smoother.smooth(norm)
                    self.last_valid_norm = norm
                    self.last_face_ts = now

                for p in res.landmarks_px[::8]:
                    cv2.circle(debug_frame, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1)

                try:
                    image_points = np.array(
                        [
                            res.landmarks_px[LANDMARK_IDS["nose_tip"]],
                            res.landmarks_px[LANDMARK_IDS["chin"]],
                            res.landmarks_px[LANDMARK_IDS["left_eye_outer"]],
                            res.landmarks_px[LANDMARK_IDS["right_eye_outer"]],
                            res.landmarks_px[LANDMARK_IDS["left_mouth"]],
                            res.landmarks_px[LANDMARK_IDS["right_mouth"]],
                        ],
                        dtype=np.float32,
                    )
                    debug_frame = draw_pose_axes(debug_frame, pose, image_points)
                except Exception:
                    pass
            
            elapsed_since_face = now - self.last_face_ts
            tracking_lost = elapsed_since_face > self.config.tracking_lost_badge_s
            if elapsed_since_face > self.config.tracking_lost_hold_s:
                dt = now - last_ts
                blend = min(1.0, dt / 0.45)
                norm = NormalizedPose(
                    x=self.last_valid_norm.x * (1.0 - blend),
                    y=self.last_valid_norm.y * (1.0 - blend),
                    z=self.last_valid_norm.z * (1.0 - blend),
                    yaw=self.last_valid_norm.yaw * (1.0 - blend),
                    pitch=self.last_valid_norm.pitch * (1.0 - blend),
                    roll=self.last_valid_norm.roll * (1.0 - blend),
                    valid=False,
                    timestamp=now,
                )
                self.last_valid_norm = norm

            frame_count += 1
            if now - fps_start >= 1.0:
                fps = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now
            else:
                fps = self.state.fps_tracking

            with self.lock:
                self.state = TrackedState(
                    raw_pose=pose,
                    normalized_pose=norm,
                    landmarks_2d=res.landmarks_px,
                    fps_tracking=fps,
                    face_confidence=res.confidence,
                    debug_frame_bgr=debug_frame,
                    tracking_lost=tracking_lost,
                    calibration_ready=self.calibration.ready,
                )
            last_ts = now


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig = DEFAULT_CONFIG):
        super().__init__()
        self.config = config
        self.worker = TrackingWorker(config)
        self.setWindowTitle("HeadTrackedParallax")
        self.resize(1360, 820)

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        assets = Path(__file__).resolve().parent.parent / "assets" / "demo_layers"
        self.gl = ParallaxGLWidget(assets)
        layout.addWidget(self.gl, 1)

        self.controls = ControlsPanel()
        layout.addWidget(self.controls)

        self.status_label = QLabel("Initializing...")
        self.statusBar().addPermanentWidget(self.status_label)

        self.controls.calibrate_clicked.connect(self.on_calibrate)
        self.controls.toggle_fullscreen.connect(self.on_fullscreen)
        self.controls.toggle_debug.connect(self.on_debug)
        self.controls.parallax_x_changed.connect(self.on_parallax_x)
        self.controls.parallax_y_changed.connect(self.on_parallax_y)
        self.controls.depth_changed.connect(self.on_parallax_z)
        self.controls.depth_spread_changed.connect(self.on_depth_spread)
        self.controls.fov_changed.connect(self.on_fov_changed)
        self.controls.render_distance_changed.connect(self.on_render_distance_changed)
        self.controls.smoothing_changed.connect(self.on_smoothing)
        self.controls.deadzone_changed.connect(self.on_deadzone)
        self.controls.depth_debug_changed.connect(self.on_depth_debug_mode)
        self.controls.eye_refine_changed.connect(self.on_eye_refine)
        self.controls.neutral_tone_changed.connect(self.on_neutral_tone_changed)

        act_mock = QAction("Toggle Mouse Mock", self)
        act_mock.triggered.connect(self.toggle_mouse_mock)
        self.menuBar().addAction(act_mock)

        started = self.worker.start()
        if not started:
            reason = self.worker.init_error or "No webcam found or permission denied."
            QMessageBox.warning(
                self,
                "Tracking Unavailable",
                f"{reason}\n\nThe app will still run. Use Menu -> Toggle Mouse Mock to test parallax.",
            )

        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self.tick)
        self.ui_timer.start(16)

    def on_calibrate(self) -> None:
        self.worker.capture_neutral()

    def on_fullscreen(self, enabled: bool) -> None:
        if enabled:
            self.showFullScreen()
        else:
            self.showNormal()

    def on_debug(self, enabled: bool) -> None:
        self.gl.set_debug_overlay(enabled)

    def on_parallax_x(self, v: float) -> None:
        self.gl.camera.parallax_strength_x = v

    def on_parallax_y(self, v: float) -> None:
        self.gl.camera.parallax_strength_y = v

    def on_parallax_z(self, v: float) -> None:
        self.gl.camera.parallax_strength_z = v

    def on_depth_spread(self, v: float) -> None:
        self.gl.set_depth_spread(v)

    def on_fov_changed(self, v: float) -> None:
        self.gl.set_fov(v)

    def on_render_distance_changed(self, v: float) -> None:
        self.gl.set_render_distance(v)

    def on_smoothing(self, v: float) -> None:
        self.worker.set_smoothing_alpha(v)

    def on_deadzone(self, v: float) -> None:
        self.worker.set_deadzone(v)

    def on_eye_refine(self, enabled: bool) -> None:
        self.worker.use_eye_refine = enabled

    def on_depth_debug_mode(self, enabled: bool) -> None:
        self.gl.set_depth_debug_mode(enabled)

    def on_neutral_tone_changed(self, enabled: bool) -> None:
        self.gl.set_neutral_tone(enabled)

    def toggle_mouse_mock(self) -> None:
        self.gl.use_mouse_mock = not self.gl.use_mouse_mock
        self.statusBar().showMessage(f"Mouse mock: {'ON' if self.gl.use_mouse_mock else 'OFF'}", 1200)

    def tick(self) -> None:
        try:
            st = self.worker.get_state()
            pose = st.normalized_pose if st.normalized_pose.valid else self.gl.pose
            self.gl.set_pose(pose)

            rp = st.raw_pose
            status = (
                f"tracking_fps={st.fps_tracking:.1f} | face={'yes' if st.face_confidence > 0 else 'no'} "
                f"| calib={'ready' if st.calibration_ready else 'pending'} | lost={'yes' if st.tracking_lost else 'no'} "
                f"| ypr=({rp.yaw:.1f},{rp.pitch:.1f},{rp.roll:.1f}) | t=({rp.tx:.1f},{rp.ty:.1f},{rp.tz:.1f})"
            )
            self.status_label.setText(status)
        except Exception as exc:
            self.status_label.setText(f"tick error: {exc}")

    def closeEvent(self, event):
        self.ui_timer.stop()
        self.worker.stop()
        return super().closeEvent(event)
