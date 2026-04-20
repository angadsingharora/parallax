"""Off-axis (head-tracked) virtual camera.

Implements the "fish-tank VR" / Johnny Lee parallax technique: the screen is
treated as a literal window into a 3D world that sits behind the monitor. As
the viewer's head moves, we move the eye in world space and recompute an
asymmetric (off-axis) frustum so that the four corners of the window stay
fixed in world space. This is what makes the scene actually feel volumetric
rather than just a 2D pan.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import NormalizedPose


def _translate(tx: float, ty: float, tz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def _scale(sx: float, sy: float, sz: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def _frustum(left: float, right: float, bottom: float, top: float,
             z_near: float, z_far: float) -> np.ndarray:
    """Standard OpenGL asymmetric frustum (column-major math, returned row-major)."""
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = (2.0 * z_near) / (right - left)
    m[1, 1] = (2.0 * z_near) / (top - bottom)
    m[0, 2] = (right + left) / (right - left)
    m[1, 2] = (top + bottom) / (top - bottom)
    m[2, 2] = -(z_far + z_near) / (z_far - z_near)
    m[2, 3] = -(2.0 * z_far * z_near) / (z_far - z_near)
    m[3, 2] = -1.0
    return m


@dataclass(slots=True)
class VirtualCamera:
    # Half-extents of the virtual "window" in world units. The window sits at
    # z = 0 and the scene lives behind it (negative z).
    window_half_w: float = 1.6
    window_half_h: float = 1.0

    # How far the head is allowed to move in world units for a full normalized
    # pose excursion of +-1. Bigger values = more dramatic parallax.
    parallax_strength_x: float = 3.0
    parallax_strength_y: float = 2.6
    parallax_strength_z: float = 1.4

    # Nominal viewing distance from the screen. This is the eye's z when the
    # head is centered. Smaller = wider FOV / stronger perspective.
    base_distance: float = 2.6
    fov_y_deg: float = 58.0

    # Near/far planes of the projection.
    z_near: float = 0.05
    z_far: float = 200.0

    # Current eye position in world space (updated from head pose).
    eye_x: float = 0.0
    eye_y: float = 0.0
    eye_z: float = 2.4
    depth_debug_mode: bool = False

    def update_from_head_pose(self, pose: NormalizedPose) -> None:
        if not pose.valid:
            # Smoothly relax toward neutral when tracking is lost.
            self.eye_x *= 0.9
            self.eye_y *= 0.9
            self.eye_z = self.base_distance + (self.eye_z - self.base_distance) * 0.9
            return
        debug_boost = 1.65 if self.depth_debug_mode else 1.0
        self.eye_x = float(pose.x) * self.parallax_strength_x * debug_boost
        self.eye_y = float(-pose.y) * self.parallax_strength_y * debug_boost
        self.eye_z = self.base_distance + float(pose.z) * self.parallax_strength_z * debug_boost
        # Don't let the eye crash through the window.
        if self.eye_z < self.z_near + 0.05:
            self.eye_z = self.z_near + 0.05

    def get_view_matrix(self) -> np.ndarray:
        """Pure translation: keep the camera axis-aligned with the screen."""
        return _translate(-self.eye_x, -self.eye_y, -self.eye_z)

    def get_projection_matrix(self, aspect: float) -> np.ndarray:
        """Off-axis frustum keyed to the eye position so the screen acts as a window."""
        # Window size is derived from FOV + nominal viewing distance for tunable perspective.
        base_half_h = float(np.tan(np.radians(self.fov_y_deg * 0.5)) * self.base_distance)
        self.window_half_h = max(0.45, base_half_h)
        self.window_half_w = self.window_half_h * max(aspect, 1e-3)
        # Adjust window aspect ratio to match the actual viewport so the four
        # window corners always project to the four screen corners.
        win_h = self.window_half_h
        win_w = self.window_half_h * max(aspect, 1e-3)
        # Make sure we don't crush the horizontal field of view on very narrow
        # windows.
        if win_w < self.window_half_w:
            win_w = self.window_half_w
            win_h = self.window_half_w / max(aspect, 1e-3)

        ez = max(self.eye_z, self.z_near + 1e-3)
        scale = self.z_near / ez
        left = (-win_w - self.eye_x) * scale
        right = (win_w - self.eye_x) * scale
        bottom = (-win_h - self.eye_y) * scale
        top = (win_h - self.eye_y) * scale
        return _frustum(left, right, bottom, top, self.z_near, self.z_far)

    @staticmethod
    def model_matrix_for_layer(z: float, half_w: float, half_h: float) -> np.ndarray:
        """Place a unit quad at world z with the given half-extents."""
        return _translate(0.0, 0.0, z) @ _scale(half_w, half_h, 1.0)
