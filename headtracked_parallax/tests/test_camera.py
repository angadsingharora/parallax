import numpy as np

from app.render.camera import VirtualCamera
from app.types import NormalizedPose


def test_view_matrix_includes_head_tilt_rotation():
    cam = VirtualCamera()
    cam.update_from_head_pose(NormalizedPose(yaw=1.0, pitch=1.0, valid=True))
    view = cam.get_view_matrix()
    # Rotation should introduce non-zero off-diagonal terms.
    assert abs(float(view[0, 2])) > 1e-5
    assert abs(float(view[1, 2])) > 1e-5


def test_view_matrix_relaxes_when_tracking_lost():
    cam = VirtualCamera()
    cam.update_from_head_pose(NormalizedPose(yaw=1.0, pitch=1.0, valid=True))
    cam.update_from_head_pose(NormalizedPose(valid=False))
    assert 0.0 < abs(cam.head_yaw) < 1.0
    assert 0.0 < abs(cam.head_pitch) < 1.0


def test_projection_has_valid_perspective_terms():
    cam = VirtualCamera()
    proj = cam.get_projection_matrix(16.0 / 9.0)
    assert np.isfinite(proj).all()
    assert proj[3, 2] == -1.0
