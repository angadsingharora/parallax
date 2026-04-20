import pytest

from app.smoothing import EMAFilter, PoseSmoother
from app.types import NormalizedPose


def test_ema_filter_basic():
    f = EMAFilter(alpha=0.25)
    assert f.update(1.0) == pytest.approx(1.0)
    assert f.update(0.0) == pytest.approx(0.75)


def test_pose_smoother_tracks_valid_pose():
    s = PoseSmoother(alpha_xy=0.5, alpha_z=0.2, alpha_angle=0.5)
    p1 = NormalizedPose(x=1.0, y=-1.0, z=0.5, yaw=0.2, pitch=-0.2, roll=0.1, valid=True)
    p2 = s.smooth(p1)
    assert p2.valid is True
    assert p2.x == pytest.approx(1.0)

    p3 = s.smooth(NormalizedPose(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0, roll=0.0, valid=True))
    assert p3.x == pytest.approx(0.5)
    assert p3.z == pytest.approx(0.4)


def test_invalid_pose_passthrough():
    s = PoseSmoother()
    invalid = NormalizedPose(valid=False)
    out = s.smooth(invalid)
    assert out.valid is False
