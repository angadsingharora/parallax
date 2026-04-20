import pytest

from app.calibration import CalibrationModel, apply_deadzone, clamp, response_curve
from app.types import HeadPose


def test_clamp_and_deadzone():
    assert clamp(2.0, -1.0, 1.0) == 1.0
    assert clamp(-2.0, -1.0, 1.0) == -1.0
    assert apply_deadzone(0.02, 0.05) == 0.0
    assert apply_deadzone(0.1, 0.05) == 0.1


def test_response_curve_sign():
    assert response_curve(0.25) > 0
    assert response_curve(-0.25) < 0


def test_calibration_normalization_and_sign_convention():
    c = CalibrationModel(range_tx=120.0, range_ty=80.0, range_tz=150.0, deadzone=0.0)
    neutral = HeadPose(tx=0.0, ty=0.0, tz=500.0, yaw=0.0, pitch=0.0, roll=0.0, valid=True)
    c.capture_neutral(neutral)

    p = HeadPose(tx=60.0, ty=-40.0, tz=575.0, yaw=10.0, pitch=-10.0, roll=0.0, valid=True)
    n = c.apply(p)

    assert n.valid is True
    assert n.x == pytest.approx((60.0 / 120.0) ** 1.5)
    assert n.y < 0
    assert n.z > 0


def test_calibration_invalid_pose():
    c = CalibrationModel()
    n = c.apply(HeadPose(valid=False))
    assert n.valid is False
