from dataclasses import dataclass
import numpy as np

from .types import HeadPose, NormalizedPose


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def apply_deadzone(v: float, deadzone: float) -> float:
    if abs(v) < deadzone:
        return 0.0
    return v


def response_curve(v: float) -> float:
    return float(np.sign(v) * (abs(v) ** 1.5))


@dataclass(slots=True)
class CalibrationModel:
    range_tx: float = 120.0
    range_ty: float = 80.0
    range_tz: float = 150.0
    deadzone: float = 0.05

    neutral_tx: float = 0.0
    neutral_ty: float = 0.0
    neutral_tz: float = 0.0
    neutral_yaw: float = 0.0
    neutral_pitch: float = 0.0
    ready: bool = False

    def start(self) -> None:
        self.ready = False

    def capture_neutral(self, pose: HeadPose) -> None:
        if not pose.valid:
            return
        self.neutral_tx = pose.tx
        self.neutral_ty = pose.ty
        self.neutral_tz = pose.tz
        self.neutral_yaw = pose.yaw
        self.neutral_pitch = pose.pitch
        self.ready = True

    def apply(self, pose: HeadPose) -> NormalizedPose:
        if not pose.valid:
            return NormalizedPose(valid=False, timestamp=pose.timestamp)

        if not self.ready:
            self.capture_neutral(pose)

        x = clamp((pose.tx - self.neutral_tx) / self.range_tx, -1.0, 1.0)
        y = clamp((pose.ty - self.neutral_ty) / self.range_ty, -1.0, 1.0)
        z = clamp((pose.tz - self.neutral_tz) / self.range_tz, -1.0, 1.0)

        x = response_curve(apply_deadzone(x, self.deadzone))
        y = response_curve(apply_deadzone(y, self.deadzone))
        z = response_curve(apply_deadzone(z, self.deadzone))

        yaw = clamp((pose.yaw - self.neutral_yaw) / 25.0, -1.0, 1.0)
        pitch = clamp((pose.pitch - self.neutral_pitch) / 20.0, -1.0, 1.0)

        return NormalizedPose(
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            pitch=pitch,
            roll=pose.roll,
            valid=True,
            timestamp=pose.timestamp,
        )
