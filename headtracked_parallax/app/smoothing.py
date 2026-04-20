from dataclasses import dataclass
from typing import Optional

from .types import NormalizedPose


@dataclass(slots=True)
class EMAFilter:
    alpha: float
    value: Optional[float] = None

    def reset(self) -> None:
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


class PoseSmoother:
    def __init__(self, alpha_xy: float = 0.25, alpha_z: float = 0.15, alpha_angle: float = 0.2):
        self.fx = EMAFilter(alpha_xy)
        self.fy = EMAFilter(alpha_xy)
        self.fz = EMAFilter(alpha_z)
        self.fyaw = EMAFilter(alpha_angle)
        self.fpitch = EMAFilter(alpha_angle)
        self.froll = EMAFilter(alpha_angle)

    def smooth(self, pose: NormalizedPose) -> NormalizedPose:
        if not pose.valid:
            return pose
        return NormalizedPose(
            x=self.fx.update(pose.x),
            y=self.fy.update(pose.y),
            z=self.fz.update(pose.z),
            yaw=self.fyaw.update(pose.yaw),
            pitch=self.fpitch.update(pose.pitch),
            roll=self.froll.update(pose.roll),
            valid=True,
            timestamp=pose.timestamp,
        )


class LostTrackingDecay:
    def __init__(self, tau_seconds: float = 0.4):
        self.tau_seconds = tau_seconds

    def decay_toward_center(self, pose: NormalizedPose, dt: float) -> NormalizedPose:
        if pose.valid:
            return pose
        k = max(0.0, min(1.0, dt / self.tau_seconds))
        return NormalizedPose(
            x=pose.x * (1.0 - k),
            y=pose.y * (1.0 - k),
            z=pose.z * (1.0 - k),
            yaw=pose.yaw * (1.0 - k),
            pitch=pose.pitch * (1.0 - k),
            roll=pose.roll * (1.0 - k),
            valid=False,
            timestamp=pose.timestamp,
        )
