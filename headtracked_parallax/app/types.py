from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass(slots=True)
class HeadPose:
    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    valid: bool = False
    timestamp: float = 0.0


@dataclass(slots=True)
class NormalizedPose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    valid: bool = False
    timestamp: float = 0.0


@dataclass(slots=True)
class TrackedState:
    raw_pose: HeadPose = field(default_factory=HeadPose)
    normalized_pose: NormalizedPose = field(default_factory=NormalizedPose)
    landmarks_2d: Optional[np.ndarray] = None
    fps_tracking: float = 0.0
    face_confidence: float = 0.0
    debug_frame_bgr: Optional[np.ndarray] = None
    tracking_lost: bool = True
    calibration_ready: bool = False
