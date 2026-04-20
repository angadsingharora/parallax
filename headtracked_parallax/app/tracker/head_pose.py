from dataclasses import dataclass
import time

import cv2
import numpy as np

from ..types import HeadPose

# MediaPipe Face Mesh landmark IDs
LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class PoseDebug:
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray


def _euler_from_rotmat(r: np.ndarray) -> tuple[float, float, float]:
    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = np.arctan2(r[1, 0], r[0, 0])
    else:
        pitch = np.arctan2(-r[1, 2], r[1, 1])
        yaw = np.arctan2(-r[2, 0], sy)
        roll = 0.0

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)


def _build_image_points(landmarks_px: np.ndarray) -> np.ndarray:
    return np.array(
        [
            landmarks_px[LANDMARK_IDS["nose_tip"]],
            landmarks_px[LANDMARK_IDS["chin"]],
            landmarks_px[LANDMARK_IDS["left_eye_outer"]],
            landmarks_px[LANDMARK_IDS["right_eye_outer"]],
            landmarks_px[LANDMARK_IDS["left_mouth"]],
            landmarks_px[LANDMARK_IDS["right_mouth"]],
        ],
        dtype=np.float32,
    )


def estimate_head_pose(image_points: np.ndarray, image_size: tuple[int, int]) -> HeadPose:
    image_height, image_width = image_size
    focal_length = float(image_width)
    center = (image_width / 2.0, image_height / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS,
        image_points.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return HeadPose(valid=False, timestamp=time.time())

    rotmat, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _euler_from_rotmat(rotmat)

    tx = float(tvec[0][0])
    ty = float(-tvec[1][0])
    tz = float(tvec[2][0])

    return HeadPose(tx=tx, ty=ty, tz=tz, yaw=float(yaw), pitch=float(pitch), roll=float(roll), valid=True, timestamp=time.time())


def estimate_from_landmarks(landmarks_px: np.ndarray, image_size: tuple[int, int]) -> HeadPose:
    try:
        points = _build_image_points(landmarks_px)
        return estimate_head_pose(points, image_size)
    except Exception:
        return HeadPose(valid=False, timestamp=time.time())


def draw_pose_axes(frame_bgr: np.ndarray, pose: HeadPose, image_points: np.ndarray) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points.astype(np.float64), camera_matrix, dist_coeffs)
    if success:
        cv2.drawFrameAxes(frame_bgr, camera_matrix, dist_coeffs, rvec, tvec, 60.0, 2)
    return frame_bgr
