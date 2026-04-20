from typing import Optional

import numpy as np


LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)
RIGHT_EYE_CORNERS = (362, 263)


def estimate_eye_offset(landmarks_px: np.ndarray) -> tuple[float, float, float]:
    if landmarks_px is None or landmarks_px.shape[0] < 478:
        return 0.0, 0.0, 0.0

    left_center = np.mean(landmarks_px[LEFT_IRIS], axis=0)
    right_center = np.mean(landmarks_px[RIGHT_IRIS], axis=0)
    iris_center = (left_center + right_center) * 0.5

    left_outer = landmarks_px[LEFT_EYE_CORNERS[0]]
    left_inner = landmarks_px[LEFT_EYE_CORNERS[1]]
    right_outer = landmarks_px[RIGHT_EYE_CORNERS[0]]
    right_inner = landmarks_px[RIGHT_EYE_CORNERS[1]]

    eye_center = (left_outer + left_inner + right_outer + right_inner) * 0.25
    eye_width = max(1.0, np.linalg.norm(left_outer - left_inner) + np.linalg.norm(right_outer - right_inner)) * 0.5

    dx = float((iris_center[0] - eye_center[0]) / eye_width)
    dy = float((iris_center[1] - eye_center[1]) / eye_width)

    conf = float(max(0.0, 1.0 - min(1.0, np.linalg.norm([dx, dy]) * 2.0)))
    return dx, dy, conf


def fuse_head_and_eye(head_xy: tuple[float, float], eye_xy: tuple[float, float], eye_confidence: float) -> tuple[float, float]:
    w_eye = 0.15 * max(0.0, min(1.0, eye_confidence))
    w_head = 1.0 - w_eye
    return (w_head * head_xy[0] + w_eye * eye_xy[0], w_head * head_xy[1] + w_eye * eye_xy[1])
