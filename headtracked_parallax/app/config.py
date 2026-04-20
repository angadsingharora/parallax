from dataclasses import dataclass


@dataclass(slots=True)
class AppConfig:
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    tracking_process_every_n: int = 1
    tracking_lost_hold_s: float = 0.5
    tracking_lost_badge_s: float = 2.0

    range_tx_mm: float = 120.0
    range_ty_mm: float = 80.0
    range_tz_mm: float = 150.0
    deadzone: float = 0.05

    alpha_xy: float = 0.25
    alpha_z: float = 0.15
    alpha_angle: float = 0.2

    parallax_strength_xy: float = 1.1
    parallax_strength_z: float = 0.8
    base_distance: float = 5.5
    fov_y_deg: float = 58.0

    debug_overlay: bool = True
    use_eye_refinement: bool = False


DEFAULT_CONFIG = AppConfig()
