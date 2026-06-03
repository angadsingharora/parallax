from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RenderPreset:
    name: str
    parallax_x: float
    parallax_y: float
    parallax_z: float
    depth_spread: float
    fov: float
    render_distance: float
    render_fps: float
    neutral_tone: bool
    cinematic_drift: bool
    drift_intensity: float
    depth_debug: bool = False


RENDER_PRESETS: dict[str, RenderPreset] = {
    "Balanced": RenderPreset(
        name="Balanced",
        parallax_x=3.0,
        parallax_y=2.6,
        parallax_z=1.4,
        depth_spread=1.0,
        fov=58.0,
        render_distance=260.0,
        render_fps=60.0,
        neutral_tone=True,
        cinematic_drift=True,
        drift_intensity=0.45,
    ),
    "Immersive": RenderPreset(
        name="Immersive",
        parallax_x=4.1,
        parallax_y=3.5,
        parallax_z=2.0,
        depth_spread=1.35,
        fov=68.0,
        render_distance=340.0,
        render_fps=90.0,
        neutral_tone=False,
        cinematic_drift=True,
        drift_intensity=0.62,
    ),
    "Studio": RenderPreset(
        name="Studio",
        parallax_x=2.0,
        parallax_y=1.8,
        parallax_z=0.9,
        depth_spread=0.85,
        fov=50.0,
        render_distance=220.0,
        render_fps=60.0,
        neutral_tone=True,
        cinematic_drift=False,
        drift_intensity=0.0,
    ),
}

