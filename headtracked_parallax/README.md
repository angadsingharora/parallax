# HeadTrackedParallax

Desktop prototype that creates pseudo-3D parallax on a normal monitor by tracking your head with a webcam and shifting a layered OpenGL scene in real time.

## Stack

- Python 3.11
- OpenCV (`solvePnP` for head pose)
- MediaPipe Face Mesh (with iris landmarks when available)
- PySide6 + `QOpenGLWidget`
- PyOpenGL

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Optional rendering platform override:

```bash
# desktop | angle | software
set PARALLAX_RENDERING_PLATFORM=desktop
python main.py
```

## Controls

- `Calibrate Neutral`: capture current posture as center.
- `Parallax XY`: left/right/up/down perspective strength.
- `Depth Z`: forward/back effect strength.
- `Smoothing`: EMA tracking smoothness.
- `Deadzone`: small movement suppression around neutral.
- `Debug Overlay`: render debug values.
- `Fullscreen`: fullscreen demo mode.
- Menu action `Toggle Mouse Mock`: use mouse+wheel instead of webcam tracking.

## Calibration Flow

1. Sit in normal viewing position.
2. Click `Calibrate Neutral`.
3. Move head slightly left/right/up/down to tune sliders.

## Troubleshooting

- Camera unavailable/permission denied: close other camera apps and allow camera permission for Python.
- Jittery motion: increase smoothing, increase deadzone, lower camera resolution.
- Slow tracking FPS: reduce camera resolution or disable eye refinement.
- Tracking lost: app holds briefly, then eases toward neutral and shows lost state in status.

## Notes

- This is pseudo-3D parallax, not stereoscopic 3D.
- Best effect is single-user at a stable viewing position.
- v1 renders its own scene (not system-wide desktop reprojection).

## Tests

```bash
pytest -q
```

Covers calibration normalization, clamp/deadzone/response curve behavior, and EMA smoothing.
