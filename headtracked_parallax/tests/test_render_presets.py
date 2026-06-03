from app.render.presets import RENDER_PRESETS


def test_render_presets_cover_expected_named_profiles():
    assert set(RENDER_PRESETS) == {"Balanced", "Immersive", "Studio"}


def test_balanced_preset_matches_current_default_tuning():
    preset = RENDER_PRESETS["Balanced"]

    assert preset.parallax_x == 3.0
    assert preset.parallax_y == 2.6
    assert preset.parallax_z == 1.4
    assert preset.depth_spread == 1.0
    assert preset.fov == 58.0
    assert preset.render_distance == 260.0
    assert preset.render_fps == 60.0
    assert preset.neutral_tone is True
    assert preset.cinematic_drift is True
    assert preset.drift_intensity == 0.45


def test_immersive_and_studio_presets_push_in_opposite_directions():
    immersive = RENDER_PRESETS["Immersive"]
    studio = RENDER_PRESETS["Studio"]

    assert immersive.parallax_x > studio.parallax_x
    assert immersive.parallax_y > studio.parallax_y
    assert immersive.parallax_z > studio.parallax_z
    assert immersive.depth_spread > studio.depth_spread
    assert immersive.fov > studio.fov
    assert immersive.render_distance > studio.render_distance
    assert immersive.render_fps >= studio.render_fps
    assert immersive.cinematic_drift is True
    assert studio.cinematic_drift is False
