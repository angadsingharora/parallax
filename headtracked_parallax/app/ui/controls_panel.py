from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ControlsPanel(QWidget):
    calibrate_clicked = Signal()
    toggle_fullscreen = Signal(bool)
    toggle_debug = Signal(bool)
    parallax_x_changed = Signal(float)
    parallax_y_changed = Signal(float)
    depth_changed = Signal(float)
    depth_spread_changed = Signal(float)
    fov_changed = Signal(float)
    render_distance_changed = Signal(float)
    smoothing_changed = Signal(float)
    deadzone_changed = Signal(float)
    depth_debug_changed = Signal(bool)
    eye_refine_changed = Signal(bool)
    neutral_tone_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        actions = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions)
        btn_calibrate = QPushButton("Calibrate Neutral")
        btn_calibrate.clicked.connect(self.calibrate_clicked.emit)
        actions_layout.addWidget(btn_calibrate)
        root.addWidget(actions)

        controls = QGroupBox("Controls")
        form = QFormLayout(controls)

        self.slider_parallax_x = QSlider(Qt.Horizontal)
        self.slider_parallax_x.setRange(50, 500)
        self.slider_parallax_x.setValue(300)
        self.slider_parallax_x.valueChanged.connect(lambda v: self.parallax_x_changed.emit(v / 100.0))
        form.addRow("Parallax X", self.slider_parallax_x)

        self.slider_parallax_y = QSlider(Qt.Horizontal)
        self.slider_parallax_y.setRange(50, 500)
        self.slider_parallax_y.setValue(260)
        self.slider_parallax_y.valueChanged.connect(lambda v: self.parallax_y_changed.emit(v / 100.0))
        form.addRow("Parallax Y", self.slider_parallax_y)

        self.slider_depth = QSlider(Qt.Horizontal)
        self.slider_depth.setRange(20, 300)
        self.slider_depth.setValue(140)
        self.slider_depth.valueChanged.connect(lambda v: self.depth_changed.emit(v / 100.0))
        form.addRow("Parallax Z", self.slider_depth)

        self.slider_depth_spread = QSlider(Qt.Horizontal)
        self.slider_depth_spread.setRange(60, 250)
        self.slider_depth_spread.setValue(100)
        self.slider_depth_spread.valueChanged.connect(lambda v: self.depth_spread_changed.emit(v / 100.0))
        form.addRow("Depth Spread", self.slider_depth_spread)

        self.slider_fov = QSlider(Qt.Horizontal)
        self.slider_fov.setRange(38, 95)
        self.slider_fov.setValue(58)
        self.slider_fov.valueChanged.connect(lambda v: self.fov_changed.emit(float(v)))
        form.addRow("FOV", self.slider_fov)

        self.slider_render_distance = QSlider(Qt.Horizontal)
        self.slider_render_distance.setRange(120, 600)
        self.slider_render_distance.setValue(260)
        self.slider_render_distance.valueChanged.connect(
            lambda v: self.render_distance_changed.emit(float(v))
        )
        form.addRow("Render Distance", self.slider_render_distance)

        self.slider_smoothing = QSlider(Qt.Horizontal)
        self.slider_smoothing.setRange(5, 80)
        self.slider_smoothing.setValue(25)
        self.slider_smoothing.valueChanged.connect(lambda v: self.smoothing_changed.emit(v / 100.0))
        form.addRow("Smoothing", self.slider_smoothing)

        self.slider_deadzone = QSlider(Qt.Horizontal)
        self.slider_deadzone.setRange(0, 20)
        self.slider_deadzone.setValue(5)
        self.slider_deadzone.valueChanged.connect(lambda v: self.deadzone_changed.emit(v / 100.0))
        form.addRow("Deadzone", self.slider_deadzone)

        self.cb_debug = QCheckBox("Debug Overlay")
        self.cb_debug.setChecked(True)
        self.cb_debug.toggled.connect(self.toggle_debug.emit)
        form.addRow(self.cb_debug)

        self.cb_fullscreen = QCheckBox("Fullscreen")
        self.cb_fullscreen.toggled.connect(self.toggle_fullscreen.emit)
        form.addRow(self.cb_fullscreen)

        self.cb_eye = QCheckBox("Use Eye Refinement")
        self.cb_eye.toggled.connect(self.eye_refine_changed.emit)
        form.addRow(self.cb_eye)

        self.cb_neutral_tone = QCheckBox("Neutral Tone")
        self.cb_neutral_tone.setChecked(True)
        self.cb_neutral_tone.toggled.connect(self.neutral_tone_changed.emit)
        form.addRow(self.cb_neutral_tone)

        self.cb_depth_debug = QCheckBox("Depth Debug Mode")
        self.cb_depth_debug.toggled.connect(self.depth_debug_changed.emit)
        form.addRow(self.cb_depth_debug)

        root.addWidget(controls)
        root.addStretch(1)
