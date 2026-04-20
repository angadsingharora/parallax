"""Scene description + procedural fallback art for the parallax demo."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QRadialGradient,
)


@dataclass(slots=True)
class Layer:
    name: str
    z: float
    alpha: float
    image_path: Optional[Path] = None


FALLBACK_W = 1536
FALLBACK_H = 1024


class DemoScene:
    def __init__(self, assets_dir: Path):
        self.assets_dir = assets_dir
        self.depth_spread = 1.0

        # Requested stronger depth spacing (screen plane is z=0, scene behind is negative).
        self._base_layers: List[Layer] = [
            Layer("far_bg", z=-20.0, alpha=1.00, image_path=assets_dir / "bg.png"),
            Layer("bg", z=-12.0, alpha=0.98, image_path=assets_dir / "mid1.png"),
            Layer("screen_anchor", z=-7.0, alpha=0.95, image_path=assets_dir / "mid2.png"),
            Layer("near", z=-3.0, alpha=0.90, image_path=assets_dir / "fg.png"),
            Layer("fg", z=-1.0, alpha=0.86, image_path=None),
        ]

        # Exaggerated debug planes with obvious labels/patterns for depth tuning.
        self._debug_layers: List[Layer] = [
            Layer("grid_backwall", z=-22.0, alpha=1.0, image_path=None),
            Layer("depth_panel_far", z=-14.0, alpha=0.92, image_path=None),
            Layer("depth_panel_mid", z=-8.0, alpha=0.92, image_path=None),
            Layer("depth_panel_near", z=-3.5, alpha=0.92, image_path=None),
            Layer("depth_panel_fg", z=-1.2, alpha=0.86, image_path=None),
        ]

    def set_depth_spread(self, spread: float) -> None:
        self.depth_spread = max(0.6, min(2.5, float(spread)))

    def get_layers(self, depth_debug_mode: bool) -> List[Layer]:
        src = self._debug_layers if depth_debug_mode else self._base_layers
        out: List[Layer] = []
        for layer in src:
            out.append(Layer(layer.name, layer.z * self.depth_spread, layer.alpha, layer.image_path))
        return out

    def load_qimages(self, depth_debug_mode: bool) -> List[QImage]:
        layers = self.get_layers(depth_debug_mode)
        images: List[QImage] = []
        for i, layer in enumerate(layers):
            if not depth_debug_mode and layer.image_path and layer.image_path.exists():
                img = QImage(str(layer.image_path))
                if not img.isNull():
                    images.append(img.convertToFormat(QImage.Format.Format_RGBA8888))
                    continue
            images.append(self._make_fallback(i, layer, depth_debug_mode))
        return images

    @staticmethod
    def _new_transparent() -> QImage:
        img = QImage(FALLBACK_W, FALLBACK_H, QImage.Format.Format_RGBA8888)
        img.fill(QColor(0, 0, 0, 0))
        return img

    def _make_fallback(self, idx: int, layer: Layer, depth_debug_mode: bool) -> QImage:
        if depth_debug_mode:
            return self._make_depth_debug_layer(idx, layer)

        name = layer.name
        if name == "far_bg":
            return self._make_sky()
        if name == "bg":
            return self._make_hills()
        if name == "screen_anchor":
            return self._make_trees()
        if name == "near":
            return self._make_grass()
        return self._make_foreground_shapes()

    @staticmethod
    def _make_sky() -> QImage:
        img = QImage(FALLBACK_W, FALLBACK_H, QImage.Format.Format_RGBA8888)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        grad = QLinearGradient(0, 0, 0, FALLBACK_H)
        grad.setColorAt(0.0, QColor(9, 16, 40))
        grad.setColorAt(0.55, QColor(40, 68, 126))
        grad.setColorAt(1.0, QColor(245, 184, 132))
        p.fillRect(img.rect(), QBrush(grad))

        sun = QRadialGradient(QPointF(FALLBACK_W * 0.72, FALLBACK_H * 0.78), FALLBACK_W * 0.2)
        sun.setColorAt(0.0, QColor(255, 245, 220, 230))
        sun.setColorAt(1.0, QColor(255, 190, 130, 0))
        p.setBrush(QBrush(sun))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(img.rect())

        rng = random.Random(7)
        p.setPen(Qt.PenStyle.NoPen)
        for _ in range(200):
            x = rng.random() * FALLBACK_W
            y = rng.random() * FALLBACK_H * 0.55
            r = rng.random() * 1.6 + 0.3
            p.setBrush(QColor(255, 255, 255, int(80 + rng.random() * 160)))
            p.drawEllipse(QPointF(x, y), r, r)
        p.end()
        return img

    @staticmethod
    def _make_hills() -> QImage:
        img = DemoScene._new_transparent()
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        rng = random.Random(11)
        for band in range(3):
            base_y = FALLBACK_H * (0.53 + band * 0.08)
            color = QColor(35 + band * 18, 52 + band * 14, 88 + band * 10, 235)
            from PySide6.QtGui import QPolygonF

            poly = QPolygonF()
            poly.append(QPointF(-50, FALLBACK_H + 50))
            x = -50.0
            while x <= FALLBACK_W + 50:
                y = base_y + math.sin(x * 0.004 + band) * 85 + (rng.random() - 0.5) * 70
                poly.append(QPointF(x, y))
                x += 58
            poly.append(QPointF(FALLBACK_W + 50, FALLBACK_H + 50))
            p.setBrush(color)
            p.drawPolygon(poly)
        p.end()
        return img

    @staticmethod
    def _make_trees() -> QImage:
        img = DemoScene._new_transparent()
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        rng = random.Random(23)
        ground_y = FALLBACK_H * 0.78
        for _ in range(28):
            x = rng.random() * FALLBACK_W
            h = rng.random() * 220 + 180
            w = h * 0.52
            color = QColor(16, 34, 30, 235)
            from PySide6.QtGui import QPolygonF

            poly = QPolygonF()
            poly.append(QPointF(x, ground_y - h))
            poly.append(QPointF(x - w * 0.5, ground_y))
            poly.append(QPointF(x + w * 0.5, ground_y))
            p.setBrush(color)
            p.drawPolygon(poly)

        # Screen-anchor guide stripes to make parallax reference obvious.
        pen = QPen(QColor(230, 230, 255, 44), 2)
        p.setPen(pen)
        for x in range(0, FALLBACK_W + 1, 96):
            p.drawLine(x, int(FALLBACK_H * 0.45), x, FALLBACK_H)
        p.end()
        return img

    @staticmethod
    def _make_grass() -> QImage:
        img = DemoScene._new_transparent()
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rng = random.Random(41)
        from PySide6.QtGui import QPolygonF

        ground_top = FALLBACK_H * 0.86
        poly = QPolygonF()
        poly.append(QPointF(-20, FALLBACK_H + 20))
        x = -20.0
        while x <= FALLBACK_W + 20:
            y = ground_top + math.sin(x * 0.02) * 11 + rng.random() * 6
            poly.append(QPointF(x, y))
            x += 18
        poly.append(QPointF(FALLBACK_W + 20, FALLBACK_H + 20))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(8, 14, 18, 250))
        p.drawPolygon(poly)

        pen = QPen(QColor(8, 14, 18, 235))
        pen.setWidthF(3.2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        for _ in range(170):
            x = rng.random() * FALLBACK_W
            base = ground_top + rng.random() * 20
            height = rng.random() * 95 + 30
            sway = (rng.random() - 0.5) * 30
            p.drawLine(QPointF(x, base), QPointF(x + sway, base - height))
        p.end()
        return img

    @staticmethod
    def _make_foreground_shapes() -> QImage:
        img = DemoScene._new_transparent()
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(30, 20, 26, 210))
        p.drawEllipse(80, 520, 460, 360)
        p.drawEllipse(1020, 560, 420, 320)
        p.end()
        return img

    @staticmethod
    def _make_depth_debug_layer(idx: int, layer: Layer) -> QImage:
        img = DemoScene._new_transparent()
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        palette = [
            QColor(18, 58, 115, 210),
            QColor(28, 115, 95, 208),
            QColor(155, 124, 28, 206),
            QColor(146, 68, 38, 206),
            QColor(150, 46, 72, 206),
        ]
        c = palette[min(idx, len(palette) - 1)]

        p.fillRect(img.rect(), QColor(0, 0, 0, 0))
        rect_w = int(FALLBACK_W * (0.86 - idx * 0.1))
        rect_h = int(FALLBACK_H * (0.7 - idx * 0.06))
        x0 = (FALLBACK_W - rect_w) // 2
        y0 = int(FALLBACK_H * 0.17 + idx * 26)

        p.setPen(QPen(QColor(245, 245, 250, 220), 3))
        p.setBrush(c)
        p.drawRoundedRect(x0, y0, rect_w, rect_h, 16, 16)

        # Perspective-like guide lines.
        p.setPen(QPen(QColor(255, 255, 255, 95), 1))
        cx = FALLBACK_W / 2
        cy = FALLBACK_H * 0.52
        for t in range(-6, 7):
            p.drawLine(int(cx + t * 70), int(cy), int(cx + t * 120), FALLBACK_H)
        for y in range(int(cy), FALLBACK_H, 40):
            p.drawLine(int(cx - 760), y, int(cx + 760), y)

        font = QFont("Segoe UI", 48)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QPen(QColor(250, 250, 250, 250), 2))
        p.drawText(x0 + 36, y0 + 76, f"{layer.name}  z={layer.z:.1f}")

        if idx == 0:
            p.setPen(QPen(QColor(255, 255, 255, 180), 2))
            p.drawText(50, 80, "DEPTH DEBUG MODE")
        p.end()
        return img
