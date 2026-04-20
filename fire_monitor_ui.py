#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np 
from PySide6.QtCore import QDateTime, QTimer, Qt, QSize, QRectF, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPainterPath
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ============================================================
# 可调参数区：你后面想改排版、比例、字体、模块开关，优先改这里
# ============================================================
UI_CONFIG = {
    "window_size": (800, 450),          # 窗口整体大幅缩小
    "left_ratio": 12,
    "center_ratio": 70,
    "right_ratio": 18,
    "video_aspect": 16 / 9,
    "video_min_width": 600,             # 视频画面缩小
    "video_min_height": 340,            # 视频高度缩小
    "left_list_width": 200,             # 左侧摄像头列表变窄
    "event_card_height": 180,           # 右侧抓拍卡片变小
    "capture_history_limit": 14,
    "show_top_time": True,
    "show_video_overlay": True,
    "show_video_timestamp": True,
    "show_video_meta": True,
}

STATE_COLORS = {
    "SAFE": "#18C37D",
    "SUSPECT": "#F7B733",
    "PREWARNING": "#FF8A26",
    "ALARM": "#FF4D5A",
    "FAILSAFE_ALARM": "#A96DFF",
    "OFFLINE": "#64748B",
}

STATE_TEXT = {
    "SAFE": "安全",
    "SUSPECT": "可疑",
    "PREWARNING": "预警",
    "ALARM": "报警",
    "FAILSAFE_ALARM": "兜底报警",
    "OFFLINE": "离线",
}

CLASS_COLORS = {
    "fire": (26, 128, 255),   # BGR
    "smoke": (64, 198, 255),
}


def fmt_num(v: Optional[float], digits: int = 2) -> str:
    if v is None:
        return "--"
    return f"{v:.{digits}f}"


def state_text(state: str) -> str:
    return STATE_TEXT.get(state, state)


def state_color(state: str) -> str:
    return STATE_COLORS.get(state, "#94A3B8")


def qpixmap_from_bgr(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def fit_pixmap(pixmap: QPixmap, size: QSize) -> QPixmap:
    return pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Detection:
    cls: str
    conf: float
    bbox: Tuple[float, float, float, float]  # normalized
    zone_label: str = ""
    parking_slot: str = ""


@dataclass
class CameraData:
    camera_id: str
    camera_name: str
    zone_name: str
    state: str = "SAFE"
    risk_score: float = 0.0
    online: bool = True
    backend_mode: str = "DEMO"
    fire_conf: float = 0.0
    smoke_conf: float = 0.0
    min_temp: Optional[float] = None
    max_temp: Optional[float] = None
    avg_temp: Optional[float] = None
    smoke_sensor_value: Optional[float] = None
    bms_online: bool = False
    bms_vdrop: Optional[float] = None
    bms_temp: Optional[float] = None
    weights: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    detections: List[Detection] = field(default_factory=list)
    fps: float = 0.0
    updated_ts: float = field(default_factory=time.time)
    parking_slots: Dict[str, str] = field(default_factory=dict)  # slot -> state


@dataclass
class CaptureItem:
    ts: float
    camera_name: str
    zone_name: str
    state: str
    summary: str
    pixmap: QPixmap


# ============================================================
# 模拟数据源 / JSONL 数据源
# ============================================================
class MultiCameraDemoSource:
    def __init__(self):
        self.start = time.time()
        self.phase = 0
        self.last_phase_switch = time.time()
        self.cameras = [
            ("cam-01", "B1北区-01", "B1北区"),
            ("cam-02", "B1北区-02", "B1北区"),
            ("cam-03", "B1中区-01", "B1中区"),
            ("cam-04", "B1中区-02", "B1中区"),
            ("cam-05", "B1南区-01", "B1南区"),
            ("cam-06", "B2充电区-01", "B2充电区"),
        ]

    def _base_slots(self) -> Dict[str, str]:
        slots = {}
        for name in [
            "A1", "A2", "A3", "A4", "A5", "A6",
            "B1", "B2", "B3", "B4", "B5", "B6",
            "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
            "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10",
            "E1", "E2", "E3", "E4", "E5", "E6",
            "F1", "F2", "F3", "F4", "F5", "F6",
        ]:
            slots[name] = "SAFE"
        return slots

    def poll(self) -> List[CameraData]:
        now = time.time()
        t = now - self.start
        if now - self.last_phase_switch > 12:
            self.phase = (self.phase + 1) % 4
            self.last_phase_switch = now

        items: List[CameraData] = []
        for i, (cid, cname, zone) in enumerate(self.cameras):
            offset = i * 0.41
            fire = 0.02 + 0.02 * abs(math.cos(t * 1.2 + offset))
            smoke = 0.03 + 0.03 * abs(math.sin(t * 1.3 + offset))
            max_temp = 31 + 3.5 * abs(math.sin(t * 0.5 + offset))
            gas = 0.08 + 0.02 * abs(math.sin(t * 0.9 + offset))
            fps = 12 + 2.4 * abs(math.sin(t * 1.1 + offset))
            risk = 8 + 4 * abs(math.sin(t * 0.8 + offset))
            state = "SAFE"
            dets: List[Detection] = []
            reasons: List[str] = []
            weights = {"vision": 0.22, "thermal": 0.34, "gas": 0.28, "bms": 0.16}
            slots = self._base_slots()

            if cid == "cam-04":
                if self.phase == 0:
                    state = "SAFE"
                elif self.phase == 1:
                    state = "SUSPECT"
                    smoke = 0.26 + 0.05 * abs(math.sin(t * 1.8))
                    max_temp = 58 + 5 * abs(math.cos(t * 0.7))
                    gas = 0.26 + 0.03 * abs(math.sin(t * 1.2))
                    risk = 34 + 4 * abs(math.sin(t * 0.7))
                    reasons = ["zone_C4_rising", "thermal_hot"]
                    dets = [Detection("smoke", smoke, (0.48, 0.28, 0.74, 0.60), "车位C4附近", "C4")]
                    slots["C4"] = "SUSPECT"
                    slots["C5"] = "SUSPECT"
                elif self.phase == 2:
                    state = "PREWARNING"
                    smoke = 0.63 + 0.05 * abs(math.sin(t * 1.8))
                    max_temp = 92 + 5 * abs(math.cos(t * 0.8))
                    gas = 0.64 + 0.03 * abs(math.cos(t * 1.5))
                    risk = 58 + 5 * abs(math.sin(t * 0.6))
                    reasons = ["vision_smoke", "gas_rising", "thermal_hot"]
                    dets = [Detection("smoke", smoke, (0.42, 0.22, 0.80, 0.66), "车位C4-C5", "C4")]
                    weights = {"vision": 0.28, "thermal": 0.30, "gas": 0.27, "bms": 0.15}
                    slots["C4"] = "PREWARNING"
                    slots["C5"] = "PREWARNING"
                else:
                    state = "ALARM"
                    fire = 0.81 + 0.05 * abs(math.sin(t * 2.4))
                    smoke = 0.74 + 0.05 * abs(math.cos(t * 1.4))
                    max_temp = 132 + 8 * abs(math.sin(t * 0.9))
                    gas = 0.78
                    risk = 87 + 3 * abs(math.sin(t * 0.7))
                    reasons = ["vision_fire", "vision_smoke", "thermal_hot", "slot_C4_located"]
                    dets = [
                        Detection("fire", fire, (0.56, 0.56, 0.67, 0.80), "车位C4", "C4"),
                        Detection("smoke", smoke, (0.38, 0.20, 0.82, 0.74), "车位C3-C5", "C4"),
                    ]
                    weights = {"vision": 0.35, "thermal": 0.31, "gas": 0.22, "bms": 0.12}
                    slots["C4"] = "ALARM"
                    slots["C5"] = "PREWARNING"
                    slots["C3"] = "SUSPECT"
            elif cid in {"cam-03", "cam-05"} and self.phase >= 2:
                state = "SUSPECT"
                smoke = 0.18 + 0.04 * abs(math.sin(t + offset))
                max_temp = 52 + 4 * abs(math.cos(t * 0.8 + offset))
                gas = 0.14
                risk = 28 + 3 * abs(math.sin(t * 0.9 + offset))
                reasons = ["neighbor_zone_link"]
                dets = [Detection("smoke", smoke, (0.34, 0.26, 0.60, 0.56), "邻近车位", "B5")]
                slots["B5"] = "SUSPECT"

            items.append(
                CameraData(
                    camera_id=cid,
                    camera_name=cname,
                    zone_name=zone,
                    state=state,
                    risk_score=risk,
                    online=True,
                    backend_mode="DEMO",
                    fire_conf=fire,
                    smoke_conf=smoke,
                    min_temp=max_temp - 8,
                    max_temp=max_temp,
                    avg_temp=max_temp - 4,
                    smoke_sensor_value=gas,
                    bms_online=False,
                    bms_vdrop=None,
                    bms_temp=None,
                    weights=weights,
                    reasons=reasons,
                    detections=dets,
                    fps=fps,
                    updated_ts=now,
                    parking_slots=slots,
                )
            )
        return items


class JSONLDataSource:
    def __init__(self, path: str):
        self.path = path
        self.offset = 0
        self.current = CameraData("cam-04", "B1中区-02", "B1中区", backend_mode="JSONL")
        self.cameras = [
            ("cam-01", "B1北区-01", "B1北区"),
            ("cam-02", "B1北区-02", "B1北区"),
            ("cam-03", "B1中区-01", "B1中区"),
            ("cam-04", "B1中区-02", "B1中区"),
        ]

    def _read_lines(self) -> List[str]:
        if not self.path or not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self.offset)
            lines = f.readlines()
            self.offset = f.tell()
        return lines

    def _extract(self, obj: dict) -> CameraData:
        sensor = obj.get("sensor_frame") or obj.get("sensor") or obj.get("frame") or {}
        decision = obj.get("decision") or {}
        state = decision.get("state", obj.get("state", "SAFE"))
        fire_conf = float(sensor.get("fire_conf", obj.get("fire_conf", 0.0)) or 0.0)
        smoke_conf = float(sensor.get("smoke_conf", obj.get("smoke_conf", 0.0)) or 0.0)
        max_temp = sensor.get("max_temp", obj.get("max_temp"))
        smoke_sensor_value = sensor.get("smoke_sensor_value", obj.get("smoke_sensor_value"))

        slots = MultiCameraDemoSource()._base_slots()
        hot_slot = "C4"
        if state == "SUSPECT":
            slots[hot_slot] = "SUSPECT"
        elif state == "PREWARNING":
            slots[hot_slot] = "PREWARNING"
        elif state in {"ALARM", "FAILSAFE_ALARM"}:
            slots[hot_slot] = "ALARM"
            slots["C5"] = "PREWARNING"

        dets: List[Detection] = []
        for d in obj.get("detections", []) or []:
            bbox = d.get("bbox", [0.45, 0.25, 0.75, 0.65])
            if len(bbox) == 4:
                dets.append(
                    Detection(
                        str(d.get("cls", "smoke")),
                        float(d.get("conf", 0.5)),
                        tuple(map(float, bbox)),
                        str(d.get("zone_label", "热点区域")),
                        str(d.get("parking_slot", d.get("camera_slot", hot_slot))),
                    )
                )
        if not dets:
            if fire_conf >= 0.55:
                dets.append(Detection("fire", fire_conf, (0.56, 0.56, 0.66, 0.80), "车位C4", hot_slot))
            if smoke_conf >= 0.35:
                dets.append(Detection("smoke", smoke_conf, (0.42, 0.22, 0.80, 0.70), "车位C4附近", hot_slot))

        return CameraData(
            camera_id=str(obj.get("camera_id", sensor.get("camera_id", "cam-04"))),
            camera_name=str(obj.get("camera_name", sensor.get("camera_name", "B1中区-02"))),
            zone_name=str(obj.get("zone_name", sensor.get("zone_name", "B1中区"))),
            state=state,
            risk_score=float(decision.get("risk_score", obj.get("risk_score", 0.0)) or 0.0),
            online=True,
            backend_mode="JSONL",
            fire_conf=fire_conf,
            smoke_conf=smoke_conf,
            min_temp=sensor.get("min_temp", obj.get("min_temp")),
            max_temp=max_temp,
            avg_temp=sensor.get("avg_temp", obj.get("avg_temp")),
            smoke_sensor_value=smoke_sensor_value,
            bms_online=bool(sensor.get("bms_online", obj.get("bms_online", False))),
            bms_vdrop=sensor.get("bms_voltage_drop_score", obj.get("bms_voltage_drop_score")),
            bms_temp=sensor.get("bms_temp_score", obj.get("bms_temp_score")),
            weights=decision.get("weights", obj.get("weights", {})) or {},
            reasons=decision.get("reasons", obj.get("reasons", [])) or [],
            detections=dets,
            fps=float(sensor.get("fps", obj.get("fps", 0.0)) or 0.0),
            updated_ts=time.time(),
            parking_slots=slots,
        )

    def poll(self) -> List[CameraData]:
        for line in self._read_lines():
            line = line.strip()
            if not line:
                continue
            try:
                self.current = self._extract(json.loads(line))
            except Exception:
                continue

        now = time.time()
        items = []
        for cid, cname, zone in self.cameras:
            if cid == self.current.camera_id:
                items.append(self.current)
            else:
                slots = MultiCameraDemoSource()._base_slots()
                items.append(
                    CameraData(
                        camera_id=cid,
                        camera_name=cname,
                        zone_name=zone,
                        state="SAFE",
                        risk_score=8.0,
                        online=True,
                        backend_mode="JSONL",
                        fire_conf=0.01,
                        smoke_conf=0.03,
                        min_temp=28.0,
                        max_temp=33.0,
                        avg_temp=30.0,
                        smoke_sensor_value=0.05,
                        bms_online=False,
                        weights={"vision": 0.20, "thermal": 0.35, "gas": 0.30, "bms": 0.15},
                        reasons=[],
                        detections=[],
                        fps=10.0,
                        updated_ts=now,
                        parking_slots=slots,
                    )
                )
        return items


# ============================================================
# UI 组件
# ============================================================
class GlowCard(QFrame):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("GlowCard")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(14, 12, 14, 12)
        self.layout.setSpacing(10)
        if title:
            self.title = QLabel(title)
            self.title.setObjectName("CardTitle")
            self.layout.addWidget(self.title)


class HeaderBanner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.title_label = QLabel("FireGuard Pro · 智能火灾预警平台")
        self.title_label.setObjectName("BannerMain")
        self.sub_label = QLabel("多摄像头车库态势监测 / 视频检测 / 位置联动 / 抓拍留痕")
        self.sub_label.setObjectName("BannerSub")
        self.time_label = QLabel("")
        self.time_label.setObjectName("BannerTime")

        left = QVBoxLayout()
        left.setSpacing(2)
        left.addWidget(self.title_label)
        left.addWidget(self.sub_label)

        root = QHBoxLayout(self)
        root.setContentsMargins(18, 8, 18, 8)
        root.addLayout(left, 1)
        root.addWidget(self.time_label, 0, Qt.AlignRight | Qt.AlignVCenter)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(rect, QColor("#07122A"))
        painter.setPen(QPen(QColor("#103A74"), 1.2))
        painter.drawRoundedRect(rect, 14, 14)
        painter.fillRect(18, 10, 140, 3, QColor("#12C6FF"))
        painter.fillRect(rect.width() - 160, 10, 140, 3, QColor("#12C6FF"))
        painter.fillRect(340, rect.height() - 14, rect.width() - 680, 2, QColor("#0B4C88"))
        super().paintEvent(event)

    def set_time_text(self, text: str):
        self.time_label.setText(text)


class CameraListWidget(QListWidget):
    cameraSelected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemSelectionChanged.connect(self._emit_selected)

    def _emit_selected(self):
        item = self.currentItem()
        if item:
            self.cameraSelected.emit(item.data(Qt.UserRole))

    def load_items(self, cameras: List[CameraData], selected_id: Optional[str]):
        self.clear()
        for cam in cameras:
            item = QListWidgetItem()
            item.setText(f"{cam.camera_name}\n{cam.zone_name} · {state_text(cam.state)}")
            item.setData(Qt.UserRole, cam.camera_id)
            item.setForeground(QColor("#E2E8F0"))
            item.setSizeHint(QSize(220, 54))
            self.addItem(item)
            if cam.camera_id == selected_id:
                self.setCurrentItem(item)


class AspectRatioVideoLabel(QLabel):
    def __init__(self, aspect: float, parent=None):
        super().__init__(parent)
        self.aspect = aspect
        self._source: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#050C18;border:1px solid #163A63;border-radius:12px;")

    def set_source_pixmap(self, pixmap: QPixmap):
        self._source = pixmap
        self._refresh()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self):
        if self._source is None or self.size().width() <= 0 or self.size().height() <= 0:
            return
        self.setPixmap(self._source.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class StateBadge(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(34)

    def set_state(self, state: str):
        self.setText(state_text(state).upper())
        c = state_color(state)
        self.setStyleSheet(
            f"background:{c}; color:white; font-size:18px; font-weight:800; border-radius:16px; padding:4px 12px;"
        )


class HalfGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.state = "SAFE"
        self.setMinimumHeight(130)

    def set_value(self, value: float, state: str):
        self.value = max(0.0, min(100.0, value))
        self.state = state
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(20, 10, -20, -10)
        arc_rect = QRectF(r.left() + 20, r.top() + 24, r.width() - 40, min(220, r.height() + 80))

        painter.setPen(QPen(QColor("#173357"), 16))
        painter.drawArc(arc_rect, 180 * 16, -180 * 16)

        pen = QPen(QColor(state_color(self.state)), 16)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        span = int(-180 * 16 * (self.value / 100.0))
        painter.drawArc(arc_rect, 180 * 16, span)

        painter.setPen(QColor("#EAF4FF"))
        painter.setFont(QFont("Microsoft YaHei", 30, QFont.Bold))
        # 风险数字 → 下移到原来 50 的位置
        painter.drawText(r.adjusted(0, 58, 0, -10), Qt.AlignHCenter | Qt.AlignTop, f"{self.value:.0f}")

        painter.setFont(QFont("Microsoft YaHei", 11, QFont.Medium))
        painter.setPen(QColor("#8FB4DB"))
        # 综合风险 → 再往下一点
        painter.drawText(r.adjusted(0, 0, 0, -10), Qt.AlignHCenter | Qt.AlignTop, "综合风险")

        ticks_y = int(r.bottom() - 8)
        painter.setFont(QFont("Segoe UI", 9, QFont.Medium))
        painter.drawText(r.left() + 6, ticks_y, "0")
        # 50 → 上移到原来数字的位置
        painter.drawText(r.center().x() - 8, int(r.top() + 30), "50")
        painter.drawText(r.right() - 20, ticks_y, "100")


class KpiTile(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("KpiTile")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        self.title = QLabel(title)
        self.title.setObjectName("KpiTitle")
        self.value = QLabel("--")
        self.value.setObjectName("KpiValue")
        self.sub = QLabel("")
        self.sub.setObjectName("KpiSub")
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.sub)

    def set_value(self, value: str, color: str, sub: str = ""):
        self.value.setText(value)
        self.value.setStyleSheet(f"color:{color};")
        self.sub.setText(sub)


class CompactRow(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("CompactRow")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(8)
        self.dot = QLabel("●")
        self.dot.setFixedWidth(14)
        self.key = QLabel(label)
        self.key.setObjectName("MiniKey")
        self.val = QLabel("--")
        self.val.setObjectName("MiniVal")
        lay.addWidget(self.dot)
        lay.addWidget(self.key, 1)
        lay.addWidget(self.val, 0, Qt.AlignRight)

    def set_values(self, value: str, color_hex: str):
        self.val.setText(value)
        self.dot.setStyleSheet(f"color:{color_hex};")
        self.val.setStyleSheet(f"color:{color_hex};")


class WeightBar(QFrame):
    def __init__(self, key_label: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.key = QLabel(key_label)
        self.key.setObjectName("MiniKey")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(9)
        self.val = QLabel("0.00")
        self.val.setObjectName("MiniVal")
        lay.addWidget(self.key, 0)
        lay.addWidget(self.bar, 1)
        lay.addWidget(self.val, 0)

    def set_value(self, num: float, accent: str):
        self.bar.setValue(int(max(0.0, min(1.0, num)) * 100))
        self.bar.setStyleSheet(
            f"QProgressBar{{background:#0B1E38;border:1px solid #183A63;border-radius:4px;}}"
            f"QProgressBar::chunk{{background:{accent};border-radius:4px;}}"
        )
        self.val.setText(f"{num:.2f}")


class TrendLineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: Deque[float] = deque(maxlen=40)
        self.setMinimumHeight(60)

    def push(self, value: float):
        self.values.append(value)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#08162F"))
        r = self.rect().adjusted(8, 8, -8, -8)
        painter.setPen(QPen(QColor("#14335A"), 1))
        for i in range(5):
            y = r.top() + i * (r.height() / 4)
            painter.drawLine(r.left(), int(y), r.right(), int(y))
        if len(self.values) < 2:
            return
        vals = list(self.values)
        lo, hi = 0.0, max(100.0, max(vals))
        pts = []
        for idx, val in enumerate(vals):
            x = r.left() + idx * (r.width() / max(1, len(vals) - 1))
            y = r.bottom() - (val - lo) / (hi - lo + 1e-6) * r.height()
            pts.append((x, y))
        painter.setPen(QPen(QColor("#16D5FF"), 2.0))
        for i in range(len(pts) - 1):
            painter.drawLine(int(pts[i][0]), int(pts[i][1]), int(pts[i + 1][0]), int(pts[i + 1][1]))


class GaragePlanWidget(QWidget):
    """简化版停车场平面图：保留你后面自己改位置和比例的空间。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_data: Optional[CameraData] = None
        self.blink = 0
        self.setMinimumHeight(220)
        # 车位布局：A/B整块右移 + C10/D10右端对齐充电区左侧，间距完全不变
        self.slot_layout: Dict[str, Tuple[float, float, float, float]] = {
            # 顶部两排 A/B：整体右移，间距保持原样
            **{f"A{i}": (0.10 + (i - 1) * 0.09, 0.08, 0.07, 0.11) for i in range(1, 7)},
            **{f"B{i}": (0.10 + (i - 1) * 0.09, 0.22, 0.07, 0.11) for i in range(1, 7)},

            # 中间长排 C/D：微调右端，C10/D10 对齐充电区左侧，内部间距不变
            **{f"C{i}": (0.10 + (i - 1) * 0.064, 0.48, 0.055, 0.10) for i in range(1, 11)},
            **{f"D{i}": (0.10 + (i - 1) * 0.064, 0.62, 0.055, 0.10) for i in range(1, 11)},

            # 右侧竖排 E（不动）
            **{f"E{i}": (0.86, 0.12 + (i - 1) * 0.10, 0.07, 0.08) for i in range(1, 7)},

            # 底部短排 F（不动）
            **{f"F{i}": (0.20 + (i - 1) * 0.07, 0.80, 0.055, 0.08) for i in range(1, 7)},
        }

    def set_camera_data(self, cam: CameraData):
        self.camera_data = cam
        self.blink = (self.blink + 1) % 2
        self.update()

    def _slot_state(self, name: str) -> str:
        if not self.camera_data:
            return "SAFE"
        return self.camera_data.parking_slots.get(name, "SAFE")

    def _slot_rect(self, box: Tuple[float, float, float, float]) -> QRectF:
        r = self.rect().adjusted(14, 16, -14, -16)
        x, y, w, h = box
        return QRectF(r.left() + r.width() * x, r.top() + r.height() * y, r.width() * w, r.height() * h)

    def _slot_fill(self, state: str) -> QColor:
        mapping = {
            "SAFE": QColor("#0E2A45"),
            "SUSPECT": QColor("#6C4C10"),
            "PREWARNING": QColor("#82400B"),
            "ALARM": QColor("#5E1622"),
            "FAILSAFE_ALARM": QColor("#4D217A"),
        }
        return mapping.get(state, QColor("#0E2A45"))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#08152C"))
        r = self.rect().adjusted(14, 14, -14, -14)

        # 主区域边框与道路
        painter.setPen(QPen(QColor("#17406E"), 1.3))
        painter.drawRoundedRect(r, 12, 12)
        painter.fillRect(int(r.left() + r.width() * 0.09), int(r.top() + r.height() * 0.36), int(r.width() * 0.72), int(r.height() * 0.08), QColor("#0A223B"))
        painter.fillRect(int(r.left() + r.width() * 0.36), int(r.top() + r.height() * 0.06), int(r.width() * 0.07), int(r.height() * 0.72), QColor("#0A223B"))
        painter.fillRect(int(r.left() + r.width() * 0.79), int(r.top() + r.height() * 0.08), int(r.width() * 0.05), int(r.height() * 0.72), QColor("#0A223B"))

        painter.setPen(QColor("#7FA6CC"))
        painter.setFont(QFont("Microsoft YaHei", 9, QFont.Medium))
        painter.drawText(int(r.left() + 12), int(r.top() + 22), "")
        painter.drawText(int(r.left() + r.width() * 0.39), int(r.top() + r.height() * 0.42), "主通道")
        painter.drawText(int(r.left() + r.width() * 0.78), int(r.top() + r.height() * 0.42), "充电区")

        # 车位
        for slot, box in self.slot_layout.items():
            rr = self._slot_rect(box)
            st = self._slot_state(slot)
            painter.fillRect(rr, self._slot_fill(st))
            painter.setPen(QPen(QColor(state_color(st)), 1.1))
            painter.drawRect(rr)
            painter.setPen(QColor("#DCE6F5"))
            painter.setFont(QFont("Segoe UI", 7, QFont.Medium))
            painter.drawText(rr, Qt.AlignCenter, slot)

        # 实时点位 / 热点红点
        if self.camera_data and self.camera_data.detections:
            for det in self.camera_data.detections:
                slot = det.parking_slot or "C4"
                if slot in self.slot_layout:
                    rr = self._slot_rect(self.slot_layout[slot])
                    center = rr.center()
                    radius = 8 if self.blink == 0 else 11
                    painter.setBrush(QColor("#FF364D"))
                    painter.setPen(QPen(QColor("#FFD5DB"), 1.2))
                    painter.drawEllipse(center, radius, radius)
                    painter.setPen(QColor("#FFFFFF"))
                    painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
                    painter.drawText(int(center.x() + 10), int(center.y() - 6), det.zone_label or slot)

        # 图例
        legend = [("安全", "#18C37D"), ("可疑", "#F7B733"), ("预警", "#FF8A26"), ("报警", "#FF4D5A")]
        x = int(r.left() + 12)
        y = int(r.bottom() - 12)
        for text, color in legend:
            painter.setBrush(QColor(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x, y - 8, 8, 8)
            painter.setPen(QColor("#DCE6F5"))
            painter.drawText(x + 14, y, text)
            x += 62


class CaptureCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CaptureCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        top = QHBoxLayout()
        self.state = QLabel("预警")
        self.time = QLabel("--")
        self.time.setObjectName("MiniVal")
        top.addWidget(self.state)
        top.addStretch(1)
        top.addWidget(self.time)
        self.thumb = QLabel()
        self.thumb.setFixedHeight(UI_CONFIG["event_card_height"])
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setStyleSheet("background:#07122A;border-radius:8px;border:1px solid #173962;")
        self.meta = QLabel("--")
        self.meta.setWordWrap(True)
        self.meta.setObjectName("CaptureMeta")
        layout.addLayout(top)
        layout.addWidget(self.thumb)
        layout.addWidget(self.meta)

    def set_item(self, item: CaptureItem):
        self.state.setText(state_text(item.state))
        self.state.setStyleSheet(f"background:{state_color(item.state)}; color:white; border-radius:10px; padding:2px 8px; font-weight:700;")
        self.time.setText(time.strftime("%m-%d %H:%M:%S", time.localtime(item.ts)))
        self.meta.setText(f"{item.camera_name} · {item.zone_name}\n{item.summary}")
        pm = item.pixmap.scaled(420, UI_CONFIG["event_card_height"], Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.thumb.setPixmap(pm)


# ============================================================
# 主窗口
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self, source_mode: str, source_path: Optional[str], camera_source: int):
        super().__init__()
        self.setWindowTitle("FireGuard Pro · 智能火灾预警平台")
        self.resize(*UI_CONFIG["window_size"])
        self.source_mode = source_mode
        self.source_path = source_path
        self.camera_source = camera_source

        self.data_source = MultiCameraDemoSource() if source_mode == "demo" else JSONLDataSource(source_path)
        self.cap = cv2.VideoCapture(camera_source)
        self.selected_camera_id: Optional[str] = None
        self.last_frame = self._make_placeholder_frame("Initializing")
        self.capture_history: Deque[CaptureItem] = deque(maxlen=UI_CONFIG["capture_history_limit"])
        self.last_capture_state: Dict[str, str] = {}

        self._build_ui()
        self._apply_style()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(140)

    # ------------------------ UI ------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(12)

        self.header = HeaderBanner()
        root.addWidget(self.header)

        body = QHBoxLayout()
        body.setSpacing(14)
        root.addLayout(body, 1)

        # 左列（缩窄）
        left_col = QVBoxLayout()
        left_col.setSpacing(12)
        body.addLayout(left_col, UI_CONFIG["left_ratio"])

        cam_card = GlowCard("区域 / 摄像头")
        self.camera_list = CameraListWidget()
        self.camera_list.setObjectName("CameraList")
        self.camera_list.setMinimumWidth(UI_CONFIG["left_list_width"])
        self.camera_list.cameraSelected.connect(self._on_camera_selected)
        cam_card.layout.addWidget(self.camera_list)
        left_col.addWidget(cam_card, 6)

        overview_card = GlowCard("全局概览")
        overview_grid = QGridLayout()
        overview_grid.setHorizontalSpacing(8)
        overview_grid.setVerticalSpacing(8)
        self.total_online_tile = KpiTile("在线摄像头")
        self.total_alert_tile = KpiTile("活跃告警")
        self.peak_zone_tile = KpiTile("重点区域")
        self.system_mode_tile = KpiTile("后端模式")
        overview_grid.addWidget(self.total_online_tile, 0, 0)
        overview_grid.addWidget(self.total_alert_tile, 0, 1)
        overview_grid.addWidget(self.peak_zone_tile, 1, 0)
        overview_grid.addWidget(self.system_mode_tile, 1, 1)
        overview_card.layout.addLayout(overview_grid)
        left_col.addWidget(overview_card, 4)

        # 中列
        center_col = QVBoxLayout()
        center_col.setSpacing(12)
        body.addLayout(center_col, UI_CONFIG["center_ratio"])

        video_card = GlowCard("主监控画面")
        video_top = QHBoxLayout()
        self.video_meta = QLabel("--")
        self.video_meta.setObjectName("PanelMeta")
        self.video_hint = QLabel("实时框选 / 摄像头全覆盖 / 多画面联动")
        self.video_hint.setObjectName("PanelMeta")
        video_top.addWidget(self.video_meta)
        video_top.addStretch(1)
        video_top.addWidget(self.video_hint)
        video_card.layout.addLayout(video_top)

        # ===================== 新增：左右小摄像头布局 =====================
        # 主视频外层水平布局（左3 + 主视频 + 右3）
        video_main_layout = QHBoxLayout()
        video_main_layout.setSpacing(8)
        video_main_layout.setContentsMargins(0,0,0,0)

        # 左侧3个小监控占位（cam01-03）
        self.left_camera_labels = []
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)
        for i in range(3):
            label = QLabel(f"摄像头 {i+1}")
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(120, 80)  # 小画面尺寸
            label.setStyleSheet("background:#050C18; border:1px solid #163A63; border-radius:6px; color:#8FB4DB;")
            self.left_camera_labels.append(label)
            left_layout.addWidget(label)

        # 右侧3个小监控占位（cam04-06）
        self.right_camera_labels = []
        right_layout = QVBoxLayout()
        right_layout.setSpacing(6)
        for i in range(3):
            label = QLabel(f"摄像头 {i+4}")
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(120, 80)  # 小画面尺寸
            label.setStyleSheet("background:#050C18; border:1px solid #163A63; border-radius:6px; color:#8FB4DB;")
            self.right_camera_labels.append(label)
            right_layout.addWidget(label)

        # 主视频
        self.video_label = AspectRatioVideoLabel(UI_CONFIG["video_aspect"])
        self.video_label.setMinimumSize(UI_CONFIG["video_min_width"], UI_CONFIG["video_min_height"])

        # 组装：左列 + 主视频 + 右列
        video_main_layout.addLayout(left_layout)
        video_main_layout.addWidget(self.video_label, 1)  # 主视频占满剩余空间
        video_main_layout.addLayout(right_layout)
        # =================================================================

        video_card.layout.addLayout(video_main_layout, 1)
        center_col.addWidget(video_card, 8)

        lower = QHBoxLayout()
        lower.setSpacing(12)
        center_col.addLayout(lower, 5)

        map_card = GlowCard("车位风险定位")
        top_bar = QHBoxLayout()
        self.map_info = QLabel("")
        self.map_info.setObjectName("PanelMeta")
        self.map_zone = QLabel("--")
        self.map_zone.setObjectName("PanelMeta")
        top_bar.addWidget(self.map_info)
        top_bar.addStretch(1)
        top_bar.addWidget(self.map_zone)
        map_card.layout.addLayout(top_bar)
        self.garage_map = GaragePlanWidget()
        map_card.layout.addWidget(self.garage_map, 1)
        lower.addWidget(map_card, 7)

        metric_card = GlowCard("关键指标")
        metric_grid = QGridLayout()
        metric_grid.setHorizontalSpacing(8)
        metric_grid.setVerticalSpacing(8)
        self.fire_tile = KpiTile("火焰置信度")
        self.smoke_tile = KpiTile("烟雾置信度")
        self.temp_tile = KpiTile("最高温度")
        self.gas_tile = KpiTile("烟雾传感器")
        metric_grid.addWidget(self.fire_tile, 0, 0)
        metric_grid.addWidget(self.smoke_tile, 0, 1)
        metric_grid.addWidget(self.temp_tile, 1, 0)
        metric_grid.addWidget(self.gas_tile, 1, 1)
        metric_card.layout.addLayout(metric_grid)
        self.trend = TrendLineWidget()
        metric_card.layout.addWidget(self.trend)
        lower.addWidget(metric_card, 4)

        # 右列
        right_col = QVBoxLayout()
        right_col.setSpacing(12)
        body.addLayout(right_col, UI_CONFIG["right_ratio"])

        summary_card = GlowCard("当前系统状态")
        self.state_badge = StateBadge()
        summary_card.layout.addWidget(self.state_badge)
        self.gauge = HalfGauge()
        summary_card.layout.addWidget(self.gauge)
        self.summary_text = QLabel("--")
        self.summary_text.setWordWrap(True)
        self.summary_text.setObjectName("PanelMeta")
        summary_card.layout.addWidget(self.summary_text)
        right_col.addWidget(summary_card, 3)

        sensor_card = GlowCard("设备与传感器 / 融合权重")
        self.row_backend = CompactRow("后端")
        self.row_camera = CompactRow("摄像头")
        self.row_thermal = CompactRow("热成像")
        self.row_gas = CompactRow("烟雾/气体")
        self.row_bms = CompactRow("BMS")
        for row in [self.row_backend, self.row_camera, self.row_thermal, self.row_gas, self.row_bms]:
            sensor_card.layout.addWidget(row)
        self.weight_rows = {
            "vision": WeightBar("视觉"),
            "thermal": WeightBar("热成像"),
            "gas": WeightBar("烟雾/气体"),
            "bms": WeightBar("BMS"),
        }
        for row in self.weight_rows.values():
            sensor_card.layout.addWidget(row)
        right_col.addWidget(sensor_card, 4)

        capture_wrap = GlowCard("预警抓拍")
        self.capture_area = QScrollArea()
        self.capture_area.setWidgetResizable(True)
        self.capture_area.setObjectName("CaptureArea")
        self.capture_holder = QWidget()
        self.capture_layout = QVBoxLayout(self.capture_holder)
        self.capture_layout.setContentsMargins(2, 2, 2, 2)
        self.capture_layout.setSpacing(10)
        self.capture_layout.addStretch(1)
        self.capture_area.setWidget(self.capture_holder)
        capture_wrap.layout.addWidget(self.capture_area)
        right_col.addWidget(capture_wrap, 6)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #04101F;
                color: #E6F0FF;
                font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            #GlowCard {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #07172F, stop:1 #091A34);
                border: 1px solid #173962;
                border-radius: 14px;
            }
            #CardTitle {
                color: #71C7FF;
                font-size: 18px;
                font-weight: 800;
                padding-bottom: 2px;
                border-bottom: 1px solid #12345A;
            }
            #BannerMain {
                color: #EAF4FF;
                font-size: 30px;
                font-weight: 900;
            }
            #BannerSub {
                color: #7FA6CC;
                font-size: 13px;
            }
            #BannerTime {
                color: #B8D7FF;
                font-size: 22px;
                font-weight: 700;
            }
            #CameraList {
                background: transparent;
                border: none;
                outline: none;
            }
            #CameraList::item {
                background: #081A34;
                border: 1px solid #17365E;
                border-radius: 10px;
                padding: 8px;
                margin: 3px 0;
            }
            #CameraList::item:selected {
                background: #0B2A4E;
                border: 1px solid #2CB8FF;
            }
            #KpiTile {
                background: #07152B;
                border: 1px solid #173962;
                border-radius: 12px;
            }
            #KpiTitle {
                color: #8FB4DB;
                font-size: 12px;
                font-weight: 600;
            }
            #KpiValue {
                color: #16D5FF;
                font-size: 26px;
                font-weight: 900;
            }
            #KpiSub {
                color: #6E8AAA;
                font-size: 12px;
            }
            #PanelMeta {
                color: #8AA7C9;
                font-size: 12px;
            }
            #CompactRow {
                background: #08162D;
                border: 1px solid #12335A;
                border-radius: 8px;
            }
            #MiniKey {
                color: #9AB7D8;
                font-size: 13px;
            }
            #MiniVal {
                color: #EAF4FF;
                font-size: 13px;
                font-weight: 700;
            }
            #CaptureCard {
                background: #07152B;
                border: 1px solid #173962;
                border-radius: 12px;
            }
            #CaptureMeta {
                color: #C7D9EE;
                font-size: 12px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #07122A;
                width: 10px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #235D92;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            """
        )

    # ------------------------ helpers ------------------------
    def _make_placeholder_frame(self, text: str) -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (6, 16, 28)
        cv2.putText(frame, text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100, 180, 240), 4, cv2.LINE_AA)
        cv2.putText(frame, "No live frame", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (130, 155, 190), 2, cv2.LINE_AA)
        return frame

    def _read_frame(self) -> np.ndarray:
        if self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.last_frame = frame
                return frame
        return self.last_frame.copy()

    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        s = hex_color.lstrip("#")
        return (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))

    def _fire_color(self, v: float) -> str:
        return "#FF4D5A" if v >= 0.65 else "#F7B733" if v >= 0.25 else "#18C37D"

    def _smoke_color(self, v: float) -> str:
        return "#FF4D5A" if v >= 0.55 else "#F7B733" if v >= 0.30 else "#18C37D"

    def _temp_color(self, v: Optional[float]) -> str:
        if v is None:
            return "#94A3B8"
        return "#FF4D5A" if v >= 100 else "#F7B733" if v >= 60 else "#18C37D"

    def _gas_color(self, v: Optional[float]) -> str:
        if v is None:
            return "#94A3B8"
        return "#FF4D5A" if v >= 0.50 else "#F7B733" if v >= 0.20 else "#18C37D"

    def _bms_color(self, cam: CameraData) -> str:
        if not cam.bms_online:
            return "#FF4D5A"
        hi = max(cam.bms_vdrop or 0.0, cam.bms_temp or 0.0)
        return "#FF4D5A" if hi >= 0.70 else "#F7B733" if hi >= 0.40 else "#18C37D"

    def _draw_video_overlay(self, frame: np.ndarray, cam: CameraData) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        if not UI_CONFIG["show_video_overlay"]:
            return out

        # 顶部简化状态条：只放必要信息，避免乱码和过度拥挤
        cv2.rectangle(out, (0, 0), (w, 62), (8, 16, 30), -1)
        st_color = self._hex_to_bgr(state_color(cam.state))
        cv2.putText(out, f"STATE {cam.state}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.86, st_color, 2, cv2.LINE_AA)
        cv2.putText(out, f"RISK {cam.risk_score:.1f}", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.72, st_color, 2, cv2.LINE_AA)
        
        if UI_CONFIG["show_video_timestamp"]:
            cv2.putText(out, QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss").replace('-', '/'), (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (220, 230, 245), 2, cv2.LINE_AA)

        # 检测框 + 定位标签
        for det in cam.detections:
            x1 = int(det.bbox[0] * w)
            y1 = int(det.bbox[1] * h)
            x2 = int(det.bbox[2] * w)
            y2 = int(det.bbox[3] * h)
            clr = CLASS_COLORS.get(det.cls, (0, 255, 255))
            cv2.rectangle(out, (x1, y1), (x2, y2), clr, 3)
            label = f"{det.cls.upper()} {det.conf:.2f} {det.zone_label}"
            tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)[0][0] + 14
            y_top = max(0, y1 - 30)
            cv2.rectangle(out, (x1, y_top), (x1 + tw, y_top + 28), clr, -1)
            cv2.putText(out, label, (x1 + 7, y_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

        # 底部只保留一行关键信息
        if UI_CONFIG["show_video_meta"]:
            cv2.rectangle(out, (0, h - 40), (w, h), (8, 16, 30), -1)
            meta = f"FIRE {cam.fire_conf:.2f}   SMOKE {cam.smoke_conf:.2f}   TEMP {fmt_num(cam.max_temp,1)}C   GAS {fmt_num(cam.smoke_sensor_value,2)}"
            cv2.putText(out, meta, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (225, 236, 255), 2, cv2.LINE_AA)

        return out

    def _on_camera_selected(self, camera_id: str):
        self.selected_camera_id = camera_id

    def _pick_selected_camera(self, cameras: List[CameraData]) -> CameraData:
        if not cameras:
            return CameraData("none", "无信号", "未知", state="OFFLINE", online=False)
        if self.selected_camera_id is None:
            cam = max(cameras, key=lambda c: c.risk_score)
            self.selected_camera_id = cam.camera_id
            return cam
        for cam in cameras:
            if cam.camera_id == self.selected_camera_id:
                return cam
        return cameras[0]

    def _snapshot_if_needed(self, cam: CameraData, display_pixmap: QPixmap):
        record_states = {"PREWARNING", "ALARM", "FAILSAFE_ALARM"}
        prev = self.last_capture_state.get(cam.camera_id)
        if cam.state in record_states and prev != cam.state:
            summary = ", ".join(cam.reasons[:3]) if cam.reasons else "状态升级"
            self.capture_history.appendleft(
                CaptureItem(time.time(), cam.camera_name, cam.zone_name, cam.state, summary, display_pixmap)
            )
            self._refresh_capture_list()
        self.last_capture_state[cam.camera_id] = cam.state

    def _refresh_capture_list(self):
        while self.capture_layout.count() > 1:
            item = self.capture_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        for cap in list(self.capture_history):
            card = CaptureCard()
            card.set_item(cap)
            self.capture_layout.insertWidget(self.capture_layout.count() - 1, card)

    def _update_overview(self, cams: List[CameraData]):
        online_n = sum(1 for c in cams if c.online)
        alert_n = sum(1 for c in cams if c.state in {"PREWARNING", "ALARM", "FAILSAFE_ALARM"})
        peak = max(cams, key=lambda c: c.risk_score) if cams else None
        self.total_online_tile.set_value(str(online_n), "#16D5FF", "在线 / 总数")
        self.total_alert_tile.set_value(str(alert_n), "#FF8A26" if alert_n else "#18C37D", "预警及以上")
        self.peak_zone_tile.set_value(peak.zone_name if peak else "--", state_color(peak.state) if peak else "#94A3B8", peak.camera_name if peak else "")
        self.system_mode_tile.set_value(cams[0].backend_mode if cams else "--", "#B9D8FF", "数据模式")

    def _update_center(self, cam: CameraData):
        raw = self._read_frame()
        overlay = self._draw_video_overlay(raw, cam)
        pixmap = qpixmap_from_bgr(overlay)
        self.video_label.set_source_pixmap(pixmap)
        self.video_meta.setText(f"{cam.camera_name} · {cam.zone_name} · 风险 {cam.risk_score:.1f} · 状态 {state_text(cam.state)}")
        self.video_hint.setText("")
        self.map_zone.setText(f"当前覆盖：{cam.zone_name}")
        self.garage_map.set_camera_data(cam)
        self.trend.push(cam.risk_score)
        self._snapshot_if_needed(cam, pixmap)

        self.fire_tile.set_value(f"{cam.fire_conf:.2f}", self._fire_color(cam.fire_conf), "视觉火焰检测")
        self.smoke_tile.set_value(f"{cam.smoke_conf:.2f}", self._smoke_color(cam.smoke_conf), "视觉烟雾检测")
        self.temp_tile.set_value(f"{fmt_num(cam.max_temp,1)} °C", self._temp_color(cam.max_temp), f"平均 {fmt_num(cam.avg_temp,1)} °C")
        self.gas_tile.set_value(f"{fmt_num(cam.smoke_sensor_value,2)}", self._gas_color(cam.smoke_sensor_value), "烟雾 / 气体输入")

    def _update_right(self, cam: CameraData):
        self.state_badge.set_state(cam.state)
        self.gauge.set_value(cam.risk_score, cam.state)
        location_text = cam.detections[0].zone_label if cam.detections else "当前未定位到明确热点"
        self.summary_text.setText(
            f"当前摄像头：{cam.camera_name}\n"
            f"覆盖区域：{cam.zone_name}\n"
            f"热点位置：{location_text}"
        )

        self.row_backend.set_values(cam.backend_mode, "#18C37D")
        self.row_camera.set_values(f"{cam.camera_name} · FPS {cam.fps:.1f}", "#18C37D" if cam.online else "#FF4D5A")
        self.row_thermal.set_values(f"max {fmt_num(cam.max_temp,1)}°C / avg {fmt_num(cam.avg_temp,1)}°C", self._temp_color(cam.max_temp))
        self.row_gas.set_values(f"{fmt_num(cam.smoke_sensor_value,2)}", self._gas_color(cam.smoke_sensor_value))
        self.row_bms.set_values(f"online={cam.bms_online}  vdrop={fmt_num(cam.bms_vdrop,2)}  temp={fmt_num(cam.bms_temp,2)}", self._bms_color(cam))
        accent = state_color(cam.state)
        for key, widget in self.weight_rows.items():
            widget.set_value(float(cam.weights.get(key, 0.0) or 0.0), accent)

    def _tick(self):
        if UI_CONFIG["show_top_time"]:
            self.header.set_time_text(QDateTime.currentDateTime().toString("yyyy-MM-dd  HH:mm:ss"))
        else:
            self.header.set_time_text("")
        cams = self.data_source.poll()
        self.camera_list.load_items(cams, self.selected_camera_id)
        self._update_overview(cams)
        cam = self._pick_selected_camera(cams)
        self._update_center(cam)
        self._update_right(cam)

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)


# ============================================================
# main
# ============================================================
def build_arg_parser():
    p = argparse.ArgumentParser(description="FireGuard Pro UI v4")
    p.add_argument("--mode", choices=["demo", "jsonl"], default="demo")
    p.add_argument("--jsonl", default="", help="events.jsonl path when mode=jsonl")
    p.add_argument("--source", type=int, default=0, help="camera source index")
    return p


def main():
    args = build_arg_parser().parse_args()
    app = QApplication(sys.argv)
    w = MainWindow(source_mode=args.mode, source_path=args.jsonl or None, camera_source=args.source)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
