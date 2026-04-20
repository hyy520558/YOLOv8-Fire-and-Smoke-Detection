#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QDateTime, QTimer, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow

from fire_monitor_ui import (
    UI_CONFIG,
    CLASS_COLORS,
    CaptureItem,
    CameraData,
    Detection,
    MainWindow as BaseMainWindow,
    MultiCameraDemoSource,
    JSONLDataSource,
    qpixmap_from_bgr,
    fmt_num,
    state_text,
    state_color,
)

# 直接复用你之前离线演示里已经手工调过的主画面框选区域
# 归一化坐标，和 build_fire_demo_video.py 保持一致
TUNED_MAIN_SMOKE_BBOX = (0.47, 0.53, 0.90, 0.98)
TUNED_MAIN_FIRE_BBOX = (0.58, 0.60, 0.92, 0.98)


class TimelineDemoSource(MultiCameraDemoSource):
    """只改演示流程，不改任何UI。时间线对齐你的视频：约2~3秒开始冒烟，随后起明火。"""

    def poll(self) -> List[CameraData]:
        now = time.time()
        t = now - self.start

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
            bms_online = False
            bms_vdrop = None
            bms_temp = None

            if cid == "cam-04":
                if t < 2.0:
                    state = "SAFE"
                    risk = 8 + 2 * abs(math.sin(t * 1.1))
                elif t < 3.6:
                    state = "SUSPECT"
                    smoke = 0.22 + 0.05 * abs(math.sin(t * 2.0))
                    max_temp = 56 + 5 * abs(math.cos(t * 0.7))
                    gas = 0.22 + 0.03 * abs(math.sin(t * 1.2))
                    risk = 32 + 5 * abs(math.sin(t * 0.9))
                    reasons = ["zone_C4_rising", "thermal_hot"]
                    dets = [Detection("smoke", smoke, TUNED_MAIN_SMOKE_BBOX, "车位C4附近", "C4")]
                    slots["C4"] = "SUSPECT"
                    slots["C5"] = "SUSPECT"
                elif t < 6.4:
                    state = "PREWARNING"
                    smoke = 0.58 + 0.05 * abs(math.sin(t * 1.8))
                    max_temp = 88 + 6 * abs(math.cos(t * 0.8))
                    gas = 0.58 + 0.04 * abs(math.cos(t * 1.5))
                    risk = 60 + 5 * abs(math.sin(t * 0.7))
                    bms_online = True
                    bms_vdrop = 0.68 + 0.05 * abs(math.sin(t * 1.3))
                    bms_temp = 0.56 + 0.05 * abs(math.cos(t * 1.1))
                    reasons = ["vision_smoke", "gas_rising", "thermal_hot", "bms_abnormal"]
                    dets = [Detection("smoke", smoke, TUNED_MAIN_SMOKE_BBOX, "车位C4-C5", "C4")]
                    weights = {"vision": 0.28, "thermal": 0.30, "gas": 0.27, "bms": 0.15}
                    slots["C4"] = "PREWARNING"
                    slots["C5"] = "PREWARNING"
                elif t < 8.8:
                    state = "ALARM"
                    fire = 0.82 + 0.05 * abs(math.sin(t * 2.4))
                    smoke = 0.76 + 0.05 * abs(math.cos(t * 1.4))
                    max_temp = 132 + 8 * abs(math.sin(t * 0.9))
                    gas = 0.78
                    risk = 88 + 3 * abs(math.sin(t * 0.8))
                    bms_online = True
                    bms_vdrop = 0.86
                    bms_temp = 0.82
                    reasons = ["vision_fire", "vision_smoke", "thermal_hot", "slot_C4_located"]
                    dets = [
                        Detection("fire", fire, TUNED_MAIN_FIRE_BBOX, "车位C4", "C4"),
                        Detection("smoke", smoke, TUNED_MAIN_SMOKE_BBOX, "车位C3-C5", "C4"),
                    ]
                    weights = {"vision": 0.35, "thermal": 0.31, "gas": 0.22, "bms": 0.12}
                    slots["C4"] = "ALARM"
                    slots["C5"] = "PREWARNING"
                    slots["C3"] = "SUSPECT"
                else:
                    state = "FAILSAFE_ALARM"
                    fire = 0.86 + 0.04 * abs(math.sin(t * 2.0))
                    smoke = 0.80 + 0.04 * abs(math.cos(t * 1.6))
                    max_temp = 146 + 8 * abs(math.sin(t * 0.9))
                    gas = 0.84
                    risk = 93 + 2 * abs(math.sin(t * 0.7))
                    reasons = ["bms_offline", "env_failsafe", "vision_fire", "thermal_hot"]
                    dets = [
                        Detection("fire", fire, TUNED_MAIN_FIRE_BBOX, "车位C4", "C4"),
                        Detection("smoke", smoke, TUNED_MAIN_SMOKE_BBOX, "车位C3-C5", "C4"),
                    ]
                    weights = {"vision": 0.42, "thermal": 0.33, "gas": 0.25, "bms": 0.00}
                    slots["C4"] = "ALARM"
                    slots["C5"] = "PREWARNING"
                    slots["C3"] = "SUSPECT"

            elif cid in {"cam-03", "cam-05"} and t >= 3.6:
                state = "SUSPECT"
                smoke = 0.16 + 0.04 * abs(math.sin(t + offset))
                max_temp = 50 + 4 * abs(math.cos(t * 0.8 + offset))
                gas = 0.14
                risk = 26 + 3 * abs(math.sin(t * 0.9 + offset))
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
                    bms_online=bms_online,
                    bms_vdrop=bms_vdrop,
                    bms_temp=bms_temp,
                    weights=weights,
                    reasons=reasons,
                    detections=dets,
                    fps=fps,
                    updated_ts=now,
                    parking_slots=slots,
                )
            )
        return items


class MainWindow(BaseMainWindow):
    """沿用你原来的 UI 构建代码，只补媒体加载、缩略图和录制。"""

    def __init__(
        self,
        source_mode: str,
        source_path: Optional[str],
        camera_source,
        side_image_paths: Optional[List[str]] = None,
        record_out: str = "",
        record_fps: float = 20.0,
        auto_close: float = 0.0,
    ):
        QMainWindow.__init__(self)
        self.setWindowTitle("FireGuard Pro · 智能火灾预警平台")
        self.resize(*UI_CONFIG["window_size"])
        self.source_mode = source_mode
        self.source_path = source_path
        self.camera_source = camera_source
        self.side_image_paths = side_image_paths or []
        self.record_out = record_out.strip()
        self.record_fps = max(1.0, float(record_fps))
        self.record_writer = None
        self.record_size = None
        self.media_start_ts = time.time()
        self.auto_close = max(0.0, float(auto_close))
        self.video_is_file = False
        self.video_finished = False
        self.source_duration_sec = 0.0
        self.source_fps = 25.0
        self.source_frame_count = 0

        self.data_source = TimelineDemoSource() if source_mode == "demo" else JSONLDataSource(source_path)
        self.cap = self._open_capture(camera_source)
        self.selected_camera_id: Optional[str] = None
        self.last_frame = self._make_placeholder_frame("Initializing")
        self.capture_history = deque(maxlen=UI_CONFIG["capture_history_limit"])
        self.last_capture_state = {}

        self._build_ui()
        self._apply_style()
        self._load_side_images()

        if self.auto_close <= 0 and self.record_out and self.source_duration_sec > 0:
            self.auto_close = self.source_duration_sec + 0.4

        timer_interval_ms = 40
        if self.video_is_file and self.source_fps > 1:
            timer_interval_ms = max(15, int(round(1000.0 / self.source_fps)))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(timer_interval_ms)
        print(f"[MEDIA] timer_interval_ms={timer_interval_ms}")

    def _cv_imread(self, path: str) -> Optional[np.ndarray]:
        try:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _open_capture(self, source):
        src = source
        if isinstance(source, str):
            s = source.strip().strip('"')
            if os.path.exists(s):
                src = s
                self.video_is_file = True
            else:
                try:
                    src = int(s)
                except Exception:
                    src = s
        cap = cv2.VideoCapture(src)
        print(f"[MEDIA] source={src} opened={cap.isOpened()} file_mode={self.video_is_file}")
        if cap.isOpened() and self.video_is_file:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 1:
                self.source_fps = fps
            self.source_frame_count = int(frame_count) if frame_count > 0 else 0
            print(f"[MEDIA] fps={fps:.2f} frames={frame_count:.0f}")
            if fps > 0 and frame_count > 0:
                self.source_duration_sec = frame_count / fps
            ok, first = cap.read()
            print(f"[MEDIA] first_frame_ok={ok} shape={None if first is None else first.shape}")
            if ok and first is not None:
                self.last_frame = first
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif not cap.isOpened():
            print(f"[MEDIA] ERROR: failed to open source -> {src}")
        return cap

    def _crop_side_image(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x2 = max(1, int(w * 0.90))
        y2 = max(1, int(h * 0.90))
        cropped = frame[:y2, :x2]
        return cropped if cropped.size else frame

    def _set_label_frame(self, label: QLabel, frame: np.ndarray):
        pix = qpixmap_from_bgr(frame)
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        label.setText("")

    def _load_side_images(self):
        labels = self.left_camera_labels + self.right_camera_labels
        for idx, label in enumerate(labels):
            if idx < len(self.side_image_paths):
                img = self._cv_imread(self.side_image_paths[idx])
                if img is not None:
                    img = self._crop_side_image(img)
                    self._set_label_frame(label, img)
                    continue
            label.setText(f"摄像头 {idx + 1}")

    def _read_frame(self) -> np.ndarray:
        if self.cap.isOpened():
            if self.video_is_file and self.source_fps > 1:
                elapsed = max(0.0, time.time() - self.media_start_ts)
                target_idx = int(elapsed * self.source_fps)
                if self.source_frame_count > 0:
                    if target_idx >= self.source_frame_count:
                        self.video_finished = True
                        target_idx = target_idx % self.source_frame_count
                    else:
                        self.video_finished = False
                current_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                if abs(target_idx - current_idx) > 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self.last_frame = frame
                    return frame
            else:
                ok, frame = self.cap.read()
                if ok and frame is not None:
                    self.last_frame = frame
                    return frame
                if self.video_is_file:
                    self.video_finished = True
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                    if ok and frame is not None:
                        self.last_frame = frame
                        return frame
        return self.last_frame.copy()

    def _draw_video_overlay(self, frame: np.ndarray, cam: CameraData) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        if not UI_CONFIG["show_video_overlay"]:
            return out

        cv2.rectangle(out, (0, 0), (w, 62), (8, 16, 30), -1)
        st_color = self._hex_to_bgr(state_color(cam.state))
        cv2.putText(out, f"STATE {cam.state}", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.86, st_color, 2, cv2.LINE_AA)
        cv2.putText(out, f"RISK {cam.risk_score:.1f}", (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.72, st_color, 2, cv2.LINE_AA)

        if UI_CONFIG["show_video_timestamp"]:
            cv2.putText(out, QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss").replace('-', '/'), (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (220, 230, 245), 2, cv2.LINE_AA)

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

        if UI_CONFIG["show_video_meta"]:
            cv2.rectangle(out, (0, h - 40), (w, h), (8, 16, 30), -1)
            meta = f"FIRE {cam.fire_conf:.2f}   SMOKE {cam.smoke_conf:.2f}   TEMP {fmt_num(cam.max_temp,1)}C   GAS {fmt_num(cam.smoke_sensor_value,2)}"
            cv2.putText(out, meta, (20, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (225, 236, 255), 2, cv2.LINE_AA)
        return out

    def _refresh_side_panels(self, cams: List[CameraData], selected_cam: CameraData):
        labels = self.left_camera_labels + self.right_camera_labels
        path_map = {f"cam-0{idx+1}": self.side_image_paths[idx] for idx in range(min(len(self.side_image_paths), 6))}
        cam_map = {c.camera_id: c for c in cams}
        for idx, cam_id in enumerate(["cam-01", "cam-02", "cam-03", "cam-04", "cam-05", "cam-06"]):
            if idx >= len(labels):
                break
            label = labels[idx]
            img_path = path_map.get(cam_id)
            if not img_path:
                continue
            img = self._cv_imread(img_path)
            if img is None:
                continue
            img = self._crop_side_image(img)
            cam_obj = cam_map.get(cam_id)
            if cam_obj is not None and cam_obj.state in {"SUSPECT", "PREWARNING", "ALARM", "FAILSAFE_ALARM"}:
                color = self._hex_to_bgr(state_color(cam_obj.state))
                cv2.rectangle(img, (4, 4), (img.shape[1] - 4, img.shape[0] - 4), color, 6)
                cv2.putText(img, state_text(cam_obj.state), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            self._set_label_frame(label, img)

    def _update_center(self, cam: CameraData, cams: List[CameraData]):
        raw = self._read_frame()
        overlay = self._draw_video_overlay(raw, cam)
        if self.video_is_file:
            cv2.putText(overlay, "VIDEO PLAYING", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2, cv2.LINE_AA)
        pixmap = qpixmap_from_bgr(overlay)
        self.video_label.set_source_pixmap(pixmap)
        self._refresh_side_panels(cams, cam)
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

    def _record_window_frame(self):
        if not self.record_out:
            return
        pix = self.grab()
        image = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = image.width(), image.height()
        bpl = image.bytesPerLine()
        ptr = image.bits()
        arr = np.frombuffer(ptr, np.uint8, count=bpl * h).reshape((h, bpl // 4, 4))[:, :w, :]
        frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR).copy()
        if self.record_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.record_writer = cv2.VideoWriter(self.record_out, fourcc, self.record_fps, (w, h))
            self.record_size = (w, h)
        self.record_writer.write(frame)

    def _tick(self):
        if UI_CONFIG["show_top_time"]:
            self.header.set_time_text(QDateTime.currentDateTime().toString("yyyy-MM-dd  HH:mm:ss"))
        else:
            self.header.set_time_text("")
        cams = self.data_source.poll()
        self.camera_list.load_items(cams, self.selected_camera_id)
        self._update_overview(cams)
        cam = self._pick_selected_camera(cams)
        self._update_center(cam, cams)
        self._update_right(cam)
        self._record_window_frame()
        if self.auto_close > 0 and (time.time() - self.media_start_ts) >= self.auto_close:
            self.close()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        if self.record_writer is not None:
            self.record_writer.release()
            self.record_writer = None
        super().closeEvent(event)


def build_arg_parser():
    p = argparse.ArgumentParser(description="FireGuard Pro UI exact media wrapper")
    p.add_argument("--mode", choices=["demo", "jsonl"], default="demo")
    p.add_argument("--jsonl", default="", help="events.jsonl path when mode=jsonl")
    p.add_argument("--source", default="0", help="camera index or video file path")
    p.add_argument("--cam_images", nargs="*", default=[], help="six parking camera images")
    p.add_argument("--record_out", default="", help="optional mp4 output path")
    p.add_argument("--record_fps", type=float, default=20.0, help="record fps")
    p.add_argument("--auto_close", type=float, default=0.0, help="seconds before auto close; 0 means manual close or video duration when recording")
    return p


def main():
    args = build_arg_parser().parse_args()
    app = QApplication(sys.argv)
    w = MainWindow(
        source_mode=args.mode,
        source_path=args.jsonl or None,
        camera_source=args.source,
        side_image_paths=args.cam_images,
        record_out=args.record_out,
        record_fps=args.record_fps,
        auto_close=args.auto_close,
    )
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
