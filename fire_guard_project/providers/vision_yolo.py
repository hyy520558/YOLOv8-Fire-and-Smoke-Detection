from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2

from core.schema import VisualResult

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


@dataclass
class VisionRuntime:
    conf: float = 0.25
    imgsz: int = 640
    device: str = "0"
    half: bool = False
    infer_every: int = 2


class ZeroVisionDetector:
    def __init__(self):
        self.names = {}

    def inspect_names(self) -> Tuple[bool, str]:
        return False, "未启用视觉检测，fire/smoke 置信度始终为 0。"

    def infer(self, frame, frame_id: int) -> VisualResult:
        return VisualResult(annotated_frame=frame)


class YoloVisionDetector:
    def __init__(self, model_path: str, runtime: VisionRuntime):
        if YOLO is None:
            raise RuntimeError("未安装 ultralytics，请先 pip install ultralytics")
        self.model = YOLO(model_path)
        self.runtime = runtime
        self.last_result = VisualResult()
        self.names: Dict[int, str] = dict(getattr(self.model, "names", {}) or {})
        self.supports_target, self.name_warning = self.inspect_names()

    def inspect_names(self) -> Tuple[bool, str]:
        values = [str(v).lower() for v in self.names.values()]
        has_fire = any(("fire" in v) or ("火" in v) for v in values)
        has_smoke = any(("smoke" in v) or ("烟" in v) for v in values)
        if has_fire and has_smoke:
            return True, "模型类别中已检测到 fire/smoke。"
        return False, "当前模型类别里未发现 fire/smoke，请确认你加载的是火焰烟雾专用 best.pt，而不是 COCO 通用权重。"

    def _is_fire(self, name: str) -> bool:
        name = name.lower()
        return ("fire" in name) or ("flame" in name) or ("火" in name)

    def _is_smoke(self, name: str) -> bool:
        name = name.lower()
        return ("smoke" in name) or ("烟" in name)

    def infer(self, frame, frame_id: int) -> VisualResult:
        if frame_id % max(self.runtime.infer_every, 1) != 0:
            cached = self.last_result
            cached.annotated_frame = frame if cached.annotated_frame is None else cached.annotated_frame
            return cached

        result = self.model.predict(
            source=frame,
            conf=self.runtime.conf,
            imgsz=self.runtime.imgsz,
            device=self.runtime.device,
            half=self.runtime.half,
            verbose=False,
        )[0]

        fire_conf = 0.0
        smoke_conf = 0.0
        detections = []
        names = dict(getattr(result, "names", {}) or self.names)
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            cls_ids = boxes.cls.tolist() if boxes.cls is not None else []
            confs = boxes.conf.tolist() if boxes.conf is not None else []
            xyxy = boxes.xyxy.tolist() if boxes.xyxy is not None else []
            for i, cls_id in enumerate(cls_ids):
                cls_idx = int(cls_id)
                name = str(names.get(cls_idx, cls_idx))
                conf = float(confs[i]) if i < len(confs) else 0.0
                box = xyxy[i] if i < len(xyxy) else None
                detections.append({"cls": cls_idx, "name": name, "conf": conf, "box": box})
                if self._is_fire(name):
                    fire_conf = max(fire_conf, conf)
                if self._is_smoke(name):
                    smoke_conf = max(smoke_conf, conf)

        annotated = result.plot()
        out = VisualResult(
            fire_conf=fire_conf,
            smoke_conf=smoke_conf,
            has_fire=fire_conf > 0,
            has_smoke=smoke_conf > 0,
            annotated_frame=annotated,
            raw_detections=detections,
            class_names=names,
            warning=None if self.supports_target else self.name_warning,
        )
        self.last_result = out
        return out


def open_capture(source: str):
    src = 0 if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")
    return cap
