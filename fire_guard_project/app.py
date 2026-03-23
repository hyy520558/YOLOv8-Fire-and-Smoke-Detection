from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2

from config import AppConfig, LoggingConfig, RuntimeConfig, ThresholdConfig
from core.fusion import fuse_and_decide
from core.logger import JSONLLogger
from core.schema import SensorFrame
from core.state_machine import decide_state
from providers.bms_mock import BMSState
from providers.smoke_mock import SmokeState
from providers.thermal_mock import ThermalState
from providers.vision_yolo import VisionRuntime, YoloVisionDetector, ZeroVisionDetector, open_capture
from transport.mock_serial import MockSerialTransport, ReplayEventSource
from transport.real_serial import RealSerialTransport
from actuators.mock_actuator import MockActuatorController
from actuators.esp32_bridge import ESP32Bridge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fire Guard PC 数字孪生主程序")
    p.add_argument("--model", default="models/best.pt", help="YOLO 权重路径")
    p.add_argument("--source", default="0", help="摄像头或视频源，none 表示关闭视觉")
    p.add_argument("--device", default="0", help="YOLO device，例如 0/cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--infer-every", type=int, default=2, help="每 N 帧真正推理一次，其余帧复用结果")
    p.add_argument("--half", action="store_true", help="YOLO 推理使用 FP16")
    p.add_argument("--max-fps", type=float, default=12.0, help="主循环上限 FPS，用于给 Jetson 降载")
    p.add_argument("--headless", action="store_true", help="不显示窗口")
    p.add_argument("--mock-stdin", action="store_true", help="开启终端 JSON 输入，模拟串口")
    p.add_argument("--mock-replay", default="", help="JSONL 回放文件，格式见 tests 目录")
    p.add_argument("--serial-port", default="", help="真串口端口，例如 COM5")
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--send-to-esp32", action="store_true", help="将决策结果通过 transport 发出")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--duration", type=float, default=0.0, help="运行时长上限（秒），0 表示不限")
    return p.parse_args()


class SensorHub:
    def __init__(self):
        self.thermal = ThermalState()
        self.smoke = SmokeState()
        self.bms = BMSState()

    def apply_message(self, msg: dict):
        self.thermal.apply_message(msg)
        self.smoke.apply_message(msg)
        self.bms.apply_message(msg)

    def snapshot(self) -> dict:
        out = {}
        out.update(self.thermal.snapshot())
        out.update(self.smoke.snapshot())
        out.update(self.bms.snapshot())
        return out


def choose_transport(args: argparse.Namespace):
    if args.serial_port:
        print(f"[INFO] 使用真串口: {args.serial_port} @ {args.baudrate}")
        return RealSerialTransport(port=args.serial_port, baudrate=args.baudrate)

    if args.mock_replay:
        print(f"[INFO] 使用回放事件源: {args.mock_replay}")
        return ReplayEventSource(args.mock_replay)

    if args.mock_stdin:
        tr = MockSerialTransport()
        tr.start()
        return tr

    return None


def choose_actuator(transport, send_to_esp32: bool):
    if transport is not None and send_to_esp32:
        return ESP32Bridge(transport)
    return MockActuatorController()



def create_vision(args: argparse.Namespace):
    if str(args.source).lower() == "none":
        print("[INFO] 已关闭视觉模块，适合纯传感器模拟联调。")
        return ZeroVisionDetector(), None

    runtime = VisionRuntime(
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        infer_every=args.infer_every,
    )
    detector = YoloVisionDetector(args.model, runtime)
    print("[INFO]", detector.name_warning)
    cap = open_capture(args.source)
    return detector, cap



def overlay_dashboard(frame, frame_obj: SensorFrame, decision, vision_warning: Optional[str] = None):
    h = frame.shape[0]
    lines = [
        f"frame={frame_obj.frame_id} fps={frame_obj.fps:.1f}",
        f"fire_conf={frame_obj.fire_conf:.2f} smoke_conf={frame_obj.smoke_conf:.2f}",
        f"min={frame_obj.min_temp} max={frame_obj.max_temp} avg={frame_obj.avg_temp}",
        f"gas={frame_obj.smoke_sensor_value} bms_vdrop={frame_obj.bms_voltage_drop_score} bms_temp={frame_obj.bms_temp_score}",
        f"risk={decision.risk_score:.2f} state={decision.state}",
        f"weights={decision.weights}",
        f"reasons={decision.reasons}",
    ]
    if vision_warning:
        lines.append(f"warning={vision_warning}")

    panel_h = 24 * len(lines) + 20
    cv2.rectangle(frame, (0, max(0, h - panel_h)), (frame.shape[1], h), (0, 0, 0), -1)
    y = h - panel_h + 24
    for line in lines:
        cv2.putText(frame, str(line), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return frame



def main():
    args = parse_args()
    cfg = AppConfig(
        thresholds=ThresholdConfig(),
        runtime=RuntimeConfig(
            model_path=args.model,
            camera_source=args.source,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            infer_every=args.infer_every,
            half=args.half,
            max_fps=args.max_fps,
            show_window=not args.headless,
        ),
        logging=LoggingConfig(log_dir=Path(args.log_dir)),
    )

    os.makedirs(cfg.logging.log_dir, exist_ok=True)
    logger = JSONLLogger(cfg.logging.log_dir / "events.jsonl")

    transport = choose_transport(args)
    actuator = choose_actuator(transport, args.send_to_esp32)
    sensor_hub = SensorHub()

    detector, cap = create_vision(args)

    last_time = time.time()
    frame_id = 0
    frame_sleep = 0.0 if args.max_fps <= 0 else (1.0 / args.max_fps)
    print("[INFO] 主循环启动。窗口模式下按 q 退出。")
    app_start = time.time()

    try:
        while True:
            loop_start = time.time()

            # 1) 读传感器输入（真串口/假串口/回放）
            if transport is not None:
                for msg in transport.read_available():
                    print("[RX]", json.dumps(msg, ensure_ascii=False))
                    sensor_hub.apply_message(msg)

            # 2) 视觉输入
            visual_warning = None
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print("[WARN] 视频源读取失败，退出。")
                    break
            else:
                frame = None

            if frame is not None:
                vis = detector.infer(frame, frame_id)
                visual_warning = vis.warning
                out_frame = vis.annotated_frame if vis.annotated_frame is not None else frame
                fire_conf = vis.fire_conf
                smoke_conf = vis.smoke_conf
                has_fire = vis.has_fire
                has_smoke = vis.has_smoke
            else:
                out_frame = None
                fire_conf = 0.0
                smoke_conf = 0.0
                has_fire = False
                has_smoke = False

            # 3) 融合为统一帧
            now = time.time()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            snapshot = sensor_hub.snapshot()
            sensor_frame = SensorFrame(
                timestamp=now,
                frame_id=frame_id,
                fire_conf=fire_conf,
                smoke_conf=smoke_conf,
                has_fire=has_fire,
                has_smoke=has_smoke,
                fps=fps,
                source="camera+transport" if cap is not None and transport is not None else ("camera" if cap is not None else "transport"),
                **snapshot,
            )

            # 4) 融合决策
            decision = fuse_and_decide(sensor_frame, cfg.thresholds, decide_state)

            # 5) 执行动作
            actuator.handle(decision)

            # 6) 日志
            logger.write({
                "sensor_frame": sensor_frame.to_dict(),
                "decision": decision.to_dict(),
            })

            # 7) 显示
            if out_frame is not None and cfg.runtime.show_window:
                out_frame = overlay_dashboard(out_frame, sensor_frame, decision, visual_warning)
                cv2.imshow("Fire Guard PC Twin", out_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            elif out_frame is None:
                # 纯传感器模式下，控制台输出简报
                print(
                    f"[TICK] risk={decision.risk_score:.2f} state={decision.state} "
                    f"max_temp={sensor_frame.max_temp} smoke={sensor_frame.smoke_sensor_value} "
                    f"bms_online={sensor_frame.bms_online} reasons={decision.reasons}"
                )

            frame_id += 1

            if args.duration > 0 and (time.time() - app_start) >= args.duration:
                print("[INFO] 已到达设定运行时长，退出。")
                break

            if cap is None and hasattr(transport, "exhausted") and getattr(transport, "exhausted"):
                if (time.time() - app_start) > 1.0:
                    print("[INFO] 回放事件已结束，退出。")
                    break

            elapsed = time.time() - loop_start
            if frame_sleep > elapsed:
                time.sleep(frame_sleep - elapsed)

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if transport is not None and hasattr(transport, "close"):
            transport.close()


if __name__ == "__main__":
    main()
