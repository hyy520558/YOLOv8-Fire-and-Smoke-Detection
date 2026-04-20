import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import serial
from serial import SerialException

try:
    from ultralytics import YOLO
    ULTRALYTICS_OK = True
except Exception:
    ULTRALYTICS_OK = False
    YOLO = None


# ============================================================
# 1) 配置区：先改这里
# ============================================================
SERIAL_PORT = "/dev/ttyUSB0"      # Jetson 连接 ESP32 的串口
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 0.02

CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 20

ENABLE_YOLO = False                # 暂时没有模型就先 False
YOLO_MODEL_PATH = "best.pt"       # 有模型后再改
YOLO_IMGSZ = 640
YOLO_CONF = 0.25

AUTO_HOME_ON_START = True
SHOW_WINDOW = True
WINDOW_NAME = "FireGuard Jetson Control"

# 风险阈值
THRESH_SUSPECT = 0.30
THRESH_PREWARNING = 0.52
THRESH_ALARM = 0.75
THRESH_RELEASE = 0.82

# 连续满足多少帧才动作，避免左右横跳
PREWARNING_HOLD_FRAMES = 3
ALARM_HOLD_FRAMES = 4
RELEASE_HOLD_FRAMES = 5
SAFE_HOME_HOLD_FRAMES = 25

# 融合权重（可调）
W_THERMAL = 0.35
W_IR = 0.15
W_SMOKE = 0.10
W_VISION_FIRE = 0.28
W_VISION_SMOKE = 0.07
W_BMS = 0.05

# 温度归一化参数（结合你当前演示环境先给一个保守值）
TEMP_NORM_LOW = 35.0
TEMP_NORM_HIGH = 120.0

# 命令去重与节流
MOVE_CMD_MIN_INTERVAL_S = 0.8
ARM_CMD_MIN_INTERVAL_S = 0.6
RELEASE_CMD_MIN_INTERVAL_S = 1.5
PERSON_CLEAR_CMD_MIN_INTERVAL_S = 0.5
HOME_CMD_MIN_INTERVAL_S = 2.0


# ============================================================
# 2) 数据结构
# ============================================================
@dataclass
class Telemetry:
    ms: int = 0
    thermal_ok: bool = False
    min_temp: float = 0.0
    max_temp: float = 0.0
    avg_temp: float = 0.0
    slot_temp: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    slot_ir: List[int] = field(default_factory=lambda: [0, 0, 0])
    hot_slot: str = "L"
    local_fire_valid: bool = False
    smoke_raw: int = 0
    smoke_norm: float = 0.0
    bms_online: bool = False
    bms_v_score: float = 0.0
    bms_t_score: float = 0.0
    bms_i_score: float = 0.0
    bms_overall_score: float = 0.0
    rail_state: str = "IDLE"
    rail_pos: int = 0
    rail_slot: str = "L"
    target_slot: str = "L"
    homed: bool = False
    home_sw: int = 0
    right_sw: int = 0
    person_clear: bool = True
    estop: bool = False
    release_active: bool = False
    release_done: bool = False
    armed_slot: str = "N"
    raw: Dict = field(default_factory=dict)


@dataclass
class VisionInfo:
    ok: bool = False
    frame: Optional[np.ndarray] = None
    frame_w: int = 0
    frame_h: int = 0
    slot_fire_conf: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    slot_smoke_conf: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    person_present: bool = False
    detections: List[Dict] = field(default_factory=list)


@dataclass
class FusionResult:
    slot_risk: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    global_risk: float = 0.0
    target_slot: str = "L"
    state: str = "SAFE"
    reasons: List[str] = field(default_factory=list)


# ============================================================
# 3) 串口通信
# ============================================================
class ESPLink:
    def __init__(self, port: str, baud: int, timeout: float = 0.02):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None
        self.last_open_try = 0.0
        self.last_rx_line = ""
        self.last_tx_time: Dict[str, float] = {}

    def connect_if_needed(self):
        now = time.time()
        if self.ser is not None and self.ser.is_open:
            return
        if now - self.last_open_try < 1.5:
            return
        self.last_open_try = now
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            print(f"[ESP] Connected: {self.port} @ {self.baud}")
        except SerialException as e:
            self.ser = None
            print(f"[ESP] Connect failed: {e}")

    def is_connected(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass

    def read_messages(self) -> List[Dict]:
        msgs = []
        if not self.is_connected():
            return msgs
        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                self.last_rx_line = line
                try:
                    obj = json.loads(line)
                    msgs.append(obj)
                except json.JSONDecodeError:
                    print(f"[ESP] Bad JSON: {line}")
        except Exception as e:
            print(f"[ESP] Read error: {e}")
            self.close()
        return msgs

    def send(self, cmd: Dict, dedup_key: Optional[str] = None, min_interval_s: float = 0.0):
        if not self.is_connected():
            return False
        now = time.time()
        if dedup_key is not None:
            last_t = self.last_tx_time.get(dedup_key, 0.0)
            if now - last_t < min_interval_s:
                return False

        line = json.dumps(cmd, ensure_ascii=False) + "\n"
        try:
            self.ser.write(line.encode("utf-8"))
            self.ser.flush()
            if dedup_key is not None:
                self.last_tx_time[dedup_key] = now
            print(f"[TX] {line.strip()}")
            return True
        except Exception as e:
            print(f"[ESP] Write error: {e}")
            self.close()
            return False


# ============================================================
# 4) 视觉模块
# ============================================================
class VisionModule:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.model = None
        self.last_infer_t = 0.0
        self.infer_interval_s = 0.10
        self.enabled_yolo = ENABLE_YOLO and ULTRALYTICS_OK

    def open(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

        if self.enabled_yolo:
            try:
                self.model = YOLO(YOLO_MODEL_PATH)
                print(f"[YOLO] Loaded model: {YOLO_MODEL_PATH}")
            except Exception as e:
                print(f"[YOLO] Load failed: {e}")
                self.model = None
                self.enabled_yolo = False

    def close(self):
        if self.cap is not None:
            self.cap.release()

    @staticmethod
    def _slot_by_x(cx: float, w: int) -> int:
        if cx < w / 3:
            return 0
        if cx < 2 * w / 3:
            return 1
        return 2

    def poll(self) -> VisionInfo:
        info = VisionInfo()
        if self.cap is None:
            return info

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return info

        h, w = frame.shape[:2]
        info.ok = True
        info.frame = frame
        info.frame_w = w
        info.frame_h = h

        if not self.enabled_yolo or self.model is None:
            return info

        now = time.time()
        if now - self.last_infer_t < self.infer_interval_s:
            return info
        self.last_infer_t = now

        try:
            results = self.model.predict(source=frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
            if not results:
                return info

            r = results[0]
            names = r.names if hasattr(r, "names") else {}
            boxes = r.boxes
            if boxes is None:
                return info

            for b in boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                cx = 0.5 * (x1 + x2)
                slot = self._slot_by_x(cx, w)

                label = str(names.get(cls_id, cls_id)).lower()
                det = {
                    "label": label,
                    "conf": conf,
                    "box": [x1, y1, x2, y2],
                    "slot": slot,
                }
                info.detections.append(det)

                if "person" in label:
                    info.person_present = True
                elif "fire" in label or "flame" in label:
                    info.slot_fire_conf[slot] = max(info.slot_fire_conf[slot], conf)
                elif "smoke" in label:
                    info.slot_smoke_conf[slot] = max(info.slot_smoke_conf[slot], conf)
        except Exception as e:
            print(f"[YOLO] Inference error: {e}")

        return info


# ============================================================
# 5) 融合与状态机
# ============================================================
class FusionController:
    def __init__(self, esp: ESPLink):
        self.esp = esp
        self.tel = Telemetry()
        self.vis = VisionInfo()
        self.result = FusionResult()

        self.current_state = "SAFE"
        self.last_target_slot = "L"
        self.prewarning_count = 0
        self.alarm_count = 0
        self.release_count = 0
        self.safe_count = 0
        self.has_released_this_alarm = False

    @staticmethod
    def clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def update_telemetry(self, obj: Dict):
        if obj.get("type") == "telemetry":
            self.tel = Telemetry(
                ms=int(obj.get("ms", 0)),
                thermal_ok=bool(obj.get("thermal_ok", 0)),
                min_temp=float(obj.get("min_temp", 0.0)),
                max_temp=float(obj.get("max_temp", 0.0)),
                avg_temp=float(obj.get("avg_temp", 0.0)),
                slot_temp=list(obj.get("slot_temp", [0.0, 0.0, 0.0])),
                slot_ir=list(obj.get("slot_ir", [0, 0, 0])),
                hot_slot=str(obj.get("hot_slot", "L")),
                local_fire_valid=bool(obj.get("local_fire_valid", 0)),
                smoke_raw=int(obj.get("smoke_raw", 0)),
                smoke_norm=float(obj.get("smoke_norm", 0.0)),
                bms_online=bool(obj.get("bms_online", 0)),
                bms_v_score=float(obj.get("bms_v_score", 0.0)),
                bms_t_score=float(obj.get("bms_t_score", 0.0)),
                bms_i_score=float(obj.get("bms_i_score", 0.0)),
                bms_overall_score=float(obj.get("bms_overall_score", 0.0)),
                rail_state=str(obj.get("rail_state", "IDLE")),
                rail_pos=int(obj.get("rail_pos", 0)),
                rail_slot=str(obj.get("rail_slot", "L")),
                target_slot=str(obj.get("target_slot", "L")),
                homed=bool(obj.get("homed", 0)),
                home_sw=int(obj.get("home_sw", 0)),
                right_sw=int(obj.get("right_sw", 0)),
                person_clear=bool(obj.get("person_clear", 1)),
                estop=bool(obj.get("estop", 0)),
                release_active=bool(obj.get("release_active", 0)),
                release_done=bool(obj.get("release_done", 0)),
                armed_slot=str(obj.get("armed_slot", "N")),
                raw=obj,
            )
        elif obj.get("type") == "ack":
            print(f"[ACK] {obj}")
        elif obj.get("type") == "error":
            print(f"[ESP-ERROR] {obj}")

    def update_vision(self, info: VisionInfo):
        self.vis = info

    def _thermal_score(self, t: float) -> float:
        return self.clip01((t - TEMP_NORM_LOW) / (TEMP_NORM_HIGH - TEMP_NORM_LOW))

    def fuse(self) -> FusionResult:
        reasons = []
        slot_risk = [0.0, 0.0, 0.0]
        smoke_global = self.clip01(self.tel.smoke_norm)
        bms_score = self.clip01(self.tel.bms_overall_score if self.tel.bms_online else 0.0)

        for i in range(3):
            thermal_score = self._thermal_score(float(self.tel.slot_temp[i]))
            ir_score = 1.0 if int(self.tel.slot_ir[i]) else 0.0
            fire_score = self.clip01(float(self.vis.slot_fire_conf[i]))
            smoke_score = self.clip01(float(self.vis.slot_smoke_conf[i]))

            risk = (
                W_THERMAL * thermal_score +
                W_IR * ir_score +
                W_SMOKE * smoke_global +
                W_VISION_FIRE * fire_score +
                W_VISION_SMOKE * smoke_score +
                W_BMS * bms_score
            )

            # 本地热像已经判断出强烈火情时，加一点额外提升
            if self.tel.local_fire_valid and self.tel.hot_slot in ["L", "M", "R"]:
                hot_idx = {"L": 0, "M": 1, "R": 2}[self.tel.hot_slot]
                if i == hot_idx:
                    risk = min(1.0, risk + 0.08)

            slot_risk[i] = self.clip01(risk)

        target_idx = int(np.argmax(np.array(slot_risk)))
        target_slot = ["L", "M", "R"][target_idx]
        global_risk = float(slot_risk[target_idx])

        if self.tel.estop:
            state = "BLOCKED"
            reasons.append("急停触发")
        elif not self.vis.ok:
            # 没相机也能先靠 ESP32 热像 + 烟雾跑起来
            if global_risk >= THRESH_ALARM:
                state = "ALARM"
            elif global_risk >= THRESH_PREWARNING:
                state = "PREWARNING"
            elif global_risk >= THRESH_SUSPECT:
                state = "SUSPECT"
            else:
                state = "SAFE"
            reasons.append("视觉暂不可用，使用下位机热像/烟雾")
        else:
            if global_risk >= THRESH_ALARM:
                state = "ALARM"
            elif global_risk >= THRESH_PREWARNING:
                state = "PREWARNING"
            elif global_risk >= THRESH_SUSPECT:
                state = "SUSPECT"
            else:
                state = "SAFE"

        if self.vis.person_present:
            reasons.append("检测到人员")
        if self.tel.local_fire_valid:
            reasons.append("ESP本地火情有效")
        if self.tel.bms_online:
            reasons.append("BMS在线")
        if smoke_global > 0.2:
            reasons.append("烟雾值升高")

        self.result = FusionResult(
            slot_risk=slot_risk,
            global_risk=global_risk,
            target_slot=target_slot,
            state=state,
            reasons=reasons,
        )
        return self.result

    def _send_person_clear(self):
        person_clear = 0 if self.vis.person_present else 1
        self.esp.send(
            {"cmd": "set_person_clear", "value": person_clear},
            dedup_key=f"person_clear_{person_clear}",
            min_interval_s=PERSON_CLEAR_CMD_MIN_INTERVAL_S,
        )

    def _send_home(self):
        self.esp.send(
            {"cmd": "home"},
            dedup_key="home",
            min_interval_s=HOME_CMD_MIN_INTERVAL_S,
        )

    def _send_move_and_arm(self, slot: str):
        self.esp.send(
            {"cmd": "move_slot", "target": slot},
            dedup_key=f"move_{slot}",
            min_interval_s=MOVE_CMD_MIN_INTERVAL_S,
        )
        self.esp.send(
            {"cmd": "arm_release", "target": slot},
            dedup_key=f"arm_{slot}",
            min_interval_s=ARM_CMD_MIN_INTERVAL_S,
        )

    def _send_release(self, slot: str):
        self.esp.send(
            {"cmd": "release", "target": slot},
            dedup_key=f"release_{slot}",
            min_interval_s=RELEASE_CMD_MIN_INTERVAL_S,
        )

    def step(self):
        result = self.fuse()
        self._send_person_clear()

        # 启动回零
        if AUTO_HOME_ON_START and self.esp.is_connected() and not self.tel.homed and self.tel.rail_state != "HOMING":
            self._send_home()

        if result.state == "BLOCKED":
            self.prewarning_count = 0
            self.alarm_count = 0
            self.release_count = 0
            self.safe_count = 0
            self.current_state = "BLOCKED"
            return

        if result.state == "SAFE":
            self.safe_count += 1
            self.prewarning_count = 0
            self.alarm_count = 0
            self.release_count = 0
            self.current_state = "SAFE"

            if self.safe_count >= SAFE_HOME_HOLD_FRAMES and self.tel.homed and self.tel.rail_state == "IDLE":
                if self.tel.rail_slot != "L":
                    self._send_home()
                self.has_released_this_alarm = False
            return

        self.safe_count = 0

        if result.state == "SUSPECT":
            self.prewarning_count = 0
            self.alarm_count = 0
            self.release_count = 0
            self.current_state = "SUSPECT"
            return

        if result.state == "PREWARNING":
            if result.target_slot == self.last_target_slot:
                self.prewarning_count += 1
            else:
                self.prewarning_count = 1
                self.last_target_slot = result.target_slot

            self.alarm_count = 0
            self.release_count = 0
            self.current_state = "PREWARNING"

            if self.prewarning_count >= PREWARNING_HOLD_FRAMES:
                self._send_move_and_arm(result.target_slot)
            return

        if result.state == "ALARM":
            if result.target_slot == self.last_target_slot:
                self.alarm_count += 1
                self.release_count += 1
            else:
                self.last_target_slot = result.target_slot
                self.alarm_count = 1
                self.release_count = 1

            self.current_state = "ALARM"

            if self.alarm_count >= ALARM_HOLD_FRAMES:
                self._send_move_and_arm(result.target_slot)

            if (
                self.release_count >= RELEASE_HOLD_FRAMES and
                not self.has_released_this_alarm and
                not self.vis.person_present and
                self.tel.homed
            ):
                self._send_release(result.target_slot)
                self.has_released_this_alarm = True
            return


# ============================================================
# 6) UI
# ============================================================
def draw_dashboard(frame: np.ndarray, tel: Telemetry, vis: VisionInfo, res: FusionResult, esp_connected: bool) -> np.ndarray:
    if frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    img = frame.copy()
    h, w = img.shape[:2]

    # 画三工位分区线
    cv2.line(img, (w // 3, 0), (w // 3, h), (255, 255, 255), 2)
    cv2.line(img, (2 * w // 3, 0), (2 * w // 3, h), (255, 255, 255), 2)

    # 画检测框
    for det in vis.detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["conf"]
        color = (0, 255, 255)
        if "person" in label:
            color = (255, 0, 0)
        elif "fire" in label or "flame" in label:
            color = (0, 0, 255)
        elif "smoke" in label:
            color = (0, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label}:{conf:.2f}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 右侧信息面板
    panel_w = 420
    canvas = np.zeros((max(h, 720), w + panel_w, 3), dtype=np.uint8)
    canvas[:h, :w] = img
    panel = canvas[:, w:]
    panel[:] = (20, 24, 28)

    def put(y, text, color=(230, 230, 230), scale=0.62, thick=2):
        cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    state_color = {
        "SAFE": (0, 200, 0),
        "SUSPECT": (0, 220, 220),
        "PREWARNING": (0, 165, 255),
        "ALARM": (0, 0, 255),
        "BLOCKED": (180, 80, 255),
    }.get(res.state, (220, 220, 220))

    put(35, f"ESP: {'ONLINE' if esp_connected else 'OFFLINE'}", (80, 220, 80) if esp_connected else (0, 0, 255), 0.8)
    put(70, f"STATE: {res.state}", state_color, 0.85, 3)
    put(105, f"TARGET SLOT: {res.target_slot}", (255, 255, 255), 0.75)
    put(140, f"GLOBAL RISK: {res.global_risk:.2f}", (255, 255, 255), 0.75)

    put(185, f"Temps   L/M/R: {tel.slot_temp[0]:.1f}  {tel.slot_temp[1]:.1f}  {tel.slot_temp[2]:.1f}")
    put(215, f"IR zone L/M/R: {tel.slot_ir[0]}  {tel.slot_ir[1]}  {tel.slot_ir[2]}")
    put(245, f"Smoke norm: {tel.smoke_norm:.2f}")
    put(275, f"Hot slot: {tel.hot_slot}   LocalFire: {int(tel.local_fire_valid)}")
    put(305, f"BMS online: {int(tel.bms_online)}  BMS score: {tel.bms_overall_score:.2f}")

    put(355, f"Vision fire : {vis.slot_fire_conf[0]:.2f}  {vis.slot_fire_conf[1]:.2f}  {vis.slot_fire_conf[2]:.2f}")
    put(385, f"Vision smoke: {vis.slot_smoke_conf[0]:.2f}  {vis.slot_smoke_conf[1]:.2f}  {vis.slot_smoke_conf[2]:.2f}")
    put(415, f"Person present: {int(vis.person_present)}", (255, 80, 80) if vis.person_present else (80, 220, 80))

    put(465, f"Rail state: {tel.rail_state}")
    put(495, f"Rail pos: {tel.rail_pos}   Rail slot: {tel.rail_slot}")
    put(525, f"Homed: {int(tel.homed)}  Armed: {tel.armed_slot}")
    put(555, f"EStop: {int(tel.estop)}  PersonClear(ESP): {int(tel.person_clear)}")
    put(585, f"Release active: {int(tel.release_active)}  Done: {int(tel.release_done)}")

    put(635, f"Risk L/M/R: {res.slot_risk[0]:.2f}  {res.slot_risk[1]:.2f}  {res.slot_risk[2]:.2f}", (200, 240, 255), 0.72)

    y = 680
    put(y, "Reasons:", (120, 200, 255), 0.7)
    for r in res.reasons[:5]:
        y += 28
        put(y, f"- {r}", (220, 220, 220), 0.6, 1)

    # 左上角提示
    cv2.putText(canvas, "Q quit | H home | J/K/L move L/M/R | R release | P toggle person_clear", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


# ============================================================
# 7) 主程序
# ============================================================
def main():
    esp = ESPLink(SERIAL_PORT, SERIAL_BAUD, SERIAL_TIMEOUT)
    vision = VisionModule()
    controller = FusionController(esp)

    vision.open()

    person_clear_manual_override: Optional[bool] = None

    try:
        while True:
            esp.connect_if_needed()

            msgs = esp.read_messages()
            for obj in msgs:
                controller.update_telemetry(obj)

            vis = vision.poll()
            controller.update_vision(vis)

            # 人员状态支持键盘手动覆盖，方便你演示时没有 YOLO 也能测联锁
            if person_clear_manual_override is not None:
                vis.person_present = not person_clear_manual_override
                controller.update_vision(vis)

            controller.step()

            if SHOW_WINDOW:
                canvas = draw_dashboard(
                    vis.frame if vis.frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8),
                    controller.tel,
                    controller.vis,
                    controller.result,
                    esp.is_connected(),
                )
                cv2.imshow(WINDOW_NAME, canvas)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('h'):
                    esp.send({"cmd": "home"}, dedup_key="manual_home", min_interval_s=0.2)
                elif key == ord('j'):
                    esp.send({"cmd": "move_slot", "target": "L"}, dedup_key="manual_move_L", min_interval_s=0.2)
                elif key == ord('k'):
                    esp.send({"cmd": "move_slot", "target": "M"}, dedup_key="manual_move_M", min_interval_s=0.2)
                elif key == ord('l'):
                    esp.send({"cmd": "move_slot", "target": "R"}, dedup_key="manual_move_R", min_interval_s=0.2)
                elif key == ord('r'):
                    esp.send({"cmd": "release", "target": controller.result.target_slot}, dedup_key="manual_release", min_interval_s=0.5)
                elif key == ord('p'):
                    if person_clear_manual_override is None:
                        person_clear_manual_override = False
                    else:
                        person_clear_manual_override = not person_clear_manual_override
                    val = 1 if person_clear_manual_override else 0
                    esp.send({"cmd": "set_person_clear", "value": val}, dedup_key=f"manual_person_{val}", min_interval_s=0.1)
            else:
                time.sleep(0.01)

    finally:
        vision.close()
        esp.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

