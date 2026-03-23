from typing import Dict, List, Tuple

from config import ThresholdConfig
from core.schema import DecisionResult, SensorFrame


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def norm_temp(max_temp, low=50.0, high=180.0) -> float:
    if max_temp is None:
        return 0.0
    return clamp01((float(max_temp) - low) / (high - low))


def norm_smoke_sensor(v, low=0.10, high=1.00) -> float:
    if v is None:
        return 0.0
    return clamp01((float(v) - low) / (high - low))


def norm_bms(voltage_drop_score, temp_score) -> float:
    a = float(voltage_drop_score or 0.0)
    b = float(temp_score or 0.0)
    return clamp01(0.6 * a + 0.4 * b)


def dynamic_weights(frame: SensorFrame) -> Dict[str, float]:
    w = {
        "bms": 0.35,
        "thermal": 0.30,
        "gas": 0.20,
        "vision": 0.15,
    }

    if (frame.bms_voltage_drop_score or 0.0) > 0.7:
        w["thermal"] += 0.10
        w["gas"] += 0.10
        w["vision"] -= 0.05
        w["bms"] -= 0.15

    if not frame.bms_online:
        w["bms"] = 0.0
        w["thermal"] += 0.15
        w["gas"] += 0.10
        w["vision"] += 0.10

    if (frame.fire_conf > 0.6) or ((frame.max_temp or 0.0) > 120):
        w["vision"] += 0.05
        w["thermal"] += 0.05
        w["gas"] -= 0.05
        w["bms"] -= 0.05

    total = sum(max(v, 0.0) for v in w.values())
    if total <= 0:
        return {k: 0.25 for k in w}
    return {k: max(v, 0.0) / total for k, v in w.items()}


def compute_risk(frame: SensorFrame, cfg: ThresholdConfig) -> Tuple[float, Dict[str, float], Dict[str, float], List[str]]:
    bms_score = norm_bms(frame.bms_voltage_drop_score, frame.bms_temp_score)
    thermal_score = norm_temp(frame.max_temp, cfg.temp_low, cfg.temp_high)
    gas_score = norm_smoke_sensor(frame.smoke_sensor_value, cfg.gas_low, cfg.gas_high)
    vision_score = max(frame.fire_conf, 0.7 * frame.smoke_conf)

    weights = dynamic_weights(frame)
    risk_score = 100.0 * (
        weights["bms"] * bms_score
        + weights["thermal"] * thermal_score
        + weights["gas"] * gas_score
        + weights["vision"] * vision_score
    )

    reasons: List[str] = []
    if frame.fire_conf > cfg.fire_alarm_conf:
        reasons.append("vision_fire")
    if frame.smoke_conf > cfg.smoke_alarm_conf:
        reasons.append("vision_smoke")
    if (frame.max_temp or 0.0) >= 100.0:
        reasons.append("thermal_hot")
    if (frame.smoke_sensor_value or 0.0) >= 0.60:
        reasons.append("gas_rising")
    if (frame.bms_voltage_drop_score or 0.0) > 0.7:
        reasons.append("bms_abnormal")
    if not frame.bms_online:
        reasons.append("bms_offline")

    normalized_scores = {
        "bms": round(bms_score, 4),
        "thermal": round(thermal_score, 4),
        "gas": round(gas_score, 4),
        "vision": round(vision_score, 4),
    }
    return risk_score, weights, normalized_scores, reasons


def build_command(state: str, risk_score: float, reasons: List[str]) -> Dict[str, object]:
    level = {
        "SAFE": "safe",
        "SUSPECT": "suspect",
        "PREWARNING": "prewarning",
        "ALARM": "high",
        "FAILSAFE_ALARM": "high",
    }.get(state, "safe")
    return {
        "cmd": "set_alarm",
        "level": level,
        "score": round(risk_score, 2),
        "reason": reasons,
    }


def fuse_and_decide(frame: SensorFrame, cfg: ThresholdConfig, decide_state_func) -> DecisionResult:
    risk_score, weights, normalized_scores, reasons = compute_risk(frame, cfg)
    state = decide_state_func(frame, risk_score, cfg)
    cmd = build_command(state, risk_score, reasons)
    return DecisionResult(
        risk_score=round(risk_score, 2),
        state=state,
        reasons=reasons,
        weights={k: round(v, 4) for k, v in weights.items()},
        normalized_scores=normalized_scores,
        command=cmd,
    )
