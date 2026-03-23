from config import ThresholdConfig
from core.schema import SensorFrame


SAFE = "SAFE"
SUSPECT = "SUSPECT"
PREWARNING = "PREWARNING"
ALARM = "ALARM"
FAILSAFE_ALARM = "FAILSAFE_ALARM"


def decide_state(frame: SensorFrame, risk_score: float, cfg: ThresholdConfig) -> str:
    if (
        (not frame.bms_online)
        and ((frame.max_temp or 0.0) >= cfg.failsafe_temp)
        and ((frame.smoke_sensor_value or 0.0) >= cfg.failsafe_smoke_sensor)
    ):
        return FAILSAFE_ALARM

    if (frame.fire_conf >= cfg.fire_alarm_conf) and ((frame.max_temp or 0.0) >= 100.0):
        return ALARM

    if (frame.smoke_conf >= cfg.smoke_alarm_conf) and ((frame.smoke_sensor_value or 0.0) >= 0.50):
        return PREWARNING

    if risk_score >= cfg.alarm_risk:
        return ALARM
    if risk_score >= cfg.prewarning_risk:
        return PREWARNING
    if risk_score >= cfg.suspect_risk:
        return SUSPECT
    return SAFE
