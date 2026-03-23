from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ThresholdConfig:
    temp_low: float = 50.0
    temp_high: float = 180.0
    gas_low: float = 0.10
    gas_high: float = 1.00
    fire_alarm_conf: float = 0.65
    smoke_alarm_conf: float = 0.55
    suspect_risk: float = 30.0
    prewarning_risk: float = 50.0
    alarm_risk: float = 75.0
    failsafe_temp: float = 140.0
    failsafe_smoke_sensor: float = 0.60


@dataclass
class RuntimeConfig:
    model_path: str = "models/best.pt"
    camera_source: str = "0"
    conf: float = 0.25
    imgsz: int = 640
    device: str = "0"
    infer_every: int = 2
    half: bool = False
    max_fps: float = 12.0
    show_window: bool = True
    stream_buffer: bool = False


@dataclass
class LoggingConfig:
    log_dir: Path = Path("logs")
    event_jsonl: str = "events.jsonl"


@dataclass
class AppConfig:
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
