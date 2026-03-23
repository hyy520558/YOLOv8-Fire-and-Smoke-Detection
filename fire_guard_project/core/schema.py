from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SensorFrame:
    timestamp: float
    frame_id: int = 0

    fire_conf: float = 0.0
    smoke_conf: float = 0.0
    has_fire: bool = False
    has_smoke: bool = False
    fps: float = 0.0

    min_temp: Optional[float] = None
    max_temp: Optional[float] = None
    avg_temp: Optional[float] = None
    thermal_online: bool = False

    smoke_sensor_value: Optional[float] = None
    gas_online: bool = False

    bms_voltage_drop_score: Optional[float] = None
    bms_temp_score: Optional[float] = None
    bms_online: bool = False

    source: str = "mock"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionResult:
    risk_score: float
    state: str
    reasons: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    normalized_scores: Dict[str, float] = field(default_factory=dict)
    command: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualResult:
    fire_conf: float = 0.0
    smoke_conf: float = 0.0
    has_fire: bool = False
    has_smoke: bool = False
    annotated_frame: Any = None
    raw_detections: List[Dict[str, Any]] = field(default_factory=list)
    class_names: Dict[int, str] = field(default_factory=dict)
    warning: Optional[str] = None
