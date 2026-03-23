from dataclasses import dataclass
from typing import Dict


@dataclass
class ThermalState:
    min_temp: float | None = None
    max_temp: float | None = None
    avg_temp: float | None = None
    thermal_online: bool = False

    def apply_message(self, msg: Dict):
        if msg.get("type") != "thermal":
            return
        self.min_temp = msg.get("min_temp", self.min_temp)
        self.max_temp = msg.get("max_temp", self.max_temp)
        self.avg_temp = msg.get("avg_temp", self.avg_temp)
        self.thermal_online = True

    def snapshot(self) -> Dict:
        return {
            "min_temp": self.min_temp,
            "max_temp": self.max_temp,
            "avg_temp": self.avg_temp,
            "thermal_online": self.thermal_online,
        }
