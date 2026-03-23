from dataclasses import dataclass
from typing import Dict


@dataclass
class SmokeState:
    smoke_sensor_value: float | None = None
    gas_online: bool = False

    def apply_message(self, msg: Dict):
        if msg.get("type") != "smoke":
            return
        self.smoke_sensor_value = msg.get("smoke_sensor_value", self.smoke_sensor_value)
        self.gas_online = True

    def snapshot(self) -> Dict:
        return {
            "smoke_sensor_value": self.smoke_sensor_value,
            "gas_online": self.gas_online,
        }
