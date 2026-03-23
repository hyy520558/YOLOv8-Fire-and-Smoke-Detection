from dataclasses import dataclass
from typing import Dict


@dataclass
class BMSState:
    bms_voltage_drop_score: float | None = None
    bms_temp_score: float | None = None
    bms_online: bool = False

    def apply_message(self, msg: Dict):
        if msg.get("type") != "bms":
            return
        self.bms_voltage_drop_score = msg.get("voltage_drop_score", self.bms_voltage_drop_score)
        self.bms_temp_score = msg.get("temp_score", self.bms_temp_score)
        self.bms_online = bool(msg.get("online", 1))

    def snapshot(self) -> Dict:
        return {
            "bms_voltage_drop_score": self.bms_voltage_drop_score,
            "bms_temp_score": self.bms_temp_score,
            "bms_online": self.bms_online,
        }
