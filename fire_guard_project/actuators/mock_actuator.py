from __future__ import annotations

from typing import Optional

from core.schema import DecisionResult


class MockActuatorController:
    def __init__(self):
        self.last_state: Optional[str] = None

    def handle(self, decision: DecisionResult):
        if decision.state != self.last_state:
            print(
                f"[ACTUATOR] 状态切换: {self.last_state or 'None'} -> {decision.state} | "
                f"risk={decision.risk_score:.2f} | reasons={decision.reasons}"
            )
            self.last_state = decision.state
