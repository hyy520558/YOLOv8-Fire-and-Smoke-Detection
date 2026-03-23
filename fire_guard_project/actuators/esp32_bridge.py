from __future__ import annotations

from core.schema import DecisionResult


class ESP32Bridge:
    """通过 transport 向 ESP32 发送指令。当前 transport 可是真串口，也可是假串口。"""

    def __init__(self, transport):
        self.transport = transport

    def handle(self, decision: DecisionResult):
        self.transport.write_json(decision.command)
