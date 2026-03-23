import json
from typing import Dict, List

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover
    serial = None


class RealSerialTransport:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.01):
        if serial is None:
            raise RuntimeError("未安装 pyserial，请先 pip install pyserial")
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)

    def read_available(self) -> List[Dict]:
        out = []
        while self.ser.in_waiting:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
        return out

    def write_json(self, obj: Dict):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        self.ser.write(line.encode("utf-8"))

    def close(self):
        self.ser.close()
