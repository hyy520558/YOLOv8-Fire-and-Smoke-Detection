import json
import time
from pathlib import Path
from typing import Dict


class JSONLLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: Dict):
        payload = dict(payload)
        payload.setdefault("logged_at", time.time())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
