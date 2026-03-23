import json
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional


class MockSerialTransport:
    """从终端实时输入 JSON，模拟串口收到的消息。"""

    def __init__(self):
        self._queue: "queue.Queue[Dict]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        examples = [
            '{"type":"thermal","min_temp":30,"max_temp":160,"avg_temp":80}',
            '{"type":"smoke","smoke_sensor_value":0.72}',
            '{"type":"bms","voltage_drop_score":0.85,"temp_score":0.60,"online":1}',
        ]
        print("\n[MOCK SERIAL] 已启动。可在终端输入 JSON 模拟串口数据，例如：")
        for line in examples:
            print("  ", line)
        print("输入 quit 可结束输入线程；主程序仍可继续运行。\n")

        while not self._stop.is_set():
            try:
                text = input("mock-serial> ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text.lower() in {"quit", "exit"}:
                break
            try:
                msg = json.loads(text)
                if isinstance(msg, dict):
                    self._queue.put(msg)
                else:
                    print("[MOCK SERIAL] 请输入 JSON 对象，而不是数组或其他类型。")
            except Exception as e:
                print(f"[MOCK SERIAL] JSON 解析失败: {e}")

    def read_available(self) -> List[Dict]:
        items: List[Dict] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def write_json(self, obj: Dict):
        print("[MOCK TX]", json.dumps(obj, ensure_ascii=False))

    def close(self):
        self._stop.set()


class ReplayEventSource:
    """按 after_ms 定时注入 JSON 事件。"""

    def __init__(self, jsonl_path: str):
        self.jsonl_path = Path(jsonl_path)
        self.events = []
        self.start_time = None
        self.index = 0
        self._load()

    def _load(self):
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.events.append(obj)
        self.events.sort(key=lambda x: x.get("after_ms", 0))

    def start(self):
        self.start_time = time.time()

    def read_available(self) -> List[Dict]:
        if self.start_time is None:
            self.start()
        elapsed_ms = (time.time() - self.start_time) * 1000.0
        out: List[Dict] = []
        while self.index < len(self.events) and self.events[self.index].get("after_ms", 0) <= elapsed_ms:
            event = dict(self.events[self.index])
            event.pop("after_ms", None)
            out.append(event)
            self.index += 1
        return out

    def write_json(self, obj: Dict):
        print("[REPLAY TX]", json.dumps(obj, ensure_ascii=False))

    @property
    def exhausted(self) -> bool:
        return self.index >= len(self.events)
