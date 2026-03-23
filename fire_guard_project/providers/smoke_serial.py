class SerialSmokeProvider:
    """预留给烟雾/气体传感器串口接入。"""

    def __init__(self, transport):
        self.transport = transport

    def read(self):
        raise NotImplementedError("后续接入 ESP32 时，在这里补充按协议解析烟雾数据。")
