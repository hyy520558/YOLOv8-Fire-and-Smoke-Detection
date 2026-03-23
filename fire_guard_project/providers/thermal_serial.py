class SerialThermalProvider:
    """预留给 ESP32 热成像串口接入。当前主流程直接通过 transport 解析 JSON。"""

    def __init__(self, transport):
        self.transport = transport

    def read(self):
        raise NotImplementedError("后续接入 ESP32 时，在这里补充按协议解析热成像数据。")
