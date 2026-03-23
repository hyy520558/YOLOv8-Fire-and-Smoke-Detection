class SerialBMSProvider:
    """预留给 BMS 串口/网关接入。"""

    def __init__(self, transport):
        self.transport = transport

    def read(self):
        raise NotImplementedError("后续接入 BMS 时，在这里补充协议解析。")
