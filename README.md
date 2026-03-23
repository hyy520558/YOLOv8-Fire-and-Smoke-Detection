# Fire Guard PC Twin

这是一个先在电脑上跑通的“火灾预警数字孪生”工程骨架，目标是：

1. 先把 **YOLO 视觉 + 传感器占位接口 + 融合决策 + 模拟串口** 跑通。
2. 后续再把模拟串口替换成 **ESP32 真串口**。
3. 最后再迁移到 **Jetson Nano**，并用低功耗参数运行。

## 目录说明

```text
fire_guard/
├─ app.py
├─ config.py
├─ requirements.txt
├─ run_mock.bat
├─ run_replay.bat
├─ run_webcam.bat
├─ core/
├─ providers/
├─ transport/
├─ actuators/
└─ tests/
```

## 安装

```bash
conda activate yolo_fire
pip install -r requirements.txt
```

如果你已经在自己的 YOLO 环境里装过 `ultralytics` 和 `opencv-python`，通常就够了。

## 先跑最小流程

### 1）纯模拟回放，不开摄像头

```bash
python app.py --source none --mock-replay tests/replay_case1.jsonl
```

你会看到控制台里风险分数和状态变化：`SAFE -> SUSPECT -> PREWARNING -> ALARM`。

### 2）纯模拟串口，手工输入 JSON

```bash
python app.py --source none --mock-stdin
```

然后在终端输入：

```json
{"type":"thermal","min_temp":30,"max_temp":160,"avg_temp":80}
{"type":"smoke","smoke_sensor_value":0.72}
{"type":"bms","voltage_drop_score":0.85,"temp_score":0.60,"online":1}
```

### 3）开摄像头 + YOLO + 模拟串口

先把你的火焰烟雾权重放到：

```text
models/best.pt
```

然后运行：

```bash
python app.py --model models/best.pt --source 0 --device 0 --imgsz 640 --infer-every 2 --mock-stdin
```

再在终端输入 JSON 模拟热成像/烟雾/BMS 数据。

> 如果你误用了 `yolov8n.pt` 这种 COCO 通用模型，程序会提示当前模型类别里未发现 `fire/smoke`。

## 未来接 ESP32 时的串口协议建议

### ESP32 -> 上位机

```json
{"type":"thermal","min_temp":31.2,"max_temp":162.8,"avg_temp":58.3,"relay":1}
{"type":"smoke","smoke_sensor_value":0.76}
{"type":"bms","voltage_drop_score":0.85,"temp_score":0.62,"online":1}
```

### 上位机 -> ESP32

```json
{"cmd":"set_alarm","level":"high","score":84.2,"reason":["vision_fire","thermal_hot"]}
```

## Jetson Nano 降载建议

第一次迁移 Nano 时，建议从这组参数起步：

```bash
python app.py --model models/best.pt --source 0 --device 0 --imgsz 416 --infer-every 3 --max-fps 8 --mock-stdin
```

如果你后面要继续减负，可以做这几件事：

- 把 `imgsz` 再降到 `320`
- 把 `infer-every` 提到 `4`
- 用导出的 TensorRT 引擎替换 `best.pt`
- 纯边缘场景下只保留 `fire/smoke` 两类

## 当前已经具备的能力

- `SensorFrame` 统一数据结构
- 热成像/烟雾/BMS 占位接口
- YOLO 实时视觉接口
- 动态加权风险评分
- 状态机：`SAFE / SUSPECT / PREWARNING / ALARM / FAILSAFE_ALARM`
- 模拟串口和回放测试
- 日志输出到 `logs/events.jsonl`

## 下一步你最可能会做的两件事

1. 把你的 ESP32 热成像代码改成串口输出 JSON 行。
2. 把 `MockSerialTransport` 换成 `RealSerialTransport(COMx)`。

## ESP32 侧建议输出函数

你后面可以把板端热成像代码补成：

```cpp
void sendThermalToHost() {
  Serial.print("{\"type\":\"thermal\",");
  Serial.print("\"min_temp\":"); Serial.print(min_temp, 1); Serial.print(",");
  Serial.print("\"max_temp\":"); Serial.print(max_temp, 1); Serial.print(",");
  Serial.print("\"avg_temp\":"); Serial.print(avg_temp, 1); Serial.print(",");
  Serial.print("\"relay\":"); Serial.print(digitalRead(RELAY_PIN));
  Serial.println("}");
}
```

这样上位机就能直接吃你板子发来的数据，不用再改协议。
