@echo off
cd /d %~dp0
python app.py --model models\best.pt --source 0 --device 0 --imgsz 640 --infer-every 2 --mock-stdin
pause
