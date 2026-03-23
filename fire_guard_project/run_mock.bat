@echo off
cd /d %~dp0
python app.py --source none --mock-stdin
pause
