@echo off
cd /d %~dp0
python app.py --source none --mock-replay tests\replay_case1.jsonl
pause
