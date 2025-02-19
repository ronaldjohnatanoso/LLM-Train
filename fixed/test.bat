@echo off
REM Change directory to the location of the batch file
cd /d %~dp0

REM Run Python script relative to the batch file location
python .\test.py
