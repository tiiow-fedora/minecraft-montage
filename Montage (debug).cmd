@echo off
rem Same app, but with a console so startup errors are visible.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m mcmontage.gui
echo.
echo (exit code %errorlevel%)
pause
