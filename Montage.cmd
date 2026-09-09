@echo off
rem Launch the Minecraft Montage desktop app.
rem pythonw = no console window; the app has its own log pane and error dialogs.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m mcmontage.gui
