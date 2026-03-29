@echo off
echo Starting Nepali OCR Gradio App...
cd /d "%~dp0src"
"C:\Program Files\Python310\python.exe" gradio_app.py
pause