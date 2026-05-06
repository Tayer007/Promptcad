@echo off
echo === PromptCAD Local Setup ===

py -3.10 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3.10 not found.
    echo Install it from: https://www.python.org/downloads/release/python-31011/
    pause
    exit /b 1
)

echo Creating virtual environment with Python 3.10...
py -3.10 -m venv cad_env

echo Activating environment...
call cad_env\Scripts\activate.bat

echo Installing PyTorch ^(CPU^)...
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

echo Installing requirements...
pip install -r requirements_full.txt

echo.
echo Setup complete. To run the app:
echo   cad_env\Scripts\activate
echo   set HF_REPO=fourat25/mesh-to-cadquery-qwen
echo   python app.py
pause
