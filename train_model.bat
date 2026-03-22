@echo off
cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
  echo Activating virtual environment...
  call venv\Scripts\activate.bat
) else (
  echo Virtual environment not found. Using system Python...
)

echo Checking required packages...
python -m pip install -r requirements.txt

if errorlevel 1 (
  echo Failed to install dependencies.
  pause
  exit /b 1
)

echo Starting model training...
python model\train_mobilenetv2.py

pause
