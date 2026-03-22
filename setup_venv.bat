@echo off
cd /d "%~dp0"

echo Creating virtual environment...
python -m venv venv

if errorlevel 1 (
  echo Failed to create virtual environment.
  pause
  exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
  echo Failed to install dependencies in the virtual environment.
  pause
  exit /b 1
)

echo Virtual environment setup complete.
echo To use it later, run: venv\Scripts\activate
pause
