@echo off
cd /d "%~dp0"

if "%~1"=="" (
  echo Usage: organize_fracatlas.bat "C:\path\to\extracted\FracAtlas"
  pause
  exit /b 1
)

if exist "venv\Scripts\activate.bat" (
  echo Activating virtual environment...
  call venv\Scripts\activate.bat
) else (
  echo Virtual environment not found. Using system Python...
)

python dataset\organize_fracatlas.py "%~1"
pause
