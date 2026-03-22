@echo off
cd /d "%~dp0"

set "FRACTUREAI_SMTP_SERVER=smtp.gmail.com"
set "FRACTUREAI_SMTP_PORT=587"
set "FRACTUREAI_SMTP_USERNAME=your_email@gmail.com"
set "FRACTUREAI_SMTP_PASSWORD=your_16_character_app_password"
set "FRACTUREAI_SENDER_EMAIL=your_email@gmail.com"

echo Starting FractureAI with email configuration...
"C:\Users\anujm\AppData\Local\Programs\Python\Python311\python.exe" backend\app.py

pause
