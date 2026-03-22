# FractureAI Deployment Guide

## Recommended Platform

Render with Docker is the safest option for this project because it uses:

- Flask
- TensorFlow
- OpenCV
- ReportLab
- SQLite file storage

## Files Added for Deployment

- `Dockerfile`
- `.dockerignore`
- `render.yaml`

## Deploy on Render

1. Push this project to GitHub.
2. Create a Render account.
3. Click `New +` -> `Blueprint`.
4. Connect your GitHub repository.
5. Render will detect `render.yaml` and create the web service.
6. After the deploy finishes, open the generated URL.

## Important Notes

- SQLite, uploaded files, reports, and saved history need persistent storage.
- This project uses a Render disk mounted at `/data`.
- Runtime files are stored through `FRACTUREAI_DATA_DIR=/data`.
- If you want email sending in production, set these environment variables in Render:
  - `FRACTUREAI_SMTP_SERVER`
  - `FRACTUREAI_SMTP_PORT`
  - `FRACTUREAI_SMTP_USERNAME`
  - `FRACTUREAI_SMTP_PASSWORD`
  - `FRACTUREAI_SENDER_EMAIL`

## Default Login

- `admin / admin123`
- `doctor / doctor123`
