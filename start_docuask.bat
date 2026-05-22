@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Expected file: .venv\Scripts\python.exe
  echo.
  echo Recreate it with:
  echo   C:\Users\LENOVO\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

if "%ANTHROPIC_API_KEY%"=="" (
echo Warning: ANTHROPIC_API_KEY is not set.
echo Uploading PDFs will work, but asking questions will fail until you set the key.
echo.
)

echo Starting DocuAsk server...
echo Your browser should open automatically in a moment.
echo Keep this window open while using the app.
echo.

set DOCUASK_AUTO_OPEN=1
".venv\Scripts\python.exe" app.py

echo.
echo DocuAsk server stopped.
echo If the app did not open, copy the message shown above from this window.
echo.
pause
