@echo off
echo Starting Upscalee servers...

REM Activate virtual environment and start backend
start cmd /k "call venv\Scripts\activate && cd backend && python app.py"

@REM REM Start frontend with admin PowerShell using full path and quoted directory
@REM start "" "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -Command "Start-Process '%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe' -ArgumentList '-NoExit', '-Command', 'Set-Location ''%~dp0frontend\react-app''; npm start' -Verb RunAs"

REM Start frontend normally
@REM start cmd /k "cd frontend\react-app && npm start"
start cmd /k "cd frontend\react-app && npm start"

echo Servers are starting in new windows...
echo Frontend will be available at: http://localhost:3000
echo Backend will be available at: http://localhost:5000
