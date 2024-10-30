@echo off
echo Starting Upscalee servers...

REM Activate virtual environment and start backend
start cmd /k "call venv\Scripts\activate && cd backend && python app.py"

REM Start frontend
start cmd /k "cd frontend/react-app && npm start"

echo Servers are starting in new windows...
echo Frontend will be available at: http://localhost:3000
echo Backend will be available at: http://localhost:5000
