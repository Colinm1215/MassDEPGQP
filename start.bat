@echo off
setlocal

if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Running installer.exe...
start "" /B installer.exe

endlocal
pause
