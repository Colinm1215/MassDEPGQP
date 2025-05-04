@echo off
REM ------------------------------------------------------------------------
REM 1) Change to your Flask project directory
REM ------------------------------------------------------------------------
cd /d C:\Users\Colin\PycharmProjects\flaskProject

REM ------------------------------------------------------------------------
REM 2) Activate your virtual environment
REM ------------------------------------------------------------------------
call C:\Users\Colin\PycharmProjects\pythonProject\.venv\Scripts\activate.bat

REM ------------------------------------------------------------------------
REM 3) Pin to GPU 0 (optional but recommended)
REM ------------------------------------------------------------------------
set CUDA_VISIBLE_DEVICES=0

REM ------------------------------------------------------------------------
REM 4) Launch Celery via the venv’s Python interpreter
REM ------------------------------------------------------------------------
C:\Users\Colin\PycharmProjects\pythonProject\.venv\Scripts\python.exe -m celery -A worker.celery worker -l info -P gevent

REM ------------------------------------------------------------------------
REM End of script
REM ------------------------------------------------------------------------
pause
