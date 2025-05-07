import subprocess, os, sys, time, venv
from pathlib import Path

if getattr(sys, 'frozen', False):
    APP_ROOT = Path(sys.executable).parent.resolve()
else:
    APP_ROOT = Path(__file__).parent.resolve()
VENV_DIR = APP_ROOT / "venv"
PYTHON_EXE = VENV_DIR / "Scripts" / "python.exe"
REDIS_EXE = APP_ROOT / "Redis-x64-3.0.504" / "redis-server.exe"

def run(cmd, cwd=None):
    return subprocess.run(cmd, shell=True, cwd=cwd or APP_ROOT)

def ensure_venv():
    venv_python = APP_ROOT / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print(f"[ERROR] No valid Python executable found in: {venv_python}")
        print("Please run: python -m venv venv")
        input("Press Enter to exit...")
        sys.exit(1)
    print(f"[OK] Virtual environment detected at: {venv_python}")
    return venv_python

def install_requirements():
    print("Installing requirements...")
    run(f'"{PYTHON_EXE}" -m pip install --upgrade pip')
    run(f'"{PYTHON_EXE}" -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.0.1+cu118')
    run(f'"{PYTHON_EXE}" -m pip install -r requirements.txt')

def install_spacy_model():
    print("Installing spaCy model...")
    run(f'"{PYTHON_EXE}" -m spacy download en_core_web_trf')

def start_redis():
    print("Starting Redis...")
    subprocess.Popen(str(REDIS_EXE), cwd=REDIS_EXE.parent)

def start_celery():
    print("Starting Celery...")
    subprocess.Popen(
        f'cmd /k "{PYTHON_EXE} -m celery -A worker.celery worker -l info -P gevent"',
        cwd=APP_ROOT
    )

def start_flask():
    print("Starting Flask...")
    subprocess.Popen(
        f'cmd /k "{PYTHON_EXE} app.py"',
        cwd=APP_ROOT
    )

if __name__ == "__main__":
    PYTHON_EXE = ensure_venv()
    install_requirements()
    install_spacy_model()
    time.sleep(2)
    start_redis()
    time.sleep(3)
    start_celery()
    time.sleep(2)
    start_flask()
