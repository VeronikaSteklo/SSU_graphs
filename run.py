import subprocess
import sys
import os
import signal
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSES = []


def ensure_directories():
    for folder in ["data", "models", "backend", "frontend"]:
        os.makedirs(os.path.join(ROOT_DIR, folder), exist_ok=True)

    for pkg in ["backend", "frontend"]:
        init_path = os.path.join(ROOT_DIR, pkg, "__init__.py")
        if not os.path.exists(init_path):
            open(init_path, "w").close()


def terminate_processes():
    print("\nЗавершение процессов.")
    for proc in PROCESSES:
        if proc.poll() is None:
            proc.terminate()

    for proc in PROCESSES:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("Все процессы остановлены.")
    sys.exit(0)


def main():
    ensure_directories()

    backend_cmd = [
        sys.executable, "-m", "uvicorn", "backend.main:app",
        "--host", "0.0.0.0", "--port", "8000", "--log-level", "info",
        "--reload"
    ]
    backend_proc = subprocess.Popen(backend_cmd, cwd=ROOT_DIR, env=os.environ)
    PROCESSES.append(backend_proc)

    time.sleep(2)

    frontend_cmd = [
        sys.executable, "-m", "streamlit", "run", "frontend/main.py",
        "--server.port", "8501", "--server.headless", "true",
        "--server.runOnSave", "false"
    ]
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=ROOT_DIR, env=os.environ)
    PROCESSES.append(frontend_proc)

    signal.signal(signal.SIGINT, lambda s, f: terminate_processes())
    signal.signal(signal.SIGTERM, lambda s, f: terminate_processes())

    try:
        while True:
            time.sleep(1)
            for proc in PROCESSES:
                if proc.poll() is not None:
                    print(f"Процесс завершился с кодом {proc.returncode}")
                    terminate_processes()
    except KeyboardInterrupt:
        terminate_processes()


if __name__ == "__main__":
    main()