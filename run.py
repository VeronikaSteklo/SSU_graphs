import subprocess
import time
import sys


def run_app():
    backend_process = subprocess.Popen(
        ["uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", "8000"]
    )

    time.sleep(3)

    try:
        subprocess.run(
            ["streamlit", "run", "frontend/app.py", "--server.port", "8501"]
        )
    except KeyboardInterrupt:
        backend_process.terminate()
        sys.exit()


if __name__ == "__main__":
    run_app()