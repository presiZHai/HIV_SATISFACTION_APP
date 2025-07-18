# ==========================
# Main entry point for the application: main.py
# This script starts both the FastAPI server and the Streamlit dashboard.
# ==========================
# main.py

import subprocess
import logging
import time
import requests
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

FASTAPI_URL = "http://localhost:8000"
MAX_RETRIES = 15
RETRY_INTERVAL = 1  # seconds

def start_fastapi():
    logging.info("Starting FastAPI server...")
    return subprocess.Popen([
        "uvicorn", "app.api:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])

def wait_for_api():
    logging.info("Waiting for FastAPI to be available...")
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{FASTAPI_URL}/", timeout=2)
            if response.status_code == 200:
                logging.info("FastAPI is up!")
                return True
        except requests.RequestException:
            logging.debug(f"FastAPI not up yet (attempt {attempt+1})")
        time.sleep(RETRY_INTERVAL)
    return False

def start_streamlit():
    logging.info("Starting Streamlit dashboard...")
    subprocess.call([
        "streamlit", "run", "streamlit_app/Home.py",
        "--server.port=8501",
        "--server.enableCORS=false"
    ])

if __name__ == "__main__":
    uvicorn_proc = start_fastapi()

    if not wait_for_api():
        logging.error("FastAPI failed to start after multiple attempts.")
        uvicorn_proc.terminate()
        sys.exit(1)

    try:
        start_streamlit()
    finally:
        logging.info("Shutting down FastAPI server.")
        uvicorn_proc.terminate()