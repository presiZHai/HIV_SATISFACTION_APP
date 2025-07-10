# ==========================
# Main entry point for the application
# This script starts both the FastAPI server and the Streamlit dashboard.
# ==========================

import subprocess
subprocess.Popen(["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"])
subprocess.call([
   "streamlit", "run", "streamlit_app/dashboard.py",
   "--server.port=8501",
   "--server.enableCORS=false"
])