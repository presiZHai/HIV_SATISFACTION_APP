#=================
# run_local.sh - Script to run FastAPI and Streamlit locally
#=================
#!/bin/bash

echo "🚀 Starting FastAPI server on http://localhost:8000 ..."
uvicorn app.api:app --host 0.0.0.0 --port 8000 &

sleep 3

echo "📊 Launching Streamlit dashboard at http://localhost:8501 ..."
streamlit run streamlit_app/home.py \
    --server.port=8501 \
    --server.enableCORS=false