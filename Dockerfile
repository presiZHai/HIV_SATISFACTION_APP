# ==========================
# Dockerfile
# ==========================
# Base image
# This Dockerfile sets up a Python environment with Uvicorn and Streamlit for serving a web application.
# It installs the necessary dependencies and exposes ports for both Uvicorn and Streamlit.
# It uses a slim version of Python 3.11 to keep the image size small.
# The application code is copied into the container, and the required Python packages are installed.
# The application will run Uvicorn on port 8000 and Streamlit on port 8501.
# The command to run both Uvicorn and Streamlit is specified at the end of the

FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

# Install dependencies
RUN pip install -r requirements.txt \
    && pip install streamlit shap python-dotenv streamlit-shap catboost scikit-learn pandas

EXPOSE 8000
EXPOSE 8501

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app/dashboard.py --server.port=8501 --server.enableCORS=false"]
# End of Dockerfile