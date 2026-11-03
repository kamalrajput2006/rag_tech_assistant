#!/bin/sh
# Start FastAPI backend in background
echo "Starting FastAPI backend on port 8000..."
uvicorn src.api:app --host 127.0.0.1 --port 8000 &

# Give FastAPI a moment to start
sleep 3

# Start Streamlit frontend on configured port (defaults to 8501)
echo "Starting Streamlit frontend on port ${PORT:-8501}..."
streamlit run frontend/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
