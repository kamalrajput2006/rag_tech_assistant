FROM python:3.10-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure startup script is executable
RUN chmod +x start.sh

# Streamlit port is 8501, FastAPI port is 8000
EXPOSE 8501 8000

ENV PORT=8501
ENV API_URL=http://127.0.0.1:8000

CMD ["./start.sh"]
