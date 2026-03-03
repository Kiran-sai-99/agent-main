# GenAI Agentic RAG - Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
# Optional: copy .env for default config (override with env at run time)
# COPY .env .env

# Persist Chroma data on a volume
ENV CHROMA_PERSIST_DIR=/data/chroma
VOLUME /data

EXPOSE 8000

# Run with uvicorn; bind to 0.0.0.0 for Docker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
