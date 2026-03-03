#!/usr/bin/env bash
# Run the GenAI Agentic RAG API (from project root).
# Ensure Ollama is running: ollama serve && ollama pull llama3.2
set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
uvicorn app.main:app --host 0.0.0.0 --port 8000 "$@"
