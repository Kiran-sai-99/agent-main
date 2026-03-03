#!/usr/bin/env bash
# Manual curl-based test script for GenAI Agentic RAG API.
# Run with server up: uvicorn app.main:app --host 0.0.0.0 --port 8000
# Usage: ./scripts/curl_tests.sh [BASE_URL]
# Default BASE_URL=http://localhost:8000

set -e
BASE_URL="${1:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SAMPLE_DOCS="$PROJECT_ROOT/tests/sample_docs"

echo "=== Base URL: $BASE_URL ==="

# 1) Health
echo ""
echo "--- 1) GET /health ---"
curl -s -w "\nHTTP %{http_code}\n" "$BASE_URL/health"
echo ""

# 2) Azure connectivity
echo "--- 2) GET /api/test-azure ---"
curl -s -w "\nHTTP %{http_code}\n" "$BASE_URL/api/test-azure"
echo ""

# 3) Upload TXT
echo "--- 3) POST /api/upload (TXT) ---"
if [ -f "$SAMPLE_DOCS/retrieval_test.txt" ]; then
  curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE_URL/api/upload" \
    -F "file=@$SAMPLE_DOCS/retrieval_test.txt"
else
  echo "Sample doc not found; creating inline."
  echo "AlphaBravoCharlieDelta2025UniqueMarker. Revenue 10 million." | curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE_URL/api/upload" \
    -F "file=@-;filename=retrieval_test.txt;type=text/plain"
fi
echo ""

# 4) Documents list
echo "--- 4) GET /api/documents ---"
curl -s -w "\nHTTP %{http_code}\n" "$BASE_URL/api/documents"
echo ""

# 5) Query (retrieval)
echo "--- 5) POST /api/query (retrieval) ---"
curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the revenue target or the key phrase AlphaBravoCharlieDelta2025UniqueMarker?"}'
echo ""

# 6) Query (general)
echo "--- 6) POST /api/query (general) ---"
curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE_URL/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
echo ""

# 7) Chat
echo "--- 7) POST /api/chat ---"
curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "curl-test-session", "message": "What is 5 + 7?"}'
echo ""

# 8) Upload unsupported type (expect 400)
echo "--- 8) POST /api/upload (unsupported .exe) - expect 400 ---"
echo "fake" | curl -s -w "\nHTTP %{http_code}\n" -X POST "$BASE_URL/api/upload" \
  -F "file=@-;filename=bad.exe;type=application/octet-stream"
echo ""

# 9) Clear
echo "--- 9) DELETE /api/clear ---"
curl -s -w "\nHTTP %{http_code}\n" -X DELETE "$BASE_URL/api/clear"
echo ""

# 10) Documents after clear
echo "--- 10) GET /api/documents (after clear) ---"
curl -s -w "\nHTTP %{http_code}\n" "$BASE_URL/api/documents"
echo ""

echo "=== curl tests finished ==="
