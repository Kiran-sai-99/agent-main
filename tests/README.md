# GenAI Agentic RAG — Test Suite

## Structure

```
tests/
  conftest.py           # Fixtures: client, sample docs, temp Chroma dir
  test_health.py        # GET /health
  test_azure.py         # GET /api/test-azure (success + auth failure mock)
  test_upload.py        # Upload: TXT/PDF/DOCX, 400 for bad type/large/empty
  test_retrieval.py     # Upload doc, query phrase, assert retrieval_used + sources
  test_general.py       # General knowledge (capital of France)
  test_hybrid.py        # Document + general (Project Phoenix + inflation)
  test_memory.py        # Multi-turn session
  test_clear.py         # DELETE /api/clear
  test_edge_cases.py    # Empty store, validation, auth mock
  test_performance.py   # Response time sanity
  sample_docs/          # Sample TXT for retrieval/hybrid
  mock_config_example.env.example
  MANUAL_QA_CHECKLIST.md
  EXPECTED_OUTPUTS.md
```

## Run tests

From project root:

```bash
# All tests (requires Azure configured and .env; LLM calls can be slow)
pytest tests/ -v

# Fast tests only (no LLM): health, upload validation, edge validation
pytest tests/test_health.py tests/test_upload.py::test_upload_unsupported_type_returns_400 tests/test_upload.py::test_upload_large_file_returns_400 tests/test_upload.py::test_upload_empty_file_returns_400 tests/test_edge_cases.py -v

# With coverage (optional)
pytest tests/ -v --cov=app --cov-report=term-missing
```

## Fixtures

- **client** — `TestClient(app)`. Cleans up (DELETE /api/clear) after each test.
- **chroma_dir** — Temporary Chroma path; set via `CHROMA_PERSIST_DIR` before app import.
- **sample_txt_file**, **sample_hybrid_txt_file** — `(filename, bytes)` for uploads.
- **sample_pdf_bytes**, **sample_docx_bytes** — Minimal PDF/DOCX for upload tests.
- **large_file_content** — >20 MB for size-limit test.

## Requirements

- **Integration tests** (upload with ingestion, query, chat, retrieval, hybrid, memory, clear content): require `LLM_PROVIDER=azure` and valid Azure env vars in `.env`.
- **Unit-style tests** (health, upload 400s, edge validation, Azure auth mock): do not require Azure.

## Manual testing

- **Curl script:** `./scripts/curl_tests.sh [BASE_URL]` (default `http://localhost:8000`). Start server first.
- **Checklist:** `tests/MANUAL_QA_CHECKLIST.md`
- **Expected outputs:** `tests/EXPECTED_OUTPUTS.md`
