"""
Pytest configuration and shared fixtures.
Sets isolated Chroma directory before app import; provides test client and sample documents.
"""
import io
import os
import tempfile
from pathlib import Path

import pytest

# Set test Chroma path before any app import so get_settings() sees it
_TEST_CHROMA_DIR = tempfile.mkdtemp(prefix="chroma_test_")
os.environ["CHROMA_PERSIST_DIR"] = _TEST_CHROMA_DIR

# Now import app (reads CHROMA_PERSIST_DIR from env)
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture(scope="session")
def chroma_dir():
    """Temporary Chroma directory used for the test session."""
    return _TEST_CHROMA_DIR


@pytest.fixture
def client():
    """Sync TestClient; triggers startup and cleans up vector store after each test."""
    with TestClient(app) as c:
        yield c
        # Clean up vector store and sessions after each test
        try:
            c.delete("/api/clear")
        except Exception:
            pass


@pytest.fixture
async def async_client():
    """Async httpx client for tests that prefer async. Startup runs via first request."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Trigger startup by hitting health (TestClient would have run it; with AsyncClient we ensure app is warm)
        await ac.get("/health")
        yield ac
        try:
            await ac.delete("/api/clear")
        except Exception:
            pass


# --- Sample document content (used for retrieval and hybrid tests) ---
RETRIEVAL_UNIQUE_PHRASE = "AlphaBravoCharlieDelta2025UniqueMarker"
RETRIEVAL_TEST_CONTENT = (
    f"Company annual report. The revenue target for 2025 is 10 million dollars. "
    f"Key phrase: {RETRIEVAL_UNIQUE_PHRASE}. End of section."
)

HYBRID_PROJECT_PHRASE = "Project Phoenix budget was 5 million dollars."
HYBRID_TEST_CONTENT = (
    "Q4 Planning Summary. Project Phoenix budget was 5 million dollars. "
    "Timeline: 12 months. Stakeholders approved."
)


@pytest.fixture
def sample_txt_content():
    """TXT file content for retrieval test."""
    return RETRIEVAL_TEST_CONTENT.encode("utf-8")


@pytest.fixture
def sample_txt_file(sample_txt_content):
    """TXT file as (filename, content) for upload."""
    return ("retrieval_test.txt", sample_txt_content)


@pytest.fixture
def sample_hybrid_txt_content():
    """TXT content for hybrid test (Project Phoenix + general)."""
    return HYBRID_TEST_CONTENT.encode("utf-8")


@pytest.fixture
def sample_hybrid_txt_file(sample_hybrid_txt_content):
    return ("hybrid_test.txt", sample_hybrid_txt_content)


@pytest.fixture
def sample_pdf_bytes():
    """Minimal valid PDF (one page) for upload test."""
    try:
        from pypdf import PdfWriter
        buf = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(612, 792)
        writer.write(buf)
        return buf.getvalue()
    except Exception:
        # Fallback: minimal PDF literal (valid structure)
        return (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
            b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF"
        )


@pytest.fixture
def sample_docx_bytes():
    """Minimal DOCX for upload test (one paragraph)."""
    try:
        from docx import Document
        doc = Document()
        doc.add_paragraph("Test document for upload validation.")
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()
    except Exception:
        # Return empty bytes; test will skip or fail gracefully
        return b""


@pytest.fixture
def large_file_content():
    """Content larger than MAX_FILE_SIZE (20 MB) for error test."""
    return b"x" * (21 * 1024 * 1024)
