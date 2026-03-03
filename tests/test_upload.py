"""
Upload endpoint validation.
Valid PDF, DOCX, TXT; unsupported type → 400; large file → 400; empty file → 400.
"""
import io
import pytest


def test_upload_valid_txt(client, sample_txt_file):
    """Upload valid TXT returns 200 and chunk count."""
    filename, content = sample_txt_file
    response = client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(content), "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("document_name") == filename
    assert "chunks_created" in data
    assert data["chunks_created"] >= 1
    assert data.get("message", "").lower().find("success") >= 0 or data.get("message") == "Document ingested successfully."


def test_upload_valid_pdf(client, sample_pdf_bytes):
    """Upload valid PDF returns 200."""
    if len(sample_pdf_bytes) < 100:
        pytest.skip("PDF generation not available")
    response = client.post(
        "/api/upload",
        files={"file": ("test_upload.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("document_name") == "test_upload.pdf"
    assert "chunks_created" in data


def test_upload_valid_docx(client, sample_docx_bytes):
    """Upload valid DOCX returns 200."""
    if not sample_docx_bytes:
        pytest.skip("DOCX generation not available")
    response = client.post(
        "/api/upload",
        files={"file": ("test_upload.docx", io.BytesIO(sample_docx_bytes), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("document_name") == "test_upload.docx"
    assert "chunks_created" in data


def test_upload_unsupported_type_returns_400(client):
    """Upload unsupported file type (e.g. .exe, .csv) returns 400."""
    response = client.post(
        "/api/upload",
        files={"file": ("script.exe", io.BytesIO(b"fake"), "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    detail = data["detail"].lower()
    assert "invalid" in detail or "allowed" in detail or "file type" in detail


def test_upload_large_file_returns_400(client, large_file_content):
    """Upload file larger than 20MB returns 400."""
    response = client.post(
        "/api/upload",
        files={"file": ("large.txt", io.BytesIO(large_file_content), "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "large" in data["detail"].lower() or "20" in data["detail"]


def test_upload_empty_file_returns_400(client):
    """Upload empty file returns 400."""
    response = client.post(
        "/api/upload",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "empty" in data["detail"].lower()
