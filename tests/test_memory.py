"""
Multi-turn session memory.
Start session, ask question, follow-up referencing previous answer; assert context preserved.
"""
import pytest


def test_memory_follow_up(client):
    """Session: ask a fact, then follow-up referencing it; answer should show context preserved."""
    client.delete("/api/clear")
    session_id = "memory-test-abc"
    # First turn: ask something specific
    r1 = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "What is 3 times 7? Reply with only the number."},
    )
    assert r1.status_code == 200
    answer1 = r1.json().get("answer", "")
    assert "21" in answer1 or "21" in "".join(c for c in answer1 if c.isdigit())
    # Second turn: reference previous
    r2 = client.post(
        "/api/chat",
        json={"session_id": session_id, "message": "Double that number."},
    )
    assert r2.status_code == 200
    answer2 = r2.json().get("answer", "")
    # Should get 42 (21 * 2) if context was preserved
    assert "42" in answer2 or "42" in "".join(c for c in answer2 if c.isdigit())
