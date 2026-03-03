# Manual QA Checklist — GenAI Agentic RAG

Use this checklist when validating the system manually (e.g. before release or after config changes).

---

## Prerequisites

- [ ] Server running: `uvicorn app.main:app --host 0.0.0.0 --port 8000` (or Docker)
- [ ] `.env` configured (Azure or chosen provider)
- [ ] No conflicting `AZURE_OPENAI_*` in shell (or start with clean env so `.env` is used)

---

## 1. Health

- [ ] **GET /health** → 200, body `{"status":"ok","service":"agentic-rag"}`
- [ ] Startup logs show correct Azure deployment (when `LLM_PROVIDER=azure`)

---

## 2. Azure Connectivity

- [ ] **GET /api/test-azure** → 200, `"status":"success"`, `"provider":"azure"` (when Azure configured)
- [ ] When provider is not Azure → 400 with message about setting `LLM_PROVIDER=azure`

---

## 3. Upload

- [ ] **POST /api/upload** valid **.txt** → 200, `chunks_created` ≥ 1
- [ ] **POST /api/upload** valid **.pdf** → 200
- [ ] **POST /api/upload** valid **.docx** → 200
- [ ] **POST /api/upload** unsupported type (e.g. .exe, .csv) → 400, message mentions allowed types
- [ ] **POST /api/upload** file > 20 MB → 400, message mentions size/large
- [ ] **POST /api/upload** empty file → 400

---

## 4. Retrieval

- [ ] Upload a document containing a **unique phrase**
- [ ] **POST /api/query** with a question that requires that phrase
- [ ] Response: `retrieval_used` = true, `sources` not empty, correct `document` name in sources
- [ ] `reasoning_trace` contains a step with `action` = `document_search`

---

## 5. General Knowledge

- [ ] **POST /api/query** with `"question": "What is the capital of France?"`
- [ ] Response: `retrieval_used` = false, `sources` = []
- [ ] Answer mentions Paris

---

## 6. Hybrid (Document + General)

- [ ] Upload a document containing e.g. "Project Phoenix budget was 5 million dollars"
- [ ] **POST /api/query** with "What was Project Phoenix budget and what is inflation?"
- [ ] Response: `retrieval_used` = true, `sources` not empty
- [ ] Answer uses both document (budget) and general knowledge (inflation)
- [ ] `reasoning_trace` includes `document_search` (and possibly `direct_llm`)

---

## 7. Memory (Multi-turn Chat)

- [ ] **POST /api/chat** with `session_id: "qa-session"`, message: "What is 3 times 7?"
- [ ] Answer contains 21
- [ ] **POST /api/chat** same `session_id`, message: "Double that number."
- [ ] Answer reflects context (e.g. 42)

---

## 8. Clear

- [ ] Upload a document, then **DELETE /api/clear** → 200
- [ ] **GET /api/documents** → `total` = 0
- [ ] **POST /api/query** with a question that would have used that document → no sources / `retrieval_used` false

---

## 9. Edge Cases

- [ ] **POST /api/query** with empty `question` → 422 or 400
- [ ] **POST /api/chat** with empty `message` or missing `session_id` → 422
- [ ] Query with **no documents** in store → 200, no crash; `retrieval_used` false or empty sources

---

## 10. Performance / Observability

- [ ] **GET /health** responds in &lt; 2 s
- [ ] **POST /api/query** (simple) completes in &lt; 30 s
- [ ] Server logs show request IDs and duration for query/chat

---

## Sign-off

- [ ] All items above passed (or documented exceptions)
- Date: _______________
- Tester: _______________
