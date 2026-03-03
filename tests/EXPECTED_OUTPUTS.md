# Expected outputs for tests

Reference for what each test or manual check should produce.

---

## 1. Health

**Request:** `GET /health`

**Expected:**  
- Status: `200`  
- Body: `{"status":"ok","service":"agentic-rag"}`  

---

## 2. Azure connectivity

**Request:** `GET /api/test-azure` (when `LLM_PROVIDER=azure` and credentials valid)

**Expected:**  
- Status: `200`  
- Body: `{"status":"success","response":"OK","deployment":"<your-deployment>","provider":"azure"}`  

**When provider not Azure:**  
- Status: `400`  
- Body: `{"detail":"Azure is not the configured provider. Set LLM_PROVIDER=azure and required Azure env vars."}`  

**When auth fails (wrong key/endpoint):**  
- Status: `500`  
- Body: `{"detail":"Azure authentication failed. Check endpoint, key, and deployment name."}`  

---

## 3. Upload

**Request:** `POST /api/upload` with valid `.txt` file

**Expected:**  
- Status: `200`  
- Body: `{"document_name":"<filename>","chunks_created":<n>,"message":"Document ingested successfully."}`  

**Unsupported type (e.g. .exe):**  
- Status: `400`  
- Body: `{"detail":"Invalid file type. Allowed: .pdf, .docx, .txt"}` (or similar)  

**File too large (> 20 MB):**  
- Status: `400`  
- Body: `{"detail":"File too large (max 20MB)."}`  

**Empty file:**  
- Status: `400`  
- Body: `{"detail":"Empty file."}`  

---

## 4. Retrieval test

**Setup:** Upload a document containing a unique phrase (e.g. `AlphaBravoCharlieDelta2025UniqueMarker`).

**Request:** `POST /api/query` with `{"question":"What is the revenue target or the key phrase ...?"}`  

**Expected:**  
- Status: `200`  
- `retrieval_used` = `true`  
- `sources` = non-empty list; each item has `document`, `chunk`, `score` (or similar)  
- `reasoning_trace` contains at least one step with `action` = `"document_search"`  
- `answer` reflects content from the uploaded document  

---

## 5. General knowledge test

**Request:** `POST /api/query` with `{"question":"What is the capital of France?"}`  

**Expected:**  
- Status: `200`  
- `retrieval_used` = `false`  
- `sources` = `[]`  
- `answer` contains "Paris" (or paris)  
- `reasoning_trace` may show `direct_llm` or similar  

---

## 6. Hybrid test

**Setup:** Upload doc with "Project Phoenix budget was 5 million dollars."

**Request:** `POST /api/query` with `{"question":"What was Project Phoenix budget and what is inflation?"}`  

**Expected:**  
- Status: `200`  
- `retrieval_used` = `true`  
- `sources` non-empty  
- `answer` mentions 5 million / Project Phoenix and something about inflation  
- `reasoning_trace` includes `document_search`  

---

## 7. Memory test

**Request 1:** `POST /api/chat` with `{"session_id":"abc","message":"What is 3 times 7?"}`  

**Expected:** Answer includes 21.  

**Request 2:** `POST /api/chat` with `{"session_id":"abc","message":"Double that number."}`  

**Expected:** Answer includes 42 (context preserved).  

---

## 8. Clear

**Request:** `DELETE /api/clear`  

**Expected:**  
- Status: `200`  
- Body: `{"message":"Vector store and sessions cleared."}` (or similar)  

**Then** `GET /api/documents`:  
- `total` = `0`, `documents` = `[]`  

**Then** query that previously used a document:  
- `retrieval_used` = `false` or `sources` = `[]`  

---

## 9. Edge cases

- **POST /api/query** with `{"question":""}` → `422` (validation error).  
- **POST /api/chat** without `session_id` or with empty `message` → `422`.  
- Query with **empty vector store** → `200`, `answer` present, `retrieval_used` false or `sources` empty.  

---

## 10. Performance

- **GET /health:** response time &lt; 2 s.  
- **POST /api/query** (simple question): response time &lt; 30 s.  
