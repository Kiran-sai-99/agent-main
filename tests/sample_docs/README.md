# Sample test documents

Clone the repo to get these files:

```bash
git clone git@github.com:varunbiluri/agent.git
cd agent
```

- **retrieval_test.txt** — Contains unique phrase `AlphaBravoCharlieDelta2025UniqueMarker` and "10 million dollars" for retrieval tests.
- **hybrid_test.txt** — Contains "Project Phoenix budget was 5 million dollars" for hybrid (document + general) tests.

Used by pytest fixtures and by `scripts/curl_tests.sh` for manual curl testing.
