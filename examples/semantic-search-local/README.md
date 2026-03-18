# Local Semantic Search with Endee (No API Keys)

This project is a **no-API-key** semantic search app built on top of the **Endee Vector Database**.

- **Vector DB**: Endee server (runs locally via Docker on Windows)
- **Embeddings**: Local HuggingFace model (`sentence-transformers/all-MiniLM-L6-v2`)
- **App**: Python scripts to ingest documents into Endee + a FastAPI server to search them

## Problem statement

Keyword search fails when the query uses different words than the document. This project enables **semantic search** (meaning-based search) over local text files by embedding both documents and queries and using Endee for fast vector retrieval.

## System design (simple)

- **Ingestion**
  - `data/*.txt` → chunk into small text blocks → embed locally → insert vectors into Endee index
- **Query**
  - user query → embed locally → Endee `/search` → return top-K most similar chunks

## How Endee is used

We call Endee’s HTTP API:

- **Create index**: `POST /api/v1/index/create`
- **Insert vectors**: `POST /api/v1/index/{index}/vector/insert` (JSON)
- **Search**: `POST /api/v1/index/{index}/search` (returns MessagePack)
- **Health**: `GET /api/v1/health`

## Prerequisites (Windows)

- Docker Desktop installed and running
- Python 3.10+ installed

## Step-by-step setup

### 1) Start Endee (Docker)

From the **Endee repo root** (`endee/`):

```bash
docker run --ulimit nofile=100000:100000 -p 8080:8080 -v ./endee-data:/data --name endee-server --restart unless-stopped endeeio/endee-server:latest
```

Verify:

```bash
curl http://localhost:8080/api/v1/health
```

### 2) Prepare sample documents

Put `.txt` files into:

`examples/semantic-search-local/data/`

Two starter files are included. You can replace them with your own.

### 3) Create Python environment + install dependencies

```bash
cd examples/semantic-search-local/backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 4) Ingest documents into Endee

```bash
python ingestion.py
```

This will:

- create an Endee index (if it doesn’t exist)
- embed your document chunks locally
- insert vectors into Endee

### 5) Run the API server

```bash
uvicorn app:app --reload --port 8000
```

Open:

- `http://localhost:8000/docs` (Swagger UI)

Try search:

```bash
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d "{\"query\":\"what is endee?\",\"top_k\":5}"
```

## Notes

- Endee runs on **port 8080** by default.
- The first run will download the embedding model (no API key required).
- If you enable Endee auth token (`NDD_AUTH_TOKEN`), set `ENDEE_AUTH_TOKEN` in `backend/.env` too.

