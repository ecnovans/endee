# Local Semantic Search + RAG with Endee (No API Keys)

This project is a **no-API-key** semantic search app built on top of the **Endee Vector Database**.

- **Vector DB**: Endee server (runs locally via Docker on Windows)
- **Embeddings**: Local HuggingFace model (`sentence-transformers/all-MiniLM-L6-v2`)
- **App**: Python scripts to ingest documents into Endee + a FastAPI server to search/ask questions (RAG)

## Problem statement

Keyword search fails when the query uses different words than the document. This project enables **semantic search** (meaning-based search) over local text files by embedding both documents and queries and using Endee for fast vector retrieval.

## System design (simple)

- **Ingestion**
  - `data/*.txt` → chunk into small text blocks → embed locally → insert vectors into Endee index
- **Query**
  - user query → embed locally → Endee `/search` → return top-K most similar chunks

## RAG (Retrieval Augmented Generation)

This repo includes a simple RAG endpoint:

- `POST /ask`: retrieves top-K chunks from Endee, then produces an answer.

**No API keys required:**

- If you install **Ollama** locally, `/ask` becomes a **true generative RAG** pipeline (local LLM).
- If Ollama is not installed, `/ask` falls back to an **extractive** answer based on the best retrieved chunk.

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

Try RAG ask:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"what is endee?\",\"top_k\":5}"
```

### Visible terminal output (for evaluation)

Minimum semantic search output:

```bash
python query.py
```

RAG-style output (retrieves context + returns final answer):

```bash
python ask_cli.py
```

### (Optional) Enable local LLM generation with Ollama

1) Install Ollama (Windows) and restart your terminal.
2) Pull a small model:

```bash
ollama pull llama3.2:3b
```

3) Keep Ollama running (default: `http://localhost:11434`).
4) Call `/ask` again. It will return `"used_llm": true`.

## Notes

- Endee runs on **port 8080** by default.
- The first run will download the embedding model (no API key required).
- If you enable Endee auth token (`NDD_AUTH_TOKEN`), set `ENDEE_AUTH_TOKEN` in `backend/.env` too.

