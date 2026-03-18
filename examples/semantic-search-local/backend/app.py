from __future__ import annotations

import json
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL_NAME, ENDEE_AUTH_TOKEN, ENDEE_BASE_URL, ENDEE_INDEX_NAME
from endee_http import EndeeHttpClient
from models import AskRequest, AskResponse, HealthResponse, SearchRequest, SearchResult
from rag import rag_answer

app = FastAPI(title="Local Semantic Search with Endee", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

endee = EndeeHttpClient(ENDEE_BASE_URL, auth_token=ENDEE_AUTH_TOKEN)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


@app.get("/")
def root():
    return {
        "message": "Local Semantic Search API is running.",
        "try": {
            "docs": "/docs",
            "health": "/health",
            "search": {"method": "POST", "path": "/search", "body": {"query": "what is Endee?", "top_k": 5}},
            "ask": {"method": "POST", "path": "/ask", "body": {"question": "What is Endee?", "top_k": 5}},
        },
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        endee_ok=endee.health(),
        endee_base_url=ENDEE_BASE_URL,
        index_name=ENDEE_INDEX_NAME,
    )


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest) -> List[SearchResult]:
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is required")

    if req.top_k < 1 or req.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    # Embed query locally
    q_vec = embed_model.encode([q], normalize_embeddings=False)[0].tolist()

    try:
        hits = endee.search_dense(ENDEE_INDEX_NAME, q_vec, k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Endee search failed: {e}")

    out: List[SearchResult] = []
    for h in hits:
        meta = h.meta_json()
        text = meta.get("text", "") if isinstance(meta, dict) else ""
        out.append(
            SearchResult(
                id=h.id,
                score=h.similarity,
                text=text,
                metadata=meta if isinstance(meta, dict) else {"_raw_meta": str(meta)},
            )
        )
    return out


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if req.top_k < 1 or req.top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    # Reuse search logic to fetch sources
    q_vec = embed_model.encode([question], normalize_embeddings=False)[0].tolist()
    try:
        hits = endee.search_dense(ENDEE_INDEX_NAME, q_vec, k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Endee search failed: {e}")

    sources: List[SearchResult] = []
    for h in hits:
        meta = h.meta_json()
        text = meta.get("text", "") if isinstance(meta, dict) else ""
        sources.append(
            SearchResult(
                id=h.id,
                score=h.similarity,
                text=text,
                metadata=meta if isinstance(meta, dict) else {"_raw_meta": str(meta)},
            )
        )

    answer, used_llm = rag_answer(endee, question=question, sources=sources)
    return AskResponse(answer=answer, used_llm=used_llm, sources=sources)

