from __future__ import annotations

import json
from typing import List, Tuple

import requests

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from endee_http import EndeeHttpClient
from models import SearchResult


def _build_context(sources: List[SearchResult], max_chars: int = 2500) -> str:
    parts: List[str] = []
    used = 0
    for i, s in enumerate(sources, start=1):
        src = s.metadata.get("source", "unknown")
        chunk_id = s.metadata.get("chunk_id", "?")
        header = f"[{i}] source={src} chunk={chunk_id}"
        body = (s.text or "").strip()
        block = f"{header}\n{body}".strip()
        if not block:
            continue
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts).strip()


def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _generate_with_ollama(question: str, context: str) -> str:
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context.\n"
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def _extractive_answer(question: str, sources: List[SearchResult]) -> str:
    # Simple fallback: return the best chunk as the "answer" plus a short note.
    if not sources:
        return "I don't know."
    best = sources[0]
    src = best.metadata.get("source", "unknown")
    text = (best.text or "").strip()
    if not text:
        return "I don't know."
    return f"Based on the documents (source: {src}), the most relevant info is:\n\n{text}"


def rag_answer(
    endee: EndeeHttpClient,
    question: str,
    sources: List[SearchResult],
) -> Tuple[str, bool]:
    """
    Returns (answer, used_llm).
    If Ollama is running locally, we use it (true RAG).
    Otherwise we fall back to extractive answering (still retrieval-based).
    """
    context = _build_context(sources)
    if context and _ollama_available():
        ans = _generate_with_ollama(question, context)
        if ans:
            return ans, True
    return _extractive_answer(question, sources), False

