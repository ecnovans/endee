"""
RAG terminal demo (no API keys).

Calls the FastAPI /ask endpoint and prints:
- User question
- Retrieved context (top chunks)
- Final answer

If Ollama is installed and running, the API will set used_llm=true.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import requests


def _post_json(url: str, body: Dict[str, Any]) -> Any:
    r = requests.post(
        url,
        data=json.dumps(body),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    api = "http://127.0.0.1:8000/ask"
    question = "What is semantic search?"
    top_k = 3

    data: Dict[str, Any] = _post_json(api, {"question": question, "top_k": top_k})
    sources: List[Dict[str, Any]] = data.get("sources") or []
    answer: str = (data.get("answer") or "").strip()
    used_llm: bool = bool(data.get("used_llm"))

    print("=" * 35)
    print("AI Knowledge Assistant (Endee)")
    print("=" * 35)
    print()
    print("User Query:")
    print(question)
    print()
    print("Retrieved Context:")

    if not sources:
        print("- (no sources returned)")
    else:
        for i, s in enumerate(sources, start=1):
            meta = s.get("metadata") or {}
            src = meta.get("source", "unknown")
            chunk_id = meta.get("chunk_id", "?")
            text = (s.get("text") or "").strip()
            preview = text if len(text) <= 240 else text[:237] + "..."
            print(f"- [{i}] {src} (chunk {chunk_id})")
            print(f"  {preview}")

    print()
    print("Final Answer:")
    print(answer if answer else "I don't know.")
    print()
    print(f"used_llm: {used_llm}")


if __name__ == "__main__":
    main()

