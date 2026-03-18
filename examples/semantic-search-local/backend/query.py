"""
Semantic search terminal demo (no API keys).

Shows clear output:
- User query
- Top retrieved results
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
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    api = "http://127.0.0.1:8000/search"
    query = "What is semantic search?"
    top_k = 3

    results: List[Dict[str, Any]] = _post_json(api, {"query": query, "top_k": top_k})

    print("=" * 35)
    print("AI Knowledge Assistant (Endee)")
    print("=" * 35)
    print()
    print("User Query:")
    print(query)
    print()
    print("Top Retrieved Results:")

    if not results:
        print("No results.")
        return

    for i, item in enumerate(results, start=1):
        text = (item.get("text") or "").strip().replace("\n", " ")
        src = (item.get("metadata") or {}).get("source", "unknown")
        score = item.get("score")
        preview = text if len(text) <= 180 else text[:177] + "..."
        print(f"{i}. ({src}, score={score:.4f}) {preview}" if isinstance(score, (int, float)) else f"{i}. ({src}) {preview}")


if __name__ == "__main__":
    main()

