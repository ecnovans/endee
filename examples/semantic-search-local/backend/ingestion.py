from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer

from config import CHUNK_MAX_CHARS, DATA_DIR, EMBED_MODEL_NAME, ENDEE_AUTH_TOKEN, ENDEE_BASE_URL, ENDEE_INDEX_NAME
from endee_http import EndeeHttpClient
from text_chunking import chunk_text_by_paragraph


def read_txt_files(data_dir: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(data_dir):
        return out

    for name in sorted(os.listdir(data_dir)):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(data_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            out.append((name, f.read()))
    return out


def build_vectors(model: SentenceTransformer, chunks: List[Dict[str, str]]) -> List[Dict[str, object]]:
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True, normalize_embeddings=False)

    payload: List[Dict[str, object]] = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        meta = {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        }
        payload.append(
            {
                "id": f"{chunk['source']}::chunk::{chunk['chunk_id']}",
                # Endee treats this as bytes; JSON route stores the string bytes as-is.
                "meta": json.dumps(meta, ensure_ascii=False),
                # Optional structured filter payload (stored on insert)
                "filter": json.dumps({"source": chunk["source"]}, ensure_ascii=False),
                "vector": vec.tolist(),
            }
        )
    return payload


def main() -> None:
    client = EndeeHttpClient(ENDEE_BASE_URL, auth_token=ENDEE_AUTH_TOKEN)

    if not client.health():
        raise SystemExit(
            f"Endee server not reachable at {ENDEE_BASE_URL}. "
            f"Start Endee first, then re-run ingestion."
        )

    docs = read_txt_files(DATA_DIR)
    if not docs:
        raise SystemExit(f"No .txt files found in {DATA_DIR}. Add files and re-run.")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    dim = int(model.get_sentence_embedding_dimension())

    client.create_index_if_missing(index_name=ENDEE_INDEX_NAME, dim=dim, space_type="cosine", precision="float32")

    chunks: List[Dict[str, str]] = []
    for filename, text in docs:
        parts = chunk_text_by_paragraph(text, max_chars=CHUNK_MAX_CHARS)
        for idx, part in enumerate(parts):
            chunks.append({"source": filename, "chunk_id": str(idx), "text": part})

    vectors_payload = build_vectors(model, chunks)
    client.insert_vectors_json(ENDEE_INDEX_NAME, vectors_payload)

    print(f"Inserted {len(vectors_payload)} vectors into Endee index '{ENDEE_INDEX_NAME}'.")


if __name__ == "__main__":
    main()

