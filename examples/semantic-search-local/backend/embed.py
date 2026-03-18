"""
Simple script to embed documents locally (no API keys).

This is mainly for the evaluation checklist / learning:
- it shows how embeddings are generated
- ingestion.py is what actually stores them in Endee
"""

from sentence_transformers import SentenceTransformer


def main() -> None:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    documents = [
        "Artificial intelligence is transforming industries.",
        "Machine learning is a subset of AI.",
        "Vector databases store embeddings.",
    ]

    embeddings = model.encode(documents)
    print("Embedding shape:", embeddings.shape)
    print("First embedding (first 10 dims):", embeddings[0][:10])


if __name__ == "__main__":
    main()

