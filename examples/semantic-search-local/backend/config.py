import os
from dotenv import load_dotenv

load_dotenv()

ENDEE_BASE_URL = os.getenv("ENDEE_BASE_URL", "http://localhost:8080").rstrip("/")
ENDEE_INDEX_NAME = os.getenv("ENDEE_INDEX_NAME", "local_docs")
ENDEE_AUTH_TOKEN = os.getenv("ENDEE_AUTH_TOKEN", "").strip()

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "600"))

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
