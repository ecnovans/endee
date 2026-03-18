from pydantic import BaseModel
from typing import Any, Dict, List


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    used_llm: bool
    sources: List[SearchResult]


class HealthResponse(BaseModel):
    endee_ok: bool
    endee_base_url: str
    index_name: str

