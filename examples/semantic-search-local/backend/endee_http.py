from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import msgpack
import requests


@dataclass(frozen=True)
class EndeeSearchHit:
    id: str
    similarity: float
    meta: bytes
    filter: str
    norm: float
    vector: List[float]

    def meta_text(self) -> str:
        try:
            return self.meta.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def meta_json(self) -> Dict[str, Any]:
        raw = self.meta_text().strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {"_raw_meta": raw}


class EndeeHttpClient:
    def __init__(self, base_url: str, auth_token: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token.strip()

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.auth_token:
            h["Authorization"] = self.auth_token
        if extra:
            h.update(extra)
        return h

    def health(self) -> bool:
        r = requests.get(f"{self.base_url}/api/v1/health", headers=self._headers(), timeout=10)
        return r.status_code == 200

    def list_indexes(self) -> List[Dict[str, Any]]:
        r = requests.get(f"{self.base_url}/api/v1/index/list", headers=self._headers(), timeout=30)
        r.raise_for_status()
        payload = r.json()
        return payload.get("indexes", [])

    def create_index_if_missing(
        self,
        index_name: str,
        dim: int,
        space_type: str = "cosine",
        precision: str = "float32",
    ) -> None:
        existing = {i.get("name") for i in self.list_indexes()}
        if index_name in existing:
            return

        body = {
            "index_name": index_name,
            "dim": int(dim),
            "space_type": space_type,
            "precision": precision,
        }
        r = requests.post(
            f"{self.base_url}/api/v1/index/create",
            headers=self._headers({"Content-Type": "application/json"}),
            data=json.dumps(body),
            timeout=60,
        )
        if r.status_code not in (200, 409):
            # 409 = already exists, acceptable if a race happened
            r.raise_for_status()

    def insert_vectors_json(self, index_name: str, vectors: List[Dict[str, Any]]) -> None:
        r = requests.post(
            f"{self.base_url}/api/v1/index/{index_name}/vector/insert",
            headers=self._headers({"Content-Type": "application/json"}),
            data=json.dumps(vectors),
            timeout=120,
        )
        r.raise_for_status()

    def search_dense(self, index_name: str, vector: List[float], k: int) -> List[EndeeSearchHit]:
        body = {"vector": vector, "k": int(k)}
        r = requests.post(
            f"{self.base_url}/api/v1/index/{index_name}/search",
            headers=self._headers({"Content-Type": "application/json"}),
            data=json.dumps(body),
            timeout=60,
        )
        r.raise_for_status()

        obj = msgpack.unpackb(r.content, raw=False)
        results = _extract_results(obj)

        hits: List[EndeeSearchHit] = []
        for item in results:
            hit = _parse_vector_result(item)
            if hit:
                hits.append(hit)
        return hits


def _extract_results(obj: Any) -> List[Any]:
    # Endee packs C++ structs via MSGPACK_DEFINE, which typically produces arrays.
    # ResultSet has a single field: results.
    if isinstance(obj, dict) and "results" in obj:
        return obj["results"] or []
    if isinstance(obj, (list, tuple)):
        # Common case: ResultSet -> [results]
        if len(obj) == 1 and isinstance(obj[0], (list, tuple)):
            return list(obj[0])
        # Some builds might return results directly
        return list(obj)
    return []


def _parse_vector_result(item: Any) -> Optional[EndeeSearchHit]:
    if isinstance(item, dict):
        similarity = float(item.get("similarity", 0.0))
        _id = str(item.get("id", ""))
        meta = item.get("meta", b"") or b""
        if isinstance(meta, str):
            meta = meta.encode("utf-8", errors="replace")
        filt = str(item.get("filter", ""))
        norm = float(item.get("norm", 0.0))
        vec = item.get("vector", []) or []
        return EndeeSearchHit(id=_id, similarity=similarity, meta=meta, filter=filt, norm=norm, vector=list(vec))

    if isinstance(item, (list, tuple)) and len(item) >= 6:
        similarity = float(item[0])
        _id = str(item[1])
        meta = item[2] or b""
        if isinstance(meta, str):
            meta = meta.encode("utf-8", errors="replace")
        filt = str(item[3] or "")
        norm = float(item[4] or 0.0)
        vec = item[5] or []
        return EndeeSearchHit(id=_id, similarity=similarity, meta=meta, filter=filt, norm=norm, vector=list(vec))

    return None

