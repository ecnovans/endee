from __future__ import annotations

from typing import Iterable, List


def chunk_text_by_paragraph(text: str, max_chars: int) -> List[str]:
    """
    Simple chunking: group paragraphs until we reach max_chars.
    Keeps chunks readable and good enough for a 2-day project.
    """
    text = (text or "").strip()
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    cur: List[str] = []

    def flush():
        if cur:
            chunks.append("\n\n".join(cur).strip())
            cur.clear()

    for p in paras:
        candidate = ("\n\n".join(cur + [p])).strip()
        if len(candidate) > max_chars and cur:
            flush()
            cur.append(p)
        else:
            cur.append(p)

    flush()
    return [c for c in chunks if c]


def iter_nonempty(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        s = (line or "").strip()
        if s:
            yield s

