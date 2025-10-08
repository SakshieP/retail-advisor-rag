import re
from typing import List, Dict, Any
from pathlib import Path

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r", " ").replace("\n", " ").strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str, target_words: int = 90, max_words: int = 140) -> List[str]:
    sents = sentences(text)
    chunks, cur, count = [], [], 0
    for s in sents:
        w = len(s.split())
        if count + w > max_words and cur:
            chunks.append(" ".join(cur)); cur, count = [], 0
        cur.append(s); count += w
        if count >= target_words:
            chunks.append(" ".join(cur)); cur, count = [], 0
    if cur: chunks.append(" ".join(cur))
    return chunks
