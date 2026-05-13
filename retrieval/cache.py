"""
Semantic cache — returns cached answers for semantically similar queries.
Uses cosine similarity on embeddings rather than exact string matching.
Falls back gracefully if cache is empty or similarity is below threshold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from core.config import settings
from core.openai_client import embed_one

_SIMILARITY_THRESHOLD = 0.95
_TTL_SECONDS = 3600  # 1 hour


@dataclass
class _Entry:
    embedding: list[float]
    result: dict
    expires_at: float


class SemanticCache:
    def __init__(self, threshold: float = _SIMILARITY_THRESHOLD, ttl: int = _TTL_SECONDS):
        self._store: list[_Entry] = []
        self.threshold = threshold
        self.ttl = ttl

    def _cosine(self, a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def _evict_expired(self) -> None:
        now = time.time()
        self._store = [e for e in self._store if e.expires_at > now]

    def get(self, question: str) -> dict | None:
        self._evict_expired()
        if not self._store:
            return None
        q_emb = embed_one(question)
        best_score = 0.0
        best_result = None
        for entry in self._store:
            score = self._cosine(q_emb, entry.embedding)
            if score > best_score:
                best_score = score
                best_result = entry.result
        if best_score >= self.threshold:
            return {**best_result, "cache_hit": True, "cache_similarity": round(best_score, 4)}
        return None

    def set(self, question: str, result: dict) -> None:
        q_emb = embed_one(question)
        self._store.append(
            _Entry(
                embedding=q_emb,
                result=result,
                expires_at=time.time() + self.ttl,
            )
        )

    def size(self) -> int:
        self._evict_expired()
        return len(self._store)


# Module-level singleton
_cache = SemanticCache(
    threshold=_SIMILARITY_THRESHOLD,
    ttl=_TTL_SECONDS,
)


def get(question: str) -> dict | None:
    return _cache.get(question)


def set(question: str, result: dict) -> None:
    _cache.set(question, result)


def stats() -> dict:
    return {"size": _cache.size(), "threshold": _cache.threshold, "ttl_seconds": _cache.ttl}
