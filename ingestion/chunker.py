"""
Two chunking strategies:

1. Semantic chunking  — split on embedding-similarity breakpoints between sentences.
2. Parent-child       — semantic chunks become parents; each is split into smaller
                        leaf chunks. Leaf carries parent_id for context expansion.
"""

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tiktoken

from core.config import settings
from core.openai_client import embed

_enc = tiktoken.encoding_for_model("gpt-4o")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def semantic_chunks(text: str, source_url: str = "") -> list[Chunk]:
    """Split text at points where sentence-embedding similarity drops sharply."""
    sents = _sentences(text)
    if len(sents) <= 2:
        return [Chunk(id=str(uuid.uuid4()), text=text, metadata={"source_url": source_url})]

    window = 3
    windows = [
        " ".join(sents[max(0, i - window) : i + window]) for i in range(len(sents))
    ]
    embeddings = embed(windows)

    breakpoints: list[int] = []
    for i in range(1, len(embeddings)):
        sim = _cosine(embeddings[i - 1], embeddings[i])
        if sim < settings.semantic_breakpoint_threshold:
            breakpoints.append(i)

    chunks: list[Chunk] = []
    prev = 0
    for bp in breakpoints:
        chunk_text = " ".join(sents[prev:bp]).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    metadata={"source_url": source_url, "chunk_type": "semantic"},
                )
            )
        prev = bp
    tail = " ".join(sents[prev:]).strip()
    if tail:
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=tail,
                metadata={"source_url": source_url, "chunk_type": "semantic"},
            )
        )
    return chunks


def _split_by_tokens(text: str, max_tokens: int) -> list[str]:
    tokens = _enc.encode(text)
    parts = []
    for i in range(0, len(tokens), max_tokens):
        parts.append(_enc.decode(tokens[i : i + max_tokens]))
    return parts


def parent_child_chunks(
    semantic: list[Chunk],
) -> tuple[list[Chunk], list[Chunk]]:
    """
    Returns (parents, leaves).
    Parents = semantic chunks (up to parent_chunk_tokens, merged if too small).
    Leaves  = sub-splits of each parent at leaf_chunk_tokens.
    """
    parents: list[Chunk] = []
    leaves: list[Chunk] = []

    for sem in semantic:
        parent_texts = _split_by_tokens(sem.text, settings.parent_chunk_tokens)
        for pt in parent_texts:
            parent = Chunk(
                id=str(uuid.uuid4()),
                text=pt,
                metadata={**sem.metadata, "chunk_type": "parent"},
            )
            parents.append(parent)

            for lt in _split_by_tokens(pt, settings.leaf_chunk_tokens):
                leaves.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        text=lt,
                        metadata={
                            **sem.metadata,
                            "chunk_type": "leaf",
                            "parent_id": parent.id,
                            "parent_text": pt[:300],
                        },
                    )
                )

    return parents, leaves


def chunks_from_file(path: Path) -> tuple[list[Chunk], list[Chunk]]:
    text = path.read_text(encoding="utf-8")
    source_url = ""  # enriched downstream if needed
    sem = semantic_chunks(text, source_url)
    return parent_child_chunks(sem)
