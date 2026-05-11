"""
Proposition indexing — Dense X Retrieval (Chen et al., 2023).

Extracts atomic, self-contained factual propositions from each chunk.
Propositions are indexed separately in Pinecone for high-precision retrieval.

Example:
  chunk: "WireMock supports JUnit 5 via @WireMockTest, introduced in v2.31"
  →  propositions:
       "WireMock supports JUnit 5."
       "JUnit 5 integration uses the @WireMockTest annotation."
       "@WireMockTest was introduced in WireMock version 2.31."
"""

import json
import uuid
from dataclasses import dataclass, field

from core.config import settings
from core.openai_client import chat
from ingestion.chunker import Chunk

_SYSTEM = (
    "You are an expert at extracting atomic facts. "
    "Given a passage, extract 3-7 self-contained propositions. "
    "Each proposition must be a single verifiable statement with enough "
    "context to be understood without the original passage. "
    'Return JSON: {"propositions": ["fact 1", "fact 2", ...]}'
)


def _extract(chunk_text: str) -> list[str]:
    try:
        raw = chat(
            model=settings.fast_model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": chunk_text},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(raw).get("propositions", [])
    except Exception:
        return []


def extract_propositions(chunks: list[Chunk]) -> list[Chunk]:
    """Return a flat list of Chunk objects, one per extracted proposition."""
    results: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        props = _extract(chunk.text)
        for prop in props:
            results.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=prop,
                    metadata={
                        **chunk.metadata,
                        "chunk_type": "proposition",
                        "source_chunk_id": chunk.id,
                        "source_chunk_text": chunk.text[:200],
                    },
                )
            )
        if (i + 1) % 10 == 0:
            print(f"  [propositions] {i + 1}/{len(chunks)} chunks processed")
    print(f"[propositions] extracted {len(results)} propositions from {len(chunks)} chunks")
    return results
