"""
Two routing strategies — both collection-aware.

1. LLM Router       — gpt-4o-mini classifies intent → namespace suffix.
2. Semantic Router  — cosine-match query against route prototypes.

Routes map to Pinecone namespace suffixes:
  "leaf"       → precise, specific chunks
  "parent"     → context-rich parent chunks
  "raptor-l2"  → high-level abstract summaries
  "props"      → atomic proposition index (highest precision)
"""

from __future__ import annotations

import json

import numpy as np

from core.config import settings
from core.openai_client import chat, embed_one

ROUTES = {
    "technical": {
        "namespace_suffix": "props",
        "description": "specific API calls, method signatures, code examples, parameters, configuration options, error messages",
    },
    "conceptual": {
        "namespace_suffix": "raptor-l2",
        "description": "high-level overviews, architecture explanations, what something is, comparisons, design decisions",
    },
    "howto": {
        "namespace_suffix": "parent",
        "description": "step-by-step guides, tutorials, how to accomplish a task, setup instructions, walkthroughs",
    },
    "precise": {
        "namespace_suffix": "leaf",
        "description": "exact fact lookup, specific version info, precise definitions, narrow technical details",
    },
}

_proto_embeddings: dict[str, list[float]] = {}


def _prototypes() -> dict[str, list[float]]:
    global _proto_embeddings
    if not _proto_embeddings:
        for label, cfg in ROUTES.items():
            _proto_embeddings[label] = embed_one(cfg["description"])
    return _proto_embeddings


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def llm_route(question: str, collection: str) -> str:
    labels = list(ROUTES.keys())
    try:
        raw = chat(
            model=settings.fast_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Classify this question into one of: {labels}.\n"
                        "technical=API/code/config detail  conceptual=what/why/overview\n"
                        "howto=step-by-step task  precise=exact narrow fact\n"
                        'Respond with JSON: {"route": "<label>"}'
                    ),
                },
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        label = json.loads(raw).get("route", "precise")
    except Exception:
        label = "precise"
    return ROUTES.get(label, ROUTES["precise"])["namespace_suffix"]


def semantic_route(question: str, collection: str) -> str:
    q_emb = embed_one(question)
    protos = _prototypes()
    best = max(protos, key=lambda l: _cosine(q_emb, protos[l]))
    return ROUTES[best]["namespace_suffix"]


def route(question: str, collection: str, strategy: str = "llm") -> str:
    """Return the namespace suffix for this question."""
    if strategy == "semantic":
        return semantic_route(question, collection)
    return llm_route(question, collection)
