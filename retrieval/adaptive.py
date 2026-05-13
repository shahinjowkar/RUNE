"""
Adaptive RAG — dynamically selects retrieval strategy based on query complexity.
(Jeong et al., 2024)

Tiers:
  simple   → single-shot dense retrieval (fast path, no CRAG overhead)
  moderate → full CRAG pipeline (retrieve → grade → generate)
  complex  → multi-hop: decompose into sub-questions, answer each, synthesize
"""

from __future__ import annotations

import json

from core.config import settings
from core.openai_client import chat

_CLASSIFY_SYSTEM = """\
Classify the complexity of this question for a RAG system.

simple:   factual lookup, single concept, one-hop answer
moderate: requires some reasoning, may span multiple concepts
complex:  multi-hop reasoning, requires combining info from multiple sources,
          comparative analysis, or chain-of-thought

Return JSON: {"complexity": "simple"|"moderate"|"complex", "reason": "..."}
"""

_DECOMPOSE_SYSTEM = """\
Break this complex question into 2-4 simpler sub-questions that can each be
answered independently. Together they should fully address the original question.
Return JSON: {"sub_questions": ["q1", "q2", ...]}
"""


def classify(question: str) -> str:
    """Returns 'simple', 'moderate', or 'complex'."""
    try:
        raw = chat(
            model=settings.fast_model,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(raw).get("complexity", "moderate")
    except Exception:
        return "moderate"


def decompose(question: str) -> list[str]:
    """Decompose a complex question into sub-questions."""
    try:
        raw = chat(
            model=settings.fast_model,
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(raw).get("sub_questions", [question])
    except Exception:
        return [question]


def synthesize(question: str, sub_answers: list[dict]) -> str:
    """Synthesize multiple sub-answers into a final coherent answer."""
    parts = "\n\n".join(
        f"Sub-question: {sa['question']}\nAnswer: {sa['answer']}"
        for sa in sub_answers
    )
    return chat(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical assistant. Synthesize the following "
                    "sub-answers into a single coherent, comprehensive response "
                    "to the original question."
                ),
            },
            {
                "role": "user",
                "content": f"Original question: {question}\n\n{parts}",
            },
        ],
    )
