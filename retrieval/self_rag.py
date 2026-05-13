"""
Self-RAG — post-generation faithfulness grading.
(Asai et al., 2023)

After generating an answer, the model grades whether it is:
  - Supported by the retrieved documents
  - Actually relevant to the question

If the answer fails grading, the pipeline re-retrieves with a refined query
and regenerates. Maximum 2 correction iterations.
"""

from __future__ import annotations

import json

from core.config import settings
from core.openai_client import chat

_GRADE_SYSTEM = """\
You are a strict factual grader. Given a question, an answer, and source documents,
evaluate the answer on two dimensions:

1. faithfulness (0-1): Is every claim in the answer supported by the documents?
2. relevance   (0-1): Does the answer actually address the question asked?

Return JSON:
{
  "faithfulness": 0.0-1.0,
  "relevance": 0.0-1.0,
  "supported": true|false,
  "issues": "description of any unsupported claims or gaps"
}
"""

_REFINE_SYSTEM = """\
The previous retrieval query failed to return useful results.
Given the original question and the issues found, write a better,
more specific search query that will find the missing information.
Return JSON: {"refined_query": "..."}
"""

FAITHFULNESS_THRESHOLD = 0.7
RELEVANCE_THRESHOLD = 0.7


def grade(question: str, answer: str, docs: list[dict]) -> dict:
    context = "\n\n".join(d.get("text", "")[:400] for d in docs[:5])
    try:
        raw = chat(
            model=settings.fast_model,
            messages=[
                {"role": "system", "content": _GRADE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Answer: {answer}\n\n"
                        f"Source documents:\n{context}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(raw)
        result["passes"] = (
            result.get("faithfulness", 0) >= FAITHFULNESS_THRESHOLD
            and result.get("relevance", 0) >= RELEVANCE_THRESHOLD
        )
        return result
    except Exception:
        return {"faithfulness": 1.0, "relevance": 1.0, "supported": True, "passes": True, "issues": ""}


def refine_query(question: str, issues: str) -> str:
    try:
        raw = chat(
            model=settings.fast_model,
            messages=[
                {"role": "system", "content": _REFINE_SYSTEM},
                {
                    "role": "user",
                    "content": f"Original question: {question}\nIssues: {issues}",
                },
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(raw).get("refined_query", question)
    except Exception:
        return question
