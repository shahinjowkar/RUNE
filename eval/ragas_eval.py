"""
RAGAs evaluation layer — automatic quality measurement.
(Shahul et al., 2023 — https://github.com/explodinggradients/ragas)

Metrics:
  faithfulness      — are claims grounded in the retrieved context?
  answer_relevancy  — does the answer address the question?
  context_precision — were the right chunks retrieved?
  context_recall    — were all necessary chunks retrieved? (requires ground truth)
"""

from __future__ import annotations


def evaluate(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict:
    """
    Run RAGAs metrics. Returns a dict of scores (0-1).
    Requires: pip install ragas
    """
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        metrics = [faithfulness, answer_relevancy, context_precision]

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }

        if ground_truth:
            from ragas.metrics import context_recall
            data["ground_truth"] = [ground_truth]
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)
        result = ragas_evaluate(dataset, metrics=metrics)

        scores = result.to_pandas().iloc[0].to_dict()
        return {k: round(float(v), 4) for k, v in scores.items() if isinstance(v, (int, float))}

    except ImportError:
        return _fallback_eval(question, answer, contexts)


def _fallback_eval(question: str, answer: str, contexts: list[str]) -> dict:
    """LLM-based fallback when ragas is not installed."""
    from core.config import settings
    from core.openai_client import chat
    import json

    context_str = "\n\n".join(contexts[:3])
    raw = chat(
        model=settings.fast_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Score this RAG output. Return JSON:\n"
                    '{"faithfulness": 0-1, "answer_relevancy": 0-1, "context_precision": 0-1}'
                    "\nfaithfulness: are all claims in the answer supported by the context?"
                    "\nanswer_relevancy: does the answer address the question?"
                    "\ncontext_precision: how relevant is the context to the question?"
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nAnswer: {answer}\nContext:\n{context_str}",
            },
        ],
        response_format={"type": "json_object"},
    )
    try:
        return {k: round(float(v), 4) for k, v in json.loads(raw).items()}
    except Exception:
        return {"faithfulness": -1, "answer_relevancy": -1, "context_precision": -1}
