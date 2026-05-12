"""
Cross-encoder re-ranking via Cohere Rerank API.
Improves precision by scoring query+chunk pairs jointly (not independently).
"""

import cohere

from core.config import settings

_co = cohere.Client(api_key=settings.cohere_api_key)


def rerank(question: str, docs: list[dict], top_n: int | None = None) -> list[dict]:
    top_n = top_n or settings.rerank_top_n
    if not docs:
        return docs

    texts = [d.get("text", "") for d in docs]
    resp = _co.rerank(
        model="rerank-english-v3.0",
        query=question,
        documents=texts,
        top_n=top_n,
    )

    reranked = []
    for hit in resp.results:
        doc = dict(docs[hit.index])
        doc["rerank_score"] = hit.relevance_score
        reranked.append(doc)

    return reranked
