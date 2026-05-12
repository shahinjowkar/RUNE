"""
Retrieval: HyDE + Hybrid (BM25 + dense) + RRF fusion.
"""

from __future__ import annotations

from pathlib import Path

from rank_bm25 import BM25Okapi

from core.config import settings
from core.openai_client import chat, embed_one
from core.pinecone_client import query as pc_query

_bm25_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}


def _build_bm25(collection: str) -> tuple[BM25Okapi, list[dict]]:
    if collection in _bm25_cache:
        return _bm25_cache[collection]
    raw_dir = Path("data/raw") / collection
    docs = []
    if raw_dir.exists():
        for path in sorted(raw_dir.glob("*.txt")):
            docs.append({"source": path.stem, "text": path.read_text(encoding="utf-8")})
    if not docs:
        # Fall back to flat raw dir for backwards compat
        flat = Path("data/raw")
        for path in sorted(flat.glob("*.txt")):
            docs.append({"source": path.stem, "text": path.read_text(encoding="utf-8")})
    tokenised = [d["text"].lower().split() for d in docs]
    index = BM25Okapi(tokenised) if tokenised else None
    _bm25_cache[collection] = (index, docs)
    return index, docs


def _rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranked in rankings:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def _hyde_embed(question: str) -> list[float]:
    hypothetical = chat(
        model=settings.fast_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a short technical paragraph (3-5 sentences) that would "
                    "directly answer the following question. Be specific and factual."
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return embed_one(hypothetical)


def retrieve(
    question: str,
    namespace: str,
    top_k: int | None = None,
    use_hyde: bool = True,
    use_hybrid: bool = True,
) -> list[dict]:
    top_k = top_k or settings.top_k

    # Collection = everything before the last "-" suffix
    collection = "-".join(namespace.split("-")[:-1]) if "-" in namespace else namespace

    query_emb = _hyde_embed(question) if use_hyde else embed_one(question)
    dense = pc_query(query_emb, namespace=namespace, top_k=top_k * 2)
    dense_ids = [r["id"] for r in dense]
    dense_map = {r["id"]: r for r in dense}

    if not use_hybrid:
        return [{**dense_map[i], "retrieval_method": "dense+hyde" if use_hyde else "dense"}
                for i in dense_ids[:top_k]]

    bm25, docs = _build_bm25(collection)
    if bm25 and docs:
        tokens = question.lower().split()
        scores = bm25.get_scores(tokens)
        bm25_ids = [docs[i]["source"] for i in sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[: top_k * 2]]
    else:
        bm25_ids = []

    fused = _rrf([dense_ids, bm25_ids])[:top_k]

    results = []
    for fid in fused:
        if fid in dense_map:
            results.append({**dense_map[fid], "retrieval_method": "hybrid+rrf+hyde"})
        else:
            match = next((d for d in docs if d["source"] == fid), None)
            if match:
                results.append({"id": fid, "text": match["text"][:500], "score": 0.0,
                                 "retrieval_method": "bm25-only"})
    return results
