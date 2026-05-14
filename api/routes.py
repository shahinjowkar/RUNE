from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import (
    CacheStats,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestStatus,
    QueryRequest,
    QueryResponse,
    Source,
)
from eval.ragas_eval import evaluate as ragas_evaluate
from retrieval import cache as semantic_cache
from retrieval.crag import answer
from retrieval.router import route
from retrieval.retriever import retrieve
from retrieval.reranker import rerank
from core.openai_client import stream_chat

router = APIRouter()

_ingestion_jobs: dict[str, dict] = {}


def _run_ingestion(url: str, collection: str) -> None:
    _ingestion_jobs[collection] = {"status": "running", "summary": None}
    try:
        from ingestion.pipeline import run
        summary = run(url, collection)
        _ingestion_jobs[collection] = {"status": "done", "summary": summary}
    except Exception as e:
        _ingestion_jobs[collection] = {"status": "error", "summary": {"error": str(e)}}


# ── Ops ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse()


@router.get("/cache/stats", response_model=CacheStats, tags=["ops"])
def cache_stats():
    return CacheStats(**semantic_cache.stats())


@router.delete("/cache", tags=["ops"])
def cache_clear():
    semantic_cache._cache._store.clear()
    return {"message": "cache cleared"}


# ── Ingestion ──────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestStatus, tags=["ingestion"])
def ingest(req: IngestRequest, background: BackgroundTasks):
    job = _ingestion_jobs.get(req.collection, {})
    if job.get("status") == "running":
        return IngestStatus(status="running", message=f"'{req.collection}' ingestion already in progress.")
    background.add_task(_run_ingestion, req.url, req.collection)
    return IngestStatus(status="started", message=f"Ingestion of '{req.collection}' started.")


@router.get("/ingest/{collection}/status", response_model=IngestStatus, tags=["ingestion"])
def ingest_status(collection: str):
    job = _ingestion_jobs.get(collection)
    if not job:
        return IngestStatus(status="not_started", message=f"No ingestion job found for '{collection}'.")
    return IngestStatus(status=job["status"], message=f"Collection: {collection}", summary=job.get("summary"))


# ── Query ──────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    try:
        result = answer(req.question, collection=req.collection, routing_strategy=req.routing)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=result["answer"],
        sources=[Source(url=s["url"], text=s["text"]) for s in result.get("sources", [])],
        retrieval_method=result.get("retrieval_method", ""),
        namespace_used=result.get("namespace", ""),
        complexity=result.get("complexity", "moderate"),
        web_fallback=result.get("web_fallback", False),
        cache_hit=result.get("cache_hit", False),
    )


@router.post("/query/stream", tags=["rag"])
async def query_stream(req: QueryRequest):
    """Streaming endpoint — returns Server-Sent Events with token-by-token output."""
    try:
        namespace_suffix = route(req.question, collection=req.collection, strategy=req.routing)
        namespace = f"{req.collection}-{namespace_suffix}"
        docs = retrieve(req.question, namespace=namespace, use_hyde=req.use_hyde, use_hybrid=req.use_hybrid)
        reranked = rerank(req.question, docs)
        context = "\n\n---\n\n".join(
            f"[Source: {d.get('source_url', '')}]\n{d.get('text', '')}" for d in reranked
        )
        messages = [
            {"role": "system", "content": "Answer using only the provided context. Be precise."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"},
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def event_stream():
        async for token in stream_chat(messages):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Evaluation ─────────────────────────────────────────────────────────────

@router.post("/eval", response_model=EvalResponse, tags=["evaluation"])
def evaluate(req: EvalRequest):
    scores = ragas_evaluate(
        question=req.question,
        answer=req.answer,
        contexts=req.contexts,
        ground_truth=req.ground_truth,
    )
    return EvalResponse(scores=scores)
