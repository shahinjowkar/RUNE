"""
Full RAG pipeline integrating:
  - Adaptive strategy selection (simple / moderate / complex)
  - Semantic cache (skip retrieval for repeated queries)
  - LLM + Semantic routing → Pinecone namespace
  - HyDE + Hybrid search (BM25 + dense + RRF)
  - Cohere cross-encoder re-ranking
  - GraphRAG knowledge graph context injection
  - Corrective RAG (CRAG) via LangGraph state machine
  - Self-RAG post-generation faithfulness grading
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph
from tavily import TavilyClient

from core.config import settings
from core.openai_client import chat
from retrieval import cache as semantic_cache
from retrieval.adaptive import classify, decompose, synthesize
from retrieval.graph_rag import graph_context
from retrieval.reranker import rerank
from retrieval.retriever import retrieve
from retrieval.router import route
from retrieval import self_rag

_tavily = TavilyClient(api_key=settings.tavily_api_key)

RELEVANCE_THRESHOLD = 0.3
MAX_SELF_RAG_ITERATIONS = 2


class RAGState(TypedDict):
    question: str
    collection: str
    namespace: str
    complexity: str
    documents: list[dict]
    generation: str
    web_fallback: bool
    retrieval_method: str
    self_rag_iterations: int
    graph_context: str


# ── Nodes ──────────────────────────────────────────────────────────────────

def node_retrieve(state: RAGState) -> RAGState:
    docs = retrieve(
        state["question"],
        namespace=state["namespace"],
        use_hyde=True,
        use_hybrid=True,
    )
    method = docs[0].get("retrieval_method", "hybrid") if docs else "hybrid"
    return {**state, "documents": docs, "retrieval_method": method}


def node_grade(state: RAGState) -> RAGState:
    scored = rerank(state["question"], state["documents"])
    relevant = [d for d in scored if d.get("rerank_score", 1.0) >= RELEVANCE_THRESHOLD]
    web_needed = len(relevant) == 0
    return {**state, "documents": relevant or scored, "web_fallback": web_needed}


def node_web_search(state: RAGState) -> RAGState:
    results = _tavily.search(query=state["question"], max_results=4)
    web_docs = [
        {
            "id": f"web-{i}",
            "text": r.get("content", r.get("snippet", "")),
            "source_url": r.get("url", ""),
            "retrieval_method": "tavily-web",
            "score": 1.0,
        }
        for i, r in enumerate(results.get("results", []))
    ]
    return {**state, "documents": web_docs, "retrieval_method": "web-fallback"}


def node_generate(state: RAGState) -> RAGState:
    graph_ctx = state.get("graph_context", "")
    context_parts = [
        f"[Source: {d.get('source_url', 'unknown')}]\n{d.get('text', '')}"
        for d in state["documents"]
    ]
    context = "\n\n---\n\n".join(context_parts)
    if graph_ctx:
        context = f"{graph_ctx}\n\n---\n\n{context}"

    answer = chat(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise technical assistant. Answer using only the provided "
                    "context. Cite sources where possible. If context is insufficient, say so."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {state['question']}",
            },
        ],
    )
    return {**state, "generation": answer}


def node_self_grade(state: RAGState) -> RAGState:
    """Grade faithfulness; if failing and under iteration limit, refine and re-retrieve."""
    grade = self_rag.grade(state["question"], state["generation"], state["documents"])

    if grade["passes"] or state["self_rag_iterations"] >= MAX_SELF_RAG_ITERATIONS:
        return state

    refined_q = self_rag.refine_query(state["question"], grade.get("issues", ""))
    new_docs = retrieve(refined_q, namespace=state["namespace"], use_hyde=True, use_hybrid=True)
    reranked = rerank(state["question"], new_docs)

    new_context = "\n\n---\n\n".join(
        f"[Source: {d.get('source_url', '')}]\n{d.get('text', '')}" for d in reranked
    )
    new_answer = chat(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": "Answer using only the context. Be factual and precise."},
            {"role": "user", "content": f"Context:\n{new_context}\n\nQuestion: {state['question']}"},
        ],
    )
    return {
        **state,
        "documents": reranked,
        "generation": new_answer,
        "self_rag_iterations": state["self_rag_iterations"] + 1,
        "retrieval_method": f"{state['retrieval_method']}+self-rag",
    }


# ── Conditional edges ──────────────────────────────────────────────────────

def _after_grade(state: RAGState) -> str:
    return "web_search" if state["web_fallback"] else "generate"


# ── Graph ──────────────────────────────────────────────────────────────────

_builder = StateGraph(RAGState)
_builder.add_node("retrieve", node_retrieve)
_builder.add_node("grade", node_grade)
_builder.add_node("web_search", node_web_search)
_builder.add_node("generate", node_generate)
_builder.add_node("self_grade", node_self_grade)

_builder.set_entry_point("retrieve")
_builder.add_edge("retrieve", "grade")
_builder.add_conditional_edges("grade", _after_grade, {
    "web_search": "web_search",
    "generate": "generate",
})
_builder.add_edge("web_search", "generate")
_builder.add_edge("generate", "self_grade")
_builder.add_edge("self_grade", END)

_graph = _builder.compile()


# ── Public interface ───────────────────────────────────────────────────────

def answer(question: str, collection: str, routing_strategy: str = "llm") -> dict:
    # 1. Semantic cache check
    cached = semantic_cache.get(question)
    if cached:
        return cached

    # 2. Adaptive complexity classification
    complexity = classify(question)

    # 3. Multi-hop for complex queries
    if complexity == "complex":
        sub_questions = decompose(question)
        sub_answers = []
        for sq in sub_questions:
            sub_result = _run_single(sq, collection, routing_strategy, complexity)
            sub_answers.append({"question": sq, "answer": sub_result["answer"]})
        final_answer = synthesize(question, sub_answers)
        result = {
            "answer": final_answer,
            "sources": [s for sa in sub_answers for s in _run_single(sa["question"], collection, routing_strategy, complexity).get("sources", [])],
            "retrieval_method": "multi-hop",
            "web_fallback": False,
            "namespace": f"{collection}-leaf",
            "complexity": complexity,
            "cache_hit": False,
        }
    else:
        result = _run_single(question, collection, routing_strategy, complexity)

    # 4. Cache result
    semantic_cache.set(question, result)
    return result


def _run_single(question: str, collection: str, routing_strategy: str, complexity: str) -> dict:
    namespace_suffix = route(question, collection=collection, strategy=routing_strategy)
    namespace = f"{collection}-{namespace_suffix}"

    g_ctx = graph_context(question, collection)

    state = _graph.invoke({
        "question": question,
        "collection": collection,
        "namespace": namespace,
        "complexity": complexity,
        "documents": [],
        "generation": "",
        "web_fallback": False,
        "retrieval_method": "",
        "self_rag_iterations": 0,
        "graph_context": g_ctx,
    })

    return {
        "answer": state["generation"],
        "sources": [
            {"url": d.get("source_url", ""), "text": d.get("text", "")[:300]}
            for d in state["documents"]
        ],
        "retrieval_method": state["retrieval_method"],
        "web_fallback": state["web_fallback"],
        "namespace": namespace,
        "complexity": complexity,
        "cache_hit": False,
    }
