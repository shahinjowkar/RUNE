# Rune RAG

> Point it at any documentation URL. It figures out the rest.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai)
![Pinecone](https://img.shields.io/badge/Pinecone-Serverless-00B388)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)

---

## What It Does

Give Rune a URL — it handles discovery, ingestion, indexing, and querying automatically:

```bash
# Ingest any docs site
curl -X POST http://localhost:8000/ingest \
  -d '{"url": "https://wiremock.org/docs/", "collection": "wiremock"}'

# Query it
curl -X POST http://localhost:8000/query \
  -d '{"question": "How do I stub a POST?", "collection": "wiremock"}'
```

No config files. No hardcoded URLs. Swap the URL, change the collection name, done.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                        INGESTION PIPELINE                        ║
║                                                                  ║
║  Any URL  ──►  Smart Crawler  ──►  Fetcher  ──►  data/raw/       ║
║                (sitemap auto-       (HTML                        ║
║                 detect + recur-      clean)                      ║
║                 sive fallback)                                   ║
║                      │                                           ║
║          ┌───────────┴────────────┐                              ║
║          ▼                        ▼                              ║
║   Semantic Chunker          Parent-Child                         ║
║   (embedding breakpoints)   (leaf 150t / parent 512t)            ║
║          │                        │                              ║
║          ├──► Proposition Extractor ──► {col}-props              ║
║          │    (atomic facts, Chen 2023)                          ║
║          │                                                       ║
║          ├──► RAPTOR Tree ──► {col}-raptor-l1 / l2               ║
║          │    (cluster → summarise → embed, Sarthi 2024)         ║
║          │                                                       ║
║          └──► Knowledge Graph ──► data/graphs/{col}.json         ║
║               (entity-relation triples, Edge 2024)               ║
╚══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════╗
║                         QUERY PIPELINE                           ║
║                                                                  ║
║  Question                                                        ║
║     │                                                            ║
║     ├──► Semantic Cache ──► [cache hit] ──► return               ║
║     │    (cosine similarity, 95% threshold)                      ║
║     │                                                            ║
║     ├──► Adaptive RAG Classifier                                 ║
║     │         simple ──► direct retrieval                        ║
║     │         moderate ──► CRAG pipeline                         ║
║     │         complex ──► sub-question decomposition             ║
║     │                                                            ║
║     ├──► LLM Router ──► namespace suffix                         ║
║     └──► Semantic Router   (props/leaf/parent/raptor-l2)         ║
║                                                                  ║
║     ┌─────────────────────────────────────┐                      ║
║     │          CRAG State Machine         │                      ║
║     │  (LangGraph)                        │                      ║
║     │                                     │                      ║
║     │  retrieve ──► grade ──► generate    │                      ║
║     │      │          │                   │                      ║
║     │   HyDE +      Cohere             Self-RAG                  ║
║     │   BM25 +      Rerank             faithfulness              ║
║     │   Dense        │                  grader                   ║
║     │   RRF      web_search                                      ║
║     │           (Tavily fallback)                                 ║
║     └─────────────────────────────────────┘                      ║
║                    +                                             ║
║             GraphRAG context                                     ║
║          (entity graph traversal)                                ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Techniques

### Ingestion

| # | Technique | Description | Paper |
|---|---|---|---|
| 1 | **Smart URL Discovery** | Auto-detects sitemap (robots.txt, `/sitemap.xml`, `/sitemap-0.xml`, index); falls back to recursive link crawl | — |
| 2 | **Semantic Chunking** | Splits at embedding-similarity breakpoints between sentence windows | — |
| 3 | **Parent-Child Chunking** | Small leaf chunks (150t) for precision retrieval; full parent chunks (512t) returned as context | — |
| 4 | **Proposition Indexing** | Extracts 3-7 atomic, self-contained facts per chunk; indexed as a separate high-precision namespace | [Dense X Retrieval, Chen 2023](https://arxiv.org/abs/2312.06648) |
| 5 | **RAPTOR** | Recursively clusters leaves → summarises clusters → embeds summaries → repeats. Multi-level abstraction tree | [Sarthi et al., Stanford 2024](https://arxiv.org/abs/2401.18059) |
| 6 | **Knowledge Graph** | Extracts (entity, relation, entity) triples from every chunk; stored as adjacency graph for structural retrieval | [GraphRAG, Edge et al. 2024](https://arxiv.org/abs/2404.16130) |

### Retrieval & Generation

| # | Technique | Description | Paper |
|---|---|---|---|
| 7 | **Adaptive RAG** | Classifies query complexity (simple/moderate/complex); selects strategy accordingly — direct retrieval, CRAG, or multi-hop decomposition | [Jeong et al., 2024](https://arxiv.org/abs/2403.14403) |
| 8 | **Semantic Cache** | Caches results by query embedding similarity (not exact match); 95% cosine threshold, 1h TTL | — |
| 9 | **LLM Router** | `gpt-4o-mini` classifies query intent → routes to optimal Pinecone namespace (props/leaf/parent/raptor-l2) | — |
| 10 | **Semantic Router** | Embeds query, cosine-matches against route prototypes — zero LLM latency alternative | — |
| 11 | **HyDE** | Generates a hypothetical answer, embeds it, uses that vector for dense retrieval — closes the query-document embedding gap | [Gao et al., CMU 2022](https://arxiv.org/abs/2212.10496) |
| 12 | **Hybrid Search + RRF** | Runs BM25 sparse + Pinecone dense in parallel; merges via Reciprocal Rank Fusion | — |
| 13 | **GraphRAG Context** | Extracts question entities, traverses the knowledge graph (2-hop), injects structured relational context into the prompt | [Edge et al., 2024](https://arxiv.org/abs/2404.16130) |
| 14 | **Cross-encoder Re-ranking** | Cohere Rerank v3 scores query+chunk pairs jointly — far more accurate than bi-encoder similarity | — |
| 15 | **Corrective RAG (CRAG)** | LangGraph state machine: retrieve → grade → generate or web-search → generate | [Yan et al., 2024](https://arxiv.org/abs/2401.15884) |
| 16 | **Self-RAG** | Post-generation faithfulness grader; re-retrieves with refined query if answer isn't grounded. Max 2 iterations | [Asai et al., 2023](https://arxiv.org/abs/2310.11511) |
| 17 | **Streaming** | Token-by-token SSE via `/query/stream` — production UX | — |
| 18 | **RAGAs Evaluation** | Automatic scoring: faithfulness, answer relevancy, context precision (+ recall with ground truth) | [Shahul et al., 2023](https://arxiv.org/abs/2309.15217) |
| 19 | **LangSmith Tracing** | Full observability — latency, token cost, retrieval quality per run. One env var to enable | — |

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM | OpenAI GPT-4o |
| Fast LLM / routing / grading | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-large (3072 dims) |
| Vector DB | Pinecone Serverless |
| Sparse search | rank-bm25 |
| Re-ranking | Cohere Rerank v3 |
| Web fallback | Tavily Search API |
| Orchestration | LangGraph |
| API | FastAPI + Uvicorn |
| Streaming | sse-starlette |
| Evaluation | RAGAs |
| Observability | LangSmith |
| Containerisation | Docker + Docker Compose |

---

## Project Structure

```
rune-rag/
├── ingestion/
│   ├── crawler.py       # smart URL discovery (sitemap → recursive fallback)
│   ├── fetcher.py       # HTML fetch + clean → .txt files
│   ├── chunker.py       # semantic chunking + parent-child
│   ├── propositions.py  # atomic fact extraction (Dense X Retrieval)
│   ├── raptor.py        # RAPTOR multi-level tree builder
│   └── pipeline.py      # orchestrates full ingestion for any URL
│
├── retrieval/
│   ├── router.py        # LLM + semantic routing → namespace
│   ├── retriever.py     # HyDE + hybrid BM25/dense + RRF
│   ├── reranker.py      # Cohere cross-encoder re-ranking
│   ├── graph_rag.py     # knowledge graph build + query
│   ├── adaptive.py      # complexity classification + multi-hop
│   ├── cache.py         # semantic cache (cosine similarity)
│   ├── self_rag.py      # faithfulness grading + query refinement
│   └── crag.py          # full pipeline orchestration (LangGraph)
│
├── eval/
│   └── ragas_eval.py    # RAGAs evaluation + LLM fallback
│
├── api/
│   ├── main.py          # FastAPI app
│   ├── routes.py        # /query, /query/stream, /ingest, /eval, /health
│   └── schemas.py       # Pydantic models
│
├── core/
│   ├── config.py        # pydantic-settings (all env vars)
│   ├── openai_client.py # sync + async OpenAI wrappers
│   └── pinecone_client.py
│
├── data/
│   ├── raw/{collection}/  # fetched .txt files (gitignored)
│   └── graphs/            # knowledge graphs (gitignored)
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install .
```

### 2. Configure

```bash
cp .env.example .env
# Fill in: OPENAI_API_KEY, PINECONE_API_KEY, COHERE_API_KEY, TAVILY_API_KEY
# Optional: LANGCHAIN_API_KEY for LangSmith tracing
```

### 3. Ingest any docs site

```bash
# Via CLI
python -m ingestion.pipeline --url https://wiremock.org/docs/ --collection wiremock

# Or start the API and POST to /ingest
uvicorn api.main:app --reload
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://docs.python.org/3/", "collection": "python"}'
```

### 4. Query

```bash
# Blocking query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I stub a POST request?", "collection": "wiremock"}'

# Streaming query (token-by-token SSE)
curl -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain stateful behaviour", "collection": "wiremock"}'
```

### 5. Docker

```bash
docker-compose up --build
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest any URL into a named collection |
| `GET` | `/ingest/{collection}/status` | Check ingestion progress |
| `POST` | `/query` | Blocking query with full pipeline |
| `POST` | `/query/stream` | Streaming SSE query |
| `POST` | `/eval` | RAGAs quality evaluation |
| `GET` | `/cache/stats` | Semantic cache stats |
| `DELETE` | `/cache` | Clear cache |
| `GET` | `/health` | Liveness check |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Pinecone Namespace Schema

Each collection gets five namespaces:

| Namespace | Contents | Best for |
|---|---|---|
| `{col}-leaf` | Semantic leaf chunks (~150t) | Precise narrow queries |
| `{col}-parent` | Full parent chunks (~512t) | How-to, context-rich answers |
| `{col}-props` | Atomic propositions | Exact fact lookup |
| `{col}-raptor-l1` | Level-1 cluster summaries | Mid-level synthesis |
| `{col}-raptor-l2` | Level-2 abstract summaries | High-level conceptual questions |

---

## References

| Paper | Authors | Year |
|---|---|---|
| [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) | Sarthi et al., Stanford | 2024 |
| [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) | Edge et al., Microsoft | 2024 |
| [Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs](https://arxiv.org/abs/2403.14403) | Jeong et al. | 2024 |
| [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) | Yan et al. | 2024 |
| [Self-RAG: Learning to Retrieve, Generate, and Critique](https://arxiv.org/abs/2310.11511) | Asai et al. | 2023 |
| [Dense X Retrieval: Proposition Indexing](https://arxiv.org/abs/2312.06648) | Chen et al. | 2023 |
| [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496) | Gao et al., CMU | 2022 |
| [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) | Shahul et al. | 2023 |
