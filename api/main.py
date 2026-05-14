from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="Rune RAG",
    description=(
        "Production-grade RAG pipeline. Ingest any docs URL, query with "
        "adaptive retrieval, GraphRAG, RAPTOR, Self-RAG, and Corrective RAG."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
