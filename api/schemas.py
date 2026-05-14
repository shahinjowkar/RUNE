from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    url: str = Field(..., description="Root URL of the docs site to ingest")
    collection: str = Field(..., description="Collection name — used as Pinecone namespace prefix")


class IngestStatus(BaseModel):
    status: str
    message: str
    summary: dict | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    collection: str = Field(..., description="Collection to query against")
    routing: str = Field("llm", description="'llm' or 'semantic'")
    top_k: int = Field(6, ge=1, le=20)
    use_hyde: bool = True
    use_hybrid: bool = True


class Source(BaseModel):
    url: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_method: str
    namespace_used: str
    complexity: str
    web_fallback: bool
    cache_hit: bool
    reranked: bool = True


class EvalRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None


class EvalResponse(BaseModel):
    scores: dict[str, float]


class CacheStats(BaseModel):
    size: int
    threshold: float
    ttl_seconds: int


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.2.0"
