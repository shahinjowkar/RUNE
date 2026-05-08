from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str
    embed_model: str = "text-embedding-3-large"
    embed_dims: int = 3072
    chat_model: str = "gpt-4o"
    fast_model: str = "gpt-4o-mini"

    # Pinecone
    pinecone_api_key: str
    pinecone_index: str = "rune-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Cohere
    cohere_api_key: str

    # Tavily
    tavily_api_key: str

    # LangSmith (optional — set to enable tracing)
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "rune-rag"

    # Chunking
    leaf_chunk_tokens: int = 150
    parent_chunk_tokens: int = 512
    semantic_breakpoint_threshold: float = 0.85

    # RAPTOR
    raptor_levels: int = 2
    raptor_min_cluster_size: int = 5

    # Retrieval
    top_k: int = 6
    rerank_top_n: int = 3
    bm25_top_k: int = 6

    # Semantic cache
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 3600


settings = Settings()

# Activate LangSmith tracing if configured
import os
if settings.langchain_tracing_v2 == "true" and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
