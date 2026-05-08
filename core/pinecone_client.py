import uuid
from pinecone import Pinecone, ServerlessSpec
from core.config import settings

_pc = Pinecone(api_key=settings.pinecone_api_key)
_index = None


def get_index():
    global _index
    if _index is not None:
        return _index
    existing = [i.name for i in _pc.list_indexes()]
    if settings.pinecone_index not in existing:
        _pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embed_dims,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
    _index = _pc.Index(settings.pinecone_index)
    return _index


def upsert(vectors: list[dict], namespace: str) -> None:
    idx = get_index()
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        idx.upsert(vectors=vectors[i : i + batch_size], namespace=namespace)


def query(
    embedding: list[float],
    namespace: str,
    top_k: int,
    filter: dict | None = None,
) -> list[dict]:
    idx = get_index()
    resp = idx.query(
        vector=embedding,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=filter,
    )
    return [
        {"id": m.id, "score": m.score, **m.metadata}
        for m in resp.matches
    ]


def make_id() -> str:
    return str(uuid.uuid4())
