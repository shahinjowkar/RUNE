"""
Full ingestion pipeline for any documentation URL.

Usage:
    python -m ingestion.pipeline --url https://wiremock.org/docs/ --collection wiremock

Or via the API:
    POST /ingest  {"url": "https://...", "collection": "my-docs"}

Steps:
  1. Discover URLs (sitemap auto-detect → recursive crawl fallback)
  2. Fetch + clean each page → data/raw/{collection}/
  3. Semantic + parent-child chunking
  4. Proposition extraction
  5. RAPTOR tree
  6. Knowledge graph
  7. Embed + upsert all to Pinecone (namespaced by collection)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from core.openai_client import embed
from core.pinecone_client import make_id, upsert
from ingestion.chunker import Chunk, chunks_from_file
from ingestion.crawler import discover
from ingestion.fetcher import fetch_all
from ingestion.propositions import extract_propositions
from ingestion.raptor import build_tree
from retrieval.graph_rag import build_graph


def _raw_dir(collection: str) -> Path:
    return Path("data/raw") / collection


def _upsert_chunks(chunks: list[Chunk], namespace: str) -> None:
    if not chunks:
        return
    print(f"[pipeline] embedding {len(chunks)} chunks → {namespace}")
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embed([c.text for c in batch])
        vectors = [
            {
                "id": c.id,
                "values": emb,
                "metadata": {**c.metadata, "text": c.text[:500]},
            }
            for c, emb in zip(batch, embeddings)
        ]
        upsert(vectors, namespace=namespace)
    print(f"[pipeline] ✓ {len(chunks)} vectors → '{namespace}'")


def _upsert_raptor(nodes, collection: str, level: int) -> None:
    if not nodes:
        return
    ns = f"{collection}-raptor-l{level}"
    vectors = [
        {
            "id": node.id,
            "values": node.embedding,
            "metadata": {
                "text": node.text[:500],
                "level": node.level,
                "chunk_type": f"raptor-l{level}",
                "collection": collection,
            },
        }
        for node in nodes
    ]
    upsert(vectors, namespace=ns)
    print(f"[pipeline] ✓ {len(vectors)} RAPTOR L{level} nodes → '{ns}'")


def run(url: str, collection: str) -> dict:
    print("=" * 60)
    print(f"INGESTION: {collection}")
    print(f"Source   : {url}")
    print("=" * 60)

    # 1. Discover URLs
    urls = discover(url)
    if not urls:
        return {"error": "No pages discovered", "collection": collection}

    # 2. Fetch pages
    raw_dir = _raw_dir(collection)
    txt_files = fetch_all(urls, raw_dir)
    print(f"\n[pipeline] {len(txt_files)} files ready\n")

    all_leaves: list[Chunk] = []
    all_parents: list[Chunk] = []

    # 3. Chunk each file
    for path in txt_files:
        slug = path.stem
        source_url = next((u for u in urls if slug in u), "")
        parents, leaves = chunks_from_file(path)
        for c in parents + leaves:
            c.metadata.setdefault("source_url", source_url)
            c.metadata["collection"] = collection
        all_parents.extend(parents)
        all_leaves.extend(leaves)

    print(f"[pipeline] {len(all_parents)} parents, {len(all_leaves)} leaves\n")

    # 4. Upsert leaf + parent chunks
    _upsert_chunks(all_leaves, namespace=f"{collection}-leaf")
    _upsert_chunks(all_parents, namespace=f"{collection}-parent")

    # 5. Proposition extraction
    print("\n[pipeline] extracting propositions...")
    props = extract_propositions(all_leaves[:200])  # cap for cost
    _upsert_chunks(props, namespace=f"{collection}-props")

    # 6. RAPTOR tree
    print("\n[pipeline] building RAPTOR tree...")
    levels = build_tree(all_leaves)
    for i, level_nodes in enumerate(levels, start=1):
        _upsert_raptor(level_nodes, collection=collection, level=i)

    # 7. Knowledge graph
    print("\n[pipeline] building knowledge graph...")
    build_graph(all_leaves[:300], collection=collection)  # cap for cost

    summary = {
        "collection": collection,
        "url": url,
        "pages": len(txt_files),
        "leaves": len(all_leaves),
        "parents": len(all_parents),
        "propositions": len(props),
        "raptor_levels": len(levels),
        "raptor_nodes": sum(len(l) for l in levels),
    }

    print("\n[pipeline] ✓ ingestion complete")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Docs URL to ingest")
    parser.add_argument("--collection", required=True, help="Collection name (used as Pinecone namespace prefix)")
    args = parser.parse_args()
    run(args.url, args.collection)
