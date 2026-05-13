"""
GraphRAG — knowledge graph alongside the vector store.
(Inspired by Microsoft GraphRAG, Edge et al. 2024)

Build phase:
  chunks → GPT-4o-mini extracts (entity, relation, entity) triples
         → stored as adjacency JSON at data/graphs/{collection}.json

Query phase:
  question → extract entities → graph traversal → return related entity context
           → used to boost Pinecone metadata filters or enrich the prompt
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from core.config import settings
from core.openai_client import chat
from ingestion.chunker import Chunk

GRAPHS_DIR = Path("data/graphs")

_EXTRACT_SYSTEM = """\
Extract all named entities and relationships from this technical text.
Return JSON:
{
  "entities": ["EntityA", "EntityB"],
  "relations": [["EntityA", "relation_verb", "EntityB"]]
}
Only include meaningful technical entities (classes, methods, concepts, products).
"""

_QUERY_SYSTEM = """\
Extract the key technical entities from this question.
Return JSON: {"entities": ["entity1", "entity2"]}
"""


# ── Build ──────────────────────────────────────────────────────────────────

def _extract_triples(text: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    try:
        raw = chat(
            model=settings.fast_model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": text[:1200]},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
        entities = [e.lower() for e in data.get("entities", [])]
        relations = [
            (r[0].lower(), r[1].lower(), r[2].lower())
            for r in data.get("relations", [])
            if len(r) == 3
        ]
        return entities, relations
    except Exception:
        return [], []


def build_graph(chunks: list[Chunk], collection: str) -> dict:
    """
    Build and persist a knowledge graph from chunks.
    Returns the graph dict (adjacency + relation lists).
    """
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    graph_path = GRAPHS_DIR / f"{collection}.json"

    if graph_path.exists():
        print(f"[graph] loaded existing graph for '{collection}'")
        return json.loads(graph_path.read_text())

    adjacency: dict[str, set[str]] = defaultdict(set)
    relations: list[dict] = []
    entity_sources: dict[str, list[str]] = defaultdict(list)

    print(f"[graph] building knowledge graph from {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        entities, triples = _extract_triples(chunk.text)
        for entity in entities:
            entity_sources[entity].append(chunk.metadata.get("source_url", ""))
        for subj, rel, obj in triples:
            adjacency[subj].add(obj)
            adjacency[obj].add(subj)
            relations.append({"subject": subj, "relation": rel, "object": obj,
                               "source_url": chunk.metadata.get("source_url", "")})
        if (i + 1) % 20 == 0:
            print(f"  [graph] {i + 1}/{len(chunks)} chunks processed")

    graph = {
        "collection": collection,
        "adjacency": {k: list(v) for k, v in adjacency.items()},
        "relations": relations,
        "entity_sources": {k: list(set(v)) for k, v in entity_sources.items()},
    }
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"[graph] saved: {len(adjacency)} entities, {len(relations)} relations")
    return graph


# ── Query ──────────────────────────────────────────────────────────────────

def _load_graph(collection: str) -> dict | None:
    path = GRAPHS_DIR / f"{collection}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _question_entities(question: str) -> list[str]:
    try:
        raw = chat(
            model=settings.fast_model,
            messages=[
                {"role": "system", "content": _QUERY_SYSTEM},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        return [e.lower() for e in json.loads(raw).get("entities", [])]
    except Exception:
        return []


def graph_context(question: str, collection: str, hops: int = 2) -> str:
    """
    Return a string describing entities related to the question,
    to be appended to the RAG prompt as structured context.
    """
    graph = _load_graph(collection)
    if not graph:
        return ""

    adjacency = graph.get("adjacency", {})
    relations = graph.get("relations", [])
    question_entities = _question_entities(question)

    if not question_entities:
        return ""

    # Multi-hop traversal
    frontier = set(question_entities)
    visited = set()
    for _ in range(hops):
        next_frontier: set[str] = set()
        for entity in frontier:
            if entity in visited:
                continue
            visited.add(entity)
            next_frontier.update(adjacency.get(entity, []))
        frontier = next_frontier - visited

    related = visited | set(question_entities)

    # Find relations connecting related entities
    relevant_relations = [
        f"{r['subject']} → {r['relation']} → {r['object']}"
        for r in relations
        if r["subject"] in related and r["object"] in related
    ][:20]

    if not relevant_relations:
        return ""

    return "Knowledge graph context:\n" + "\n".join(relevant_relations)
