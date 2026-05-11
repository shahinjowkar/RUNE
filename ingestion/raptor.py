"""
RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.
(Sarthi et al., Stanford 2024)

Algorithm:
  1. Embed all leaf chunks.
  2. K-means cluster them (k ≈ sqrt(n)).
  3. Summarise each cluster with gpt-4o-mini.
  4. Embed summaries → these become Level-1 nodes.
  5. Repeat up to `raptor_levels` times.

Each summary is upserted to Pinecone under namespace raptor-l{level}.
"""

import math
import uuid
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans

from core.config import settings
from core.openai_client import chat, embed
from ingestion.chunker import Chunk


@dataclass
class RaptorNode:
    id: str
    text: str
    level: int
    child_ids: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)


def _summarise(texts: list[str]) -> str:
    joined = "\n\n---\n\n".join(texts[:20])  # cap to avoid token limits
    return chat(
        model=settings.fast_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical writer. Produce a dense, factual summary "
                    "of the following document excerpts in 3-5 sentences."
                ),
            },
            {"role": "user", "content": joined},
        ],
    )


def _optimal_k(n: int) -> int:
    return max(2, min(int(math.sqrt(n)), n // settings.raptor_min_cluster_size))


def build_tree(leaves: list[Chunk]) -> list[list[RaptorNode]]:
    """
    Returns a list of levels, each a list of RaptorNodes.
    levels[0] = Level-1 summaries of leaves
    levels[1] = Level-2 summaries of Level-1 nodes, etc.
    """
    if not leaves:
        return []

    print(f"[raptor] embedding {len(leaves)} leaf chunks for clustering...")
    embeddings = embed([c.text for c in leaves])
    ids = [c.id for c in leaves]

    current_embeddings = embeddings
    current_ids = ids
    current_texts = [c.text for c in leaves]

    all_levels: list[list[RaptorNode]] = []

    for level in range(1, settings.raptor_levels + 1):
        n = len(current_embeddings)
        if n < settings.raptor_min_cluster_size:
            print(f"[raptor] level {level}: only {n} nodes, stopping.")
            break

        k = _optimal_k(n)
        print(f"[raptor] level {level}: {n} nodes → {k} clusters")

        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(np.array(current_embeddings))

        nodes: list[RaptorNode] = []
        node_embeddings: list[list[float]] = []
        node_ids: list[str] = []
        node_texts: list[str] = []

        for cluster_id in range(k):
            indices = [i for i, l in enumerate(labels) if l == cluster_id]
            cluster_texts = [current_texts[i] for i in indices]
            cluster_node_ids = [current_ids[i] for i in indices]

            summary = _summarise(cluster_texts)
            [emb] = embed([summary])

            node = RaptorNode(
                id=str(uuid.uuid4()),
                text=summary,
                level=level,
                child_ids=cluster_node_ids,
                embedding=emb,
            )
            nodes.append(node)
            node_embeddings.append(emb)
            node_ids.append(node.id)
            node_texts.append(summary)
            print(f"  cluster {cluster_id}: {len(indices)} children → summarised")

        all_levels.append(nodes)
        current_embeddings = node_embeddings
        current_ids = node_ids
        current_texts = node_texts

    return all_levels
