"""In-memory vector store backed by a pickle file on disk.

At import time we load `data/index.pkl` (path from settings.index_path), normalize
the embedding matrix for cosine similarity, and keep it in memory. Queries are a
single matrix-vector product — microseconds for any plausible portfolio corpus.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings

_STORE: dict[str, Any] = {"chunks": [], "matrix": np.zeros((0, 0), dtype=np.float32)}


def _load_from_disk() -> None:
    p = Path(settings.index_path)
    if not p.exists():
        print(f"[retrieval] index not found at {p.resolve()} — serving empty store")
        _STORE["chunks"] = []
        _STORE["matrix"] = np.zeros((0, settings.embedding_dim), dtype=np.float32)
        return

    with p.open("rb") as f:
        data = pickle.load(f)

    chunks: list[dict] = data["chunks"]
    raw = np.asarray(data["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    _STORE["chunks"] = chunks
    _STORE["matrix"] = raw / norms
    print(f"[retrieval] loaded {len(chunks)} chunks from {p.name}")


_load_from_disk()


def match_documents(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    k = top_k or settings.top_k
    chunks: list[dict] = _STORE["chunks"]
    if not chunks:
        return []
    q = np.asarray(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []
    q = q / q_norm

    sims = _STORE["matrix"] @ q
    top = np.argsort(-sims)[:k]
    return [
        {
            "content": chunks[i]["content"],
            "metadata": chunks[i]["metadata"],
            "similarity": float(sims[i]),
        }
        for i in top
    ]


def save_index(chunks: list[dict], embeddings: list[list[float]]) -> Path:
    """Persist chunks + embeddings to index_path. Called by the ingest script."""
    p = Path(settings.index_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(
            {
                "chunks": chunks,
                "embeddings": embeddings,
                "model": settings.embedding_model,
                "dim": settings.embedding_dim,
            },
            f,
        )
    return p


def reload_index() -> int:
    """Re-read the index file from disk (called after POST /api/ingest)."""
    _load_from_disk()
    return len(_STORE["chunks"])
