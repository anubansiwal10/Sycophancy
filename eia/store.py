"""Local vector store (Chroma by default, embedded Qdrant optional) over the KnowledgeBase sheet.
Metadata filters: club_id, event_type, doc_type (per the dataset README). No graph database anywhere."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

import numpy as np

from .config import S
from .data import load_dataset

# ------------------------------------------------------------------ embeddings -----------


class Embedder:
    def __init__(self):
        self.provider = S.embed_provider
        if self.provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            self._emb = OllamaEmbeddings(model=S.embed_model, base_url=S.ollama_base_url)
        elif self.provider == "hash":  # offline / tests only - lexical, not semantic
            from sklearn.feature_extraction.text import HashingVectorizer
            self._emb = HashingVectorizer(n_features=1024, alternate_sign=False, norm="l2", ngram_range=(1, 2))
        else:
            raise ValueError(f"unknown embed provider {self.provider}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "ollama":
            return self._emb.embed_documents(texts)
        return self._emb.transform(texts).toarray().astype(float).tolist()


# ------------------------------------------------------------------ backends -------------


def _hit(doc_id, text, meta, score) -> dict:
    return {"doc_id": doc_id, "text": text, "doc_type": meta["doc_type"], "club_id": meta["club_id"],
            "event_id": meta.get("event_id", ""), "event_type": meta.get("event_type", ""),
            "doc_date": meta.get("doc_date", ""), "title": meta.get("title", ""), "score": float(score)}


class ChromaStore:
    def __init__(self):
        import chromadb
        S.vector_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(S.vector_dir / "chroma"))
        self.name = f"kb_{S.embed_provider}_{S.embed_model.replace(':', '_').replace('-', '_')}"[:60]
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.col.count()

    def reset(self):
        self.client.delete_collection(self.name)
        self.col = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def add(self, ids, vectors, metas, docs):
        for i in range(0, len(ids), 200):
            self.col.add(ids=ids[i:i + 200], embeddings=vectors[i:i + 200], metadatas=metas[i:i + 200], documents=docs[i:i + 200])

    def search(self, vec, k, club_ids=None, event_type=None, doc_types=None) -> list[dict]:
        clauses = []
        if club_ids:
            clauses.append({"club_id": {"$in": list(club_ids)}})
        if event_type:
            clauses.append({"event_type": event_type})
        if doc_types:
            clauses.append({"doc_type": {"$in": list(doc_types)}})
        where = None if not clauses else (clauses[0] if len(clauses) == 1 else {"$and": clauses})
        r = self.col.query(query_embeddings=[vec], n_results=k, where=where)
        return [_hit(i, d, m, 1 - dist) for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]


class QdrantStore:
    def __init__(self):
        from qdrant_client import QdrantClient
        S.vector_dir.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(S.vector_dir / "qdrant"))  # embedded mode, no server needed
        self.name = f"kb_{S.embed_provider}_{S.embed_model.replace(':', '_').replace('-', '_')}"[:60]

    def count(self) -> int:
        return self.client.count(self.name).count if self.client.collection_exists(self.name) else 0

    def reset(self):
        if self.client.collection_exists(self.name):
            self.client.delete_collection(self.name)
        self._dim = None

    def add(self, ids, vectors, metas, docs):
        from qdrant_client.models import Distance, PointStruct, VectorParams
        if not self.client.collection_exists(self.name):
            self.client.create_collection(self.name, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
        pts = [PointStruct(id=n, vector=v, payload={**m, "doc_id": i, "text": d}) for n, (i, v, m, d) in enumerate(zip(ids, vectors, metas, docs))]
        for i in range(0, len(pts), 200):
            self.client.upsert(self.name, pts[i:i + 200])

    def search(self, vec, k, club_ids=None, event_type=None, doc_types=None) -> list[dict]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
        must = []
        if club_ids:
            must.append(FieldCondition(key="club_id", match=MatchAny(any=list(club_ids))))
        if event_type:
            must.append(FieldCondition(key="event_type", match=MatchValue(value=event_type)))
        if doc_types:
            must.append(FieldCondition(key="doc_type", match=MatchAny(any=list(doc_types))))
        res = self.client.query_points(self.name, query=vec, query_filter=Filter(must=must) if must else None, limit=k).points
        return [_hit(p.payload["doc_id"], p.payload["text"], p.payload, p.score) for p in res]


# ------------------------------------------------------------------ public API -----------


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


@lru_cache(maxsize=1)
def get_store():
    store = QdrantStore() if S.vector_backend == "qdrant" else ChromaStore()
    kb = load_dataset().kb
    if store.count() != len(kb):
        build_index(store)
    return store


def build_index(store=None) -> int:
    """(Re)build the index from the KnowledgeBase sheet. Idempotent."""
    store = store or (QdrantStore() if S.vector_backend == "qdrant" else ChromaStore())
    kb = load_dataset().kb
    store.reset()
    emb = get_embedder()
    ids, metas, docs = [], [], []
    for r in kb.itertuples():
        ids.append(str(r.doc_id))
        docs.append(f"{r.title}. {r.content}")
        metas.append({"doc_id": str(r.doc_id), "doc_type": str(r.doc_type), "club_id": str(r.club_id),
                      "event_id": "" if str(r.event_id) == "nan" else str(r.event_id),
                      "event_type": "" if str(r.event_type) == "nan" else str(r.event_type),
                      "doc_date": str(r.doc_date)[:10], "title": str(r.title)})
    vectors = []
    for i in range(0, len(docs), 64):
        vectors += emb.embed(docs[i:i + 64])
    store.add(ids, vectors, metas, docs)
    return len(ids)
