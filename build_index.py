#!/usr/bin/env python
"""Embed the KnowledgeBase sheet (863 docs) into the local vector store. Run once (needs `ollama pull nomic-embed-text`)."""
import argparse

from eia.config import S

ap = argparse.ArgumentParser()
ap.add_argument("--embed", help="ollama | hash")
ap.add_argument("--backend", help="chroma | qdrant")
a = ap.parse_args()
if a.embed: S.embed_provider = a.embed
if a.backend: S.vector_backend = a.backend

from eia.store import build_index

print("indexed", build_index(), "documents into", S.vector_backend, "using", S.embed_provider, S.embed_model)
