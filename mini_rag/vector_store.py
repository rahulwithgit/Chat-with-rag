"""Thin wrapper around ChromaDB persistent storage."""

from __future__ import annotations

import uuid

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from mini_rag.config import Settings

# One PersistentClient per DB path: creating clients reloads ONNX embedder metadata.
_clients: dict[str, ClientAPI] = {}


def get_collection(settings: Settings) -> Collection:
    path = settings.chroma_path
    if path not in _clients:
        _clients[path] = chromadb.PersistentClient(path=path)
    return _clients[path].get_or_create_collection(
        name=settings.collection_name,
        metadata={"description": "mini_rag document chunks"},
    )


def add_documents(
    collection: Collection,
    chunks: list[str],
    source_label: str,
) -> int:
    if not chunks:
        return 0
    ids = [f"{source_label}:{uuid.uuid4().hex}" for _ in chunks]
    metadatas = [{"source": source_label} for _ in chunks]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def query_similar(collection: Collection, query: str, top_k: int) -> list[tuple[str, str | None]]:
    result = collection.query(query_texts=[query], n_results=top_k)
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    out: list[tuple[str, str | None]] = []
    for doc, meta in zip(docs, metas):
        source = None
        if isinstance(meta, dict):
            source = meta.get("source")
        out.append((doc, source if isinstance(source, str) else None))
    return out
