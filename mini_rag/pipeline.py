"""High-level ingest and query operations."""

from __future__ import annotations

from pathlib import Path

from mini_rag.chunking import chunk_text
from mini_rag.config import Settings, get_settings
from mini_rag.ollama_llm import chat
from mini_rag.vector_store import add_documents, get_collection, query_similar


def ingest_file(path: Path, settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    collection = get_collection(settings)
    label = str(path.resolve())
    return add_documents(collection, chunks, source_label=label)


def ingest_directory(dir_path: Path, patterns: tuple[str, ...], settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    total = 0
    for pattern in patterns:
        for p in sorted(dir_path.glob(pattern)):
            if p.is_file():
                total += ingest_file(p, settings=settings)
    return total


SYSTEM_PROMPT = """You are a careful assistant. Answer using ONLY the provided context.
If the context does not contain enough information, say you do not know and suggest what might be missing.
Keep answers concise (at most a few sentences unless the question requires lists)."""


def build_user_prompt(
    question: str,
    contexts: list[tuple[str, str | None]],
    max_context_chars: int,
) -> str:
    blocks = []
    for i, (doc, source) in enumerate(contexts, start=1):
        src = f" (source: {source})" if source else ""
        blocks.append(f"[{i}]{src}\n{doc}")
    joined = "\n\n---\n\n".join(blocks)
    if len(joined) > max_context_chars:
        joined = joined[: max_context_chars - 40].rstrip() + "\n\n…(context truncated for speed)"
    return f"Context:\n\n{joined}\n\nQuestion: {question}"


def answer_question(question: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    collection = get_collection(settings)
    hits = query_similar(collection, question, settings.top_k)
    if not hits:
        return (
            "No documents are indexed yet. Ingest `.txt` or `.md` files first "
            "(see README)."
        )
    user = build_user_prompt(question, hits, settings.max_context_chars)
    return chat(settings, SYSTEM_PROMPT, user)
