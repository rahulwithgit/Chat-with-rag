"""Load settings from environment (optional `.env` in project root)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_model: str
    chroma_path: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def get_settings() -> Settings:
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
        chroma_path=os.getenv("CHROMA_PATH", ".chroma"),
        collection_name=os.getenv("COLLECTION_NAME", "mini_rag_docs"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "400")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        top_k=int(os.getenv("TOP_K", "4")),
    )
