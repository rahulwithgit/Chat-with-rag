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
    ollama_keep_alive: str
    ollama_num_predict: int
    ollama_num_ctx: int
    ollama_temperature: float
    chroma_path: str
    collection_name: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_context_chars: int


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def get_settings() -> Settings:
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b"),
        ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        ollama_num_predict=_env_int("OLLAMA_NUM_PREDICT", 256),
        ollama_num_ctx=_env_int("OLLAMA_NUM_CTX", 2048),
        ollama_temperature=_env_float("OLLAMA_TEMPERATURE", 0.2),
        chroma_path=os.getenv("CHROMA_PATH", ".chroma"),
        collection_name=os.getenv("COLLECTION_NAME", "mini_rag_docs"),
        chunk_size=_env_int("CHUNK_SIZE", 400),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 80),
        top_k=_env_int("TOP_K", 4),
        max_context_chars=_env_int("MAX_CONTEXT_CHARS", 2000),
    )
