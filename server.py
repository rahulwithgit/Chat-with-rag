"""Lightweight FastAPI server exposing the RAG pipeline over HTTP."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mini_rag.config import get_settings
from mini_rag.ollama_llm import OllamaError
from mini_rag.pipeline import answer_question, ingest_directory, ingest_file
from mini_rag.vector_store import get_collection

app = FastAPI(title="Mini RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    elapsed_ms: int


class IngestRequest(BaseModel):
    path: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


@app.get("/health", response_model=HealthResponse)
def health():
    settings = get_settings()
    try:
        col = get_collection(settings)
        count = col.count()
    except Exception:
        count = 0
    return HealthResponse(status="ok", indexed_chunks=count)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.perf_counter()
    try:
        ans = answer_question(req.question)
    except OllamaError as e:
        raise HTTPException(status_code=502, detail=str(e))
    elapsed = int((time.perf_counter() - t0) * 1000)
    return AskResponse(answer=ans, elapsed_ms=elapsed)


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")
    settings = get_settings()
    if p.is_file():
        n = ingest_file(p, settings=settings)
    elif p.is_dir():
        n = ingest_directory(p, ("*.txt", "*.md"), settings=settings)
    else:
        raise HTTPException(status_code=400, detail="Unsupported path type")
    return IngestResponse(chunks=n, message=f"Ingested {n} chunk(s) from {req.path}")
