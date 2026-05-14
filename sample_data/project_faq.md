# Mini RAG project FAQ

## What is this folder?

This is a small retrieval-augmented generation demo. Documents are split into
semantic chunks based on paragraph boundaries, embedded with a lightweight
ONNX model bundled by ChromaDB, and stored locally. Questions retrieve the
closest chunks and are answered by a local Ollama model.

## Hardware expectations

The default stack is tuned for laptops without a discrete GPU. The generator
model size is the main RAM driver; embedding uses Chroma's default ONNX model.
Tested on a MacBook Air M1 with 8 GB RAM using qwen2.5:0.5b.

## Features

- Semantic chunking with paragraph-boundary splitting
- In-memory query caching for instant repeated answers
- FastAPI backend with CORS-enabled REST API
- Beautiful dark-themed chat frontend (zero build step)
- Configurable via environment variables
