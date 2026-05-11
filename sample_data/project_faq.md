# Mini RAG project FAQ

## What is this folder?

This is a small retrieval-augmented generation demo. Documents are split into
chunks, embedded with a lightweight model bundled by ChromaDB, and stored
locally. Questions retrieve the closest chunks and are answered by a local
Ollama model.

## Hardware expectations

The default stack is tuned for laptops without a discrete GPU. The generator
model size is the main RAM driver; embedding uses Chroma's default ONNX model.
