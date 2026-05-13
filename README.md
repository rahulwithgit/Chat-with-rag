# Mini RAG (Python)

A **small, modular** retrieval-augmented generation stack meant for a **MacBook Air M1** with **Ollama** installed locally. It keeps dependencies light: **ChromaDB** (persistent vector store + default small ONNX embeddings) and **httpx** for talking to Ollama.

## What this project does

1. **Ingest** plain text or Markdown files: split into overlapping chunks, embed, and store in a local Chroma database under `.chroma/`.
2. **Ask** a question: retrieve the top‑K most similar chunks, pack them into a prompt, and get an answer from your chosen Ollama model.

No GPU is required; everything runs on CPU. The heaviest part is usually the **chat model** you pick in Ollama.

## Repository layout

| Path | Role |
|------|------|
| `mini_rag/config.py` | Environment-based settings (`OLLAMA_*`, chunk sizes, paths). |
| `mini_rag/chunking.py` | Fixed-size overlapping text chunks. |
| `mini_rag/vector_store.py` | ChromaDB collection helpers (add + similarity search). |
| `mini_rag/ollama_llm.py` | POST `/api/chat` to Ollama. |
| `mini_rag/pipeline.py` | `ingest_*` and `answer_question` orchestration. |
| `mini_rag/__main__.py` | CLI entrypoint (`python -m mini_rag`). |
| `sample_data/` | Example `.txt` / `.md` files for a first run. |
| `.env.example` | Copy to `.env` to override defaults. |

## Prerequisites

- **Python 3.10+** recommended (3.9 may work; not explicitly tested here).
- **[Ollama](https://ollama.com/)** running locally (`ollama serve` is usually automatic after install).
- A **small** pullable model. Defaults in this repo target a **very small** chat model:

```bash
ollama pull qwen2.5:0.5b
```

Other reasonable lightweight options (pick one and set `OLLAMA_MODEL`):

| Model | Notes |
|-------|--------|
| `qwen2.5:0.5b` | **Default** — very small, fast on M1 Air. |
| `tinyllama` | Popular tiny model; still modest size vs 7B+ stacks. |
| `llama3.2:1b` | Slightly larger; often better phrasing than the smallest tier. |

ChromaDB will download its **default embedding** model on first use (small ONNX; suitable for laptops). That is separate from Ollama.

## Setup

From this directory (`rag/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: copy environment defaults and edit.

```bash
cp .env.example .env
```

## Usage

All commands assume your shell is in `rag/` and the venv is activated.

### 1. Index documents

Ingest a **single file**:

```bash
python -m mini_rag ingest sample_data/coffee_notes.txt
```

Ingest **every** `*.txt` and `*.md` in a folder (non-recursive; only that directory):

```bash
python -m mini_rag ingest sample_data
```

Re-ingesting **adds** new chunks; it does not deduplicate identical text. To start clean, delete the `.chroma/` folder (see Configuration).

### 2. Ask questions

```bash
python -m mini_rag ask 'What water temperature is suggested for pour-over?'
```

On **zsh**, put the question in **single quotes** if it contains `?`, otherwise the shell treats `?` as a glob pattern.

Optional: change how many chunks are retrieved for one query:

```bash
python -m mini_rag ask -k 6 Why store beans in an airtight container?
```

## Configuration

Environment variables (or `.env` next to `requirements.txt`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP API base. |
| `OLLAMA_MODEL` | `qwen2.5:0.5b` | Model name for `/api/chat`. |
| `CHROMA_PATH` | `.chroma` | Persistent Chroma directory (created automatically). |
| `COLLECTION_NAME` | `mini_rag_docs` | Logical collection name inside Chroma. |
| `CHUNK_SIZE` | `400` | Characters per chunk (rough length). |
| `CHUNK_OVERLAP` | `80` | Overlap between consecutive chunks. |
| `TOP_K` | `4` | Chunks to retrieve per question. |
| `MAX_CONTEXT_CHARS` | `2000` | Max characters of retrieved context sent to Ollama (smaller = less prompt work). |
| `OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model resident after a request (`30m`, `0`, etc.). Avoids cold reload between asks. |
| `OLLAMA_NUM_PREDICT` | `256` | Max tokens to generate (lower = faster, shorter answers). |
| `OLLAMA_NUM_CTX` | `2048` | Context window in **tokens**; raise if Ollama errors on long prompts. |
| `OLLAMA_TEMPERATURE` | `0.2` | Lower = faster / more deterministic sampling on small models. |

### Faster replies without changing the model

Most latency is usually **Ollama generation** and **loading the model** if it was unloaded.

1. **`OLLAMA_KEEP_ALIVE`** — Default `30m` keeps the same model in RAM between CLI runs so the next question does not pay a full load cost. Set to `0` only if you need to free RAM after each ask.
2. **`OLLAMA_NUM_PREDICT`** — Default `256` caps output length; try `128` for snappier short answers.
3. **`TOP_K`** or **`python -m mini_rag ask -k 2 ...`** — Fewer chunks → shorter prompt → less work per token.
4. **`MAX_CONTEXT_CHARS`** — Default `2000` trims huge retrievals before the LLM sees them.
5. **`OLLAMA_NUM_CTX`** — Only lower this (e.g. `1024`) if prompts are small and you want a bit less KV work; if you see context errors, raise it again.

The code also **reuses** the HTTP client to Ollama and one **Chroma persistent client per DB path** inside a single Python process (e.g. a REPL or a small API you add later).

## How retrieval + generation work

1. **Chunking** — Each file is read as UTF-8 text, normalized lightly, and split with overlap so sentences at chunk boundaries are less likely to be cut in half without context.
2. **Embedding + storage** — Chroma computes embeddings and stores `(id, document text, metadata)` where metadata includes the source file path.
3. **Query** — Your question is embedded the same way; Chroma returns the nearest chunks by cosine distance in embedding space.
4. **Generation** — Those chunks are concatenated into a **context** block. Ollama receives a short system instruction (“answer only from context”) and the user message with context + question.

This is intentionally minimal: no hybrid BM25, no re-ranking, no streaming UI.

## Troubleshooting

- **`Could not reach Ollama`** — Start Ollama or fix `OLLAMA_BASE_URL`. Confirm with `curl http://127.0.0.1:11434/api/tags`.
- **First run is slow** — Chroma downloads a small ONNX embedding model (~79 MB) into `~/.cache/chroma/` once.
- **`Failed to send telemetry event` / `capture() takes 1 positional argument`** — Chroma 0.6.x is incompatible with **posthog 6+**. This repo pins **`posthog>=2.4,<6`** in `requirements.txt`. Reinstall deps: `pip install -r requirements.txt`.
- **HTTP 404 from Ollama** — Model not pulled: `ollama pull <OLLAMA_MODEL>`.
- **RAM pressure** — Use a smaller `OLLAMA_MODEL`, reduce `TOP_K`, or shorten `CHUNK_SIZE` so prompts stay smaller.
- **Stale or wrong answers after edits** — Remove `.chroma` and re-ingest, or use a new `COLLECTION_NAME` / `CHROMA_PATH` for isolation.

## Extending the codebase

Reasonable next steps (not implemented here) without turning this into a large framework:

- Recursive directory ingest with ignore rules.
- **Delete / upsert** by `source` metadata before re-ingesting a file.
- Swap Chroma’s default embedding for **Ollama embeddings** (`/api/embeddings`) if you prefer a single vendor stack.
- Add FastAPI + one endpoint for `POST /query` for local demos.

## License

Internal / educational use in this repo; add a license file at the monorepo root if you need redistribution terms.
