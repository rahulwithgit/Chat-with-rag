"""CLI: `python -m mini_rag` from the `rag` directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mini_rag.config import get_settings
from mini_rag.ollama_llm import OllamaError
from mini_rag.pipeline import answer_question, ingest_directory, ingest_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mini RAG with ChromaDB + Ollama")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index text/markdown files")
    p_ingest.add_argument(
        "path",
        type=Path,
        help="File or directory to ingest",
    )

    p_ask = sub.add_parser("ask", help="Ask a question over indexed documents")
    p_ask.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=None,
        help="Override TOP_K for this query only (not persisted)",
    )
    p_ask.add_argument("words", nargs="+", help="Your question (multiple words ok)")

    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "ingest":
        path: Path = args.path
        if not path.exists():
            print(f"Not found: {path}", file=sys.stderr)
            return 2
        if path.is_file():
            n = ingest_file(path, settings=settings)
            print(f"Ingested {n} chunk(s) from {path}")
            return 0
        if path.is_dir():
            n = ingest_directory(path, ("*.txt", "*.md"), settings=settings)
            print(f"Ingested {n} chunk(s) from {path} (*.txt, *.md)")
            return 0
        print(f"Unsupported path type: {path}", file=sys.stderr)
        return 2

    if args.command == "ask":
        q = " ".join(args.words).strip()
        if not q:
            print("Usage: python -m mini_rag ask your question here", file=sys.stderr)
            return 2
        if args.top_k is not None:
            from dataclasses import replace

            settings = replace(settings, top_k=args.top_k)
        try:
            print(answer_question(q, settings=settings))
        except OllamaError as e:
            print(str(e), file=sys.stderr)
            print(
                "Hint: start the Ollama app (or `ollama serve`), then `ollama pull "
                + settings.ollama_model
                + "`.",
                file=sys.stderr,
            )
            return 3
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
