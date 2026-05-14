"""Split text into semantic chunks for embedding and retrieval.

Primary strategy: split on paragraph boundaries (double newlines) so each
coherent section stays together.  When a single paragraph exceeds
*chunk_size* characters it is sub-split using a fixed-size sliding window
with *overlap* as a fallback.
"""

from __future__ import annotations


def _fixed_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback: split *text* into fixed-size overlapping windows."""
    chunks: list[str] = []
    start = 0
    n = len(text)
    step = chunk_size - overlap

    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += step

    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* into chunks, preferring paragraph boundaries.

    Paragraphs are delimited by two or more consecutive newlines.  Short
    paragraphs are kept whole; paragraphs longer than *chunk_size* are
    sub-split with :func:`_fixed_split`.

    Parameters keep the same meaning as the previous fixed-window version so
    callers do not need to change.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            chunks.extend(_fixed_split(para, chunk_size, overlap))

    return chunks
