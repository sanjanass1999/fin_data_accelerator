"""Content-aware text chunking for the RAG knowledge base.

The platform stores two very different kinds of content, so it deliberately
uses a *hybrid* chunking strategy:

* **Structured records** - a per-company financial narrative or a per-table
  "schema card" - are already short, self-contained semantic units. Each one
  becomes exactly one chunk ("record-based" chunking). Splitting them would
  only sever facts that belong together (revenue, margin and assets for the
  same company-year must stay in one passage to retrieve and cite cleanly).

* **Long-form prose** - the ``earnings_reports`` documents and any ingested
  PDF/TXT - is split here with a **sentence-aware recursive** strategy: it
  packs whole sentences up to a target size, never cuts mid-sentence, and
  carries a real character **overlap** from the tail of one chunk into the
  head of the next so a fact that straddles a boundary is still retrievable
  from both sides.

This module is the single source of truth for prose chunking; both the offline
seed script (:mod:`scripts.seed_data`) and the live transform agent
(:mod:`app.agents.transform`) import :func:`chunk_text` so the two paths can
never drift apart.
"""
from __future__ import annotations

import re
from typing import List

# Split on whitespace that follows sentence-ending punctuation, keeping the
# punctuation attached to the preceding sentence.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

DEFAULT_TARGET = 600
DEFAULT_OVERLAP = 80
DEFAULT_MIN_CHUNK = 120


def _normalize(text: str) -> str:
    """Collapse all runs of whitespace into single spaces."""
    return " ".join((text or "").split())


def _hard_split(text: str, target: int, overlap: int) -> List[str]:
    """Last-resort splitter for a single sentence longer than ``target``.

    Falls back to a fixed-size sliding window (with a *working* overlap) when a
    sentence cannot be kept whole.
    """
    chunks: List[str] = []
    start = 0
    n = len(text)
    step = max(target - overlap, 1)
    while start < n:
        chunks.append(text[start:start + target].strip())
        if start + target >= n:
            break
        start += step
    return [c for c in chunks if c]


def _overlap_tail(sentences: List[str], overlap: int) -> List[str]:
    """Return the trailing sentences whose combined length is about ``overlap``.

    These are replayed at the start of the next chunk to preserve context
    continuity across the boundary.
    """
    if overlap <= 0:
        return []
    tail: List[str] = []
    length = 0
    for sent in reversed(sentences):
        if length and length + len(sent) + 1 > overlap:
            break
        tail.insert(0, sent)
        length += len(sent) + 1
    return tail


def chunk_text(
    text: str,
    target: int = DEFAULT_TARGET,
    overlap: int = DEFAULT_OVERLAP,
    min_chunk: int = DEFAULT_MIN_CHUNK,
) -> List[str]:
    """Split ``text`` into sentence-aware, overlapping chunks of ~``target`` chars.

    * Whole sentences are never broken (a sentence longer than ``target`` is the
      only exception and is hard-split as a fallback).
    * Consecutive chunks share ~``overlap`` characters of context.
    * A short trailing remainder (< ``min_chunk``) is merged into the previous
      chunk so we never emit a tiny orphan passage.
    """
    text = _normalize(text)
    if not text:
        return []
    if len(text) <= target:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        if len(sent) > target:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(_hard_split(sent, target, overlap))
            continue
        if current and current_len + len(sent) + 1 > target:
            chunks.append(" ".join(current))
            current = _overlap_tail(current, overlap)
            current_len = sum(len(s) + 1 for s in current)
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        tail = " ".join(current)
        if chunks and len(tail) < min_chunk:
            chunks[-1] = f"{chunks[-1]} {tail}"
        else:
            chunks.append(tail)
    return [c for c in chunks if c]
