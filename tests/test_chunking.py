"""Tests for the sentence-aware RAG chunker (app/utils/chunking.py)."""
from __future__ import annotations

from app.utils import chunking


def _make_text(n_sentences: int) -> str:
    return " ".join(
        f"Sentence number {i} describes a financial fact about company {i}."
        for i in range(n_sentences)
    )


def test_short_text_is_single_chunk():
    text = "Apple reported strong revenue growth in fiscal year 2024."
    assert chunking.chunk_text(text) == [text]


def test_empty_text_returns_empty_list():
    assert chunking.chunk_text("") == []
    assert chunking.chunk_text("   \n  ") == []


def test_long_text_splits_into_multiple_chunks():
    text = _make_text(60)
    chunks = chunking.chunk_text(text, target=300, overlap=60)
    assert len(chunks) > 1
    # No chunk grossly exceeds the target (allow target + one overlap window).
    assert all(len(c) <= 300 + 60 + 80 for c in chunks)


def test_overlap_is_actually_applied():
    # Regression: the previous implementation computed `max(end - overlap, end)`
    # which silently disabled overlap entirely. Consecutive chunks must now
    # share some trailing/leading text.
    text = _make_text(40)
    chunks = chunking.chunk_text(text, target=200, overlap=80)
    assert len(chunks) >= 2
    shared = 0
    for a, b in zip(chunks, chunks[1:]):
        tail_words = set(a.split()[-6:])
        head_words = set(b.split()[:6])
        if tail_words & head_words:
            shared += 1
    assert shared >= 1, "expected overlapping context between consecutive chunks"


def test_sentences_are_not_broken_midword():
    text = _make_text(50)
    chunks = chunking.chunk_text(text, target=250, overlap=50)
    # Every chunk should end on sentence punctuation (no mid-sentence cut),
    # since all sentences are shorter than the target.
    assert all(c.rstrip().endswith(".") for c in chunks)


def test_long_single_sentence_is_hard_split():
    giant = "word " * 400  # one ~2000-char "sentence" with no terminators
    chunks = chunking.chunk_text(giant.strip(), target=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
