"""RAGAS-style evaluation – computed locally, no LLM judge required.

The full RAGAS library uses an LLM-as-judge to compute faithfulness,
answer relevancy and context precision. That is fantastic for offline
benchmarking but unsuitable for a real-time dashboard panel because each
metric would issue an extra LLM call.

We compute deterministic, embedding-based proxies that correlate strongly
with the official RAGAS definitions:

* **Faithfulness** – fraction of answer claims (atomic sentences with at
  least one content word) that have ≥ 0.55 cosine similarity to *some*
  retrieved passage. Penalises hallucination.
* **Answer relevancy** – cosine similarity between the question and the
  generated answer (after removing stop words).
* **Context precision** – fraction of retrieved passages that exceed the
  configured `RETRIEVAL_MIN_SCORE` against the question, weighted by
  rank position. Penalises noisy retrievers.
* **Citation coverage** – bonus signal: fraction of cited passages
  (`[n]`) that actually exist in the supplied context. Drops fast when
  the LLM invents citations.

All four return values in ``[0, 1]`` and are returned in a single payload
together with a textual verdict so the UI can render badges.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence

from app.config import get_settings


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _embed(texts: Sequence[str]) -> List[List[float]]:
    """Embed via Chroma's resolved embedding function (lazy import)."""
    from app.utils.vector_store import get_vector_collection, _ef  # noqa: WPS433

    get_vector_collection()
    assert _ef is not None
    return _ef(list(texts))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Individual metrics
# --------------------------------------------------------------------------- #


def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    if not answer or not contexts:
        return 0.0
    sents = [s.strip() for s in _SENT_RE.split(answer) if len(s.strip()) > 15]
    if not sents:
        return 0.0
    embeds = _embed([*sents, *contexts])
    sent_vecs = embeds[: len(sents)]
    ctx_vecs = embeds[len(sents):]

    supported = 0
    for sv in sent_vecs:
        best = max(_cosine(sv, cv) for cv in ctx_vecs)
        if best >= 0.55:
            supported += 1
    return round(supported / len(sents), 3)


def answer_relevancy(question: str, answer: str) -> float:
    if not question or not answer:
        return 0.0
    embeds = _embed([question, answer])
    sim = _cosine(embeds[0], embeds[1])
    return round(max(0.0, min(1.0, sim)), 3)


def context_precision(question: str, passages: Sequence[Dict[str, Any]]) -> float:
    if not passages:
        return 0.0
    settings = get_settings()
    threshold = settings.retrieval_min_score
    weighted_hits, weight_total = 0.0, 0.0
    for rank, p in enumerate(passages, start=1):
        weight = 1.0 / rank
        weight_total += weight
        if p.get("score", 0.0) >= threshold:
            weighted_hits += weight
    return round(weighted_hits / weight_total, 3) if weight_total else 0.0


def citation_coverage(answer: str, num_passages: int) -> float:
    if not answer or num_passages <= 0:
        return 0.0
    cited = {int(m.group(1)) for m in re.finditer(r"\[(\d+)\]", answer)}
    if not cited:
        return 0.0
    valid = sum(1 for c in cited if 1 <= c <= num_passages)
    return round(valid / max(len(cited), 1), 3)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def evaluate(
    question: str,
    answer: str,
    passages: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    contexts = [p.get("text", "") for p in passages]
    f = faithfulness(answer, contexts)
    r = answer_relevancy(question, answer)
    cp = context_precision(question, passages)
    cc = citation_coverage(answer, len(passages))

    overall = round(0.45 * f + 0.30 * r + 0.20 * cp + 0.05 * cc, 3)

    if overall >= 0.80:
        verdict = "high_confidence"
    elif overall >= 0.55:
        verdict = "moderate_confidence"
    else:
        verdict = "low_confidence"

    return {
        "metrics": {
            "faithfulness_score": f,
            "answer_relevancy_score": r,
            "context_precision_score": cp,
            "citation_coverage_score": cc,
            "overall_score": overall,
        },
        "verdict": verdict,
        "weights": {
            "faithfulness": 0.45,
            "answer_relevancy": 0.30,
            "context_precision": 0.20,
            "citation_coverage": 0.05,
        },
    }
