"""ChromaDB-backed retrieval layer with MMR re-ranking and hybrid keyword boost.

Improvements over the original implementation:

* Lazy singleton initialisation (so imports stay cheap inside Uvicorn workers).
* Real distance scores returned alongside documents.
* MMR (Maximal Marginal Relevance) re-ranker that diversifies retrieved
  passages – this dramatically improves answer quality on multi-faceted
  questions like "compare margins across sectors".
* Hybrid keyword boost: ticker mentions (AAPL, NVDA, ...) and quoted phrases
  in the query bias the score upwards, fixing the common RAG failure where
  a semantically similar but factually unrelated chunk wins.
* Metadata-aware ingestion + filtered search.
"""
from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
from chromadb.utils import embedding_functions

from app.config import get_settings
from app.logging_config import get_logger

log = get_logger("vector_store")

_chroma_client = None
_collection = None
_schema_collection = None
_ef = None


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #


def _build_embedding_function():
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "gemini" and settings.gemini_api_key:
        try:
            return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=settings.gemini_api_key,
                model_name="models/text-embedding-004",
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("gemini embeddings unavailable, falling back to local", extra={"error": str(exc)})

    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
    except Exception as exc:
        log.warning(
            "sentence-transformers unavailable, using Chroma default",
            extra={"error": str(exc)},
        )
        return embedding_functions.DefaultEmbeddingFunction()


def get_vector_collection():
    """Lazy-load ChromaDB. Safe to call from any thread / Uvicorn worker."""
    global _chroma_client, _collection, _ef
    if _collection is not None:
        return _collection

    settings = get_settings()
    log.info(
        "initialising chromadb",
        extra={"path": settings.chroma_path, "collection": settings.chroma_collection},
    )

    _chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    _ef = _build_embedding_function()
    _collection = _chroma_client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def reset_collection() -> None:
    """Drop and recreate the collection. Used by seed scripts and tests."""
    global _chroma_client, _collection
    settings = get_settings()
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        _chroma_client.delete_collection(settings.chroma_collection)
    except Exception:
        pass
    _collection = None
    get_vector_collection()


def get_schema_collection():
    """Lazy-load the dedicated collection that stores table 'schema cards'.

    This collection is what powers semantic *table selection*: each row is a
    natural-language description of one database table, so a question embeds
    close to the table(s) that can answer it.
    """
    global _schema_collection
    if _schema_collection is not None:
        return _schema_collection

    get_vector_collection()  # ensures _chroma_client and _ef are initialised
    settings = get_settings()
    assert _chroma_client is not None
    _schema_collection = _chroma_client.get_or_create_collection(
        name=settings.chroma_schema_collection,
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _schema_collection


def reset_schema_collection() -> None:
    global _chroma_client, _schema_collection
    settings = get_settings()
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_path)
    try:
        _chroma_client.delete_collection(settings.chroma_schema_collection)
    except Exception:
        pass
    _schema_collection = None
    get_schema_collection()


def index_schema_cards(cards: Sequence[Dict[str, str]]) -> int:
    """Upsert table cards. Each card is ``{"table": str, "text": str}``."""
    if not cards:
        return 0
    col = get_schema_collection()
    col.upsert(
        documents=[c["text"] for c in cards],
        metadatas=[{"table": c["table"]} for c in cards],
        ids=[f"schema::{c['table']}" for c in cards],
    )
    return len(cards)


def search_schema_cards(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Return ``[{"table", "score"}]`` ranked by semantic similarity to query."""
    col = get_schema_collection()
    count = col.count()
    if not count:
        return []
    raw = col.query(
        query_texts=[query],
        n_results=max(1, min(num_results, count)),
        include=["metadatas", "distances"],
    )
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    out: List[Dict[str, Any]] = []
    for md, dist in zip(metas, dists):
        sim = float(max(0.0, 1.0 - float(dist) / 2.0))
        out.append({"table": str((md or {}).get("table", "")), "score": sim})
    return out


def collection_stats() -> Dict[str, Any]:
    col = get_vector_collection()
    count = col.count()
    sources: Dict[str, int] = {}
    if count > 0:
        sample = col.get(limit=min(count, 1000), include=["metadatas"])
        for md in sample.get("metadatas") or []:
            src = str((md or {}).get("source", "unknown"))
            sources[src] = sources.get(src, 0) + 1
    return {"chunks": count, "sources": sources}


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


def add_document_chunks(
    texts: Sequence[str],
    document_id: str,
    metadatas: Optional[Sequence[Dict[str, Any]]] = None,
) -> int:
    """Insert text chunks. Returns the number of chunks actually written."""
    if not texts:
        return 0

    col = get_vector_collection()
    timestamp_ms = int(time.time() * 1000)
    ids = [f"{document_id}::{timestamp_ms}::{i}" for i in range(len(texts))]
    if metadatas is None:
        metadatas = [{"source": document_id} for _ in texts]
    else:
        metadatas = [
            {**(m or {}), "source": (m or {}).get("source", document_id)} for m in metadatas
        ]

    col.upsert(documents=list(texts), metadatas=list(metadatas), ids=ids)
    log.info(
        "indexed chunks",
        extra={"document_id": document_id, "count": len(texts)},
    )
    return len(texts)


# --------------------------------------------------------------------------- #
# Retrieval helpers
# --------------------------------------------------------------------------- #


_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_TOKENS_RE = re.compile(r"[A-Za-z0-9$%\.]+")


def _extract_keyword_signals(query: str) -> Tuple[List[str], List[str]]:
    """Return (ticker mentions, quoted phrases) used for the keyword boost."""
    tickers = _TICKER_RE.findall(query)
    quoted = re.findall(r"\"([^\"]+)\"|'([^']+)'", query)
    quoted_phrases = [a or b for a, b in quoted]
    return tickers, quoted_phrases


def _keyword_boost(text: str, tickers: List[str], quoted: List[str]) -> float:
    boost = 0.0
    upper = text.upper()
    for t in tickers:
        if t.upper() in upper:
            boost += 0.06
    for phrase in quoted:
        if phrase and phrase.lower() in text.lower():
            boost += 0.10
    return min(boost, 0.30)


def _embed_text(text: str) -> List[float]:
    get_vector_collection()  # ensure _ef is built
    assert _ef is not None
    # Explicitly cast to python floats
    return [float(x) for x in _ef([text])[0]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return float(dot / (na * nb))


def _mmr_rerank(
    query_vec: Sequence[float],
    candidates: List[Dict[str, Any]],
    k: int,
    lam: float,
) -> List[Dict[str, Any]]:
    """Maximal Marginal Relevance: trades off relevance vs. diversity.

    score(d) = lam * sim(q,d) - (1 - lam) * max(sim(d, d') for d' in selected)
    """
    if not candidates:
        return []
    selected: List[Dict[str, Any]] = []
    remaining = candidates[:]

    while remaining and len(selected) < k:
        best_idx, best_score = 0, -math.inf
        for i, c in enumerate(remaining):
            relevance = c["score"]
            diversity_pen = 0.0
            if selected:
                diversity_pen = max(
                    _cosine(c["embedding"], s["embedding"]) for s in selected
                )
            mmr = float(lam * relevance - (1.0 - lam) * diversity_pen)
            if mmr > best_score:
                best_score, best_idx = mmr, i
        selected.append(remaining.pop(best_idx))
    return selected


# --------------------------------------------------------------------------- #
# Public search
# --------------------------------------------------------------------------- #


def search_financial_docs(
    query: str,
    num_results: Optional[int] = None,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Hybrid semantic + keyword retrieval with MMR re-ranking.

    Returns a list of dicts: ``{"text", "score", "source", "metadata", "id"}``
    sorted by descending blended score.
    """
    settings = get_settings()
    top_k = num_results or settings.retrieval_top_k

    col = get_vector_collection()
    fetch_k = max(top_k * 4, 16)

    raw = col.query(
        query_texts=[query],
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    embs = (raw.get("embeddings") or [[]])[0]
    ids = (raw.get("ids") or [[]])[0]

    if not docs:
        return []

    tickers, quoted = _extract_keyword_signals(query)
    candidates: List[Dict[str, Any]] = []
    for text, md, dist, emb, _id in zip(docs, metas, dists, embs, ids):
        
        # --- EXPLICIT PYTHON TYPE CASTING FOR FASTAPI ---
        dist_val = float(dist)
        sim = float(max(0.0, 1.0 - dist_val / 2.0))
        boost = float(_keyword_boost(str(text), tickers, quoted))
        blended = float(min(sim + boost, 1.0))
        
        candidates.append(
            {
                "id": str(_id),
                "text": str(text),
                "metadata": md or {},
                "source": str((md or {}).get("source", "unknown")),
                "score": blended,
                "raw_similarity": sim,
                "keyword_boost": boost,
                "embedding": [float(x) for x in emb], # Strip numpy floats out of embeddings array
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)

    query_vec = _embed_text(query)
    reranked = _mmr_rerank(
        query_vec, candidates, k=top_k, lam=settings.retrieval_mmr_lambda
    )

    threshold = settings.retrieval_min_score
    final = [c for c in reranked if c["score"] >= threshold] or reranked[:1]

    for c in final:
        c.pop("embedding", None)
    return final


def format_context(passages: Iterable[Dict[str, Any]]) -> str:
    """Render retrieved passages as a numbered, citation-friendly block."""
    blocks = []
    for i, p in enumerate(passages, start=1):
        src = str(p.get("source", "unknown"))
        score = float(p.get("score", 0.0))
        blocks.append(f"[{i}] (source={src}, score={score:.2f})\n{p['text']}")
    return "\n\n".join(blocks) if blocks else "No relevant context found."