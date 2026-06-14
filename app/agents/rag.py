"""RAG agent – persists narratives + metadata into ChromaDB."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from app.agents.state import PipelineState
from app.logging_config import get_logger
from app.utils.vector_store import add_document_chunks

log = get_logger("agent.rag")


def _trace(state: PipelineState, started: int, status: str, summary: str, **metrics: Any) -> Dict[str, Any]:
    finished = int(time.time() * 1000)
    trace = {
        "agent": "rag",
        "started_ms": started,
        "finished_ms": finished,
        "duration_ms": finished - started,
        "status": status,
        "summary": summary,
        "metrics": metrics,
    }
    traces = list(state.get("agent_traces", []) or [])
    traces.append(trace)
    return {"agent_traces": traces}


def rag_agent(state: PipelineState) -> dict:
    started = int(time.time() * 1000)
    errors = list(state.get("errors", []) or [])

    narratives: List[str] = state.get("narratives") or []
    metadata: List[Dict[str, Any]] = state.get("narrative_metadata") or []

    if not narratives:
        errors.append("RAG: no narratives to index")
        return {
            "is_indexed": False,
            "indexed_chunks": 0,
            "errors": errors,
            "next_node": "end",
            **_trace(state, started, "failed", "no narratives"),
        }

    document_id = (
        os.path.splitext(os.path.basename(state.get("file_path", "pipeline_run")))[0]
        or "pipeline_run"
    )

    try:
        count = add_document_chunks(narratives, document_id=document_id, metadatas=metadata)
        return {
            "is_indexed": True,
            "indexed_chunks": count,
            "errors": errors,
            "next_node": "end",
            **_trace(state, started, "ok", f"indexed {count} chunks", chunks=count, document_id=document_id),
        }
    except Exception as exc:
        log.exception("rag_indexing_failed")
        errors.append(f"RAG indexing exception: {exc}")
        return {
            "is_indexed": False,
            "indexed_chunks": 0,
            "errors": errors,
            "next_node": "end",
            **_trace(state, started, "failed", str(exc)),
        }
