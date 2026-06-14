"""Ingestion agent – polymorphic loader for CSV / Parquet / PDF / JSON / TXT."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import pandas as pd

from app.agents.state import PipelineState
from app.logging_config import get_logger

log = get_logger("agent.ingestion")


_SUPPORTED_TABULAR = {".csv", ".parquet", ".tsv"}
_SUPPORTED_TEXT = {".pdf", ".txt", ".md"}
_SUPPORTED_STRUCT = {".json"}


def _trace(state: PipelineState, started: int, status: str, summary: str, **metrics: Any) -> Dict[str, Any]:
    finished = int(time.time() * 1000)
    trace = {
        "agent": "ingestion",
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


def _load_tabular(path: str, suffix: str) -> List[Dict[str, Any]]:
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:                                              # pragma: no cover
        raise ValueError(f"Unsupported tabular extension: {suffix}")
    return df.to_dict(orient="records")


def _load_text(path: str, suffix: str) -> List[str]:
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:                     # pragma: no cover
            raise RuntimeError("pypdf is required for PDF ingestion") from exc
        reader = PdfReader(path)
        return [p.extract_text() or "" for p in reader.pages]
    with open(path, "r", encoding="utf-8") as fh:
        return [fh.read()]


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("JSON must contain an object or array of objects")


def ingestion_agent(state: PipelineState) -> dict:
    started = int(time.time() * 1000)
    file_path = state.get("file_path", "")
    errors = list(state.get("errors", []) or [])

    log.info("ingestion_started", extra={"file_path": file_path})

    if not file_path or not os.path.exists(file_path):
        errors.append(f"File not found: {file_path}")
        return {
            "errors": errors,
            "is_indexed": False,
            "indexed_chunks": 0,
            "next_node": "end",
            **_trace(state, started, "failed", "file not found", file_path=file_path),
        }

    suffix = os.path.splitext(file_path)[1].lower()

    try:
        if suffix in _SUPPORTED_TABULAR:
            raw_data = _load_tabular(file_path, suffix)
            file_type = "tabular"
            summary = f"loaded {len(raw_data)} rows from {os.path.basename(file_path)}"
            return {
                "file_type": file_type,
                "raw_data": raw_data,
                "errors": errors,
                "next_node": "quality",
                **_trace(state, started, "ok", summary, rows=len(raw_data), suffix=suffix),
            }

        if suffix in _SUPPORTED_STRUCT:
            raw_data = _load_json(file_path)
            file_type = "tabular" if all(isinstance(r, dict) for r in raw_data) else "structured"
            summary = f"loaded {len(raw_data)} JSON records"
            return {
                "file_type": file_type,
                "raw_data": raw_data,
                "errors": errors,
                "next_node": "quality" if file_type == "tabular" else "transform",
                **_trace(state, started, "ok", summary, records=len(raw_data), suffix=suffix),
            }

        if suffix in _SUPPORTED_TEXT:
            pages = _load_text(file_path, suffix)
            file_type = "text"
            summary = f"loaded {len(pages)} page(s) of text from {os.path.basename(file_path)}"
            return {
                "file_type": file_type,
                "raw_data": pages,
                "errors": errors,
                "next_node": "transform",
                **_trace(state, started, "ok", summary, pages=len(pages), suffix=suffix),
            }

        errors.append(f"Unsupported file extension: {suffix}")
        return {
            "errors": errors,
            "is_indexed": False,
            "indexed_chunks": 0,
            "next_node": "end",
            **_trace(state, started, "failed", f"unsupported extension {suffix}"),
        }

    except Exception as exc:
        errors.append(f"Ingestion failure: {exc}")
        log.exception("ingestion_failed")
        return {
            "errors": errors,
            "is_indexed": False,
            "indexed_chunks": 0,
            "next_node": "end",
            **_trace(state, started, "failed", str(exc)),
        }
