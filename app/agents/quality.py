"""Quality agent – schema, completeness and type checks for tabular data."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd

from app.agents.state import PipelineState
from app.logging_config import get_logger

log = get_logger("agent.quality")

REQUIRED_FIELDS = ["ticker", "revenue", "net_income"]
NUMERIC_FIELDS = ["revenue", "net_income", "operating_income", "total_assets",
                  "total_liabilities", "employees"]


def _trace(state: PipelineState, started: int, status: str, summary: str, **metrics: Any) -> Dict[str, Any]:
    finished = int(time.time() * 1000)
    trace = {
        "agent": "quality",
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


def quality_agent(state: PipelineState) -> dict:
    started = int(time.time() * 1000)
    raw_data = state.get("raw_data") or []
    errors = list(state.get("errors", []) or [])
    warnings = list(state.get("warnings", []) or [])

    if state.get("file_type") != "tabular":
        return {
            "cleaned_data": raw_data,
            "errors": errors,
            "warnings": warnings,
            "next_node": "transform",
            **_trace(state, started, "skipped", "non-tabular input bypassed quality gate"),
        }

    if not raw_data:
        errors.append("Quality: no data to validate")
        return {
            "errors": errors,
            "warnings": warnings,
            "next_node": "end",
            **_trace(state, started, "failed", "no rows to validate"),
        }

    cleaned: List[Dict[str, Any]] = []
    rejected = 0
    for index, row in enumerate(raw_data):
        missing = [
            f for f in REQUIRED_FIELDS
            if f not in row or row[f] is None or (isinstance(row[f], float) and pd.isna(row[f]))
            or (isinstance(row[f], str) and not row[f].strip())
        ]
        if missing:
            warnings.append(f"row {index} ({row.get('ticker','?')}): missing {missing}")
            rejected += 1
            continue

        ok = True
        for f in NUMERIC_FIELDS:
            if f in row and row[f] not in (None, ""):
                try:
                    row[f] = float(row[f])
                except (ValueError, TypeError):
                    warnings.append(f"row {index} ({row.get('ticker','?')}): non-numeric {f}={row[f]}")
                    ok = False
                    break
        if not ok:
            rejected += 1
            continue

        cleaned.append(row)

    log.info(
        "quality_done",
        extra={"rows_in": len(raw_data), "rows_out": len(cleaned), "rejected": rejected},
    )

    summary = f"{len(cleaned)} of {len(raw_data)} rows passed validation"
    status = "ok" if cleaned and rejected == 0 else ("warning" if cleaned else "failed")

    return {
        "cleaned_data": cleaned,
        "errors": errors,
        "warnings": warnings,
        "next_node": "transform" if cleaned else "end",
        **_trace(state, started, status, summary,
                 rows_in=len(raw_data), rows_out=len(cleaned), rejected=rejected),
    }
