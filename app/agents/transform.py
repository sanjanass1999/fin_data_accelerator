"""Transform agent – derives financial ratios and builds narratives.

For tabular input we calculate net profit margin, operating margin and
leverage ratios. We also synthesise a per-row narrative that the RAG
agent can index directly. For text/PDF/JSON input we just pass the
content through into ``narratives``.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from app.agents.state import PipelineState
from app.logging_config import get_logger
from app.utils.chunking import chunk_text

log = get_logger("agent.transform")


def _trace(state: PipelineState, started: int, status: str, summary: str, **metrics: Any) -> Dict[str, Any]:
    finished = int(time.time() * 1000)
    trace = {
        "agent": "transform",
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


def _enrich_row(row: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(row)
    revenue = float(row.get("revenue") or 0)
    net_income = float(row.get("net_income") or 0)
    op_income = float(row.get("operating_income") or 0)
    assets = float(row.get("total_assets") or 0)
    liabilities = float(row.get("total_liabilities") or 0)

    enriched["net_profit_margin_pct"] = round(net_income / revenue * 100, 2) if revenue else 0.0
    enriched["operating_margin_pct"] = round(op_income / revenue * 100, 2) if revenue else 0.0
    enriched["debt_to_assets_pct"] = round(liabilities / assets * 100, 2) if assets else 0.0
    enriched["equity"] = round(assets - liabilities, 2) if assets and liabilities else None
    return enriched


def _row_narrative(row: Dict[str, Any]) -> str:
    return (
        f"{row.get('company_name', row.get('ticker'))} ({row.get('ticker')}) "
        f"FY{int(row.get('fiscal_year', 2024))} financials: revenue "
        f"${float(row['revenue']):,.0f}M, net income "
        f"${float(row['net_income']):,.0f}M, net profit margin "
        f"{row['net_profit_margin_pct']}% and operating margin "
        f"{row['operating_margin_pct']}%. Total assets "
        f"${float(row.get('total_assets', 0)):,.0f}M with debt-to-assets "
        f"of {row['debt_to_assets_pct']}%. Sector: {row.get('sector','n/a')} / "
        f"industry: {row.get('industry','n/a')}."
    )


def transform_agent(state: PipelineState) -> dict:
    started = int(time.time() * 1000)
    errors = list(state.get("errors", []) or [])
    file_type = state.get("file_type", "tabular")

    if file_type == "tabular":
        cleaned = state.get("cleaned_data") or state.get("raw_data") or []
        if not cleaned:
            errors.append("Transform: no cleaned data available")
            return {
                "errors": errors,
                "next_node": "end",
                **_trace(state, started, "failed", "no cleaned data"),
            }

        transformed: List[Dict[str, Any]] = []
        narratives: List[str] = []
        meta: List[Dict[str, Any]] = []
        for row in cleaned:
            try:
                enriched = _enrich_row(row)
                transformed.append(enriched)
                narratives.append(_row_narrative(enriched))
                meta.append({
                    "source": "pipeline_run_tabular",
                    "doc_type": "company_financials",
                    "ticker": str(enriched.get("ticker", "UNK")),
                    "sector": str(enriched.get("sector", "")),
                    "fiscal_year": int(enriched.get("fiscal_year", 2024) or 2024),
                })
            except Exception as exc:
                errors.append(f"Transform error for {row.get('ticker','?')}: {exc}")

        summary = f"calculated metrics for {len(transformed)} firms"
        return {
            "transformed_data": transformed,
            "narratives": narratives,
            "narrative_metadata": meta,
            "errors": errors,
            "next_node": "rag",
            **_trace(state, started, "ok", summary,
                     transformed=len(transformed),
                     errors_added=len(errors) - len(state.get("errors", []) or [])),
        }

    if file_type == "text":
        # raw_data is a list[str] with one entry per page / file
        pages: List[str] = state.get("raw_data") or []
        narratives: List[str] = []
        meta: List[Dict[str, Any]] = []
        for i, page in enumerate(pages):
            page = (page or "").strip()
            if not page:
                continue
            for chunk in chunk_text(page):
                narratives.append(chunk)
                meta.append({
                    "source": state.get("file_path", "unstructured"),
                    "doc_type": "unstructured_text",
                    "page": i,
                })
        summary = f"prepared {len(narratives)} text chunks from {len(pages)} page(s)"
        return {
            "narratives": narratives,
            "narrative_metadata": meta,
            "errors": errors,
            "next_node": "rag",
            **_trace(state, started, "ok", summary, chunks=len(narratives)),
        }

    if file_type == "structured":
        # Free-form JSON dump – just stringify each top-level item
        items = state.get("raw_data") or []
        narratives = [str(item) for item in items if item]
        meta = [{"source": state.get("file_path", "structured"),
                 "doc_type": "structured_json"} for _ in narratives]
        return {
            "narratives": narratives,
            "narrative_metadata": meta,
            "errors": errors,
            "next_node": "rag",
            **_trace(state, started, "ok", f"prepared {len(narratives)} JSON items"),
        }

    errors.append(f"Transform: unknown file_type '{file_type}'")
    return {
        "errors": errors,
        "next_node": "end",
        **_trace(state, started, "failed", f"unknown file_type {file_type}"),
    }
