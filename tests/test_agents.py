"""Pipeline / agent tests."""
from __future__ import annotations

import os

from app.graph import app_graph

CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "data", "sample_companies.csv")


def _run(file_path: str) -> dict:
    return app_graph.invoke({
        "file_path": file_path, "errors": [], "warnings": [],
        "agent_traces": [], "next_node": "ingest",
    })


def test_pipeline_runs_end_to_end():
    out = _run(CSV)
    assert out["is_indexed"] is True
    assert out["indexed_chunks"] > 0
    agents = [t["agent"] for t in out["agent_traces"]]
    assert agents == ["ingestion", "quality", "transform", "rag"]


def test_pipeline_missing_file_fails_gracefully():
    out = _run("does_not_exist.csv")
    assert out["is_indexed"] is False
    assert any("not found" in e.lower() for e in out["errors"])


def test_quality_filters_bad_rows(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("ticker,revenue,net_income\nAAA,100,10\nBBB,MISSING,5\nCCC,,\n")
    out = _run(str(bad))
    # Only the first valid row should survive into the transform stage.
    transformed = out.get("transformed_data") or []
    assert len(transformed) == 1
    assert transformed[0]["ticker"] == "AAA"
    assert out["warnings"]
