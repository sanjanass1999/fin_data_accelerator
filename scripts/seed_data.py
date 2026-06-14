"""Seed the ChromaDB knowledge base with rich synthetic financial data.

This script is the single source of truth for the demo dataset:

* Structured CSV financials  -> per-row narratives (one per company-year)
* Long-form earnings reports -> chunked into ~600-char passages
* Macro / sector overviews   -> indexed verbatim

Run from the project root::

    python scripts/seed_data.py [--reset]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.utils.vector_store import (  # noqa: E402  (path tweak above)
    add_document_chunks,
    collection_stats,
    get_vector_collection,
    reset_collection,
)

CSV_PATH = os.path.join(PROJECT_ROOT, "app", "data", "sample_companies.csv")
REPORTS_PATH = os.path.join(PROJECT_ROOT, "app", "data", "earnings_reports.json")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _company_narrative(row: Dict[str, Any]) -> str:
    revenue = float(row["revenue"])
    net_income = float(row["net_income"])
    op_income = float(row["operating_income"])
    margin = round(net_income / revenue * 100, 2) if revenue else 0.0
    op_margin = round(op_income / revenue * 100, 2) if revenue else 0.0
    return (
        f"{row['company_name']} ({row['ticker']}) is a {row['sector']} "
        f"sector company in the {row['industry']} industry, headquartered "
        f"in {row['hq_country']}. In fiscal year {int(row['fiscal_year'])}, "
        f"{row['ticker']} generated total revenue of ${revenue:,.0f}M and "
        f"net income of ${net_income:,.0f}M, producing a net profit margin "
        f"of {margin}%. Operating income was ${op_income:,.0f}M for an "
        f"operating margin of {op_margin}%. The company employed "
        f"approximately {int(row['employees']):,} people and reported total "
        f"assets of ${float(row['total_assets']):,.0f}M against total "
        f"liabilities of ${float(row['total_liabilities']):,.0f}M."
    )


def _chunk(text: str, target: int = 600, overlap: int = 80) -> List[str]:
    """Naive sentence-aware chunker (no langchain dep needed)."""
    text = " ".join(text.split())
    if len(text) <= target:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + target, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + 100, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def seed(reset: bool = False) -> Dict[str, Any]:
    if reset:
        print("Reset requested: dropping existing collection.")
        reset_collection()
    else:
        get_vector_collection()

    started = time.time()

    df = pd.read_csv(CSV_PATH)
    csv_texts: List[str] = []
    csv_meta: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        csv_texts.append(_company_narrative(row))
        csv_meta.append({
            "source": "sample_companies.csv",
            "doc_type": "company_financials",
            "ticker": row["ticker"],
            "sector": row["sector"],
            "fiscal_year": int(row["fiscal_year"]),
        })
    csv_count = add_document_chunks(csv_texts, document_id="company_financials_2024", metadatas=csv_meta)

    with open(REPORTS_PATH, "r", encoding="utf-8") as fh:
        reports = json.load(fh)

    report_texts: List[str] = []
    report_meta: List[Dict[str, Any]] = []
    for r in reports:
        for chunk in _chunk(r["content"]):
            report_texts.append(f"{r['title']}\n\n{chunk}")
            report_meta.append({
                "source": "earnings_reports.json",
                "doc_type": r["doc_type"],
                "ticker": r["ticker"],
                "fiscal_year": int(r["fiscal_year"]),
                "doc_id": r["id"],
            })
    report_count = add_document_chunks(
        report_texts, document_id="earnings_reports", metadatas=report_meta
    )

    elapsed = round(time.time() - started, 2)
    stats = collection_stats()
    print(
        f"Seed complete in {elapsed}s. "
        f"CSV narratives: {csv_count}, report chunks: {report_count}. "
        f"Total chunks now in collection: {stats['chunks']}."
    )
    print(f"Source breakdown: {stats['sources']}")
    return {
        "csv_narratives": csv_count,
        "report_chunks": report_count,
        "elapsed_seconds": elapsed,
        "stats": stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the FinDataAccelerator knowledge base")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collection first")
    args = parser.parse_args()
    seed(reset=args.reset)
