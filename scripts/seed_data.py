"""Seed the ChromaDB knowledge base from the relational SQLite database.

This is the single source of truth for the *vector* side of the platform. It
reads ``app/data/findata.db`` (built by ``scripts/build_database.py``) and:

* embeds one natural-language **schema card** per table into a dedicated
  collection -> this is what lets the agent *choose the right table* for a
  question via semantic similarity;
* builds a joined per-company financial **narrative** (one per company-year)
  for qualitative retrieval;
* chunks the long-form ``earnings_reports`` documents into ~600-char passages.

Run from the project root::

    python scripts/seed_data.py [--reset]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import schema_catalog  # noqa: E402
from app.utils import sql_db  # noqa: E402
from app.utils.chunking import chunk_text  # noqa: E402
from app.utils.vector_store import (  # noqa: E402
    add_document_chunks,
    collection_stats,
    get_vector_collection,
    index_schema_cards,
    reset_collection,
    reset_schema_collection,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _company_narrative(row: Dict[str, Any]) -> str:
    return (
        f"{row['company_name']} ({row['ticker']}) is a {row['sector_name']} "
        f"sector company in the {row['industry_name']} industry, headquartered "
        f"in {row['hq_country']}. In fiscal year {int(row['fiscal_year'])}, "
        f"{row['ticker']} generated total revenue of ${float(row['revenue']):,.0f}M and "
        f"net income of ${float(row['net_income']):,.0f}M, producing a net profit margin "
        f"of {row['net_profit_margin_pct']}%. Operating income was "
        f"${float(row['operating_income']):,.0f}M for an operating margin of "
        f"{row['operating_margin_pct']}%. The company employed approximately "
        f"{int(row['employees']):,} people and reported total assets of "
        f"${float(row['total_assets']):,.0f}M against total liabilities of "
        f"${float(row['total_liabilities']):,.0f}M (debt-to-assets "
        f"{row['debt_to_assets_pct']}%)."
    )


# Long-form report prose is split with the shared sentence-aware chunker
# (see app/utils/chunking.py). Per-company narratives are short, self-contained
# records and are intentionally indexed whole (one row -> one chunk).


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def seed(reset: bool = False) -> Dict[str, Any]:
    if reset:
        print("Reset requested: dropping existing collections.")
        reset_collection()
        reset_schema_collection()
    else:
        get_vector_collection()

    started = time.time()

    # 1) Schema cards -> table-selection collection.
    card_count = index_schema_cards(schema_catalog.all_schema_cards())

    # 2) Joined per-company financial narratives -> main collection.
    company_rows, _ = sql_db.run_select(
        """
        SELECT c.ticker, c.company_name, c.hq_country, c.employees,
               i.industry_name, sec.sector_name,
               s.fiscal_year, s.revenue, s.net_income, s.operating_income,
               s.total_assets, s.total_liabilities,
               r.net_profit_margin_pct, r.operating_margin_pct, r.debt_to_assets_pct
        FROM companies c
        JOIN industries i ON c.industry_id = i.industry_id
        JOIN sectors sec ON i.sector_id = sec.sector_id
        JOIN financial_statements s ON s.company_id = c.company_id
        JOIN financial_ratios r ON r.statement_id = s.statement_id
        """,
        limit=1000,
    )
    narr_texts: List[str] = []
    narr_meta: List[Dict[str, Any]] = []
    for row in company_rows:
        narr_texts.append(_company_narrative(row))
        narr_meta.append({
            "source": "findata.db",
            "doc_type": "company_financials",
            "ticker": row["ticker"],
            "sector": row["sector_name"],
            "fiscal_year": int(row["fiscal_year"]),
        })
    narr_count = add_document_chunks(
        narr_texts, document_id="company_financials", metadatas=narr_meta
    )

    # 3) Long-form narrative documents -> main collection (chunked).
    report_rows, _ = sql_db.run_select(
        """
        SELECT er.report_id, er.fiscal_year, er.doc_type, er.title, er.content,
               c.ticker
        FROM earnings_reports er
        LEFT JOIN companies c ON er.company_id = c.company_id
        """,
        limit=1000,
    )
    report_texts: List[str] = []
    report_meta: List[Dict[str, Any]] = []
    for r in report_rows:
        for chunk in chunk_text(r["content"]):
            report_texts.append(f"{r['title']}\n\n{chunk}")
            report_meta.append({
                "source": "earnings_reports",
                "doc_type": r["doc_type"],
                "ticker": r["ticker"] or "n/a",
                "fiscal_year": int(r["fiscal_year"]),
                "doc_id": int(r["report_id"]),
            })
    report_count = add_document_chunks(
        report_texts, document_id="earnings_reports", metadatas=report_meta
    )

    elapsed = round(time.time() - started, 2)
    stats = collection_stats()
    print(
        f"Seed complete in {elapsed}s. "
        f"Schema cards: {card_count}, company narratives: {narr_count}, "
        f"report chunks: {report_count}."
    )
    print(f"Total chunks now in main collection: {stats['chunks']}.")
    print(f"Source breakdown: {stats['sources']}")
    return {
        "schema_cards": card_count,
        "company_narratives": narr_count,
        "report_chunks": report_count,
        "elapsed_seconds": elapsed,
        "stats": stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the FinDataAccelerator knowledge base")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the collections first")
    args = parser.parse_args()
    seed(reset=args.reset)
