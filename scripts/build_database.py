"""Build the FinDataAccelerator SQLite relational database from seed CSVs.

This is the single builder for the relational *source of truth*. It creates
``app/data/findata.db`` with real ``PRIMARY KEY`` / ``FOREIGN KEY`` constraints
(``PRAGMA foreign_keys=ON``) and bulk-loads the normalized CSVs from
``app/data/relational/``. A post-load referential-integrity check fails loudly
if any foreign key is dangling.

Run from the project root::

    python scripts/build_database.py [--reset]
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

REL_DIR = os.path.join(PROJECT_ROOT, "app", "data", "relational")
DB_PATH = os.path.join(PROJECT_ROOT, "app", "data", "findata.db")


# --------------------------------------------------------------------------- #
# Schema DDL (order matters: parents before children)
# --------------------------------------------------------------------------- #

SCHEMA_DDL: List[Tuple[str, str]] = [
    ("sectors", """
        CREATE TABLE sectors (
            sector_id   INTEGER PRIMARY KEY,
            sector_name TEXT NOT NULL UNIQUE,
            description TEXT
        )
    """),
    ("industries", """
        CREATE TABLE industries (
            industry_id   INTEGER PRIMARY KEY,
            industry_name TEXT NOT NULL,
            sector_id     INTEGER NOT NULL REFERENCES sectors(sector_id)
        )
    """),
    ("companies", """
        CREATE TABLE companies (
            company_id   INTEGER PRIMARY KEY,
            ticker       TEXT NOT NULL UNIQUE,
            company_name TEXT NOT NULL,
            industry_id  INTEGER NOT NULL REFERENCES industries(industry_id),
            hq_country   TEXT,
            employees    INTEGER,
            founded_year INTEGER
        )
    """),
    ("financial_statements", """
        CREATE TABLE financial_statements (
            statement_id      INTEGER PRIMARY KEY,
            company_id        INTEGER NOT NULL REFERENCES companies(company_id),
            fiscal_year       INTEGER NOT NULL,
            revenue           REAL,
            net_income        REAL,
            operating_income  REAL,
            gross_profit      REAL,
            total_assets      REAL,
            total_liabilities REAL,
            free_cash_flow    REAL,
            capex             REAL,
            UNIQUE(company_id, fiscal_year)
        )
    """),
    ("financial_ratios", """
        CREATE TABLE financial_ratios (
            ratio_id              INTEGER PRIMARY KEY,
            statement_id          INTEGER NOT NULL REFERENCES financial_statements(statement_id),
            net_profit_margin_pct REAL,
            operating_margin_pct  REAL,
            gross_margin_pct      REAL,
            debt_to_assets_pct    REAL,
            roe_pct               REAL
        )
    """),
    ("business_segments", """
        CREATE TABLE business_segments (
            segment_id      INTEGER PRIMARY KEY,
            company_id      INTEGER NOT NULL REFERENCES companies(company_id),
            fiscal_year     INTEGER NOT NULL,
            segment_name    TEXT NOT NULL,
            segment_revenue REAL,
            yoy_growth_pct  REAL
        )
    """),
    ("earnings_events", """
        CREATE TABLE earnings_events (
            event_id        INTEGER PRIMARY KEY,
            company_id      INTEGER NOT NULL REFERENCES companies(company_id),
            fiscal_year     INTEGER NOT NULL,
            fiscal_quarter  TEXT NOT NULL,
            report_date     TEXT,
            eps_actual      REAL,
            eps_estimate    REAL,
            revenue_actual  REAL,
            surprise_pct    REAL
        )
    """),
    ("risk_factors", """
        CREATE TABLE risk_factors (
            risk_id       INTEGER PRIMARY KEY,
            company_id    INTEGER NOT NULL REFERENCES companies(company_id),
            fiscal_year   INTEGER NOT NULL,
            risk_category TEXT,
            description   TEXT
        )
    """),
    ("executives", """
        CREATE TABLE executives (
            exec_id    INTEGER PRIMARY KEY,
            company_id INTEGER NOT NULL REFERENCES companies(company_id),
            name       TEXT NOT NULL,
            title      TEXT NOT NULL,
            since_year INTEGER
        )
    """),
    ("earnings_reports", """
        CREATE TABLE earnings_reports (
            report_id   INTEGER PRIMARY KEY,
            company_id  INTEGER REFERENCES companies(company_id),
            sector_id   INTEGER REFERENCES sectors(sector_id),
            fiscal_year INTEGER NOT NULL,
            doc_type    TEXT,
            title       TEXT,
            content     TEXT
        )
    """),
    ("macro_indicators", """
        CREATE TABLE macro_indicators (
            indicator_id INTEGER PRIMARY KEY,
            name         TEXT NOT NULL,
            period       TEXT,
            value        REAL,
            unit         TEXT,
            description  TEXT
        )
    """),
]

# Columns that should become NULL when the CSV cell is empty.
_NULLABLE_INT = {"founded_year", "since_year", "company_id", "sector_id"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _coerce(column: str, value: str) -> Optional[Any]:
    value = (value or "").strip()
    if value == "":
        return None
    return value


def _load_csv_rows(table: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    path = os.path.join(REL_DIR, f"{table}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing seed CSV for table '{table}': {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            rows.append(tuple(_coerce(col, row.get(col, "")) for col in fieldnames))
    return fieldnames, rows


def _foreign_keys() -> List[Tuple[str, str, str, str]]:
    """(child_table, child_column, parent_table, parent_column) for the integrity scan."""
    return [
        ("industries", "sector_id", "sectors", "sector_id"),
        ("companies", "industry_id", "industries", "industry_id"),
        ("financial_statements", "company_id", "companies", "company_id"),
        ("financial_ratios", "statement_id", "financial_statements", "statement_id"),
        ("business_segments", "company_id", "companies", "company_id"),
        ("earnings_events", "company_id", "companies", "company_id"),
        ("risk_factors", "company_id", "companies", "company_id"),
        ("executives", "company_id", "companies", "company_id"),
        ("earnings_reports", "company_id", "companies", "company_id"),
        ("earnings_reports", "sector_id", "sectors", "sector_id"),
    ]


def build(reset: bool = False) -> Dict[str, int]:
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing database at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    counts: Dict[str, int] = {}

    try:
        for table, ddl in SCHEMA_DDL:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(ddl)

        for table, _ in SCHEMA_DDL:
            fieldnames, rows = _load_csv_rows(table)
            placeholders = ", ".join("?" for _ in fieldnames)
            cols = ", ".join(fieldnames)
            conn.executemany(
                f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", rows
            )
            counts[table] = len(rows)
            print(f"  loaded {table:<24} {len(rows):>4} rows")

        conn.commit()

        # Referential-integrity validation.
        problems = conn.execute("PRAGMA foreign_key_check").fetchall()
        if problems:
            raise RuntimeError(f"Foreign key violations detected: {problems}")

        for child, child_col, parent, parent_col in _foreign_keys():
            dangling = conn.execute(
                f"SELECT COUNT(*) FROM {child} c "
                f"WHERE c.{child_col} IS NOT NULL AND NOT EXISTS "
                f"(SELECT 1 FROM {parent} p WHERE p.{parent_col} = c.{child_col})"
            ).fetchone()[0]
            if dangling:
                raise RuntimeError(
                    f"{dangling} dangling {child}.{child_col} -> {parent}.{parent_col}"
                )

        print(f"\nDatabase built at {DB_PATH}")
        print("Referential integrity: OK")
        return counts
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the FinDataAccelerator SQLite DB")
    parser.add_argument("--reset", action="store_true", help="Delete the DB file first")
    args = parser.parse_args()
    build(reset=args.reset)
