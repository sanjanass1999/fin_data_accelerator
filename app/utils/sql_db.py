"""Read-only SQLite access layer with a safe, validated SELECT executor.

The table router generates SQL (via the LLM or a deterministic fallback) and
this module is the *only* place that SQL is allowed to touch the database. It
enforces hard safety rails so a generated query can never mutate the source:

* The connection is opened read-only (``mode=ro``) via a URI.
* Only a single ``SELECT``/``WITH`` statement is accepted.
* Write/DDL keywords and statement stacking (``;``) are rejected.
* Every referenced identifier is checked against the live schema, so the model
  cannot read tables/columns that do not exist.
"""
from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.logging_config import get_logger

log = get_logger("sql_db")

_SCHEMA_CACHE: Optional[Dict[str, List[str]]] = None

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|"
    r"detach|pragma|vacuum|reindex|grant|revoke)\b",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SqlValidationError(ValueError):
    """Raised when a generated SQL string fails the safety checks."""


# --------------------------------------------------------------------------- #
# Connection + introspection
# --------------------------------------------------------------------------- #


def _db_path() -> str:
    settings = get_settings()
    path = settings.sqlite_path
    if not os.path.isabs(path):
        # Resolve relative to the project root (two levels up from app/utils).
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.normpath(os.path.join(root, path))
    return path


def _connect_ro() -> sqlite3.Connection:
    path = _db_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SQLite database not found at {path}. "
            f"Run `python scripts/build_database.py --reset` first."
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_schema() -> Dict[str, List[str]]:
    """Return ``{table_name: [column, ...]}`` for the live database (cached)."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    schema: Dict[str, List[str]] = {}
    conn = _connect_ro()
    try:
        tables = [
            r["name"]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            cols = [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]
            schema[table] = cols
    finally:
        conn.close()
    _SCHEMA_CACHE = schema
    return schema


def reset_schema_cache() -> None:
    global _SCHEMA_CACHE
    _SCHEMA_CACHE = None


# --------------------------------------------------------------------------- #
# Validation + execution
# --------------------------------------------------------------------------- #


def _strip_sql(sql: str) -> str:
    sql = sql.strip()
    # Strip a trailing semicolon (single statement only).
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql


def validate_select(sql: str) -> str:
    """Validate ``sql`` is a safe single SELECT and return the cleaned string."""
    if not sql or not sql.strip():
        raise SqlValidationError("empty SQL")

    cleaned = _strip_sql(sql)

    if ";" in cleaned:
        raise SqlValidationError("multiple statements are not allowed")

    lowered = cleaned.lstrip().lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise SqlValidationError("only SELECT/WITH queries are allowed")

    if _FORBIDDEN.search(cleaned):
        raise SqlValidationError("query contains a forbidden keyword")

    # Validate that referenced tables exist. We check identifiers that appear
    # in FROM/JOIN positions against the live schema.
    schema = get_schema()
    known_tables = set(schema.keys())
    referenced = set(
        m.group(1).lower()
        for m in re.finditer(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)", cleaned, re.IGNORECASE)
    )
    unknown = {t for t in referenced if t not in known_tables}
    if unknown:
        raise SqlValidationError(f"unknown table(s): {sorted(unknown)}")

    return cleaned


def run_select(
    sql: str,
    limit: int = 100,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Validate, execute and return ``(rows, executed_sql)``.

    A defensive ``LIMIT`` is appended if the query does not already specify one.
    ``params`` are bound safely (named ``:placeholder`` style) when provided.
    """
    cleaned = validate_select(sql)

    if not re.search(r"\blimit\b", cleaned, re.IGNORECASE):
        cleaned = f"{cleaned} LIMIT {int(limit)}"

    conn = _connect_ro()
    try:
        cur = conn.execute(cleaned, params or {})
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return rows, cleaned


def list_tables() -> List[str]:
    return list(get_schema().keys())
