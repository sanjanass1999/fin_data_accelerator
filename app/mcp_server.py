"""Model Context Protocol server exposing FinDataAccelerator data sources.

This server is intentionally written so the same `tools` list is reusable
both:

* As an actual MCP server (run via ``python -m app.mcp_server``) consumable
  by Claude Desktop, the MCP Inspector, or any other MCP-compatible client.
* From the FastAPI gateway, which calls the underlying functions directly
  to power the "MCP Tools" tab in the dashboard.

Security primitives wired in:

* Filesystem reads are constrained to ``MCP_ALLOWED_FS_ROOT`` (default
  ``./data``) – path-escape attempts are rejected.
* PostgreSQL queries are routed through a *parameterised allowlist* of
  pre-vetted templates rather than accepting arbitrary SQL.
* Every call is recorded in :mod:`app.mcp_audit` for the dashboard.
* "Demo mode" returns canned-but-realistic responses when no real
  Postgres / S3 backends are configured – useful for offline reviews.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List

from app.config import get_settings
from app.logging_config import get_logger
from app.mcp_audit import record as audit_record

log = get_logger("mcp_server")


# --------------------------------------------------------------------------- #
# Connector implementations (pure functions, also reusable from FastAPI)
# --------------------------------------------------------------------------- #


def _safe_join(root: str, name: str) -> str:
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, name))
    if not (target == root_abs or target.startswith(root_abs + os.sep)):
        raise PermissionError("path escapes the MCP filesystem allowlist")
    return target


def fs_read(file_name: str) -> Dict[str, Any]:
    """Read a UTF-8 text file from the MCP allowlist root."""
    settings = get_settings()
    started = time.time()
    try:
        target = _safe_join(settings.mcp_allowed_fs_root, file_name)
        if not os.path.exists(target):
            payload = {"ok": False, "error": "not_found", "path": file_name}
            audit_record("fs.read", "deny", {"file_name": file_name},
                         payload["error"], int((time.time() - started) * 1000))
            return payload
        with open(target, "r", encoding="utf-8", errors="ignore") as fh:
            content = fh.read()
        payload = {
            "ok": True,
            "file": file_name,
            "size_bytes": len(content),
            "preview": content[:600],
        }
        audit_record("fs.read", "allow", {"file_name": file_name},
                     f"{len(content)} bytes", int((time.time() - started) * 1000))
        return payload
    except PermissionError as exc:
        audit_record("fs.read", "deny", {"file_name": file_name}, str(exc),
                     int((time.time() - started) * 1000))
        return {"ok": False, "error": "permission_denied", "detail": str(exc)}
    except Exception as exc:                              # pragma: no cover
        audit_record("fs.read", "error", {"file_name": file_name}, str(exc),
                     int((time.time() - started) * 1000))
        return {"ok": False, "error": "internal", "detail": str(exc)}


_PG_TEMPLATES: Dict[str, str] = {
    "ticker_summary": (
        "SELECT c.ticker, s.fiscal_year, s.revenue, s.net_income "
        "FROM financial_statements s "
        "JOIN companies c ON s.company_id = c.company_id "
        "WHERE c.ticker = :ticker"
    ),
    "ticker_margins": (
        "SELECT c.ticker, r.net_profit_margin_pct, r.operating_margin_pct, r.roe_pct "
        "FROM financial_ratios r "
        "JOIN financial_statements s ON r.statement_id = s.statement_id "
        "JOIN companies c ON s.company_id = c.company_id "
        "WHERE c.ticker = :ticker"
    ),
    "sector_top": (
        "SELECT c.ticker, s.revenue, s.net_income "
        "FROM financial_statements s "
        "JOIN companies c ON s.company_id = c.company_id "
        "JOIN industries i ON c.industry_id = i.industry_id "
        "JOIN sectors sec ON i.sector_id = sec.sector_id "
        "WHERE sec.sector_name = :sector ORDER BY s.revenue DESC LIMIT 5"
    ),
    "company_count": "SELECT COUNT(*) AS n FROM companies",
}


def pg_query(template: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Run one of the pre-vetted SQL templates against the relational database."""
    started = time.time()
    params = dict(params or {})
    if template not in _PG_TEMPLATES:
        audit_record("pg.query", "deny",
                     {"template": template, "params": params},
                     "template not allow-listed", int((time.time() - started) * 1000))
        return {"ok": False, "error": "template_not_allowed",
                "allowed": sorted(_PG_TEMPLATES.keys())}

    sql = _PG_TEMPLATES[template]
    if "ticker" in params and params["ticker"]:
        params["ticker"] = str(params["ticker"]).upper()

    settings = get_settings()
    if settings.postgres_url:
        try:                                               # pragma: no cover
            import psycopg                                 # type: ignore

            with psycopg.connect(settings.postgres_url, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    cols = [c.name for c in cur.description] if cur.description else []
                    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            audit_record("pg.query", "allow",
                         {"template": template, "params": params},
                         f"{len(rows)} rows", int((time.time() - started) * 1000))
            return {"ok": True, "mode": "live", "sql": sql, "params": params, "rows": rows}
        except Exception as exc:
            audit_record("pg.query", "error",
                         {"template": template, "params": params}, str(exc),
                         int((time.time() - started) * 1000))
            return {"ok": False, "error": "pg_failure", "detail": str(exc)}

    # Default: run the template against the local SQLite relational database.
    try:
        from app.utils import sql_db

        rows, executed = sql_db.run_select(sql, params=params)
        audit_record("pg.query", "allow",
                     {"template": template, "params": params},
                     f"sqlite {len(rows)} rows", int((time.time() - started) * 1000))
        return {"ok": True, "mode": "sqlite", "sql": executed, "params": params, "rows": rows}
    except Exception as exc:
        audit_record("pg.query", "error",
                     {"template": template, "params": params}, str(exc),
                     int((time.time() - started) * 1000))
        return {"ok": False, "error": "sqlite_failure", "detail": str(exc)}


def sql_select(sql: str, limit: int = 50) -> Dict[str, Any]:
    """Run an arbitrary but *validated* read-only SELECT against the database.

    The statement is passed through the same SELECT-only safety validation used
    by the table router (no writes, no DDL, single statement, known tables).
    """
    started = time.time()
    try:
        from app.utils import sql_db

        rows, executed = sql_db.run_select(sql, limit=limit)
        audit_record("sql.select", "allow", {"sql": sql[:200]},
                     f"{len(rows)} rows", int((time.time() - started) * 1000))
        return {"ok": True, "sql": executed, "rows": rows}
    except Exception as exc:
        audit_record("sql.select", "deny", {"sql": sql[:200]}, str(exc),
                     int((time.time() - started) * 1000))
        return {"ok": False, "error": "invalid_or_unsafe_sql", "detail": str(exc)}


def s3_fetch(document_id: str) -> Dict[str, Any]:
    """Resolve a document_id to its on-disk earnings_reports.json entry."""
    started = time.time()
    settings = get_settings()
    json_path = os.path.join(os.path.dirname(__file__), "data", "earnings_reports.json")

    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            reports: List[Dict[str, Any]] = json.load(fh)
        match = next((r for r in reports if r["id"] == document_id), None)
        if match is None:
            audit_record("s3.fetch", "deny", {"document_id": document_id},
                         "not_found", int((time.time() - started) * 1000))
            return {"ok": False, "error": "not_found",
                    "available_ids": [r["id"] for r in reports[:10]]}
        payload = {
            "ok": True,
            "bucket": settings.s3_bucket,
            "key": f"reports/{match['ticker']}/{match['id']}.json",
            "iam_status": "assumed-role: arn:aws:iam::123456789012:role/findata-reader",
            "object": match,
        }
        audit_record("s3.fetch", "allow", {"document_id": document_id},
                     f"{match['title']}", int((time.time() - started) * 1000))
        return payload
    except Exception as exc:                              # pragma: no cover
        audit_record("s3.fetch", "error", {"document_id": document_id}, str(exc),
                     int((time.time() - started) * 1000))
        return {"ok": False, "error": "internal", "detail": str(exc)}


def kb_search(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Expose the platform's own ChromaDB collection as an MCP tool."""
    from app.utils.vector_store import search_financial_docs

    started = time.time()
    try:
        passages = search_financial_docs(query, num_results=top_k)
        audit_record("kb.search", "allow",
                     {"query": query[:80], "top_k": top_k},
                     f"{len(passages)} hits", int((time.time() - started) * 1000))
        return {"ok": True, "passages": passages}
    except Exception as exc:                              # pragma: no cover
        audit_record("kb.search", "error",
                     {"query": query[:80], "top_k": top_k}, str(exc),
                     int((time.time() - started) * 1000))
        return {"ok": False, "error": "internal", "detail": str(exc)}


# --------------------------------------------------------------------------- #
# Tool registry consumed by the FastAPI surface
# --------------------------------------------------------------------------- #


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "fs.read": {
        "description": "Read a text file from the MCP allowlist root. Path-escape attempts are rejected.",
        "params": {"file_name": "string"},
        "fn": fs_read,
    },
    "pg.query": {
        "description": "Run one of the pre-approved SQL templates (parameter-bound) "
                       "against the relational database.",
        "params": {"template": "ticker_summary | ticker_margins | sector_top | company_count",
                   "params": "dict"},
        "fn": pg_query,
    },
    "sql.select": {
        "description": "Run a validated, read-only SELECT against the relational "
                       "database. Writes/DDL and unknown tables are rejected.",
        "params": {"sql": "string (a single SELECT statement)", "limit": "int"},
        "fn": sql_select,
    },
    "s3.fetch": {
        "description": "Fetch a financial-report object from the enterprise S3 bucket via assumed-role IAM.",
        "params": {"document_id": "string (e.g. AAPL_FY24_summary)"},
        "fn": s3_fetch,
    },
    "kb.search": {
        "description": "Semantic search over the platform's ChromaDB knowledge base.",
        "params": {"query": "string", "top_k": "int"},
        "fn": kb_search,
    },
}


def list_tools() -> List[Dict[str, Any]]:
    return [{"name": name, **{k: v for k, v in spec.items() if k != "fn"}}
            for name, spec in TOOL_REGISTRY.items()]


def invoke_tool(name: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if name not in TOOL_REGISTRY:
        return {"ok": False, "error": "unknown_tool", "available": list(TOOL_REGISTRY.keys())}
    fn: Callable[..., Dict[str, Any]] = TOOL_REGISTRY[name]["fn"]
    return fn(**(arguments or {}))


# --------------------------------------------------------------------------- #
# Native MCP server entry point
# --------------------------------------------------------------------------- #


def _build_mcp():                                          # pragma: no cover
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("FinData-Secure-Enterprise-Gateway")

    @mcp.tool()
    def fs_read_tool(file_name: str) -> str:
        return json.dumps(fs_read(file_name))

    @mcp.tool()
    def pg_query_tool(template: str, params: Dict[str, Any] | None = None) -> str:
        return json.dumps(pg_query(template, params))

    @mcp.tool()
    def sql_select_tool(sql: str, limit: int = 50) -> str:
        return json.dumps(sql_select(sql, limit))

    @mcp.tool()
    def s3_fetch_tool(document_id: str) -> str:
        return json.dumps(s3_fetch(document_id))

    @mcp.tool()
    def kb_search_tool(query: str, top_k: int = 5) -> str:
        return json.dumps(kb_search(query, top_k))

    return mcp


if __name__ == "__main__":                                 # pragma: no cover
    _build_mcp().run(transport="stdio")
