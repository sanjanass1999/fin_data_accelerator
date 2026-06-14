"""Automatic table router: the heart of the relational agent.

Given a natural-language question this module decides, with no manual rules from
the caller, *which table(s)* in the relational database can answer it, writes a
safe SQL query against them and returns the exact rows.

Pipeline
--------
1. ``select_tables`` -- blends semantic similarity over the embedded schema
   cards (see :mod:`app.schema_catalog`) with a deterministic keyword/alias
   boost to rank the tables.
2. ``build_sql`` -- asks the LLM (NL2SQL) to write a SELECT constrained to the
   chosen tables; if no LLM is available it falls back to a deterministic
   template builder so the path still works fully offline.
3. ``answer_structured`` -- orchestrates selection -> SQL -> validation ->
   execution and returns ``{selected_tables, sql, rows, ...}``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app import schema_catalog
from app.logging_config import get_logger
from app.utils import sql_db
from app.utils.llm_service import generate_sql
from app.utils.vector_store import index_schema_cards, search_schema_cards

log = get_logger("table_router")


# --------------------------------------------------------------------------- #
# Schema index bootstrap
# --------------------------------------------------------------------------- #


def ensure_schema_index() -> None:
    """Make sure the schema-card collection is populated (idempotent)."""
    try:
        hits = search_schema_cards("revenue", num_results=1)
        if hits:
            return
    except Exception:
        pass
    index_schema_cards(schema_catalog.all_schema_cards())


# --------------------------------------------------------------------------- #
# Lightweight DB-derived lookups (cached)
# --------------------------------------------------------------------------- #


_TICKERS: Optional[set] = None
_NAME_TOKENS: Optional[Dict[str, str]] = None
_SECTOR_NAMES: Optional[List[str]] = None

_NAME_STOPWORDS = {
    "inc", "incorporated", "corporation", "corp", "company", "co", "plc",
    "group", "the", "and", "platforms", "communications", "holdings", "ltd",
    "limited", "international", "& co.", "&",
}

_SECTOR_ALIASES = {
    "tech": "Technology",
    "technology": "Technology",
    "bank": "Financial Services",
    "banks": "Financial Services",
    "banking": "Financial Services",
    "financial": "Financial Services",
    "energy": "Energy",
    "oil": "Energy",
    "healthcare": "Healthcare",
    "health": "Healthcare",
    "pharma": "Healthcare",
    "industrial": "Industrials",
    "industrials": "Industrials",
    "materials": "Basic Materials",
    "telecom": "Communication Services",
    "media": "Communication Services",
}


def _load_lookups() -> None:
    global _TICKERS, _NAME_TOKENS, _SECTOR_NAMES
    if _TICKERS is not None:
        return
    tickers: set = set()
    name_tokens: Dict[str, str] = {}
    try:
        rows, _ = sql_db.run_select(
            "SELECT ticker, company_name FROM companies", limit=1000
        )
        for r in rows:
            ticker = str(r["ticker"]).strip()
            tickers.add(ticker.upper())
            name = str(r["company_name"]).lower()
            for tok in re.findall(r"[a-z][a-z0-9\-]+", name):
                if len(tok) >= 4 and tok not in _NAME_STOPWORDS:
                    name_tokens.setdefault(tok, ticker)
        sectors, _ = sql_db.run_select("SELECT sector_name FROM sectors", limit=100)
        _SECTOR_NAMES = [str(s["sector_name"]) for s in sectors]
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("lookup load failed", extra={"error": str(exc)})
        _SECTOR_NAMES = []
    _TICKERS = tickers
    _NAME_TOKENS = name_tokens


def reset_lookups() -> None:
    global _TICKERS, _NAME_TOKENS, _SECTOR_NAMES
    _TICKERS = _NAME_TOKENS = _SECTOR_NAMES = None


# --------------------------------------------------------------------------- #
# Entity extraction
# --------------------------------------------------------------------------- #


_DESC_WORDS = ("highest", "top", "largest", "most", "best", "greatest",
               "leading", "biggest", "max", "maximum")
_ASC_WORDS = ("lowest", "least", "smallest", "worst", "bottom", "min", "minimum")

_FS_METRICS = {
    "revenue": ["revenue", "sales", "top line", "turnover"],
    "net_income": ["net income", "profit", "earnings", "bottom line", "net profit"],
    "operating_income": ["operating income", "operating profit"],
    "gross_profit": ["gross profit"],
    "total_assets": ["total assets", "assets"],
    "total_liabilities": ["total liabilities", "liabilities", "debt"],
    "free_cash_flow": ["free cash flow", "fcf", "cash flow"],
    "capex": ["capex", "capital expenditure", "capital expenditures"],
}

_RATIO_METRICS = {
    "net_profit_margin_pct": ["net profit margin", "net margin", "profit margin", "margin"],
    "operating_margin_pct": ["operating margin"],
    "gross_margin_pct": ["gross margin"],
    "debt_to_assets_pct": ["debt to assets", "debt-to-assets", "leverage"],
    "roe_pct": ["return on equity", "roe"],
}


def _resolve_companies(query: str) -> List[str]:
    _load_lookups()
    found: List[str] = []
    # Exact ticker tokens (uppercase in the original query).
    for tok in re.findall(r"\b[A-Z]{1,5}\b", query):
        if tok in (_TICKERS or set()) and tok not in found:
            found.append(tok)
    # Company-name tokens.
    low = query.lower()
    for tok, ticker in (_NAME_TOKENS or {}).items():
        if re.search(rf"\b{re.escape(tok)}\b", low) and ticker not in found:
            found.append(ticker)
    return found


def _detect_year(query: str) -> Optional[int]:
    m = re.search(r"\b(19|20)\d{2}\b", query)
    return int(m.group(0)) if m else None


def _detect_direction(query: str) -> Optional[str]:
    low = query.lower()
    if any(w in low for w in _DESC_WORDS):
        return "DESC"
    if any(w in low for w in _ASC_WORDS):
        return "ASC"
    return None


def _detect_sector(query: str) -> Optional[str]:
    _load_lookups()
    low = query.lower()
    for name in (_SECTOR_NAMES or []):
        if name.lower() in low:
            return name
    for alias, name in _SECTOR_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", low):
            return name
    return None


def _detect_metric(query: str, mapping: Dict[str, List[str]]) -> Optional[str]:
    low = query.lower()
    best: Optional[str] = None
    best_len = 0
    for col, phrases in mapping.items():
        for p in phrases:
            if p in low and len(p) > best_len:
                best, best_len = col, len(p)
    return best


def _detect_quarter(query: str) -> Optional[str]:
    m = re.search(r"\bq([1-4])\b", query, re.IGNORECASE)
    return f"Q{m.group(1)}" if m else None


def _sql_str_list(values: List[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


# --------------------------------------------------------------------------- #
# Table selection
# --------------------------------------------------------------------------- #


def select_tables(query: str, top_k: int = 4) -> Dict[str, Any]:
    """Rank tables for ``query`` blending semantic + keyword/alias signals."""
    ensure_schema_index()

    semantic = {h["table"]: h["score"] for h in search_schema_cards(query, num_results=20)}
    alias_map = schema_catalog.keyword_alias_map()
    low = query.lower()

    scored: List[Tuple[str, float, float, int]] = []
    for table in schema_catalog.list_tables():
        sem = float(semantic.get(table, 0.0))
        hits = 0
        for alias in alias_map.get(table, []):
            if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", low):
                hits += 1
        kw = min(hits * 0.18, 0.54)
        blended = sem + kw
        scored.append((table, blended, sem, hits))

    scored.sort(key=lambda t: t[1], reverse=True)
    ranked = [
        {"table": t, "score": round(b, 4), "semantic": round(s, 4), "keyword_hits": h}
        for (t, b, s, h) in scored
    ]
    selected = [r["table"] for r in ranked[:top_k] if r["score"] > 0][:top_k] or [ranked[0]["table"]]
    return {"selected": selected, "ranked": ranked}


def _tables_for_sql(selected: List[str], query: str) -> List[str]:
    """Augment the selected tables with the join tables SQL will need."""
    tables = list(selected)
    needs_company = any(
        t in tables for t in (
            "financial_statements", "financial_ratios", "business_segments",
            "earnings_events", "risk_factors", "executives", "earnings_reports",
        )
    )
    if needs_company and "companies" not in tables:
        tables.append("companies")
    if "financial_ratios" in tables and "financial_statements" not in tables:
        tables.append("financial_statements")
    if _detect_sector(query):
        for extra in ("industries", "sectors"):
            if extra not in tables:
                tables.append(extra)
    return tables


# --------------------------------------------------------------------------- #
# Deterministic SQL templates (offline fallback)
# --------------------------------------------------------------------------- #


def _template_sql(query: str, primary: str) -> Optional[str]:
    tickers = _resolve_companies(query)
    year = _detect_year(query)
    direction = _detect_direction(query)
    sector = _detect_sector(query)
    ticker_filter = f"c.ticker IN ({_sql_str_list(tickers)})" if tickers else None

    if primary == "financial_ratios":
        metric = _detect_metric(query, _RATIO_METRICS) or "net_profit_margin_pct"
        sql = (
            f"SELECT c.ticker, c.company_name, r.{metric} "
            "FROM financial_ratios r "
            "JOIN financial_statements s ON r.statement_id = s.statement_id "
            "JOIN companies c ON s.company_id = c.company_id "
            "JOIN industries i ON c.industry_id = i.industry_id "
            "JOIN sectors sec ON i.sector_id = sec.sector_id"
        )
        where = []
        if ticker_filter:
            where.append(ticker_filter)
        if sector:
            where.append(f"sec.sector_name = '{sector}'")
        if year:
            where.append(f"s.fiscal_year = {year}")
        if where:
            sql += " WHERE " + " AND ".join(where)
        if not tickers:
            sql += f" ORDER BY r.{metric} {direction or 'DESC'} LIMIT 5"
        return sql

    if primary == "financial_statements":
        metric = _detect_metric(query, _FS_METRICS)
        select_cols = f"c.ticker, c.company_name, s.{metric}" if metric else (
            "c.ticker, c.company_name, s.fiscal_year, s.revenue, s.net_income, "
            "s.operating_income, s.total_assets, s.total_liabilities"
        )
        sql = (
            f"SELECT {select_cols} FROM financial_statements s "
            "JOIN companies c ON s.company_id = c.company_id "
            "JOIN industries i ON c.industry_id = i.industry_id "
            "JOIN sectors sec ON i.sector_id = sec.sector_id"
        )
        where = []
        if ticker_filter:
            where.append(ticker_filter)
        if sector:
            where.append(f"sec.sector_name = '{sector}'")
        if year:
            where.append(f"s.fiscal_year = {year}")
        if where:
            sql += " WHERE " + " AND ".join(where)
        if not tickers and metric:
            sql += f" ORDER BY s.{metric} {direction or 'DESC'} LIMIT 5"
        return sql

    if primary == "companies":
        if re.search(r"\bhow many\b", query.lower()) and not tickers:
            return "SELECT COUNT(*) AS company_count FROM companies"
        sql = (
            "SELECT c.ticker, c.company_name, c.hq_country, c.employees, "
            "c.founded_year, i.industry_name, sec.sector_name "
            "FROM companies c "
            "JOIN industries i ON c.industry_id = i.industry_id "
            "JOIN sectors sec ON i.sector_id = sec.sector_id"
        )
        where = []
        if ticker_filter:
            where.append(ticker_filter)
        if sector:
            where.append(f"sec.sector_name = '{sector}'")
        if where:
            sql += " WHERE " + " AND ".join(where)
        return sql

    if primary == "business_segments":
        sql = (
            "SELECT c.ticker, b.segment_name, b.segment_revenue, b.yoy_growth_pct "
            "FROM business_segments b JOIN companies c ON b.company_id = c.company_id"
        )
        if ticker_filter:
            sql += f" WHERE {ticker_filter}"
        sql += " ORDER BY b.segment_revenue DESC"
        return sql

    if primary == "earnings_events":
        quarter = _detect_quarter(query)
        sql = (
            "SELECT c.ticker, e.fiscal_year, e.fiscal_quarter, e.eps_actual, "
            "e.eps_estimate, e.surprise_pct, e.revenue_actual "
            "FROM earnings_events e JOIN companies c ON e.company_id = c.company_id"
        )
        where = []
        if ticker_filter:
            where.append(ticker_filter)
        if quarter:
            where.append(f"e.fiscal_quarter = '{quarter}'")
        if where:
            sql += " WHERE " + " AND ".join(where)
        if not tickers:
            sql += f" ORDER BY e.surprise_pct {direction or 'DESC'} LIMIT 5"
        else:
            sql += " ORDER BY e.report_date"
        return sql

    if primary == "risk_factors":
        sql = (
            "SELECT c.ticker, rf.risk_category, rf.description "
            "FROM risk_factors rf JOIN companies c ON rf.company_id = c.company_id"
        )
        if ticker_filter:
            sql += f" WHERE {ticker_filter}"
        return sql

    if primary == "executives":
        sql = (
            "SELECT c.ticker, c.company_name, x.name, x.title "
            "FROM executives x JOIN companies c ON x.company_id = c.company_id"
        )
        where = []
        if ticker_filter:
            where.append(ticker_filter)
        low = query.lower()
        if "cfo" in low or "financial officer" in low:
            where.append("x.title = 'Chief Financial Officer'")
        elif "ceo" in low or "chief executive" in low or "who runs" in low or "who is the" in low:
            where.append("x.title = 'Chief Executive Officer'")
        if where:
            sql += " WHERE " + " AND ".join(where)
        return sql

    if primary == "sectors":
        sql = "SELECT sector_name, description FROM sectors"
        if sector:
            sql += f" WHERE sector_name = '{sector}'"
        return sql

    if primary == "industries":
        sql = (
            "SELECT i.industry_name, sec.sector_name FROM industries i "
            "JOIN sectors sec ON i.sector_id = sec.sector_id"
        )
        if sector:
            sql += f" WHERE sec.sector_name = '{sector}'"
        return sql

    if primary == "earnings_reports":
        sql = "SELECT title, doc_type, fiscal_year, content FROM earnings_reports"
        if tickers:
            sql += (
                " WHERE company_id IN (SELECT company_id FROM companies "
                f"WHERE ticker IN ({_sql_str_list(tickers)}))"
            )
        return sql

    if primary == "macro_indicators":
        return "SELECT name, period, value, unit, description FROM macro_indicators"

    return None


# --------------------------------------------------------------------------- #
# SQL build + orchestration
# --------------------------------------------------------------------------- #


def build_sql(query: str, selected: List[str], provider_override: Optional[str] = None) -> Dict[str, Any]:
    """Produce a validated SELECT for ``query`` using the LLM, else templates."""
    sql_tables = _tables_for_sql(selected, query)
    schema_snippet = schema_catalog.render_schema_for_sql(sql_tables)

    # 1) Try the LLM NL2SQL path.
    llm = generate_sql(query, schema_snippet, provider_override=provider_override)
    sql_text = llm.get("sql")
    if sql_text and "unanswerable" not in sql_text.lower():
        try:
            sql_db.validate_select(sql_text)
            return {"sql": sql_text, "strategy": "llm",
                    "provider_used": llm.get("provider_used")}
        except sql_db.SqlValidationError as exc:
            log.warning("llm sql rejected", extra={"error": str(exc), "sql": sql_text})

    # 2) Deterministic template fallback on the highest-ranked table.
    primary = selected[0] if selected else "financial_statements"
    template = _template_sql(query, primary)
    if template:
        return {"sql": template, "strategy": "template", "provider_used": "deterministic"}

    return {"sql": None, "strategy": "none", "provider_used": "none"}


def answer_structured(
    query: str,
    top_k_tables: int = 4,
    row_limit: int = 50,
    provider_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Select tables, build + run SQL, and return rows with diagnostics."""
    selection = select_tables(query, top_k=top_k_tables)
    selected = selection["selected"]

    build = build_sql(query, selected, provider_override=provider_override)
    sql_text = build.get("sql")

    rows: List[Dict[str, Any]] = []
    executed_sql: Optional[str] = None
    error: Optional[str] = None
    if sql_text:
        try:
            rows, executed_sql = sql_db.run_select(sql_text, limit=row_limit)
        except Exception as exc:
            error = str(exc)
            log.warning("sql execution failed", extra={"error": error, "sql": sql_text})

    return {
        "selected_tables": selected,
        "ranked_tables": selection["ranked"],
        "sql": executed_sql or sql_text,
        "sql_strategy": build.get("strategy"),
        "sql_provider": build.get("provider_used"),
        "rows": rows,
        "row_count": len(rows),
        "error": error,
    }


def _row_phrase(row: Dict[str, Any]) -> str:
    """Compact, period-free one-line summary of a row (safe for sentence split)."""
    ident = None
    for key in ("ticker", "company_name", "sector_name", "name", "segment_name"):
        if key in row and row[key] not in (None, ""):
            ident = str(row[key])
            break
    parts = [f"{k}={v}" for k, v in row.items() if k != "ticker"]
    body = ", ".join(parts)
    return f"{ident}: {body}" if ident else body


def format_rows_as_context(result: Dict[str, Any], query: Optional[str] = None) -> str:
    """Render fetched rows as a grounded, citation-friendly context block.

    The first line is a query-aligned 'direct answer' sentence: it repeats the
    question and summarises the top rows so that even the offline extractive
    reasoner ranks the database facts above any narrative passage.
    """
    tables = ", ".join(result.get("selected_tables", [])) or "n/a"
    sql = result.get("sql") or "n/a"
    rows = result.get("rows") or []

    lines: List[str] = []
    if rows and query:
        summary = "; ".join(_row_phrase(r) for r in rows[:3])
        # Strip sentence-terminating punctuation from the echoed question so the
        # whole lead stays a single sentence (the extractive reasoner splits on
        # '.', '?' and '!'), keeping the facts attached to the query keywords.
        q_clean = re.sub(r"[.?!]+", " ", query).strip()
        lines.append(f"Database answer for {q_clean}: {summary}")
    lines.append(f"[SQL] Selected table(s): {tables}")
    lines.append(f"[SQL] Query: {sql}")

    if not rows:
        lines.append("[SQL] Result: no matching rows were found.")
        return "\n".join(lines)

    lines.append(f"[SQL] Result rows ({len(rows)}):")
    for i, row in enumerate(rows[:25], start=1):
        rendered = ", ".join(f"{k}={v}" for k, v in row.items())
        lines.append(f"  ({i}) {rendered}")
    return "\n".join(lines)
