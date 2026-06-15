"""Structured NL -> SQL planner built around a typed QuerySpec.

The planner separates *understanding* a question from *generating* SQL:

1. A :class:`QuerySpec` is the intermediate representation - it captures the
   entities, metric, dimension, aggregation, filters, sort and limit a question
   implies.
2. :func:`extract_spec` fills a QuerySpec deterministically (offline-safe);
   :func:`plan` prefers an LLM-produced spec when a provider is configured.
3. :func:`check_coverage` compares the spec against what the database actually
   holds (currently a single fiscal year) and rewrites it - attaching an honest
   note - rather than letting the system fabricate an answer.
4. :func:`compile_spec` turns the (possibly rewritten) spec into a single,
   always-valid SELECT with correct columns, JOINs, GROUP BY, ORDER BY and
   LIMIT.

Only quantitative / metric questions are owned by the planner; qualitative
tables (risk factors, executives, sectors, ...) still fall back to the
deterministic templates in :mod:`app.utils.table_router`.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.logging_config import get_logger
from app.utils import sql_db

log = get_logger("query_planner")


# --------------------------------------------------------------------------- #
# Metric vocabulary (column -> owning table + trigger phrases)
# --------------------------------------------------------------------------- #

# Each metric column and the table that owns it.
_METRIC_TABLE: Dict[str, str] = {
    # financial_statements (absolute $M figures)
    "revenue": "financial_statements",
    "net_income": "financial_statements",
    "operating_income": "financial_statements",
    "gross_profit": "financial_statements",
    "total_assets": "financial_statements",
    "total_liabilities": "financial_statements",
    "free_cash_flow": "financial_statements",
    "capex": "financial_statements",
    # financial_ratios (percentages)
    "net_profit_margin_pct": "financial_ratios",
    "operating_margin_pct": "financial_ratios",
    "gross_margin_pct": "financial_ratios",
    "debt_to_assets_pct": "financial_ratios",
    "roe_pct": "financial_ratios",
    # business_segments
    "segment_revenue": "business_segments",
    "yoy_growth_pct": "business_segments",
    # earnings_events
    "eps_actual": "earnings_events",
    "eps_estimate": "earnings_events",
    "revenue_actual": "earnings_events",
    "surprise_pct": "earnings_events",
}

# Trigger phrases per metric (longest match wins so multi-word phrases such as
# "segment revenue" beat "revenue").
_METRIC_PHRASES: Dict[str, List[str]] = {
    "revenue": ["revenue", "sales", "top line", "turnover"],
    "net_income": ["net income", "net profit", "bottom line", "profit", "earnings"],
    "operating_income": ["operating income", "operating profit"],
    "gross_profit": ["gross profit"],
    "total_assets": ["total assets", "assets"],
    "total_liabilities": ["total liabilities", "liabilities", "debt"],
    "free_cash_flow": ["free cash flow", "fcf", "cash flow"],
    "capex": ["capex", "capital expenditure", "capital expenditures"],
    "net_profit_margin_pct": ["net profit margin", "net margin", "profit margin", "margin"],
    "operating_margin_pct": ["operating margin"],
    "gross_margin_pct": ["gross margin"],
    "debt_to_assets_pct": ["debt to assets", "debt-to-assets", "leverage"],
    "roe_pct": ["return on equity", "roe"],
    "segment_revenue": ["segment revenue"],
    "yoy_growth_pct": ["yoy growth", "year over year growth", "segment growth"],
    "eps_actual": ["eps", "earnings per share", "actual eps"],
    "eps_estimate": ["eps estimate", "estimated eps", "consensus eps"],
    "revenue_actual": ["quarterly revenue", "reported revenue"],
    "surprise_pct": ["surprise", "beat", "miss", "earnings surprise"],
}

# How each metric table reaches the companies table (+ the alias holding the
# metric and the alias that holds fiscal_year).
_BASE: Dict[str, Tuple[str, List[str], str, str]] = {
    "financial_statements": (
        "financial_statements s",
        ["JOIN companies c ON s.company_id = c.company_id"],
        "s", "s",
    ),
    "financial_ratios": (
        "financial_ratios r",
        ["JOIN financial_statements s ON r.statement_id = s.statement_id",
         "JOIN companies c ON s.company_id = c.company_id"],
        "r", "s",
    ),
    "business_segments": (
        "business_segments b",
        ["JOIN companies c ON b.company_id = c.company_id"],
        "b", "b",
    ),
    "earnings_events": (
        "earnings_events e",
        ["JOIN companies c ON e.company_id = c.company_id"],
        "e", "e",
    ),
}


# --------------------------------------------------------------------------- #
# QuerySpec
# --------------------------------------------------------------------------- #


@dataclass
class QuerySpec:
    """Typed plan describing what SQL should compute for a question."""

    intent: str = "lookup"                       # list|rank|compare|aggregate|lookup
    entities: List[str] = field(default_factory=list)
    metric: Optional[str] = None
    metric_table: Optional[str] = None
    dimension: Optional[str] = None              # fiscal_year|segment_name|sector_name|industry_name|fiscal_quarter
    aggregation: Optional[str] = None            # avg|sum|count
    year: Optional[int] = None
    quarter: Optional[str] = None
    sector: Optional[str] = None
    direction: Optional[str] = None              # DESC|ASC (ranking)
    limit: Optional[int] = None
    source: str = "deterministic"                # deterministic|llm


# --------------------------------------------------------------------------- #
# Detection helpers
# --------------------------------------------------------------------------- #


def _detect_metric_any(query: str) -> Optional[str]:
    """Detect the single best metric across all tables (longest phrase wins)."""
    low = query.lower()
    best: Optional[str] = None
    best_len = 0
    for col, phrases in _METRIC_PHRASES.items():
        for p in phrases:
            # Allow an optional trailing plural "s" ("margins", "revenues").
            if re.search(rf"(?<![a-z]){re.escape(p)}s?(?![a-z])", low) and len(p) > best_len:
                best, best_len = col, len(p)
    return best


def _detect_dimension(low: str) -> Optional[str]:
    if re.search(r"\b(by|per|each)\s+segment\b", low) or "segment" in low or "breakdown" in low or "by product" in low:
        return "segment_name"
    if re.search(r"\b(by|per|each)\s+quarter\b", low) or re.search(r"\bquarter(?:ly|s)?\b", low):
        return "fiscal_quarter"
    if re.search(r"\b(by|per|each)\s+sector\b", low):
        return "sector_name"
    if re.search(r"\b(by|per|each)\s+industry\b", low):
        return "industry_name"
    if re.search(r"\byears?\b", low) or re.search(r"\b(annual|yearly)\b", low):
        return "fiscal_year"
    return None


def _detect_aggregation(low: str) -> Optional[str]:
    if re.search(r"\b(average|avg|mean)\b", low):
        return "avg"
    if re.search(r"\bcount\b|\bhow many\b|\bnumber of\b", low):
        return "count"
    if re.search(r"\b(combined|aggregate|sum of|total of|altogether)\b", low):
        return "sum"
    return None


def _detect_intent(low: str, entities: List[str], metric: Optional[str],
                   aggregation: Optional[str], direction: Optional[str]) -> str:
    if aggregation in ("avg", "sum", "count"):
        return "aggregate"
    if direction:
        return "rank"
    if re.search(r"\b(compare|comparison|versus|vs\.?|against|difference between)\b", low) or len(entities) >= 2:
        return "compare"
    return "lookup"


def _resolve_limit(query: str, intent: str, direction: Optional[str]) -> Optional[int]:
    from app.utils import table_router as tr

    explicit = tr._detect_limit(query, default=0)
    if explicit:
        return explicit
    if intent == "rank" or direction:
        return 5
    return None


# --------------------------------------------------------------------------- #
# Deterministic extraction
# --------------------------------------------------------------------------- #


def extract_spec(query: str) -> QuerySpec:
    """Fill a :class:`QuerySpec` from ``query`` using offline heuristics."""
    from app.utils import table_router as tr

    low = query.lower()
    entities = tr._resolve_companies(query)
    metric = _detect_metric_any(query)
    metric_table = _METRIC_TABLE.get(metric) if metric else None
    dimension = _detect_dimension(low)
    aggregation = _detect_aggregation(low)
    year = tr._detect_year(query)
    quarter = tr._detect_quarter(query)
    sector = tr._detect_sector(query)
    direction = tr._detect_direction(query)
    # Explicit companies are the filter; ignore an incidental sector match
    # (e.g. "Bank of America" should not also add a Financial Services filter).
    if entities:
        sector = None
    intent = _detect_intent(low, entities, metric, aggregation, direction)

    # A ranked / year-dimensioned / avg|sum question without an explicit metric
    # defaults to revenue (a sensible proxy for "best"/"biggest").
    if metric_table is None and (dimension == "fiscal_year" or direction or aggregation in ("avg", "sum")):
        metric, metric_table = "revenue", "financial_statements"

    limit = _resolve_limit(query, intent, direction)

    return QuerySpec(
        intent=intent,
        entities=entities,
        metric=metric,
        metric_table=metric_table,
        dimension=dimension,
        aggregation=aggregation,
        year=year,
        quarter=quarter,
        sector=sector,
        direction=direction,
        limit=limit,
        source="deterministic",
    )


# --------------------------------------------------------------------------- #
# LLM-produced spec (used when a provider is configured)
# --------------------------------------------------------------------------- #


def _llm_spec(query: str, provider_override: Optional[str]) -> Optional[QuerySpec]:
    from app.utils.llm_service import generate_query_spec

    raw = generate_query_spec(query, provider_override=provider_override)
    if not raw:
        return None
    try:
        metric = raw.get("metric") or None
        if metric and metric not in _METRIC_TABLE:
            metric = None
        metric_table = _METRIC_TABLE.get(metric) if metric else None
        dimension = raw.get("dimension") or None
        if dimension not in (None, "fiscal_year", "segment_name", "sector_name",
                             "industry_name", "fiscal_quarter"):
            dimension = None
        aggregation = raw.get("aggregation") or None
        if aggregation not in (None, "avg", "sum", "count"):
            aggregation = None
        direction = (raw.get("direction") or "").upper() or None
        if direction not in (None, "ASC", "DESC"):
            direction = None
        entities = [str(e).upper() for e in (raw.get("entities") or []) if e]
        limit = raw.get("limit")
        limit = int(limit) if isinstance(limit, (int, float)) and 1 <= int(limit) <= 50 else None
        year = raw.get("year")
        year = int(year) if isinstance(year, (int, float)) else None
        if metric_table is None:
            return None
        return QuerySpec(
            intent=raw.get("intent") or "lookup",
            entities=entities,
            metric=metric,
            metric_table=metric_table,
            dimension=dimension,
            aggregation=aggregation,
            year=year,
            quarter=raw.get("quarter") or None,
            sector=raw.get("sector") or None,
            direction=direction,
            limit=limit,
            source="llm",
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("llm spec parse failed", extra={"error": str(exc)})
        return None


def plan(query: str, provider_override: Optional[str] = None) -> Optional[QuerySpec]:
    """Return a QuerySpec the planner can own, or ``None`` to use legacy templates."""
    spec = _llm_spec(query, provider_override)
    if spec is None:
        spec = extract_spec(query)
    # The planner only owns questions with a concrete metric to compute.
    if not spec.metric_table:
        return None
    return spec


# --------------------------------------------------------------------------- #
# Coverage guard
# --------------------------------------------------------------------------- #

_COVERAGE: Optional[Dict[str, Any]] = None


def get_coverage() -> Dict[str, Any]:
    """Cached facts about what the database actually contains (fiscal years)."""
    global _COVERAGE
    if _COVERAGE is not None:
        return _COVERAGE
    years: set = set()
    for table in ("financial_statements", "business_segments", "earnings_events"):
        try:
            rows, _ = sql_db.run_select(f"SELECT DISTINCT fiscal_year FROM {table}", limit=100)
            years.update(int(r["fiscal_year"]) for r in rows if r.get("fiscal_year") is not None)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("coverage probe failed", extra={"table": table, "error": str(exc)})
    _COVERAGE = {"years": years}
    return _COVERAGE


def reset_coverage() -> None:
    global _COVERAGE
    _COVERAGE = None


def check_coverage(spec: QuerySpec) -> Tuple[QuerySpec, Optional[str]]:
    """Rewrite ``spec`` to what the data can support and return an honest note."""
    cov = get_coverage()
    years = cov.get("years") or set()
    if not years:
        return spec, None

    only = sorted(years)[-1] if len(years) == 1 else None
    notes: List[str] = []

    # A year-by-year breakdown / ranking is impossible with a single year.
    if spec.dimension == "fiscal_year" and only is not None:
        notes.append(
            f"The dataset currently only covers fiscal year {only}, so I can't "
            f"rank or compare across years - showing the FY{only} figure instead."
        )
        spec.dimension = None
        if not spec.year:
            spec.year = only

    # A specific requested year that we don't have.
    if spec.year and spec.year not in years:
        if only is not None:
            notes.append(
                f"I don't have FY{spec.year} data; the dataset only covers "
                f"FY{only}, so this answer is for FY{only}."
            )
            spec.year = only
        else:
            notes.append(
                f"I don't have FY{spec.year} data; available years are "
                f"{sorted(years)}."
            )

    return spec, (" ".join(notes) if notes else None)


# --------------------------------------------------------------------------- #
# SQL compiler
# --------------------------------------------------------------------------- #


def _sql_str_list(values: List[str]) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def _dimension_column(table: str, dim: Optional[str], year_alias: str) -> Optional[str]:
    if dim == "fiscal_year":
        return f"{year_alias}.fiscal_year"
    if dim == "segment_name":
        return "b.segment_name" if table == "business_segments" else None
    if dim == "fiscal_quarter":
        return "e.fiscal_quarter" if table == "earnings_events" else None
    if dim == "sector_name":
        return "sec.sector_name"
    if dim == "industry_name":
        return "i.industry_name"
    return None


def compile_spec(spec: QuerySpec) -> Optional[str]:
    """Compile a QuerySpec into a single, valid SQLite SELECT."""
    table = spec.metric_table
    if not table or table not in _BASE:
        return None

    base, joins, alias, year_alias = _BASE[table]
    metric_expr = f"{alias}.{spec.metric}" if spec.metric else None
    dim = spec.dimension
    dim_col = _dimension_column(table, dim, year_alias)
    need_company_sector = dim in ("sector_name", "industry_name") or bool(spec.sector)

    where: List[str] = []
    if spec.entities:
        where.append(f"c.ticker IN ({_sql_str_list(spec.entities)})")
    if spec.sector:
        where.append(f"sec.sector_name = '{spec.sector}'")
    if spec.year:
        where.append(f"{year_alias}.fiscal_year = {int(spec.year)}")
    if spec.quarter and table == "earnings_events":
        where.append(f"e.fiscal_quarter = '{spec.quarter}'")

    select: List[str] = []
    group: List[str] = []
    order: Optional[str] = None

    if spec.aggregation in ("avg", "sum", "count"):
        fn = {"avg": "AVG", "sum": "SUM", "count": "COUNT"}[spec.aggregation]
        if dim_col:
            select.append(dim_col)
            group.append(dim_col)
        elif spec.entities:
            select.extend(["c.ticker", "c.company_name"])
            group.extend(["c.ticker", "c.company_name"])
        out_col = f"{spec.aggregation}_{spec.metric or 'count'}"
        if spec.aggregation == "count":
            agg_expr = f"{fn}(*)"
        else:
            agg_expr = f"ROUND({fn}({metric_expr}), 2)"
        select.append(f"{agg_expr} AS {out_col}")
        if spec.aggregation != "count":
            order = f"{agg_expr} {spec.direction or 'DESC'}"
    else:
        select.extend(["c.ticker", "c.company_name"])
        if dim_col:
            select.append(dim_col)
        if metric_expr:
            select.append(metric_expr)
        if spec.direction or spec.intent == "rank":
            order = f"{metric_expr} {spec.direction or 'DESC'}"
        elif dim == "fiscal_year":
            order = f"{year_alias}.fiscal_year"

    sql = f"SELECT {', '.join(select)} FROM {base} " + " ".join(joins)
    if need_company_sector:
        sql += (" JOIN industries i ON c.industry_id = i.industry_id "
                "JOIN sectors sec ON i.sector_id = sec.sector_id")
    if where:
        sql += " WHERE " + " AND ".join(where)
    if group:
        sql += " GROUP BY " + ", ".join(group)
    if order:
        sql += " ORDER BY " + order
    if spec.limit:
        sql += f" LIMIT {int(spec.limit)}"
    return sql


def spec_to_dict(spec: Optional[QuerySpec]) -> Optional[Dict[str, Any]]:
    return asdict(spec) if spec is not None else None
