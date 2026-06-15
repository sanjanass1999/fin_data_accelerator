"""Tests for the QuerySpec NL->SQL planner."""
from __future__ import annotations

from app.utils import query_planner as qp
from app.utils import table_router as tr


# --------------------------------------------------------------------------- #
# Spec extraction
# --------------------------------------------------------------------------- #


def test_extract_metric_and_table():
    spec = qp.extract_spec("what is nvidia net profit margin")
    assert spec.metric == "net_profit_margin_pct"
    assert spec.metric_table == "financial_ratios"
    assert "NVDA" in spec.entities


def test_extract_plural_metric_matches():
    # "margins" (plural) must still resolve to the ratio column, not net_income.
    spec = qp.extract_spec("compare the net profit margins of JPMorgan and Bank of America")
    assert spec.metric == "net_profit_margin_pct"
    assert spec.intent == "compare"
    # Explicit entities should suppress an incidental sector filter.
    assert spec.sector is None


def test_extract_rank_limit():
    spec = qp.extract_spec("top 3 companies which have the highest revenue")
    assert spec.intent == "rank"
    assert spec.direction == "DESC"
    assert spec.limit == 3
    assert spec.metric == "revenue"


def test_extract_year_dimension_and_default_metric():
    spec = qp.extract_spec("best years for amazon")
    assert spec.dimension == "fiscal_year"
    # No explicit metric -> defaults to revenue so "best" is computable.
    assert spec.metric == "revenue"
    assert spec.metric_table == "financial_statements"


def test_extract_aggregation_avg_by_sector():
    spec = qp.extract_spec("average net profit margin by sector")
    assert spec.aggregation == "avg"
    assert spec.dimension == "sector_name"
    assert spec.intent == "aggregate"


def test_plan_returns_none_for_non_metric_question():
    # Qualitative questions are left to the legacy templates.
    assert qp.plan("who is the CEO of apple", provider_override="simulation") is None
    assert qp.plan("list all the companies", provider_override="simulation") is None


# --------------------------------------------------------------------------- #
# SQL compilation
# --------------------------------------------------------------------------- #


def test_compile_rank_has_order_and_limit():
    spec = qp.extract_spec("top 3 companies by revenue")
    sql = qp.compile_spec(spec)
    assert "ORDER BY s.revenue DESC" in sql
    assert "LIMIT 3" in sql
    assert "FROM financial_statements s" in sql


def test_compile_aggregate_has_group_by():
    spec = qp.extract_spec("average net profit margin by sector")
    sql = qp.compile_spec(spec)
    assert "GROUP BY sec.sector_name" in sql
    assert "AVG(r.net_profit_margin_pct)" in sql
    assert "ROUND(" in sql


def test_compile_compare_filters_entities():
    spec = qp.extract_spec("compare net profit margins of JPMorgan and Wells Fargo")
    sql = qp.compile_spec(spec)
    assert "r.net_profit_margin_pct" in sql
    assert "c.ticker IN (" in sql
    assert "'JPM'" in sql and "'WFC'" in sql


# --------------------------------------------------------------------------- #
# Coverage guard
# --------------------------------------------------------------------------- #


def test_coverage_degrades_year_dimension():
    spec = qp.extract_spec("best years for amazon")
    spec, note = qp.check_coverage(spec)
    # Only FY2024 exists -> year dimension dropped + honest note attached.
    assert spec.dimension is None
    assert spec.year == 2024
    assert note and "2024" in note


def test_coverage_handles_missing_year():
    spec = qp.QuerySpec(metric="revenue", metric_table="financial_statements", year=2019)
    spec, note = qp.check_coverage(spec)
    assert spec.year == 2024
    assert note and "2019" in note


# --------------------------------------------------------------------------- #
# End-to-end via the router
# --------------------------------------------------------------------------- #


def test_router_uses_planner_and_respects_limit():
    routed = tr.answer_structured("top 3 companies by revenue", provider_override="simulation")
    assert routed["sql_strategy"] == "planner:deterministic"
    assert routed["row_count"] == 3
    assert "LIMIT 3" in routed["sql"]


def test_router_best_years_is_honest():
    routed = tr.answer_structured(
        "top 3 best years for amazon in the last 10 years", provider_override="simulation"
    )
    assert routed["coverage_note"] and "2024" in routed["coverage_note"]
    answer = tr.rows_to_answer(routed, query="top 3 best years for amazon")
    assert "AMZN" in answer or "Amazon" in answer
