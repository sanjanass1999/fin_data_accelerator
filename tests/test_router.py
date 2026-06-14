"""Tests for the automatic table router and the safe SQL layer.

These run fully offline: with no LLM API keys configured the router falls back
to its deterministic template builder, so assertions are stable.
"""
from __future__ import annotations

import pytest

from app.utils import sql_db, table_router


def _tickers(rows):
    return {r.get("ticker") for r in rows}


# --------------------------------------------------------------------------- #
# Safe SQL layer
# --------------------------------------------------------------------------- #


def test_select_executes_and_is_read_only():
    rows, sql = sql_db.run_select("SELECT ticker FROM companies WHERE ticker = 'AAPL'")
    assert rows and rows[0]["ticker"] == "AAPL"
    assert "limit" in sql.lower()


@pytest.mark.parametrize("bad_sql", [
    "DELETE FROM companies",
    "UPDATE companies SET ticker='X'",
    "DROP TABLE companies",
    "SELECT * FROM companies; DROP TABLE companies",
    "SELECT * FROM not_a_real_table",
])
def test_unsafe_sql_is_rejected(bad_sql):
    with pytest.raises(sql_db.SqlValidationError):
        sql_db.validate_select(bad_sql)


# --------------------------------------------------------------------------- #
# Table selection accuracy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("query,expected_table", [
    ("Which technology company had the highest net profit margin in 2024?", "financial_ratios"),
    ("Summarise NVIDIA's risk factors.", "risk_factors"),
    ("Who is the CEO of Microsoft?", "executives"),
    ("What was Apple's Q3 EPS?", "earnings_events"),
    ("What are Apple's business segments?", "business_segments"),
    ("How many employees does Amazon have?", "companies"),
    ("What is the expected GDP growth for 2025?", "macro_indicators"),
])
def test_router_selects_expected_table(query, expected_table):
    selection = table_router.select_tables(query)
    assert expected_table in selection["selected"], (
        f"{expected_table} not in {selection['selected']} for query: {query}"
    )


# --------------------------------------------------------------------------- #
# End-to-end answer accuracy (against known seed values)
# --------------------------------------------------------------------------- #


def test_highest_margin_tech_company_is_nvidia():
    result = table_router.answer_structured(
        "Which technology company had the highest net profit margin in 2024?"
    )
    assert result["rows"], "expected rows"
    assert result["rows"][0]["ticker"] == "NVDA"


def test_ceo_of_microsoft():
    result = table_router.answer_structured("Who is the CEO of Microsoft?")
    assert any(r.get("name") == "Satya Nadella" for r in result["rows"])


def test_nvidia_risk_factors_returned():
    result = table_router.answer_structured("Summarise NVIDIA's risk factors.")
    assert _tickers(result["rows"]) == {"NVDA"}
    assert len(result["rows"]) >= 3


def test_generated_sql_is_validated_select():
    result = table_router.answer_structured("What was Apple's FY2024 revenue?")
    # Whatever strategy produced it, the executed SQL must pass validation.
    assert result["sql"]
    sql_db.validate_select(result["sql"])
    assert result["error"] is None
