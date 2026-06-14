"""Declarative schema catalog for the FinDataAccelerator relational database.

This module is the single source of truth that describes every table in
``findata.db`` in natural language. It powers three things:

* **Table selection** - each table's "card" (description + columns + example
  questions) is embedded into a dedicated ChromaDB collection so the agent can
  semantically choose the right table for a question.
* **Keyword/alias routing** - a deterministic boost layer that nudges obvious
  questions to the obvious table even when embeddings are ambiguous.
* **NL2SQL grounding** - the catalog renders a compact schema + relationship
  snippet that is handed to the LLM (or the deterministic fallback) so the
  generated SQL only references real tables, columns and join keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Relationship:
    column: str
    references: str          # "parent_table.parent_column"
    description: str = ""


@dataclass(frozen=True)
class TableInfo:
    name: str
    primary_key: str
    description: str
    columns: Dict[str, str]
    relationships: List[Relationship] = field(default_factory=list)
    example_questions: List[str] = field(default_factory=list)
    keyword_aliases: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #


CATALOG: Dict[str, TableInfo] = {
    "sectors": TableInfo(
        name="sectors",
        primary_key="sector_id",
        description=(
            "Reference list of the high-level GICS-style economic sectors "
            "(Technology, Financial Services, Energy, Healthcare, etc.) with a "
            "short description of each sector."
        ),
        columns={
            "sector_id": "Primary key for the sector.",
            "sector_name": "Human-readable sector name, e.g. 'Technology'.",
            "description": "Short description of what the sector contains.",
        },
        example_questions=[
            "What sectors are covered?",
            "Describe the Technology sector.",
            "List all sectors.",
        ],
        keyword_aliases=["sector", "sectors", "industry group", "gics"],
    ),
    "industries": TableInfo(
        name="industries",
        primary_key="industry_id",
        description=(
            "Reference list of specific industries (e.g. Semiconductors, "
            "Banks - Diversified, Drug Manufacturers). Each industry belongs to "
            "exactly one sector."
        ),
        columns={
            "industry_id": "Primary key for the industry.",
            "industry_name": "Industry name, e.g. 'Semiconductors'.",
            "sector_id": "Foreign key to the parent sector.",
        },
        relationships=[
            Relationship("sector_id", "sectors.sector_id", "Each industry belongs to one sector."),
        ],
        example_questions=[
            "Which industries are in the Technology sector?",
            "What industry is NVIDIA in?",
            "List the industries.",
        ],
        keyword_aliases=["industry", "industries"],
    ),
    "companies": TableInfo(
        name="companies",
        primary_key="company_id",
        description=(
            "Master record for each company: ticker symbol, legal name, the "
            "industry it operates in, headquarters country, employee headcount "
            "and the year it was founded. Join here to resolve a ticker or "
            "company name to its financials, segments, risks or executives."
        ),
        columns={
            "company_id": "Primary key for the company.",
            "ticker": "Stock ticker symbol, e.g. 'AAPL'.",
            "company_name": "Full company name, e.g. 'Apple Inc.'.",
            "industry_id": "Foreign key to the company's industry.",
            "hq_country": "Country of the company headquarters.",
            "employees": "Approximate total employee headcount.",
            "founded_year": "Year the company was founded.",
        },
        relationships=[
            Relationship("industry_id", "industries.industry_id", "Each company belongs to one industry."),
        ],
        example_questions=[
            "How many employees does Amazon have?",
            "Where is NVIDIA headquartered?",
            "When was Microsoft founded?",
            "What is the ticker for Walmart?",
        ],
        keyword_aliases=[
            "company", "companies", "ticker", "headquarters", "hq",
            "employees", "headcount", "founded", "where is", "based",
        ],
    ),
    "financial_statements": TableInfo(
        name="financial_statements",
        primary_key="statement_id",
        description=(
            "Annual income-statement and balance-sheet figures per company and "
            "fiscal year: revenue, net income, operating income, gross profit, "
            "total assets, total liabilities, free cash flow and capital "
            "expenditure. All monetary values are in millions of USD. Use this "
            "for absolute dollar figures and to rank companies by size."
        ),
        columns={
            "statement_id": "Primary key for the annual statement.",
            "company_id": "Foreign key to the company.",
            "fiscal_year": "Fiscal year of the figures.",
            "revenue": "Total revenue in $M.",
            "net_income": "Net income (bottom-line profit) in $M.",
            "operating_income": "Operating income in $M.",
            "gross_profit": "Gross profit in $M.",
            "total_assets": "Total assets in $M.",
            "total_liabilities": "Total liabilities in $M.",
            "free_cash_flow": "Free cash flow in $M.",
            "capex": "Capital expenditures in $M.",
        },
        relationships=[
            Relationship("company_id", "companies.company_id", "Each statement belongs to one company."),
        ],
        example_questions=[
            "What was Apple's FY2024 revenue?",
            "How much net income did Microsoft make?",
            "Which company had the highest revenue?",
            "What were NVIDIA's total assets?",
            "How much did Amazon spend on capex?",
        ],
        keyword_aliases=[
            "revenue", "sales", "net income", "profit", "earnings",
            "operating income", "gross profit", "total assets", "liabilities",
            "free cash flow", "fcf", "capex", "capital expenditure", "balance sheet",
        ],
    ),
    "financial_ratios": TableInfo(
        name="financial_ratios",
        primary_key="ratio_id",
        description=(
            "Derived profitability and leverage ratios for each annual "
            "statement: net profit margin %, operating margin %, gross margin %, "
            "debt-to-assets % and return on equity %. Use this for any question "
            "about margins, profitability ratios or leverage rather than raw "
            "dollars. Join through financial_statements to reach the company."
        ),
        columns={
            "ratio_id": "Primary key for the ratio row.",
            "statement_id": "Foreign key to the annual statement.",
            "net_profit_margin_pct": "Net income / revenue, as a percentage.",
            "operating_margin_pct": "Operating income / revenue, as a percentage.",
            "gross_margin_pct": "Gross profit / revenue, as a percentage.",
            "debt_to_assets_pct": "Total liabilities / total assets, as a percentage.",
            "roe_pct": "Return on equity, net income / equity, as a percentage.",
        },
        relationships=[
            Relationship("statement_id", "financial_statements.statement_id",
                         "Each ratio row maps to one annual statement (then to a company)."),
        ],
        example_questions=[
            "Which technology company had the highest net profit margin in 2024?",
            "What is NVIDIA's net profit margin?",
            "Compare operating margins across banks.",
            "What is Apple's return on equity?",
        ],
        keyword_aliases=[
            "margin", "margins", "net profit margin", "operating margin",
            "gross margin", "profitability", "roe", "return on equity",
            "debt to assets", "leverage", "ratio", "ratios",
        ],
    ),
    "business_segments": TableInfo(
        name="business_segments",
        primary_key="segment_id",
        description=(
            "Revenue broken down by business segment / product line for each "
            "company and fiscal year (e.g. Apple iPhone vs Services, Amazon AWS "
            "vs Retail), including each segment's revenue and year-over-year "
            "growth percentage."
        ),
        columns={
            "segment_id": "Primary key for the segment row.",
            "company_id": "Foreign key to the company.",
            "fiscal_year": "Fiscal year of the segment figures.",
            "segment_name": "Name of the business segment / product line.",
            "segment_revenue": "Revenue attributed to the segment in $M.",
            "yoy_growth_pct": "Year-over-year growth of the segment, as a percentage.",
        },
        relationships=[
            Relationship("company_id", "companies.company_id", "Each segment belongs to one company."),
        ],
        example_questions=[
            "What are Apple's business segments?",
            "How much revenue did AWS generate?",
            "Which segment grew the fastest at NVIDIA?",
            "Break down Microsoft revenue by segment.",
        ],
        keyword_aliases=[
            "segment", "segments", "business segment", "product line",
            "division", "aws", "iphone", "breakdown", "by segment",
        ],
    ),
    "earnings_events": TableInfo(
        name="earnings_events",
        primary_key="event_id",
        description=(
            "Quarterly earnings events per company: fiscal quarter, report date, "
            "actual vs estimated EPS, actual revenue and the EPS surprise "
            "percentage. Use this for quarter-level questions, EPS, estimates, "
            "beats/misses and earnings surprises."
        ),
        columns={
            "event_id": "Primary key for the earnings event.",
            "company_id": "Foreign key to the company.",
            "fiscal_year": "Fiscal year of the quarter.",
            "fiscal_quarter": "Fiscal quarter label, e.g. 'Q3'.",
            "report_date": "Date the quarter was reported (YYYY-MM-DD).",
            "eps_actual": "Reported (actual) earnings per share.",
            "eps_estimate": "Consensus estimated earnings per share.",
            "revenue_actual": "Reported quarterly revenue in $M.",
            "surprise_pct": "EPS surprise = (actual - estimate)/estimate, as a percentage.",
        },
        relationships=[
            Relationship("company_id", "companies.company_id", "Each earnings event belongs to one company."),
        ],
        example_questions=[
            "What was Apple's Q3 EPS?",
            "Did NVIDIA beat earnings estimates?",
            "What was the biggest earnings surprise?",
            "Show Microsoft quarterly EPS.",
        ],
        keyword_aliases=[
            "eps", "earnings per share", "quarter", "quarterly", "q1", "q2",
            "q3", "q4", "estimate", "consensus", "surprise", "beat", "miss",
        ],
    ),
    "risk_factors": TableInfo(
        name="risk_factors",
        primary_key="risk_id",
        description=(
            "Principal risk factors disclosed by each company, grouped by risk "
            "category (e.g. Supply Chain, Regulatory, Competition) with a "
            "description of each risk. Use this for any question about risks, "
            "threats or 10-K risk factors."
        ),
        columns={
            "risk_id": "Primary key for the risk factor.",
            "company_id": "Foreign key to the company.",
            "fiscal_year": "Fiscal year the risk was disclosed.",
            "risk_category": "Category of the risk, e.g. 'Regulatory'.",
            "description": "Description of the specific risk factor.",
        },
        relationships=[
            Relationship("company_id", "companies.company_id", "Each risk factor belongs to one company."),
        ],
        example_questions=[
            "Summarise NVIDIA's risk factors.",
            "What are Apple's main risks?",
            "What regulatory risks does Microsoft face?",
        ],
        keyword_aliases=[
            "risk", "risks", "risk factor", "risk factors", "threat",
            "exposure", "10-k risk", "headwind",
        ],
    ),
    "executives": TableInfo(
        name="executives",
        primary_key="exec_id",
        description=(
            "Key executives (CEO, CFO) for each company, with their name, title "
            "and the year they took the role. Use this for leadership questions "
            "such as 'who is the CEO of ...'."
        ),
        columns={
            "exec_id": "Primary key for the executive row.",
            "company_id": "Foreign key to the company.",
            "name": "Executive's full name.",
            "title": "Executive's title, e.g. 'Chief Executive Officer'.",
            "since_year": "Year the executive took the role (may be null).",
        },
        relationships=[
            Relationship("company_id", "companies.company_id", "Each executive belongs to one company."),
        ],
        example_questions=[
            "Who is the CEO of Microsoft?",
            "Who runs Apple?",
            "Who is NVIDIA's CFO?",
        ],
        keyword_aliases=[
            "ceo", "cfo", "executive", "executives", "chief executive",
            "leader", "leadership", "who runs", "who is the ceo", "management",
        ],
    ),
    "earnings_reports": TableInfo(
        name="earnings_reports",
        primary_key="report_id",
        description=(
            "Long-form narrative documents: annual summaries, risk-factor "
            "narratives, sector overviews, macro outlooks and a metrics "
            "glossary. Use this table for qualitative, descriptive or "
            "'summarise / explain / describe' questions where free text is "
            "needed rather than a precise number. company_id and sector_id may "
            "be null for sector/macro/glossary documents."
        ),
        columns={
            "report_id": "Primary key for the document.",
            "company_id": "Foreign key to the company (null for sector/macro docs).",
            "sector_id": "Foreign key to the sector (set for sector overviews).",
            "fiscal_year": "Fiscal year the document refers to.",
            "doc_type": "Document type, e.g. annual_summary, risk_factors, sector_overview, macro_outlook, glossary.",
            "title": "Document title.",
            "content": "Full narrative text of the document.",
        },
        relationships=[
            Relationship("company_id", "companies.company_id", "Optional link to a company."),
            Relationship("sector_id", "sectors.sector_id", "Optional link to a sector."),
        ],
        example_questions=[
            "Summarise Apple's annual report.",
            "Give an overview of the technology sector.",
            "What is the macro outlook for 2025?",
            "Define net profit margin.",
        ],
        keyword_aliases=[
            "summary", "summarise", "summarize", "overview", "narrative",
            "report", "outlook", "describe", "explain", "glossary", "definition",
        ],
    ),
    "macro_indicators": TableInfo(
        name="macro_indicators",
        primary_key="indicator_id",
        description=(
            "Top-down macroeconomic indicators and forecasts (GDP growth, CPI, "
            "Fed Funds rate, Treasury yields, S&P 500 EPS growth) with their "
            "value, unit and period."
        ),
        columns={
            "indicator_id": "Primary key for the indicator.",
            "name": "Indicator name, e.g. 'US Real GDP Growth'.",
            "period": "Period the value refers to, e.g. '2025E'.",
            "value": "Numeric value of the indicator.",
            "unit": "Unit of the value, e.g. '%'.",
            "description": "Short description of the indicator.",
        },
        example_questions=[
            "What is the expected GDP growth for 2025?",
            "What is the forecast Fed Funds rate?",
            "What is consensus S&P 500 EPS growth?",
        ],
        keyword_aliases=[
            "gdp", "cpi", "inflation", "fed funds", "interest rate",
            "treasury yield", "macro", "macroeconomic", "s&p 500", "forecast",
        ],
    ),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def list_tables() -> List[str]:
    return list(CATALOG.keys())


def get_table(name: str) -> TableInfo:
    return CATALOG[name]


def schema_card_text(table: str) -> str:
    """Render a table 'card' for semantic embedding and table selection."""
    info = CATALOG[table]
    cols = "; ".join(f"{c} ({d})" for c, d in info.columns.items())
    rels = "; ".join(f"{r.column} -> {r.references}" for r in info.relationships)
    examples = " | ".join(info.example_questions)
    parts = [
        f"TABLE: {info.name}",
        f"PURPOSE: {info.description}",
        f"COLUMNS: {cols}",
    ]
    if rels:
        parts.append(f"RELATIONSHIPS: {rels}")
    if examples:
        parts.append(f"EXAMPLE QUESTIONS: {examples}")
    return "\n".join(parts)


def all_schema_cards() -> List[Dict[str, str]]:
    """Return one card dict per table for seeding the schema collection."""
    return [{"table": t, "text": schema_card_text(t)} for t in CATALOG]


def render_schema_for_sql(tables: List[str]) -> str:
    """Render a compact CREATE-TABLE-like snippet for the NL2SQL prompt.

    Only the requested tables are included so the model stays focused, but the
    relationships are spelled out so it can JOIN correctly.
    """
    blocks: List[str] = []
    for table in tables:
        info = CATALOG.get(table)
        if not info:
            continue
        col_lines = [f"    {c}  -- {d}" for c, d in info.columns.items()]
        block = [f"TABLE {info.name} (PK: {info.primary_key})", *col_lines]
        for r in info.relationships:
            block.append(f"    FOREIGN KEY {r.column} REFERENCES {r.references}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def keyword_alias_map() -> Dict[str, List[str]]:
    """Map each table to its list of trigger phrases (lower-cased)."""
    return {t: [a.lower() for a in info.keyword_aliases] for t, info in CATALOG.items()}
