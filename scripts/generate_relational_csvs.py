"""Generate the normalized relational seed CSVs for FinDataAccelerator.

This script is the *reproducible* origin of the relational source. It reads the
legacy flat sources (``app/data/sample_companies.csv`` and
``app/data/earnings_reports.json``) plus a handful of curated lookup
dictionaries, and emits one CSV per table into ``app/data/relational/``.

Those CSVs are the human-readable, version-controlled source of truth that
``scripts/build_database.py`` loads into the SQLite database (``findata.db``)
with real PRIMARY KEY / FOREIGN KEY constraints.

Run from the project root::

    python scripts/generate_relational_csvs.py
"""
from __future__ import annotations

import csv
import json
import os
import random
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "app", "data")
REL_DIR = os.path.join(DATA_DIR, "relational")
LEGACY_CSV = os.path.join(DATA_DIR, "sample_companies.csv")
LEGACY_JSON = os.path.join(DATA_DIR, "earnings_reports.json")

# Deterministic synthesis so the generated source is stable across runs.
random.seed(20240614)


# --------------------------------------------------------------------------- #
# Curated lookup data
# --------------------------------------------------------------------------- #

SECTOR_DESCRIPTIONS: Dict[str, str] = {
    "Technology": "Hardware, software, semiconductors and IT services companies.",
    "Consumer Cyclical": "Discretionary retail, autos, restaurants and apparel.",
    "Financial Services": "Banks, capital-markets firms and diversified financials.",
    "Energy": "Integrated oil & gas and energy producers.",
    "Healthcare": "Pharmaceuticals, biotech and managed-care providers.",
    "Consumer Defensive": "Staples, beverages, household products and discount retail.",
    "Communication Services": "Telecom, media, entertainment and streaming.",
    "Basic Materials": "Chemicals, mining and materials producers.",
    "Industrials": "Aerospace, defense, machinery and diversified industrials.",
}

# Cleaned revenue overrides for rows that are corrupt in the legacy CSV.
REVENUE_OVERRIDES: Dict[str, float] = {
    "AAPL": 391035.0,  # legacy CSV had the literal string "uh"
}

GROSS_MARGIN_BY_SECTOR: Dict[str, float] = {
    "Technology": 0.58,
    "Consumer Cyclical": 0.34,
    "Financial Services": 0.55,
    "Energy": 0.30,
    "Healthcare": 0.62,
    "Consumer Defensive": 0.38,
    "Communication Services": 0.52,
    "Basic Materials": 0.33,
    "Industrials": 0.31,
}

CAPEX_INTENSITY_BY_SECTOR: Dict[str, float] = {
    "Technology": 0.09,
    "Consumer Cyclical": 0.05,
    "Financial Services": 0.02,
    "Energy": 0.12,
    "Healthcare": 0.05,
    "Consumer Defensive": 0.04,
    "Communication Services": 0.15,
    "Basic Materials": 0.10,
    "Industrials": 0.04,
}

FOUNDED_YEAR: Dict[str, int] = {
    "AAPL": 1976, "MSFT": 1975, "GOOGL": 1998, "NVDA": 1993, "AMZN": 1994,
    "META": 2004, "TSLA": 2003, "JPM": 2000, "BAC": 1998, "WFC": 1852,
    "GS": 1869, "MS": 1935, "XOM": 1870, "CVX": 1879, "PFE": 1849,
    "JNJ": 1886, "UNH": 1977, "ABBV": 2013, "WMT": 1962, "PG": 1837,
    "KO": 1892, "PEP": 1898, "DIS": 1923, "NFLX": 1997, "ADBE": 1982,
    "CRM": 1999, "ORCL": 1977, "INTC": 1968, "AMD": 1969, "QCOM": 1985,
    "AVGO": 1991, "ACN": 1989, "IBM": 1911, "COST": 1983, "HD": 1978,
    "LOW": 1921, "NKE": 1964, "SBUX": 1971, "MCD": 1955, "TMUS": 1994,
    "T": 1885, "VZ": 2000, "LIN": 1879, "APD": 1940, "GE": 1892,
    "CAT": 1925, "HON": 1906, "MMM": 1902, "LMT": 1995, "RTX": 1922,
}

CEO_BY_TICKER: Dict[str, str] = {
    "AAPL": "Tim Cook", "MSFT": "Satya Nadella", "GOOGL": "Sundar Pichai",
    "NVDA": "Jensen Huang", "AMZN": "Andy Jassy", "META": "Mark Zuckerberg",
    "TSLA": "Elon Musk", "JPM": "Jamie Dimon", "BAC": "Brian Moynihan",
    "WFC": "Charlie Scharf", "GS": "David Solomon", "MS": "Ted Pick",
    "XOM": "Darren Woods", "CVX": "Mike Wirth", "PFE": "Albert Bourla",
    "JNJ": "Joaquin Duato", "UNH": "Andrew Witty", "ABBV": "Robert Michael",
    "WMT": "Doug McMillon", "PG": "Jon Moeller", "KO": "James Quincey",
    "PEP": "Ramon Laguarta", "DIS": "Bob Iger", "NFLX": "Ted Sarandos",
    "ADBE": "Shantanu Narayen", "CRM": "Marc Benioff", "ORCL": "Safra Catz",
    "INTC": "Pat Gelsinger", "AMD": "Lisa Su", "QCOM": "Cristiano Amon",
    "AVGO": "Hock Tan", "ACN": "Julie Sweet", "IBM": "Arvind Krishna",
    "COST": "Ron Vachris", "HD": "Ted Decker", "LOW": "Marvin Ellison",
    "NKE": "Elliott Hill", "SBUX": "Brian Niccol", "MCD": "Chris Kempczinski",
    "TMUS": "Mike Sievert", "T": "John Stankey", "VZ": "Hans Vestberg",
    "LIN": "Sanjiv Lamba", "APD": "Seifi Ghasemi", "GE": "Larry Culp",
    "CAT": "Jim Umpleby", "HON": "Vimal Kapur", "MMM": "William Brown",
    "LMT": "Jim Taiclet", "RTX": "Chris Calio",
}

CFO_BY_TICKER: Dict[str, str] = {
    "AAPL": "Luca Maestri", "MSFT": "Amy Hood", "GOOGL": "Anat Ashkenazi",
    "NVDA": "Colette Kress", "AMZN": "Brian Olsavsky", "META": "Susan Li",
    "TSLA": "Vaibhav Taneja",
}

# Curated named segments for the mega-caps (others get generic split segments).
CURATED_SEGMENTS: Dict[str, List[Dict[str, Any]]] = {
    "AAPL": [
        {"segment_name": "iPhone", "share": 0.51, "yoy_growth_pct": 0.3},
        {"segment_name": "Services", "share": 0.25, "yoy_growth_pct": 13.0},
        {"segment_name": "Mac, iPad & Wearables", "share": 0.24, "yoy_growth_pct": 1.5},
    ],
    "MSFT": [
        {"segment_name": "Intelligent Cloud", "share": 0.44, "yoy_growth_pct": 20.0},
        {"segment_name": "Productivity & Business Processes", "share": 0.32, "yoy_growth_pct": 12.0},
        {"segment_name": "More Personal Computing", "share": 0.24, "yoy_growth_pct": 4.0},
    ],
    "GOOGL": [
        {"segment_name": "Google Search", "share": 0.56, "yoy_growth_pct": 12.0},
        {"segment_name": "Google Cloud", "share": 0.12, "yoy_growth_pct": 30.0},
        {"segment_name": "YouTube & Network Ads", "share": 0.20, "yoy_growth_pct": 13.0},
        {"segment_name": "Other Bets & Subscriptions", "share": 0.12, "yoy_growth_pct": 8.0},
    ],
    "NVDA": [
        {"segment_name": "Data Center", "share": 0.78, "yoy_growth_pct": 217.0},
        {"segment_name": "Gaming", "share": 0.15, "yoy_growth_pct": 15.0},
        {"segment_name": "Professional Visualization & Auto", "share": 0.07, "yoy_growth_pct": 20.0},
    ],
    "AMZN": [
        {"segment_name": "North America Retail", "share": 0.61, "yoy_growth_pct": 10.0},
        {"segment_name": "International Retail", "share": 0.22, "yoy_growth_pct": 9.0},
        {"segment_name": "AWS", "share": 0.17, "yoy_growth_pct": 19.0},
    ],
    "META": [
        {"segment_name": "Family of Apps", "share": 0.99, "yoy_growth_pct": 22.0},
        {"segment_name": "Reality Labs", "share": 0.01, "yoy_growth_pct": -5.0},
    ],
}

# Curated risk factors for the mega-caps (others get sector-template risks).
CURATED_RISKS: Dict[str, List[Dict[str, str]]] = {
    "AAPL": [
        {"risk_category": "Product Concentration", "description": "iPhone still accounts for over half of total revenue, concentrating results in a single product line."},
        {"risk_category": "Supply Chain", "description": "Dependence on a small number of contract manufacturers in mainland China and Vietnam."},
        {"risk_category": "Regulatory", "description": "App Store regulatory risk under the EU Digital Markets Act and ongoing US antitrust litigation."},
        {"risk_category": "Foreign Exchange", "description": "Roughly 60% of revenue is generated outside the United States, exposing results to FX volatility."},
        {"risk_category": "AI Execution", "description": "Risk of delivering Apple Intelligence features at the cadence promised at WWDC 2024."},
    ],
    "MSFT": [
        {"risk_category": "AI Capacity", "description": "GPU capacity demand continues to outpace supply through at least the first half of FY2025."},
        {"risk_category": "Partner Dependence", "description": "Heavy dependence on the OpenAI partnership exposes Microsoft to model availability and pricing changes."},
        {"risk_category": "Regulatory", "description": "Scrutiny around the Activision Blizzard acquisition and bundling of Teams with Office in the EU."},
        {"risk_category": "Cybersecurity", "description": "Cybersecurity exposure following the 2023 Storm-0558 incident."},
        {"risk_category": "Foreign Exchange", "description": "About half of revenue is generated outside the US."},
    ],
    "NVDA": [
        {"risk_category": "Customer Concentration", "description": "Top three hyperscaler customers contributed roughly 40% of Data Center revenue."},
        {"risk_category": "Export Controls", "description": "US export controls restrict sales of advanced GPUs to China, historically ~20% of Data Center revenue."},
        {"risk_category": "Competition", "description": "Competitive risk from AMD MI300, Intel Gaudi, and in-house custom silicon at hyperscalers."},
        {"risk_category": "Supply Chain", "description": "Supply-chain concentration on TSMC for leading-edge wafers."},
        {"risk_category": "Demand Normalization", "description": "Demand normalization risk as the first wave of generative AI training capex matures."},
    ],
}

SECTOR_RISK_TEMPLATES: Dict[str, List[Dict[str, str]]] = {
    "Technology": [
        {"risk_category": "Competition", "description": "Intense competition and rapid technological change can compress margins."},
        {"risk_category": "Talent", "description": "Reliance on attracting and retaining scarce engineering talent."},
    ],
    "Financial Services": [
        {"risk_category": "Credit", "description": "Provisions for credit losses can rise as consumer delinquencies normalize."},
        {"risk_category": "Interest Rate", "description": "Net interest income is sensitive to the rate cycle and deposit beta."},
        {"risk_category": "Regulatory", "description": "Capital and liquidity requirements constrain balance-sheet flexibility."},
    ],
    "Energy": [
        {"risk_category": "Commodity Price", "description": "Earnings are highly sensitive to Brent and WTI crude price swings."},
        {"risk_category": "Regulatory", "description": "Climate regulation and emissions policy create long-term demand uncertainty."},
    ],
    "Healthcare": [
        {"risk_category": "Patent Cliff", "description": "Loss of exclusivity on key drugs exposes revenue to biosimilar/generic erosion."},
        {"risk_category": "Regulatory", "description": "Drug pricing reform and FDA approval risk affect the pipeline."},
    ],
    "Consumer Defensive": [
        {"risk_category": "Input Costs", "description": "Commodity and freight inflation can pressure gross margins."},
        {"risk_category": "Demand", "description": "Volume softness in developing markets as pricing actions are absorbed."},
    ],
    "Consumer Cyclical": [
        {"risk_category": "Demand", "description": "Discretionary demand is sensitive to the consumer-spending cycle."},
        {"risk_category": "Competition", "description": "Pricing pressure from low-cost and online competitors."},
    ],
    "Communication Services": [
        {"risk_category": "Competition", "description": "Streaming and telecom markets face intense subscriber competition."},
        {"risk_category": "Capital Intensity", "description": "Network and content investment requires sustained heavy capex."},
    ],
    "Basic Materials": [
        {"risk_category": "Commodity Price", "description": "Input and output prices are exposed to commodity cycles."},
        {"risk_category": "Regulatory", "description": "Environmental and emissions regulation increases compliance costs."},
    ],
    "Industrials": [
        {"risk_category": "Cyclicality", "description": "Demand is tied to capital-investment and defense-budget cycles."},
        {"risk_category": "Supply Chain", "description": "Complex supply chains expose production to component shortages."},
    ],
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_legacy_companies() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(LEGACY_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ticker = row["ticker"].strip()
            raw_rev = row["revenue"].strip()
            if ticker in REVENUE_OVERRIDES:
                revenue = REVENUE_OVERRIDES[ticker]
            else:
                revenue = float(raw_rev)
            rows.append({
                "ticker": ticker,
                "company_name": row["company_name"].strip(),
                "sector": row["sector"].strip(),
                "industry": row["industry"].strip(),
                "fiscal_year": int(row["fiscal_year"]),
                "revenue": revenue,
                "net_income": float(row["net_income"]),
                "operating_income": float(row["operating_income"]),
                "total_assets": float(row["total_assets"]),
                "total_liabilities": float(row["total_liabilities"]),
                "employees": int(row["employees"]),
                "hq_country": row["hq_country"].strip(),
            })
    return rows


def _write_csv(name: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path = os.path.join(REL_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {name:<28} {len(rows):>4} rows")


def _round(value: float, places: int = 2) -> float:
    return round(float(value), places)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate() -> None:
    os.makedirs(REL_DIR, exist_ok=True)
    companies = _read_legacy_companies()

    # --- sectors -------------------------------------------------------------
    sector_names = sorted({c["sector"] for c in companies})
    sector_id = {name: i + 1 for i, name in enumerate(sector_names)}
    _write_csv(
        "sectors.csv",
        ["sector_id", "sector_name", "description"],
        [{"sector_id": sid, "sector_name": name,
          "description": SECTOR_DESCRIPTIONS.get(name, name)}
         for name, sid in sector_id.items()],
    )

    # --- industries ----------------------------------------------------------
    industry_pairs = sorted({(c["industry"], c["sector"]) for c in companies})
    industry_id = {pair: i + 1 for i, pair in enumerate(industry_pairs)}
    _write_csv(
        "industries.csv",
        ["industry_id", "industry_name", "sector_id"],
        [{"industry_id": iid, "industry_name": ind, "sector_id": sector_id[sec]}
         for (ind, sec), iid in industry_id.items()],
    )

    # --- companies -----------------------------------------------------------
    company_id = {c["ticker"]: i + 1 for i, c in enumerate(companies)}
    company_rows = []
    for c in companies:
        company_rows.append({
            "company_id": company_id[c["ticker"]],
            "ticker": c["ticker"],
            "company_name": c["company_name"],
            "industry_id": industry_id[(c["industry"], c["sector"])],
            "hq_country": c["hq_country"],
            "employees": c["employees"],
            "founded_year": FOUNDED_YEAR.get(c["ticker"], ""),
        })
    _write_csv(
        "companies.csv",
        ["company_id", "ticker", "company_name", "industry_id",
         "hq_country", "employees", "founded_year"],
        company_rows,
    )

    # --- financial_statements + financial_ratios -----------------------------
    stmt_rows = []
    ratio_rows = []
    statement_id = {}
    for i, c in enumerate(companies):
        sid = i + 1
        statement_id[c["ticker"]] = sid
        revenue = c["revenue"]
        net_income = c["net_income"]
        op_income = c["operating_income"]
        assets = c["total_assets"]
        liabilities = c["total_liabilities"]
        gross_margin = GROSS_MARGIN_BY_SECTOR.get(c["sector"], 0.4)
        gross_profit = _round(revenue * gross_margin, 0)
        capex = _round(revenue * CAPEX_INTENSITY_BY_SECTOR.get(c["sector"], 0.05), 0)
        free_cash_flow = _round(net_income * 0.9 + op_income * 0.1 - capex * 0.2, 0)
        equity = assets - liabilities
        stmt_rows.append({
            "statement_id": sid,
            "company_id": company_id[c["ticker"]],
            "fiscal_year": c["fiscal_year"],
            "revenue": _round(revenue, 0),
            "net_income": _round(net_income, 0),
            "operating_income": _round(op_income, 0),
            "gross_profit": gross_profit,
            "total_assets": _round(assets, 0),
            "total_liabilities": _round(liabilities, 0),
            "free_cash_flow": free_cash_flow,
            "capex": capex,
        })
        ratio_rows.append({
            "ratio_id": sid,
            "statement_id": sid,
            "net_profit_margin_pct": _round(net_income / revenue * 100) if revenue else 0.0,
            "operating_margin_pct": _round(op_income / revenue * 100) if revenue else 0.0,
            "gross_margin_pct": _round(gross_profit / revenue * 100) if revenue else 0.0,
            "debt_to_assets_pct": _round(liabilities / assets * 100) if assets else 0.0,
            "roe_pct": _round(net_income / equity * 100) if equity else 0.0,
        })
    _write_csv(
        "financial_statements.csv",
        ["statement_id", "company_id", "fiscal_year", "revenue", "net_income",
         "operating_income", "gross_profit", "total_assets", "total_liabilities",
         "free_cash_flow", "capex"],
        stmt_rows,
    )
    _write_csv(
        "financial_ratios.csv",
        ["ratio_id", "statement_id", "net_profit_margin_pct", "operating_margin_pct",
         "gross_margin_pct", "debt_to_assets_pct", "roe_pct"],
        ratio_rows,
    )

    # --- business_segments ---------------------------------------------------
    segment_rows = []
    seg_id = 0
    for c in companies:
        revenue = c["revenue"]
        curated = CURATED_SEGMENTS.get(c["ticker"])
        if curated:
            specs = curated
        else:
            # Generic two-segment split based on a deterministic ratio.
            split = round(random.uniform(0.55, 0.72), 2)
            specs = [
                {"segment_name": "Core Products", "share": split,
                 "yoy_growth_pct": round(random.uniform(-2, 10), 1)},
                {"segment_name": "Services & Other", "share": round(1 - split, 2),
                 "yoy_growth_pct": round(random.uniform(2, 20), 1)},
            ]
        for spec in specs:
            seg_id += 1
            segment_rows.append({
                "segment_id": seg_id,
                "company_id": company_id[c["ticker"]],
                "fiscal_year": c["fiscal_year"],
                "segment_name": spec["segment_name"],
                "segment_revenue": _round(revenue * spec["share"], 0),
                "yoy_growth_pct": spec["yoy_growth_pct"],
            })
    _write_csv(
        "business_segments.csv",
        ["segment_id", "company_id", "fiscal_year", "segment_name",
         "segment_revenue", "yoy_growth_pct"],
        segment_rows,
    )

    # --- earnings_events (4 quarters per company) ----------------------------
    event_rows = []
    event_id = 0
    for c in companies:
        # Synthesize a per-quarter EPS path from annual net income.
        # Use a sector-plausible synthetic share count.
        share_count_b = max(0.3, round(c["employees"] / 50000 + 1.0, 2))  # billions (synthetic)
        annual_eps = c["net_income"] / 1000.0 / share_count_b  # net_income is in $M
        for q in range(1, 5):
            event_id += 1
            seasonal = [0.22, 0.24, 0.25, 0.29][q - 1]
            eps_actual = _round(annual_eps * seasonal * 4)
            eps_estimate = _round(eps_actual / (1 + random.uniform(-0.06, 0.04)))
            surprise = _round((eps_actual - eps_estimate) / abs(eps_estimate) * 100) if eps_estimate else 0.0
            rev_actual = _round(c["revenue"] * seasonal, 0)
            event_rows.append({
                "event_id": event_id,
                "company_id": company_id[c["ticker"]],
                "fiscal_year": c["fiscal_year"],
                "fiscal_quarter": f"Q{q}",
                "report_date": f"{c['fiscal_year']}-{q * 3:02d}-15",
                "eps_actual": eps_actual,
                "eps_estimate": eps_estimate,
                "revenue_actual": rev_actual,
                "surprise_pct": surprise,
            })
    _write_csv(
        "earnings_events.csv",
        ["event_id", "company_id", "fiscal_year", "fiscal_quarter", "report_date",
         "eps_actual", "eps_estimate", "revenue_actual", "surprise_pct"],
        event_rows,
    )

    # --- risk_factors --------------------------------------------------------
    risk_rows = []
    risk_id = 0
    for c in companies:
        risks = CURATED_RISKS.get(c["ticker"]) or SECTOR_RISK_TEMPLATES.get(
            c["sector"], [{"risk_category": "General", "description": "General business and market risks."}]
        )
        for r in risks:
            risk_id += 1
            risk_rows.append({
                "risk_id": risk_id,
                "company_id": company_id[c["ticker"]],
                "fiscal_year": c["fiscal_year"],
                "risk_category": r["risk_category"],
                "description": r["description"],
            })
    _write_csv(
        "risk_factors.csv",
        ["risk_id", "company_id", "fiscal_year", "risk_category", "description"],
        risk_rows,
    )

    # --- executives ----------------------------------------------------------
    exec_rows = []
    exec_id = 0
    for c in companies:
        ceo = CEO_BY_TICKER.get(c["ticker"])
        if ceo:
            exec_id += 1
            exec_rows.append({
                "exec_id": exec_id,
                "company_id": company_id[c["ticker"]],
                "name": ceo,
                "title": "Chief Executive Officer",
                "since_year": "",
            })
        cfo = CFO_BY_TICKER.get(c["ticker"])
        if cfo:
            exec_id += 1
            exec_rows.append({
                "exec_id": exec_id,
                "company_id": company_id[c["ticker"]],
                "name": cfo,
                "title": "Chief Financial Officer",
                "since_year": "",
            })
    _write_csv(
        "executives.csv",
        ["exec_id", "company_id", "name", "title", "since_year"],
        exec_rows,
    )

    # --- earnings_reports (migrated from legacy JSON) ------------------------
    with open(LEGACY_JSON, "r", encoding="utf-8") as fh:
        reports = json.load(fh)

    sector_pseudo_ticker = {
        "SECTOR_TECH": "Technology",
        "SECTOR_BANKS": "Financial Services",
        "SECTOR_ENERGY": "Energy",
    }
    report_rows = []
    for i, r in enumerate(reports, start=1):
        ticker = r["ticker"]
        comp_id = company_id.get(ticker, "")
        sec_id = ""
        if ticker in sector_pseudo_ticker:
            sec_id = sector_id.get(sector_pseudo_ticker[ticker], "")
        report_rows.append({
            "report_id": i,
            "company_id": comp_id,
            "sector_id": sec_id,
            "fiscal_year": int(r["fiscal_year"]),
            "doc_type": r["doc_type"],
            "title": r["title"],
            "content": r["content"],
        })
    _write_csv(
        "earnings_reports.csv",
        ["report_id", "company_id", "sector_id", "fiscal_year", "doc_type",
         "title", "content"],
        report_rows,
    )

    # --- macro_indicators ----------------------------------------------------
    macro_rows = [
        {"indicator_id": 1, "name": "US Real GDP Growth", "period": "2025E",
         "value": 2.0, "unit": "%", "description": "Consensus US real GDP growth for 2025."},
        {"indicator_id": 2, "name": "US Headline CPI", "period": "2025E",
         "value": 2.5, "unit": "%", "description": "Headline CPI moderating toward target."},
        {"indicator_id": 3, "name": "Fed Funds Upper Bound", "period": "2025E",
         "value": 3.5, "unit": "%", "description": "Expected year-end 2025 Fed Funds upper bound."},
        {"indicator_id": 4, "name": "10Y Treasury Yield", "period": "2025E",
         "value": 4.4, "unit": "%", "description": "Expected rangebound 10-year Treasury yield."},
        {"indicator_id": 5, "name": "S&P 500 EPS Growth", "period": "2025E",
         "value": 12.0, "unit": "%", "description": "Consensus S&P 500 earnings growth for 2025."},
    ]
    _write_csv(
        "macro_indicators.csv",
        ["indicator_id", "name", "period", "value", "unit", "description"],
        macro_rows,
    )

    print("Relational seed CSVs generated under app/data/relational/.")


if __name__ == "__main__":
    generate()
