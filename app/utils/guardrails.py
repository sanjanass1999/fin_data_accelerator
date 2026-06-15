"""Input + output guardrails for the RAG chat surface.

The platform needs to demonstrate *responsible AI* practices to enterprise
reviewers, so we enforce four guardrails out-of-the-box:

1. **PII blocking** – queries containing SSNs, credit-card-like numbers
   or emails are rejected.
2. **Prompt-injection detection** – common jailbreak phrases short-circuit
   the pipeline and never reach the LLM.
3. **Topic allowlist** – the assistant only answers questions about the
   financial knowledge base; off-topic chatter gets a polite refusal.
4. **Output grounding check** – generated answers are scored against the
   retrieved context and a financial-advice disclaimer is appended when
   the user implicitly asks for a recommendation.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.config import get_settings


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


@dataclass
class GuardrailResult:
    allowed: bool
    reason: Optional[str] = None
    rule: Optional[str] = None
    severity: str = "info"          # info | warning | block
    message_for_user: Optional[str] = None


@dataclass
class OutputCheck:
    grounded_ratio: float           # fraction of answer tokens supported by ctx
    contains_disclaimer: bool
    appended_disclaimer: bool
    final_text: str
    warnings: List[str]


# --------------------------------------------------------------------------- #
# Input guardrail
# --------------------------------------------------------------------------- #


_FINANCIAL_KEYWORDS = {
    "revenue", "earnings", "profit", "margin", "loss", "income", "cash",
    "balance", "asset", "liability", "equity", "dividend", "share", "stock",
    "ticker", "quarter", "fiscal", "yoy", "qoq", "guidance", "outlook",
    "ebitda", "ebit", "cogs", "capex", "opex", "p/e", "valuation",
    "company", "companies", "firm", "firms", "industry", "sector",
    "report", "filing", "10-k", "10-q", "sec", "analyst", "risk",
    "growth", "decline", "increase", "decrease", "compare", "summarise",
    "summarize", "overview", "performance", "metric", "metrics",
    # relational entities exposed by the table router
    "employee", "employees", "headcount", "headquarter", "headquarters", "hq",
    "ceo", "cfo", "executive", "executives", "leadership", "founded",
    "segment", "segments", "eps", "estimate", "surprise", "roe",
    "macro", "gdp", "cpi", "inflation", "treasury", "yield",
    # company names (resolved by the router even without a ticker)
    "apple", "microsoft", "alphabet", "google", "nvidia", "amazon",
    "tesla", "walmart", "netflix", "disney", "intel", "broadcom",
    # common tickers from the seed dataset
    "aapl", "msft", "googl", "nvda", "amzn", "meta", "tsla", "jpm",
    "bac", "wfc", "gs", "xom", "cvx", "pfe", "jnj", "wmt", "pg",
}


# Keywords long enough to fuzzy-match reliably against a typo'd word.
_FUZZY_KEYWORDS = [kw for kw in _FINANCIAL_KEYWORDS if len(kw) >= 4]


def _looks_financial(query: str) -> bool:
    q = query.lower()
    if any(kw in q for kw in _FINANCIAL_KEYWORDS):
        return True
    # Capitalised tickers (≥ 2 letters) count as financial intent.
    if re.search(r"\b[A-Z]{2,5}\b", query):
        return True
    # Typo tolerance: accept the query if any word is a near-miss of a known
    # financial keyword (e.g. "revenuge" -> "revenue", "comapanies" ->
    # "companies"). This keeps simple misspellings from being refused.
    for tok in re.findall(r"[a-z]{4,}", q):
        if difflib.get_close_matches(tok, _FUZZY_KEYWORDS, n=1, cutoff=0.82):
            return True
    return False


def check_input(query: str) -> GuardrailResult:
    """Run all input guardrails and return the *first* failing rule."""
    settings = get_settings()
    text = query or ""
    if not text.strip():
        return GuardrailResult(False, "Empty query", "non_empty", "block",
                               "Please enter a question.")

    if len(text) > 2000:
        return GuardrailResult(False, "Query too long", "max_length", "block",
                               "Question exceeds 2000 character limit.")

    for pattern in settings.pii_patterns:
        if re.search(pattern, text):
            return GuardrailResult(
                False, "PII detected", "pii_block", "block",
                "Your question appears to include personal information "
                "(SSN, email, or card number). Please rephrase without it.",
            )

    lowered = text.lower()
    for phrase in settings.prompt_injection_phrases:
        if phrase in lowered:
            return GuardrailResult(
                False, f"Prompt-injection phrase: '{phrase}'",
                "prompt_injection", "block",
                "I can only answer questions about the financial knowledge "
                "base – I won't follow instructions that override my role.",
            )

    if not _looks_financial(text):
        return GuardrailResult(
            False, "Off-topic query", "topic_allowlist", "warning",
            "I'm a financial-data assistant. Try asking about company "
            "metrics, earnings, margins, or any indexed report.",
        )

    return GuardrailResult(True, None, None, "info")


# --------------------------------------------------------------------------- #
# Output guardrail
# --------------------------------------------------------------------------- #


_ADVICE_TRIGGERS = (
    "should i buy", "should i sell", "is it a good investment",
    "recommend", "predict", "will it go up", "will it go down",
)
_DISCLAIMER = (
    "\n\n_Note: This response is generated from indexed financial documents "
    "and is for informational purposes only. It is not investment advice._"
)


def _tokenise(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text)]


def check_output(query: str, answer: str, context_blocks: List[str]) -> OutputCheck:
    warnings: List[str] = []
    final = (answer or "").strip()

    if not final:
        return OutputCheck(0.0, False, False, "I don't have enough information to answer that.", ["empty_answer"])

    context_blob = " \n ".join(context_blocks).lower()
    answer_tokens = _tokenise(final)
    if answer_tokens:
        supported = sum(1 for t in answer_tokens if t in context_blob)
        grounded = supported / len(answer_tokens)
    else:
        grounded = 0.0

    if grounded < 0.25 and context_blocks:
        warnings.append("low_grounding")

    contains_disclaimer = "not investment advice" in final.lower()
    appended = False

    if any(trigger in query.lower() for trigger in _ADVICE_TRIGGERS) and not contains_disclaimer:
        final = final + _DISCLAIMER
        appended = True
        contains_disclaimer = True

    return OutputCheck(
        grounded_ratio=round(grounded, 3),
        contains_disclaimer=contains_disclaimer,
        appended_disclaimer=appended,
        final_text=final,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Convenience helpers exposed to the API layer
# --------------------------------------------------------------------------- #


def describe_guardrails() -> List[dict]:
    """Used by the dashboard to render a "trust" panel."""
    return [
        {"name": "PII filter", "stage": "input",
         "details": "Blocks SSN, credit-card-like numbers and email addresses."},
        {"name": "Prompt-injection detector", "stage": "input",
         "details": "Blocks jailbreak phrases such as 'ignore previous instructions'."},
        {"name": "Topic allowlist", "stage": "input",
         "details": "Allows only financial / company / KB questions."},
        {"name": "Grounding check", "stage": "output",
         "details": "Scores generated tokens against retrieved context."},
        {"name": "Advice disclaimer", "stage": "output",
         "details": "Auto-appends an informational disclaimer when the user asks for a recommendation."},
    ]
