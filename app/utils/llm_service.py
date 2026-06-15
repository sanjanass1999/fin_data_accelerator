"""Multi-provider LLM router with deterministic fallbacks.

Provider routing order:

    primary  -> fallback -> simulation

`simulation` is a self-contained extractive reasoner that builds answers
*directly from the retrieved context*. It has no dependencies and exists
solely so the demo never breaks during a live presentation when the
laptop is offline or rate-limited.

The system prompt forces the LLM to (1) ground every fact in the numbered
context, (2) cite passages by `[n]`, and (3) refuse when the answer is
not supported.
"""
from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.logging_config import get_logger

log = get_logger("llm_service")


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a senior financial-data analyst inside the FinDataAccelerator
    enterprise platform.

    Strict rules:
    1. Only use the numbered passages in the CONTEXT block. Do not invent
       numbers, tickers, dates, or company names.
    2. Cite every factual claim with a bracketed citation that matches the
       passage number, e.g. "Apple's FY24 revenue was $391B [2]".
    3. If the context does not contain the answer, say:
       "I don't have enough information in the indexed knowledge base to
        answer that confidently."
    4. Never give personal investment advice. If asked, redirect to the
       factual data and add a clearly labelled disclaimer.
    5. Prefer concise, well-structured answers (bullets or short paragraphs).
    """
).strip()


def build_user_prompt(query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER (cite passages by [n]):"
    )


# --------------------------------------------------------------------------- #
# NL2SQL prompt (table routing -> SQL generation)
# --------------------------------------------------------------------------- #


SQL_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a precise SQLite text-to-SQL generator for a financial database.

    Strict rules:
    1. Output ONLY a single SQLite SELECT statement. No prose, no explanation,
       no markdown fences, no trailing semicolon.
    2. Use ONLY the tables and columns provided in the SCHEMA block. Never
       invent tables or columns.
    3. Resolve a ticker or company name by JOINing through the `companies`
       table using the foreign keys shown in the SCHEMA.
    4. For margins / ratios use `financial_ratios` joined via
       `financial_statements`. For raw dollar amounts use
       `financial_statements`.
    5. Always SELECT a human-readable identifier (e.g. companies.ticker or
       companies.company_name) alongside the requested value.
    6. Add ORDER BY and LIMIT when the question implies "highest", "top",
       "largest", "most", etc.
    7. If the question cannot be answered from the schema, output exactly:
       SELECT 'unanswerable' AS note
    """
).strip()


def build_sql_prompt(query: str, schema_snippet: str) -> str:
    return (
        f"SCHEMA:\n{schema_snippet}\n\n"
        f"QUESTION: {query}\n\n"
        f"SQLITE SELECT:"
    )


# --------------------------------------------------------------------------- #
# Provider implementations
# --------------------------------------------------------------------------- #


def _call_groq(query: str, context: str) -> Optional[str]:
    settings = get_settings()
    if not settings.groq_api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.groq_api_key,
        )
        resp = client.chat.completions.create(
            model=settings.groq_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, context)},
            ],
        )
        return resp.choices[0].message.content
    except Exception as exc:
        log.warning("groq call failed", extra={"error": str(exc)})
        return None


def _call_gemini(query: str, context: str) -> Optional[str]:
    settings = get_settings()
    if not settings.gemini_api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            system_instruction=SYSTEM_PROMPT,
        )
        resp = model.generate_content(build_user_prompt(query, context))
        return getattr(resp, "text", None)
    except Exception as exc:
        log.warning("gemini call failed", extra={"error": str(exc)})
        return None


def _call_ollama(query: str, context: str) -> Optional[str]:
    settings = get_settings()
    try:
        import ollama  # type: ignore

        client = ollama.Client(host=settings.ollama_base_url)
        resp = client.chat(
            model=settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, context)},
            ],
            options={"temperature": 0.1},
        )
        return resp.get("message", {}).get("content")
    except Exception as exc:
        log.warning("ollama call failed", extra={"error": str(exc)})
        return None


# --------------------------------------------------------------------------- #
# Simulation provider (offline-safe)
# --------------------------------------------------------------------------- #


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def _stopwords() -> set:
    return {
        "the", "a", "an", "and", "or", "of", "in", "on", "for", "to",
        "is", "are", "was", "were", "be", "been", "being", "as", "at",
        "by", "with", "that", "this", "these", "those", "it", "its",
        "what", "which", "who", "whom", "how", "why", "when", "where",
        "do", "does", "did", "tell", "me", "about", "have", "has", "had",
    }


def _score_sentence(sentence: str, q_terms: set) -> float:
    words = {w.lower() for w in _WORD.findall(sentence)}
    if not words:
        return 0.0
    overlap = len(words & q_terms)
    if overlap == 0:
        return 0.0
    digits = 1.0 if re.search(r"\d", sentence) else 0.0
    return overlap + 0.4 * digits


# Question intents that call for prose pulled from the narrative passages
# rather than a single database figure.
_QUALITATIVE = (
    "risk", "risks", "outlook", "strategy", "guidance", "macro", "why",
    "explain", "describe", "summary", "summarise", "summarize", "overview",
    "driver", "drivers", "headwind", "tailwind", "factor", "factors",
)
# Intents whose answer reads best as a short list.
_LIST_INTENT = ("risk", "factor", "driver", "headwind", "tailwind", "segment", "list")


def _split_blocks(context: str) -> List[Dict[str, str]]:
    """Split a numbered context string into ``{num, header, body}`` blocks."""
    blocks: List[Dict[str, str]] = []
    for block in re.split(r"\n\n+", context.strip()):
        m = re.match(r"^\[(\d+)\]([^\n]*)", block)
        num = m.group(1) if m else ""
        header = m.group(2) if m else ""
        body = re.sub(r"^\[\d+\][^\n]*\n?", "", block, count=1)
        blocks.append({"num": num, "header": header, "body": body})
    return blocks


def _relational_cite(blocks: List[Dict[str, str]]) -> str:
    for b in blocks:
        if "source=relational_db" in b["header"] and b["num"]:
            return f"[{b['num']}]"
    return ""


def _collect_sentences(blocks: List[Dict[str, str]], q_terms: set) -> List[tuple]:
    """Score narrative sentences by query overlap, skipping the DB block."""
    scored: List[tuple] = []
    seen: set = set()
    for b in blocks:
        if "source=relational_db" in b["header"]:
            continue
        cite = f"[{b['num']}]" if b["num"] else ""
        for sent in _SENTENCE_SPLIT.split(b["body"]):
            sent = sent.strip()
            if len(sent) < 20 or "[SQL]" in sent:
                continue
            # Skip title/heading fragments (e.g. "FY2024 Key Risk Factors")
            # which carry no real prose -- they lack lowercase content words.
            if len(re.findall(r"\b[a-z]{3,}\b", sent)) < 2:
                continue
            key = sent.lower()
            if key in seen:
                continue
            score = _score_sentence(sent, q_terms)
            if score > 0:
                seen.add(key)
                scored.append((score, sent, cite))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored


def _simulate(query: str, context: str, primary_answer: Optional[str] = None) -> str:
    """Offline reasoner that returns a clean, direct answer.

    Strategy:
      * Quantitative questions with a database answer -> return that answer
        verbatim (already natural language), with a single provenance citation.
      * Qualitative questions (risks, outlook, strategy ...) -> stitch the most
        relevant narrative sentences into a short, readable response.

    No SQL/table telemetry ever leaks into the prose; that lives in the
    separate provenance panel.
    """
    ql = query.lower()
    q_terms = {w.lower() for w in _WORD.findall(query)} - _stopwords()
    if not q_terms:
        q_terms = {w.lower() for w in _WORD.findall(query)}

    blocks = _split_blocks(context)
    qualitative = any(k in ql for k in _QUALITATIVE)

    # Quantitative + a deterministic DB answer -> answer directly.
    if primary_answer and not qualitative:
        cite = _relational_cite(blocks)
        return f"{primary_answer} {cite}".strip()

    top = _collect_sentences(blocks, q_terms)[:4]

    if not top:
        if primary_answer:
            cite = _relational_cite(blocks)
            return f"{primary_answer} {cite}".strip()
        return ("I don't have enough information in the indexed knowledge "
                "base to answer that confidently.")

    if any(k in ql for k in _LIST_INTENT):
        lead = None
        if "risk" in ql:
            lead = "The key risk factors are:"
        elif any(k in ql for k in ("outlook", "guidance", "macro")):
            lead = "Here is the outlook:"
        body = "\n".join(f"- {sent} {cite}".strip() for _, sent, cite in top)
        answer = f"{lead}\n{body}" if lead else body
    else:
        answer = " ".join(f"{sent} {cite}".strip() for _, sent, cite in top)

    # For qualitative questions the narrative is the answer; only lead with the
    # database figure when it is a single concise fact (not a verbose row dump).
    if primary_answer and len(primary_answer) <= 160 and ";" not in primary_answer:
        cite = _relational_cite(blocks)
        answer = f"{primary_answer} {cite}\n\n{answer}".strip()
    return answer


# --------------------------------------------------------------------------- #
# Public router
# --------------------------------------------------------------------------- #


_PROVIDERS = {
    "groq": _call_groq,
    "gemini": _call_gemini,
    "ollama": _call_ollama,
    "simulation": lambda q, c: _simulate(q, c),
}


def generate_rag_response(
    user_query: str,
    retrieved_context: str,
    provider_override: Optional[str] = None,
    primary_answer: Optional[str] = None,
) -> Dict[str, Any]:
    """Return ``{"answer", "provider_used", "providers_tried"}``.

    ``primary_answer`` is an optional pre-computed natural-language answer from
    the relational database. It is used by the offline simulation provider so
    quantitative questions resolve to a clean, direct sentence.
    """
    settings = get_settings()
    order: List[str] = []
    primary = (provider_override or settings.primary_provider or "simulation").lower()
    fallback = (settings.fallback_provider or "simulation").lower()

    for p in (primary, fallback, "simulation"):
        if p and p in _PROVIDERS and p not in order:
            order.append(p)

    tried: List[str] = []
    for p in order:
        tried.append(p)
        try:
            if p == "simulation":
                answer = _simulate(user_query, retrieved_context, primary_answer=primary_answer)
            else:
                answer = _PROVIDERS[p](user_query, retrieved_context)
        except Exception as exc:
            log.warning("provider raised", extra={"provider": p, "error": str(exc)})
            answer = None
        if answer:
            return {
                "answer": answer.strip(),
                "provider_used": p,
                "providers_tried": tried,
            }

    return {
        "answer": "Unable to generate a response.",
        "provider_used": "none",
        "providers_tried": tried,
    }


# --------------------------------------------------------------------------- #
# NL2SQL generation
# --------------------------------------------------------------------------- #


_SQL_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _provider_complete(provider: str, system_prompt: str, user_prompt: str) -> Optional[str]:
    """Generic single-shot completion used by the NL2SQL path."""
    settings = get_settings()
    try:
        if provider == "groq":
            if not settings.groq_api_key:
                return None
            from openai import OpenAI

            client = OpenAI(base_url="https://api.groq.com/openai/v1",
                            api_key=settings.groq_api_key)
            resp = client.chat.completions.create(
                model=settings.groq_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content
        if provider == "gemini":
            if not settings.gemini_api_key:
                return None
            import google.generativeai as genai

            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel(
                model_name=settings.gemini_model, system_instruction=system_prompt
            )
            resp = model.generate_content(user_prompt)
            return getattr(resp, "text", None)
        if provider == "ollama":
            import ollama  # type: ignore

            client = ollama.Client(host=settings.ollama_base_url)
            resp = client.chat(
                model=settings.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.0},
            )
            return resp.get("message", {}).get("content")
    except Exception as exc:
        log.warning("nl2sql provider failed", extra={"provider": provider, "error": str(exc)})
        return None
    return None


def _extract_sql(raw: str) -> Optional[str]:
    if not raw:
        return None
    fenced = _SQL_FENCE.search(raw)
    candidate = fenced.group(1) if fenced else raw
    candidate = candidate.strip()
    # Keep from the first SELECT/WITH onwards.
    m = re.search(r"\b(select|with)\b", candidate, re.IGNORECASE)
    if not m:
        return None
    candidate = candidate[m.start():].strip()
    if candidate.endswith(";"):
        candidate = candidate[:-1].strip()
    return candidate or None


def generate_sql(
    query: str,
    schema_snippet: str,
    provider_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a SQLite SELECT for ``query`` grounded in ``schema_snippet``.

    Returns ``{"sql", "provider_used", "providers_tried"}``. ``sql`` is ``None``
    when no real LLM provider is available (offline / no API keys); in that case
    the table router falls back to its deterministic template builder.
    """
    settings = get_settings()
    primary = (provider_override or settings.primary_provider or "").lower()
    fallback = (settings.fallback_provider or "").lower()

    order: List[str] = []
    for p in (primary, fallback):
        if p in ("groq", "gemini", "ollama") and p not in order:
            order.append(p)

    user_prompt = build_sql_prompt(query, schema_snippet)
    tried: List[str] = []
    for p in order:
        tried.append(p)
        raw = _provider_complete(p, SQL_SYSTEM_PROMPT, user_prompt)
        sql = _extract_sql(raw or "")
        if sql:
            return {"sql": sql, "provider_used": p, "providers_tried": tried}

    return {"sql": None, "provider_used": "none", "providers_tried": tried}
