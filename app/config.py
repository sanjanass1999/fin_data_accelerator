"""Centralised, env-driven configuration for FinDataAccelerator.

Using a single :class:`Settings` object means every module pulls its
configuration from the same place. The defaults are tuned so the platform
runs end-to-end *without any API keys* – it falls back to a deterministic
simulation provider, which keeps live demos resilient.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- LLM provider keys ---
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))

    # --- Provider routing ---
    primary_provider: str = field(default_factory=lambda: _env("LLM_PRIMARY_PROVIDER", "groq"))
    fallback_provider: str = field(default_factory=lambda: _env("LLM_FALLBACK_PROVIDER", "simulation"))
    embedding_provider: str = field(default_factory=lambda: _env("LLM_EMBEDDING_PROVIDER", "local"))

    # --- Model names ---
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "phi3"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-1.5-flash"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    # --- Vector store ---
    chroma_path: str = field(default_factory=lambda: _env("CHROMA_PATH", "./chroma_db"))
    chroma_collection: str = field(
        default_factory=lambda: _env("CHROMA_COLLECTION", "findata_financial_docs")
    )

    # --- Retrieval tuning ---
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 6))
    retrieval_mmr_lambda: float = field(default_factory=lambda: _env_float("RETRIEVAL_MMR_LAMBDA", 0.6))
    retrieval_min_score: float = field(default_factory=lambda: _env_float("RETRIEVAL_MIN_SCORE", 0.20))

    # --- MCP / data sources ---
    mcp_allowed_fs_root: str = field(default_factory=lambda: _env("MCP_ALLOWED_FS_ROOT", "./data"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "enterprise-financial-docs-prod"))
    postgres_url: str = field(default_factory=lambda: _env("POSTGRES_URL", ""))

    # --- Guardrails ---
    blocked_topics: tuple = (
        "personal investment advice",
        "buy or sell recommendation",
    )
    pii_patterns: tuple = (
        r"\b\d{3}-\d{2}-\d{4}\b",                      # US SSN
        r"\b(?:\d[ -]*?){13,16}\b",                    # credit card-like
        r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b",              # email
    )
    prompt_injection_phrases: tuple = (
        "ignore previous instructions",
        "ignore the above",
        "forget the prior",
        "system prompt",
        "you are now",
        "disregard previous",
        "reveal your instructions",
    )

    # --- Branding ---
    platform_name: str = "FinDataAccelerator"
    platform_version: str = "1.1.0"


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return a process-wide singleton instance of :class:`Settings`."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def list_supported_providers() -> List[str]:
    return ["groq", "gemini", "ollama", "simulation"]
