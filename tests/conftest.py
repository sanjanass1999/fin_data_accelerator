"""Test fixtures: isolate ChromaDB into a temp dir so tests never touch
the demo collection."""
from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("CHROMA_PATH", tempfile.mkdtemp(prefix="findata_test_chroma_"))
os.environ.setdefault("CHROMA_COLLECTION", "findata_test")


@pytest.fixture(scope="session", autouse=True)
def _seed_once():
    from app.utils.vector_store import reset_collection, add_document_chunks
    reset_collection()
    add_document_chunks(
        [
            "NVIDIA (NVDA) FY2024 net income was $55,000M on revenue of $118,000M, "
            "a net profit margin of 46.6%, the highest in the technology sector.",
            "JPMorgan (JPM) FY2024 net income was $58,471M with a return on tangible "
            "common equity of 22 percent.",
            "Wells Fargo (WFC) continues to operate under a Federal Reserve asset cap.",
        ],
        document_id="test_seed",
        metadatas=[
            {"source": "test", "ticker": "NVDA"},
            {"source": "test", "ticker": "JPM"},
            {"source": "test", "ticker": "WFC"},
        ],
    )
    yield
