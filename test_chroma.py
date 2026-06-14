"""Manual smoke test for the retrieval layer (run directly, not via pytest).

    python test_chroma.py
"""
from app.utils.vector_store import (
    add_document_chunks,
    format_context,
    search_financial_docs,
)

print("Step 1: indexing sample financial sentences...")
add_document_chunks(
    [
        "Wells Fargo net income for Q1 reached 4.9 billion dollars driven by strong interest rates.",
        "The corporate risk team updated data privacy guidelines to match strict regulations.",
        "Apple announced a revenue increase due to high demand for AI-enabled devices.",
    ],
    document_id="smoke_report_2026",
    metadatas=[{"source": "smoke"} for _ in range(3)],
)

print("\nStep 2: semantic search for 'banking profits and earnings'...")
hits = search_financial_docs("Tell me about banking profits and earnings", num_results=3)
for h in hits:
    print(f"  score={h['score']:.3f}  source={h['source']}  -> {h['text'][:80]}...")

print("\nStep 3: formatted context block:\n")
print(format_context(hits))
