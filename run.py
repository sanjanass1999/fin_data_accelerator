"""Local launcher. Seeds the knowledge base (if empty) then serves the API."""
from __future__ import annotations

import uvicorn

from app.logging_config import get_logger
from app.utils.vector_store import collection_stats

log = get_logger("run")


def _ensure_seeded() -> None:
    try:
        stats = collection_stats()
        if stats["chunks"] == 0:
            log.info("knowledge base empty, seeding now")
            from scripts.seed_data import seed
            seed(reset=False)
        else:
            log.info("knowledge base ready", extra={"chunks": stats["chunks"]})
    except Exception as exc:  # pragma: no cover
        log.warning("seed check failed; continuing", extra={"error": str(exc)})


if __name__ == "__main__":
    _ensure_seeded()
    print("Launching FinDataAccelerator at http://127.0.0.1:8000/dashboard")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
