"""Lightweight structured logging used across all agents.

We deliberately avoid third-party dependencies here so the platform stays
small. Each log record is emitted as a single JSON line which plays nicely
with Loki / Datadog / Splunk style ingestion.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in {"args", "msg", "levelname", "name", "exc_info", "exc_text",
                       "stack_info", "lineno", "funcName", "created", "msecs",
                       "relativeCreated", "thread", "threadName", "processName",
                       "process", "pathname", "filename", "module", "levelno"}:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("chromadb", "httpx", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
