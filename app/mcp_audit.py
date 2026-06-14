"""Tiny in-memory audit log for MCP tool calls.

The FastAPI dashboard reads this so reviewers can *see* the security
behaviour (allow / deny / latency) of every MCP invocation in real-time.
The log is intentionally bounded so it never grows without bound.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List

_LOCK = threading.Lock()
_BUFFER: Deque[Dict[str, Any]] = deque(maxlen=200)


def record(tool: str, status: str, args: Dict[str, Any], result_preview: str, latency_ms: int) -> None:
    with _LOCK:
        _BUFFER.append({
            "ts_ms": int(time.time() * 1000),
            "tool": tool,
            "status": status,            # allow | deny | error
            "args": args,
            "result_preview": (result_preview or "")[:240],
            "latency_ms": latency_ms,
        })


def snapshot() -> List[Dict[str, Any]]:
    with _LOCK:
        return list(_BUFFER)


def clear() -> None:
    with _LOCK:
        _BUFFER.clear()
