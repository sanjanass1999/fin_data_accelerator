"""Shared TypedDict state object passed between LangGraph nodes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentTrace(TypedDict, total=False):
    agent: str
    started_ms: int
    finished_ms: int
    duration_ms: int
    status: str           # ok | warning | failed | skipped
    summary: str
    metrics: Dict[str, Any]


class PipelineState(TypedDict, total=False):
    file_path: str
    file_type: str

    raw_data: Optional[Any]               # list[dict] for tabular, list[str] for text
    cleaned_data: Optional[Any]
    transformed_data: Optional[Any]
    narratives: Optional[List[str]]       # textual chunks ready for ChromaDB
    narrative_metadata: Optional[List[Dict[str, Any]]]

    errors: List[str]
    warnings: List[str]
    next_node: str

    is_indexed: bool
    indexed_chunks: int

    agent_traces: List[AgentTrace]        # ordered execution log for the dashboard
