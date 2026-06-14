"""LangGraph wiring for the 4-agent FinDataAccelerator pipeline."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.ingestion import ingestion_agent
from app.agents.quality import quality_agent
from app.agents.rag import rag_agent
from app.agents.state import PipelineState
from app.agents.transform import transform_agent

workflow = StateGraph(PipelineState)
workflow.add_node("ingest_node", ingestion_agent)
workflow.add_node("quality_node", quality_agent)
workflow.add_node("transform_node", transform_agent)
workflow.add_node("rag_node", rag_agent)


def _route(state: PipelineState) -> str:
    target = state.get("next_node", "end")
    return {
        "quality": "quality_node",
        "transform": "transform_node",
        "rag": "rag_node",
    }.get(target, END)


workflow.set_entry_point("ingest_node")
workflow.add_conditional_edges("ingest_node", _route, {
    "quality_node": "quality_node",
    "transform_node": "transform_node",
    END: END,
})
workflow.add_conditional_edges("quality_node", _route, {
    "transform_node": "transform_node",
    END: END,
})
workflow.add_conditional_edges("transform_node", _route, {
    "rag_node": "rag_node",
    END: END,
})
workflow.add_edge("rag_node", END)

app_graph = workflow.compile()
