# Architecture diagrams (C4 model)

[draw.io](https://app.diagrams.net) / diagrams.net source files describing the
FinDataAccelerator architecture using the [C4 model](https://c4model.com)
(Context -> Container -> Component).

| File | Level | Shows |
|---|---|---|
| [`c1_system_context.drawio`](c1_system_context.drawio) | C1 - System Context | The platform as one box, its users, and the external systems it talks to (LLM providers, data sources, MCP clients) |
| [`c2_container.drawio`](c2_container.drawio) | C2 - Container | The deployable/runtime pieces: React dashboard, FastAPI gateway, LangGraph pipeline, query engine, retrieval+safety, LLM router, MCP server, ChromaDB and the SQLite DB |
| [`c3_component_query_engine.drawio`](c3_component_query_engine.drawio) | C3 - Component | A zoom into the `POST /api/v1/chat` request path: guardrails -> table router -> query planner -> SQL access -> vector retrieval -> LLM router -> output guardrails -> evaluation |
| [`c3_component_ingestion_pipeline.drawio`](c3_component_ingestion_pipeline.drawio) | C3 - Component | A zoom into the LangGraph ingestion pipeline: Ingestion -> Quality -> Transform (+ chunking) -> RAG agents, the shared pipeline state, and the write into ChromaDB |

## Opening / editing

- Open https://app.diagrams.net and choose **Open Existing Diagram**, or
- Install the **Draw.io Integration** VS Code/Cursor extension and open the
  `.drawio` files directly in the editor, or
- Use the desktop draw.io app.

## Exporting

From draw.io: **File -> Export as -> PNG / SVG / PDF** to embed the diagrams in
docs or slides.

## Keeping them current

These are hand-maintained. When the architecture changes (new container,
component, or external dependency), update the matching `.drawio` file so the
C1/C2/C3 views stay consistent with the code.
