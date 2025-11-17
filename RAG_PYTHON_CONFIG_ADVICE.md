# Using Python structures instead of JSON for RAG configuration

The current RAG feasibility flow reads configuration and financial snapshots from JSON payloads. If you want to shift toward Python-native configuration while keeping the pipeline stable, consider the following approach:

1. **Define a typed settings module.** Create a small `rag_settings.py` that exposes dataclasses for project metadata, LLM options, and financial snapshots. Include `from_json`/`to_json` helpers so existing endpoints can still consume and emit JSON while the internal code uses typed Python objects.
2. **Centralize schema validation.** Use `pydantic` or dataclasses with custom validators to enforce required fields (e.g., `project_id`, `workbook_hash`, `financial_snapshot` keys) before any processing. This gives you Python-side guarantees instead of ad-hoc JSON checks.
3. **Adjust the ingestion pathway gradually.** Let API handlers accept JSON as they do today but immediately hydrate the Python models. Downstream functions (chunking, embedding, LLM prompts) operate on the typed objects, reducing key/typo risk and making refactors safer.
4. **Maintain backward compatibility.** Keep serialization boundaries at the API layer so external clients don’t need to change. Internally you gain type safety and autocompletion while preserving the current JSON contract for uploads/downloads.
5. **Plan incremental migration.** Start by wrapping only the financial snapshot in a Python model, then extend to retrieval and generation settings. Document the mapping between JSON fields and Python attributes so future changes remain transparent.

This path keeps the interface JSON-friendly for clients but lets the app rely on structured Python objects for validation, defaults, and IDE support.
