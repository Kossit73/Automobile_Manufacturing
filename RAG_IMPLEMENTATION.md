# RAG Feasibility Study Generator

This repository includes `rag_app.py`, a standalone FastAPI service that implements the Retrieval‑Augmented Generation (RAG) blueprint for generating feasibility studies from your financial model outputs and supporting documents.

## Architecture & Workflow
- **Endpoints**
  - `POST /collect` – store a structured `financial_snapshot` (plus optional `cell_map`, workbook hash, and version) under `projects/<id>/financial/`.
  - `POST /ingest` – stream up to 1 GB of uploads (PDF/DOCX/PPTX/TXT/CSV/XLSX), parse text, chunk, embed with Sentence‑Transformers, and persist a FAISS index plus metadata under `projects/<id>/index/`.
  - `POST /generate` – retrieve top passages (optionally reranked), combine them with the financial snapshot, and draft each section using an authenticated OpenAI chat completion.
- **Project layout**
  - `projects/<id>/uploads/` raw files, `parsed/` cached text, `index/` FAISS and metadata, `financial/` snapshots and cell maps, `charts/` generated PNGs, plus `report.json` and `report.md` outputs.
- **Charts**: Optional NPV curve, DSCR trend (from Excel), and cash-flow waterfall are rendered to `projects/<id>/charts/` and referenced in the markdown report.

## Quickstart
```bash
pip install fastapi uvicorn[standard] sentence-transformers faiss-cpu pypdf python-docx python-pptx pandas openpyxl matplotlib openai
export OPENAI_API_KEY="sk-..."  # required for external LLM calls
python rag_app.py  # serves on 0.0.0.0:8000
```

Example usage:
```bash
# 1) Send the model snapshot
curl -X POST http://localhost:8000/collect \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"demo","financial_snapshot":{"npv":1_000_000,"irr":0.18}}'

# 2) Ingest documents
curl -F "project_id=demo" -F "files=@/path/to/report.pdf" http://localhost:8000/ingest

# 3) Generate the feasibility study
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"project_id":"demo","query_hint":"industrial project feasibility"}'
```

## Prompting & Grounding
- Global system prompt enforces grounded, cited output (`[Source: file p.n]` or `[Sheet: Name!Cell]`).
- Section prompts include the structured `financial_snapshot` plus the top retrieved passages; missing data is called out explicitly.
- Reranking uses an optional CrossEncoder (`RERANK_MODEL`) when available; otherwise retrieval falls back to the top dense results.

## Deployment Notes
- Streaming uploads are written in 1 MB chunks to avoid memory pressure; front with Nginx for 1 GB bodies (`client_max_body_size 1024m`).
- The service relies on environment variables for model names and temperature; defaults are compatible with `gpt-4o`.
- Outputs are cached per project so reruns reuse existing indexes and snapshots.

## Extensibility
- Swap FAISS for a managed vector DB if desired.
- Replace the OpenAI client in `LLMClient` with another provider by adapting the auth call in `complete`.
- Enrich `financial_snapshot` parsing by extending `parse_excel_metrics` or posting a richer `cell_map` via `/collect`.
