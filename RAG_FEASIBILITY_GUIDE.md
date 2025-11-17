# RAG Feasibility Study Generator Blueprint

A production-ready blueprint (plus reference code) for a Retrieval-Augmented Generation (RAG) system that ingests up to **1 GB** of project materials and automatically drafts a comprehensive feasibility study grounded in your financial model outputs and accompanying documents.

This document covers:

1. Architecture & design choices
2. Data schema & parsing of financial model (Excel)
3. Prompt strategy & section templates
4. Reference implementation (FastAPI + FAISS + Sentence-Transformers, optional reranker)
5. Quality, auditing & reproducibility
6. Deployment notes (including handling 1 GB uploads)

---

## 0) RAC: Model-Integrated Design (RAG inside the Financial Model)

**What changed** – The Excel workbook is treated as the system of record and orchestrator. The Retrieval–Aggregation–Composer (RAC) service is triggered from the model (button/macros/Office Script/xlwings) and does two things:

1. Collects results directly from defined cells/ranges in the model (NPV/IRR/DSCR, capex/opex, scenarios, sensitivities, flags) and writes a structured `financial_snapshot`.
2. Ingests and retrieves external evidence from up to **1 GB** of uploaded sources (PDF/DOCX/PPTX/CSV/TXT) to ground narrative sections with citations.

**Why this pattern**

* Ensures a single source of truth for numbers (no drift between the model and the report).
* Reduces user steps: analysts stay in Excel; one *Generate Feasibility Study* action runs end-to-end.
* Enables repeatable runs: report always tied to a workbook hash + timestamp.

### RAC Components

1. **Workbook Adapter (in-workbook)**
   * Implementation options: Office Scripts (Excel on web), VBA macro, or xlwings (Python bridge).
   * Provides a mapping config of named ranges/cell addresses → semantic keys (npv, irr, dscr_min, etc.).
   * Exposes a *Generate Report* control that calls the RAC API with the extracted snapshot.
2. **Results Collector (API)**
   * Validates workbook hash; normalizes units/currency; runs sanity checks (e.g., DSCR > 0, IRR ∈ [−1, 1]).
   * Saves `projects/<id>/financial/snapshot.json` with lineage: `{workbook_hash, as_of, model_version, cell_map}`.
3. **Evidence Ingestor (API)**
   * Streams large uploads to disk; parses & chunks; embeds; stores in FAISS with rich metadata.
   * Optional OCR pass for scanned PDFs (add Tesseract layer if needed).
4. **Retriever + Reranker**
   * Section-specific queries (Market, Technical, Legal, etc.) pull top-*k* passages; cross-encoder reranks.
5. **Composer**
   * Section templates that inject the `financial_snapshot` + retrieved passages; forces inline `[Source: …]` and `[Sheet: …]` citations; flags gaps.
6. **Auditor**
   * Produces a provenance table (file → hash → cited pages/sheets) and reprints key model outputs in the appendix.

### End-to-End Flow (in workbook)

1. Analyst updates assumptions → workbook computes.
2. Click *Generate Feasibility Study*.
3. Workbook Adapter reads named ranges/rules → posts to `/collect` with `financial_snapshot` + `cell_map`.
4. Evidence Ingestor already holds uploads (or accepts them in the same action up to 1 GB).
5. Retriever + Composer generate each section → returns `report.md` & `report.json` to the workbook (or a download link).

### API Shape (revised)

* **POST `/collect`** – accepts `{project_id, financial_snapshot, cell_map, workbook_hash}`.
* **POST `/ingest`** – streamed uploads for external sources.
* **POST `/generate`** – accepts `{project_id, section_outline?}`; uses provided `financial_snapshot` (does not guess).

---

## 1) High-level Architecture

**Goals**

* Upload up to 1 GB of files (PDF, DOCX, XLSX/CSV, PPTX, text) without exhausting memory.
* Parse the financial model (Excel-based) to extract key outputs (NPV, IRR, DSCR, payback, capex/opex tables, sensitivities, scenarios).
* Build a local vector store (FAISS) for RAG retrieval with metadata filters (file, section, page).
* Assemble a full feasibility study (Executive Summary → Market → Technical → Implementation → Financial → Risk/ESG → Conclusion) with verbatim citations and appendices.

**Pipeline**

1. Upload & Ingest: Stream large files to disk → parse text (per type) → chunk (token-aware) → embed → store in FAISS; persist metadata (SQLite/JSONL).
2. Financial Model Extraction: Load Excel → extract standardized metrics/tables via a schema (configurable sheet/label mapping).
3. Retrieval: Hybrid (BM25 + dense) optional; the reference uses dense + MMR + cross-encoder rerank.
4. Planning: Build section-by-section plan informed by model metrics.
5. Generation: For each section, craft grounded prompts with citations from retrieved chunks + structured financial bullets.
6. Audit: Attach a sources table and financial snapshot (as-of timestamp, workbook hash) for reproducibility.

**Core tech (reference)**

* FastAPI for API server
* Sentence-Transformers for embeddings (e.g., all-MiniLM-L6-v2 or bge-base)
* FAISS vector store (local)
* Cross-Encoder for reranking (optional)
* pandas / openpyxl for Excel parsing
* pypdf, python-docx, pptx for document extraction
* tiktoken or a simple tokenizer for chunk sizing

You can swap in Pinecone/Qdrant for the vector DB, or Azure/OpenAI/Anthropic for the LLM. The design abstracts the LLM client so you can extend it.

---

## 2) Data Model & Financial Schema

**Project store**

* `projects/<project_id>/uploads/` – raw files (streamed)
* `projects/<project_id>/parsed/` – extracted text, JSON for tables
* `projects/<project_id>/index/` – FAISS index + `meta.jsonl`
* `projects/<project_id>/financial/` – `snapshot.json`

**Metadata per chunk**

```json
{
  "project_id": "string",
  "file_path": "string",
  "file_type": "pdf|docx|xlsx|csv|pptx|txt",
  "page_or_sheet": "number|string",
  "section": "optional heading",
  "char_start": 0,
  "char_end": 1024,
  "hash": "sha256 of source file"
}
```

**Financial snapshot (`projects/<id>/financial/snapshot.json`)**

```json
{
  "as_of": "2025-11-17T08:00:00Z",
  "workbook_path": ".../model.xlsx",
  "workbook_hash": "sha256",
  "currency": "USD",
  "assumptions": {
    "discount_rate": 0.12,
    "inflation": 0.03,
    "tax_rate": 0.28
  },
  "capex_total": 120000000,
  "opex_annual": 8500000,
  "revenue_annual": 23500000,
  "npv": 54000000,
  "irr": 0.19,
  "payback_years": 5.6,
  "dscr_min": 1.35,
  "sensitivities": [
    {"variable": "price", "delta": 0.1, "npv": 60000000, "irr": 0.205},
    {"variable": "price", "delta": -0.1, "npv": 48000000, "irr": 0.175}
  ],
  "scenarios": [
    {"name": "Base", "npv": 54000000, "irr": 0.19},
    {"name": "Downside", "npv": 30000000, "irr": 0.14},
    {"name": "Upside", "npv": 78000000, "irr": 0.24}
  ]
}
```

**Excel parsing strategy** – Config-driven mapping of sheet names and anchor labels (e.g., lookup tables for NPV/IRR/DSCR). Fallback heuristics scan for keywords in the first column, regex for %, IRR, NPV, DSCR. Preserve units & currency, record workbook hash and sheet cell addresses for traceability.

---

## 3) Prompt Strategy & Section Templates

**System prompt (global)**

> You are a financial analyst producing a feasibility study. Only use facts from provided CONTEXT and FINANCIAL_SNAPSHOT. Cite sources inline like [Source: filename p.12] or [Sheet: Assumptions!B7]. If a claim is unsupported, say so. Keep each section structured, concise, and decision-oriented.

**Section outline**

1. Executive Summary
2. Project Description & Scope
3. Market & Demand Analysis
4. Technical & Operations
5. Legal, Permitting & Environmental
6. Implementation Plan (Schedule, Procurement, Organization)
7. Financial Analysis (NPV/IRR/DSCR, sensitivities, scenarios)
8. Risk Assessment & Mitigations (including ESG)
9. Conclusion & Recommendation
10. Appendices (Detailed assumptions, tables, source list)

**Section prompt template (per section)**

```
[GOAL]
Draft the <SECTION_NAME> for the feasibility study.

[GUIDANCE]
- Use FINANCIAL_SNAPSHOT metrics explicitly where relevant.
- Use CONTEXT passages with inline citations [Source: <file> p.<n>] or [Sheet: <name>!<cell>].
- State uncertainties and missing data.
- Avoid boilerplate; keep it specific to the project.

[FINANCIAL_SNAPSHOT]
{{structured bullets}}

[CONTEXT]
{{top_k passages with metadata}}

[OUTPUT]
- 3–7 well-structured paragraphs
- Subheadings
- Bullet lists for key metrics & risks
- Citations inline
```

**Citation style**

* Text sources: `[Source: filename p.12]`
* Excel: `[Sheet: SheetName!CellRef]`

---

## 4) Reference Implementation (single-file API)

Minimal working example in one file (FastAPI + FAISS + Sentence-Transformers). Swap models/LLM as needed and adapt to your deployment stack.

```python
import os
import io
import json
import hashlib
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import pandas as pd

from docx import Document as DocxDocument
from pptx import Presentation
import pypdf

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_TOKENS = 500
CHUNK_OVERLAP = 100
TOP_K = 12
RERANK_K = 5

DATA_DIR = os.getenv("DATA_DIR", "./projects")

class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai")

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        return "[LLM OUTPUT PLACEHOLDER]\n" + prompt[:500]

# Helper functions omitted for brevity (streamed saves, chunking, parsing)

app = FastAPI(title="RAG Feasibility Study Generator")
llm = LLMClient()

@app.post("/ingest")
async def ingest(project_id: str = Form(...), files: List[UploadFile] = File(...)):
    # Stream to disk, parse, chunk, embed, persist FAISS index, extract Excel snapshot
    ...

@app.post("/generate")
def generate(req: GenerateRequest):
    # Load snapshot, retrieve context, craft prompts, generate sections, write report.json/md
    ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 5) Quality, Auditing & Reproducibility

* **Grounding & citations:** Retrieval → rerank → limit to top 3–5 rich chunks per section. Enforce inline citations in prompts; flag unknowns instead of hallucinating.
* **Determinism:** Fix embed/rerank model versions; record workbook hash and timestamp.
* **Validation checks:** Numeric sanity (IRR ∈ [−100%, 100%], DSCR > 0, currency consistency). Red-flag financial sections if DSCR < 1.
* **Human-in-the-loop:** Return `report.md` for redlining; include `report.json` for BI export and auditing.

---

## 6) Handling 1 GB Uploads

* Streamed writes (1 MB chunks) avoid memory spikes; tune reverse proxy (e.g., `client_max_body_size 1024m`).
* Recommended Nginx front with `proxy_request_buffering off` and fast disk (NVMe) for `projects/` storage.
* Run multiple Uvicorn workers for concurrency.

---

## 7) Extending the Blueprint

* Hybrid retrieval (BM25 + dense), structured table extraction, section-specific retrievers.
* Guardrails: regex-check generated claims vs. snapshot values.
* Custom LLM provider/model; convert `report.md` to PDF via WeasyPrint or pandoc.

---

## 8) Section-specific Retrieval Queries (ready-made)

* Executive Summary: “materiality of results, decision drivers, showstoppers”
* Market: “market size, demand forecast, price assumptions, offtake”
* Technical: “process design, throughput, yield, utilities, site layout”
* Legal/Env: “permits, EIA/ESIA, land rights, community”
* Implementation: “schedule, capex phasing, procurement strategy, org”
* Financial: “NPV, IRR, DSCR, payback, sensitivities”
* Risk/ESG: “risk register, mitigation, ESG metrics”

---

## 9) Report Composition & Appendices

**Financial Statements and Schedules**

* Income Statement (projected)
* Balance Sheet
* Cash Flow Statement (including financing flows)
* Figures pulled directly from the Excel model via the defined `cell_map` and refreshed each run.

**Schedules and Supporting Tables**

* Capex, Opex, Revenue, Debt, Depreciation, Sensitivity, Scenario Analysis.
* Graphs (NPV curve, DSCR timeline, cash-flow waterfall, cost breakdown) embedded in `report.md`.

**Appendices**

* Full reproduction of key tables and charts.
* Sensitivity matrices (e.g., Price vs. IRR, Cost vs. NPV).
* Scenario summaries (Base, Downside, Upside) with comparative metrics.
* Audit trail: workbook hash, timestamp, and cell reference map.

---

## 10) Security & Privacy

* All data stays local by default. No outbound calls unless you wire an external LLM.
* Hash all source files; keep a manifest for chain-of-custody.
* Optional: encrypt `projects/` at rest; restrict file permissions.

---

## 11) What to Customize

* Excel sheet & label mappings to your specific financial model outputs.
* Section outline and tone.
* LLM provider and model.
* Any industry-specific compliance/ESG frameworks.

---

## 12) Recommended Improvements

* **Retrieval robustness** – Add hybrid retrieval (BM25 + dense) with section-specific queries, apply language detection to route embeddings to the right model, and deduplicate near-identical chunks before indexing to reduce noise.
* **Chunk quality** – Introduce layout-aware parsing for PDFs/PowerPoint (tables, headers, bullet hierarchy) and structured Excel table extraction so section prompts receive semantically coherent passages rather than raw text dumps.
* **Grounding safeguards** – Enforce a rerank-and-verify step: cross-encoder top-*k* followed by faithfulness checks (e.g., string matching for cited figures) to filter hallucinated numbers, with automatic “insufficient evidence” fallbacks in generation.
* **Performance & scale** – Move embedding and rerank jobs to a background worker queue, cache embeddings by file hash, and add batch ingest endpoints to keep 1 GB uploads responsive under concurrent usage.
* **Observability** – Emit structured logs/metrics for retrieval hits, citation density, generation latency, and section-level token usage; wire them into a lightweight dashboard for run-to-run drift detection.
* **Security & governance** – Add MIME allowlists, antivirus/PII scans on upload, and optional encryption-at-rest for the project store; record an audit manifest (file hash → cited sections) alongside `report.json` for downstream compliance.
* **Evaluation harness** – Build regression tests using golden reports plus automated RAG metrics (answer similarity, groundedness) so model, prompt, or embedding upgrades can be shipped with confidence.

You now have a working foundation to ingest large files, extract key model results, and generate a grounded, audit-friendly feasibility study. Plug in your model, load your documents, and run `/generate`. ✅
