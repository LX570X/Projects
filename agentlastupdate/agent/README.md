# Incident Analysis Agent

An API-capable, folder-driven multimodal extraction and reporting system.

This project processes files from disk, classifies each file, extracts structured data, generates one validated report per file, then synthesizes a final cross-file report.

---

## Table of Contents

- [What this project does](#what-this-project-does)
- [How the system works (end-to-end)](#how-the-system-works-end-to-end)
- [Project structure](#project-structure)
- [File-by-file explanation](#file-by-file-explanation)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to run](#how-to-run)
- [Incremental processing behavior](#incremental-processing-behavior)
- [Outputs you should expect](#outputs-you-should-expect)
- [Pydantic + LangChain piping in this project](#pydantic--langchain-piping-in-this-project)
- [Reliability safeguards](#reliability-safeguards)
- [Troubleshooting](#troubleshooting)

---

## What this project does

Given one or more files placed in `source-files/`, the system:

1. Classifies each file type (`document`, `image`, `audio`, `video`, `spreadsheet`, `unknown`)
2. Routes each file to the correct extraction pipeline
3. Saves extraction output JSON to `raw-data/`
4. Generates a structured per-file report via LangChain + Pydantic
5. Saves per-file report JSON to `single_reports/`
6. Builds/updates one aggregated report in `final_report.json`

Primary run mode is **folder-based processing** (simple + repeatable).

---

## How the system works (end-to-end)

```text
source-files/
   │
   ├─(classify + route)
   ▼
pipelines/* (audio, video, images, documents, spreadsheet)
   │
   ▼
raw-data/<file>.json
   │
   ├─ single_report_prompt | llm | parser
   ▼
single_reports/<file>.report.json
   │
   ├─ final_report_prompt | llm | parser
   ▼
final_report.json
```

---

## Project structure

```text
agent/
├── .env
├── .gitignore
├── .venv/
├── app/
│   ├── main.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── templates.py
│   │   └── report_agent.py
│   ├── pipelines/
│   │   ├── audio.py
│   │   ├── documents.py
│   │   ├── images.py
│   │   ├── spreadsheet.py
│   │   └── video.py
│   ├── routers/
│   │   ├── audio_api.py
│   │   ├── document_api.py
│   │   ├── image_api.py
│   │   ├── spreadsheet_api.py
│   │   └── video_api.py
│   └── services/
│       ├── classifier.py
│       ├── llm_client.py
│       ├── source_file_processor.py
│       └── storage.py
├── final_report.json
├── outputs/
├── raw-data/
├── requirements.txt
├── single_reports/
├── source-files/
└── uploads/
```

---

## File-by-file explanation

### Root files

#### `.env`
Runtime configuration and secrets:
- OpenRouter API key/base URL
- model names
- app metadata


#### `.gitignore`
Prevents committing secrets, virtual env, and generated outputs (`raw-data/`, `single_reports/`, `final_report.json`, etc.).

#### `requirements.txt`
Dependency list for extraction + reporting stack (FastAPI, OpenAI/OpenRouter, LangChain, Pydantic, CV/document libs).

#### `final_report.json`
Generated aggregated report across all single reports.

#### `README.md`
This document.

---

### `app/` code modules

#### `app/main.py`
FastAPI app entrypoint (kept for API endpoints/testing).  
Folder processor does not require running FastAPI server.

---

### `app/services/`

#### `services/classifier.py`
Maps file extensions to normalized file categories used for routing.

#### `services/llm_client.py`
Initializes OpenAI client configured to call OpenRouter models.

#### `services/storage.py`
Path + persistence helpers:
- ensure directories exist
- save extraction JSON (`raw-data/`)
- save single reports (`single_reports/`)
- save final report (`final_report.json`)

#### `services/source_file_processor.py` (main orchestrator)
Main runtime pipeline:
1. Scan files in `source-files/`
2. Skip already processed files (if matching single report exists)
3. Classify + extract + save raw JSON
4. Generate/save single report
5. Load all single reports and regenerate final report
6. Apply quality guard + fallback if final report is weak/empty

---

### `app/pipelines/`

Extraction engines by modality:
- `audio.py`: transcription/analysis
- `video.py`: frame/audio-based extraction
- `images.py`: OCR + image understanding
- `documents.py`: PDF/DOCX/TXT extraction
- `spreadsheet.py`: CSV/XLS/XLSX extraction

---

### `app/agents/`

#### `agents/schemas.py`
Pydantic output contracts:
- `SingleFileReport`
- `FinalReport`

These enforce strong structured output.

#### `agents/templates.py`
Prompt templates:
- single file reporting prompt
- final aggregation prompt

Includes schema formatting instructions and quality constraints.

#### `agents/report_agent.py`
LangChain execution layer using pipe composition:
- `single_report_prompt | llm | parser`
- `final_report_prompt | llm | parser`

---

### `app/routers/`

API endpoints per modality (audio/document/image/spreadsheet/video).  
These are available for API use, but current primary operational flow is folder-based processing.

---

### Data folders

#### `source-files/`
Drop new input files here.

#### `raw-data/`
Extraction output JSON files.

#### `single_reports/`
Per-file generated report JSON files.

#### `uploads/`, `outputs/`
Legacy folders from earlier API upload workflow (kept for compatibility/history).

---

## Installation

1. Create/activate virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure `.env` exists and contains valid OpenRouter credentials.

---

## Configuration

Minimum `.env` keys:

```ini
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=qwen/qwen3-vl-235b-a22b-instruct
OPENROUTER_AUDIO_MODEL=google/gemini-3-flash-preview
SITE_URL=http://localhost:8000
SITE_NAME=Incident Analysis Agent
```

Optional MinIO keys (Phase 2 object storage):

```ini
MINIO_ENABLED=false
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=incident-analysis
MINIO_SECURE=false
```

When `MINIO_ENABLED=true`, the processor performs dual-write for artifacts:
- keeps local filesystem outputs (for compatibility)
- also uploads source/raw/single/final artifacts to MinIO
- stores MinIO artifact pointers in SQLite `artifacts` table (`storage_backend='minio'`)

---

## How to run

From project root (`.../projects`):

```bash
cd agent
.venv\Scripts\python -m app.services.source_file_processor
```

---

## Incremental processing behavior

The processor is incremental by design:

- For each file in `source-files/`, it checks whether
  `single_reports/<file_name>.report.json` already exists.
- If it exists: file is marked `skipped` (no re-extraction, no re-reporting).
- If not: process as new.
- `final_report.json` is regenerated from **all** single reports (old + new).

This reduces cost and avoids repeated work.

---

## Outputs you should expect

For each new source file:

- Raw extraction:
  `raw-data/<file_name>.json`
- Single report:
  `single_reports/<file_name>.report.json`

For the whole run:

- Aggregated report:
  `final_report.json`

---

## Pydantic + LangChain piping in this project

### What Pydantic does
Defines strict report schemas and validates model output.

### What parser does
Converts LLM output text into validated schema objects.

### What this means

```python
chain = final_report_prompt | llm | parser
```

Pipeline stages:
1. Prompt template builds instruction
2. LLM generates candidate output
3. Parser validates/parses into `FinalReport`

Result: reliable structured output, not random free-text.

---

## Reliability safeguards

To prevent empty/low-quality final reports:

1. Strong final prompt quality constraints
2. Final report quality gate (`_is_final_report_meaningful`)
3. Deterministic fallback aggregator (`_build_fallback_final_report`) if LLM output is weak

So `final_report.json` should not be overwritten by empty content.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'langchain'`
Install dependencies:

```bash
cd agent
.venv\Scripts\pip install -r requirements.txt
```

### Parser import errors
Use modern import path in code:
- `from langchain_core.output_parsers import PydanticOutputParser`

### Final report appears weak/empty
The processor now applies quality checks + fallback synthesis. Re-run processor after ensuring single reports exist.

---

## Optional API mode (Swagger)

You can still run FastAPI endpoints:

```bash
cd agent
.venv\Scripts\python -m uvicorn app.main:app --reload
```

Open:
- `http://127.0.0.1:8000/docs`



