# ResearchMind 🔬

> **A Multi-Agent AI Research System** — Four specialized AI agents collaborate (Search, Read, Write, Critique) to deliver polished, citation-backed research reports on any topic.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54+-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1.4+-green.svg)](https://langchain.com)
[![Google Gemini](https://img.shields.io/badge/AI-Gemini%203.1%20Flash-blue)](https://ai.google.dev)

---

## 📖 Overview

ResearchMind is an autonomous research pipeline built with **LangChain**, **ChromaDB**, and **Streamlit** that orchestrates four specialized AI agents to conduct deep research on any topic:

| Agent | Role | Tools |
|-------|------|-------|
| **Search Agent** | Finds recent, reliable web sources | Tavily Search API |
| **Reader Agent** | Scrapes & extracts deep content from top URLs | BeautifulSoup + Requests |
| **RAG Layer** | Chunks scraped pages and retrieves the top-5 evidence chunks | ChromaDB + Gemini/HuggingFace embeddings |
| **Writer Chain** | Synthesizes retrieved evidence into structured reports | Google Gemini (gemini-3.1-flash-lite) |
| **Critic Chain** | Reviews, scores & citation-checks the report | Google Gemini (gemini-3.1-flash-lite) |

The system produces a **complete research report** with:
- Introduction & Key Findings (minimum 4 detailed points)
- Methodology
- Analysis & Discussion (including Limitations and Counterarguments)
- Conclusion
- Source citations (all URLs discovered)
- **Critic feedback** with score (X/10), strengths, areas to improve, and verdict

---

## Features

- 🤖 **Multi-Agent Pipeline** — Four agents work in sequence, each building on the previous agent's output
- 🧩 **Vector RAG (ChromaDB)** — Scraped pages are chunked (1,000/150) and embedded; the writer receives only the top-5 semantically relevant chunks with source URLs
- 📊 **Quantitative Telemetry** — End-to-end and per-stage latency, token reduction vs. a full-context baseline, critic score, and citation groundedness
- 🔍 **Real-time Web Search** — Uses Tavily API for current, reliable information
- 📄 **Deep Content Extraction** — Scrapes full article content for nuanced understanding
- 🎨 **Beautiful Streamlit UI** — Dark theme, live pipeline status, metrics row, expandable sources and retrieved chunks
- 📥 **Export Reports** — Download finished reports as Markdown files and metrics as JSON
- 🚀 **FastAPI + Docker** — Pydantic-typed `POST /research` endpoint with a production-style container image
- ⚡ **Fast Inference** — Powered by Google Gemini for responsive LLM responses
- 🐳 **Dev Container Ready** — One-click GitHub Codespaces / VS Code Remote Containers support

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         RESEARCHMIND PIPELINE                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  USER INPUT: "Quantum computing breakthroughs 2026"                  │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐   │
│  │ SEARCH AGENT │───▶│ READER AGENT │───▶│ VECTOR RAG (ChromaDB) │   │
│  │              │    │              │    │                       │   │
│  │ • Tavily     │    │ • Select URL │    │ • Split 1000 / 150    │   │
│  │ • Top 5      │    │ • Scrape     │    │ • Embed pages         │   │
│  │   results    │    │   top-N      │    │ • Retrieve top-5      │   │
│  └──────────────┘    └──────────────┘    └──────────┬────────────┘   │
│                                                     │                │
│                                                     ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐   │
│  │ EVALUATION & │◀───│ CRITIC CHAIN │◀───│ WRITER CHAIN          │   │
│  │ TELEMETRY    │    │              │    │                       │   │
│  │              │    │ • Score X/10 │    │ • Uses retrieved      │   │
│  │ • Latency    │    │ • Strengths  │    │   evidence only       │   │
│  │ • Tokens     │    │ • Gaps       │    │ • Cites [SOURCE N]    │   │
│  │ • Grounding  │    │              │    │ • Drafts full report  │   │
│  └──────────────┘    └──────────────┘    └───────────────────────┘   │
│                                                                      │
│  OUTPUT: report + critic feedback + sources + evaluation + metrics   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
researchmind/
├── app.py                      # 🎨 Streamlit web application (main entry point)
├── api.py                      # 🚀 FastAPI endpoint + Pydantic models
├── src/                        # 🧠 Core engine package
│   ├── __init__.py
│   ├── pipeline.py             # 🔄 Pipeline orchestration + metrics
│   ├── agents.py               # 🤖 Lazy Gemini model, agents and chains
│   ├── rag.py                  # 🧩 ChromaDB vector RAG layer
│   ├── tools.py                # 🔧 Tavily search, BeautifulSoup scraping
│   └── evaluation.py           # ✅ Evaluation, groundedness, telemetry, text utils
├── benchmarks/                 # 📈 Benchmark harness + measured results
│   ├── benchmark.py
│   └── token_benchmark.json
├── tests/                      # 🧪 Offline unit tests
├── Dockerfile                  # 🐳 API container image
├── requirements.txt            # 📦 Python dependencies (+ optional local embeddings)
├── INTERVIEW_GUIDE.md          # 📖 Architecture, benchmarks and interview notes
├── README.md
├── .env.example                # Template for environment variables
└── .gitignore
```

### File Descriptions

| File | Purpose |
|------|---------|
| **app.py** | Streamlit app with metrics row, live pipeline status, sources, retrieved chunks, and report download |
| **api.py** | FastAPI service exposing `POST /research` and `GET /health`, with merged Pydantic schemas |
| **src/pipeline.py** | `run_research_pipeline(topic)` — orchestrates all stages, RAG, evaluation, telemetry |
| **src/agents.py** | Lazy Gemini model, Search/Reader agents, Writer/Critic chains |
| **src/rag.py** | Recursive chunking, Gemini/HuggingFace embeddings, ephemeral ChromaDB, diversified top-k retrieval |
| **src/tools.py** | `web_search(query)` via Tavily, `scrape_url(url)` via BeautifulSoup, concurrent page scraping |
| **src/evaluation.py** | Critic score parsing, groundedness/citation checks, token accounting, JSONL telemetry, content normalization |
| **benchmarks/benchmark.py** | Offline token benchmark and full-pipeline latency/quality benchmark |
| **Dockerfile** | Non-root API image with healthcheck |
| **requirements.txt** | Core dependencies; optional local embedding packages are commented at the bottom |

---

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Google API Key** — Get at [ai.google.dev](https://ai.google.dev)
- **Tavily API Key** — Get free at [tavily.com](https://tavily.com)

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd researchmind

# 2. Create virtual environment
python -m venv .venv

# 3. Activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# Optional: free/local HuggingFace embeddings instead of Gemini embeddings
# pip install langchain-huggingface sentence-transformers

# 5. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys (see Configuration below)
```

### Configuration

Edit `.env` with your API keys:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx

# Optional RAG tuning
EMBEDDING_PROVIDER=gemini
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=150
RAG_TOP_K=5
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | Google API key for Gemini LLM inference (gemini-3.1-flash-lite) and embeddings |
| `TAVILY_API_KEY` | ✅ Yes | Tavily API key for web search |
| `EMBEDDING_PROVIDER` | No | `gemini` (default) or `huggingface` for local/free embeddings |
| `GEMINI_EMBEDDING_MODEL` | No | Defaults to `models/gemini-embedding-001` |
| `EMBEDDING_DIMENSIONS` | No | Optional reduced Gemini embedding size (e.g. `768`) |
| `HF_EMBEDDING_MODEL` | No | Defaults to `sentence-transformers/all-MiniLM-L6-v2` |
| `RAG_CHUNK_SIZE` | No | Defaults to `1000` |
| `RAG_CHUNK_OVERLAP` | No | Defaults to `150` |
| `RAG_TOP_K` | No | Defaults to `5` |
| `RAG_RETRIEVAL_MODE` | No | `diversified` (per-URL cap), `mmr`, or `similarity` |
| `RAG_FETCH_K` | No | Candidate pool size for diversified retrieval; default `20` |
| `RAG_MAX_CHUNKS_PER_SOURCE` | No | Max chunks from one URL in the top-k; default `2` |
| `EMBEDDING_BATCH_SIZE` | No | Chunks per embedding batch; default `32` |
| `EMBEDDING_CONCURRENCY` | No | Concurrent embedding batches for Gemini; default `4` |
| `TELEMETRY_LOG_DIR` | No | Defaults to `logs` (JSONL run metrics) |
| `CORS_ORIGINS` | No | Comma-separated origins for the API; default `*` |

---

## ▶️ Running the Application

### Option 1: Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` — Beautiful UI with live pipeline visualization.

### Option 2: CLI Pipeline

```bash
python -m src.pipeline
```

Prompts for a topic, runs the full pipeline, and prints the final report + critic feedback to console.

### Option 3: Using the Pipeline Programmatically

```python
from src.pipeline import run_research_pipeline

result = run_research_pipeline("Your research topic here")

# Original outputs (still available)
print(result["search_results"])   # Raw search results
print(result["scraped_content"])  # Reader agent output
print(result["report"])           # Final research report
print(result["feedback"])         # Critic feedback with score

# New RAG + evaluation outputs
print(result["sources"])          # Scraped source metadata (URL, title, chars)
print(result["retrieved_chunks"]) # Top-k ChromaDB chunks with relevance
print(result["metrics"])          # Latency, token savings, quality metrics
print(result["evaluation"])       # Critic score + groundedness / citation check
```

### Option 4: FastAPI Backend

```bash
uvicorn api:app --reload --port 8000
```

Open <http://localhost:8000/docs> for Swagger UI.

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "quantum error correction 2026", "max_sources": 3, "top_k": 5}'
```

Returns the report, critic feedback, sources, retrieved chunks, evaluation, and metrics as JSON.

### Option 5: Docker

```bash
docker build -t researchmind-api .
docker run --rm -p 8000:8000 --env-file .env researchmind-api
```

To include the local HuggingFace embedding fallback in the image (larger, pulls PyTorch):

```bash
docker build --build-arg INSTALL_LOCAL_EMBEDDINGS=true -t researchmind-api:local .
```

For the Streamlit UI in the same image:

```bash
docker run --rm -p 8501:8501 --env-file .env researchmind-api \
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

---

## Usage Examples

### Web UI (Streamlit)

1. Enter a topic: `"LLM agents 2026"`, `"CRISPR gene editing"`, `"Fusion energy progress"`
2. Click **⚡ Run Research Pipeline**
3. Watch the live stage log (search, scraping, ChromaDB indexing, writer, critic)
4. Review the metrics row: latency, token reduction, critic score, groundedness, chunks
5. Inspect retrieved evidence and sources in the expanders
6. Read the final report + critic feedback
7. Download the report as `.md` or metrics as `.json`

### CLI

```bash
$ python -m src.pipeline

 Enter a research topic : Quantum computing error correction 2024

 ==================================================
step 1 - search agent is working ...
==================================================

 search result  Title: Quantum Error Correction Breakthrough...
URL: https://example.com/quantum-ec
Snippet: Researchers at MIT have demonstrated...

 ==================================================
step 2 - Reader agent is scraping top resources ...
==================================================

scraped content:
 Researchers at MIT have demonstrated a new quantum error correction...

 ==================================================
step 3 - Writer is drafting the report ...
==================================================

 Final Report
 # Quantum Computing Error Correction: 2024 Breakthroughs

 ## Executive Summary
 ...

 ## Introduction
 ...

 ## Methodology
 ...

 ## Key Findings
 ...

 ## Analysis & Discussion
 ...

 ## Conclusion
 ...

 ## References
 - https://example.com/quantum-ec
 ...

 ==================================================
step 4 - critic is reviewing the report
==================================================

 critic report
 Overall Score: 8/10

 Evidence & Citations:
 - Comprehensive coverage of recent breakthroughs
 - Clear structure with citations

 Structure & Logic:
 - ...

 Analysis & Synthesis:
 - ...

 Transparency / Limitations:
 - ...

 Presentation:
 - ...

 Critical Gaps (must address before publication):
 - Could include more technical depth on surface codes
 - Add timeline of developments

 One line verdict: Strong report with minor gaps in technical depth.
```

---

## 📈 Benchmarking

```bash
# Offline token/chunking benchmark (no API keys required)
python benchmarks/benchmark.py --token-only --runs 3 --topic "large language model agents"

# Full end-to-end benchmark (requires both API keys)
python benchmarks/benchmark.py --runs 3 --topic "LLM agents 2026"
```

The offline run writes `benchmarks/token_benchmark.json`; the full run writes `benchmarks/latest.json`. See **[INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)** for architecture details, metric definitions, and the measured token-reduction numbers.

---

## 🔧 Development

### Dev Containers (VS Code / GitHub Codespaces)

The project includes a complete dev container setup:

1. Open in VS Code: `code .`
2. Command Palette → **Dev Containers: Reopen in Container**
3. Auto-installs dependencies & starts Streamlit on port 8501
4. Browser preview opens automatically

**Dev Container Features:**
- Python 3.11 (bookworm)
- Auto-installs `requirements.txt` + `streamlit`
- Port 8501 forwarded with auto-preview
- Pre-configured Python extensions

### Tests

```bash
python -m pytest tests -q
```

The suite is fully offline: fake embeddings exercise real ChromaDB chunking/retrieval, mocked LLMs exercise pipeline orchestration, and the evaluation tests cover citation precision and claim coverage.

### Adding New Tools/Agents

1. **New Tool** → Add to `src/tools.py` with `@tool` decorator
2. **New Agent** → Add to `src/agents.py` using `create_agent(model, tools=[...])`
3. **New Chain** → Build the prompt in `src/agents.py` and expose a `build_*_chain()` factory
4. **Integrate** → Update `src/pipeline.py` and/or `app.py` to use new component

---

## 📦 Dependencies

### Core (requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥1.4.0 | Agent orchestration framework |
| `langchain-core` | ≥1.6.0 | Core LangChain abstractions |
| `langchain-community` | ≥0.4.0 | Community integrations |
| `langchain-google-genai` | ≥4.4.0 | Google Gemini LLM + embeddings |
| `langchain-chroma` | ≥1.1.0 | Chroma vector store integration |
| `langchain-text-splitters` | ≥1.0.0 | Recursive chunking |
| `chromadb` | ≥1.5.0 | Ephemeral vector database |
| `tiktoken` | ≥0.8.0 | Token accounting |
| `fastapi` | ≥0.115.0 | API framework |
| `uvicorn[standard]` | ≥0.30.0 | ASGI server |
| `streamlit` | ≥1.54.0 | Web UI framework |
| `tavily-python` | ≥0.8.0 | Web search API client |
| `beautifulsoup4` | ≥4.14.0 | HTML parsing for scraping |
| `requests` | ≥2.32.0 | HTTP requests |
| `lxml` | ≥6.1.0 | Fast XML/HTML parser |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |
| `rich` | ≥15.0.0 | Rich terminal output (CLI) |
| `pydantic` | ≥2.13.0 | Data validation / API schemas |
| `tenacity` | ≥9.1.0 | Retry logic for API calls |

### Optional local embeddings (commented in `requirements.txt`)

| Package | Purpose |
|---------|---------|
| `langchain-huggingface` | LangChain wrapper for sentence-transformers |
| `sentence-transformers` | Local/free embedding model runtime |

---

## 🔐 API Keys Setup

### Google API Key
1. Go to [ai.google.dev](https://ai.google.dev)
2. Sign up / log in
3. Create API Key
4. Model used: `gemini-3.1-flash-lite` (fast, cost-effective)

### Tavily API Key
1. Go to [tavily.com](https://tavily.com)
2. Sign up for free tier (1000 searches/month)
3. Copy API key from dashboard

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` in activated venv |
| `GOOGLE_API_KEY not set` | Check `.env` file exists and has correct key |
| `TAVILY_API_KEY not set` | Same as above for Tavily |
| Streamlit won't start | Ensure port 8501 is free: `streamlit run app.py --server.port 8502` |
| Scraping fails | Some sites block bots; the tool handles errors gracefully |
| Slow responses | Check network and API rate limits |
| Gemini embedding error | Set `EMBEDDING_PROVIDER=huggingface` and uncomment the optional local packages in `requirements.txt` |
| `rag_enabled: false` in metrics | The pipeline fell back to full context; check `rag_error` in the metrics/logs |
| No `/research` response | Ensure both `GOOGLE_API_KEY` and `TAVILY_API_KEY` are set; `/health` works without them |

---

## Acknowledgments

- **LangChain** — Agent orchestration framework
- **Google Gemini** — LLM inference for writer and critic chains
- **Tavily** — Real-time web search API
- **Streamlit** — Beautiful data apps in pure Python
- **BeautifulSoup** — HTML parsing made easy
