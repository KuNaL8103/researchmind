# ResearchMind 🔬

> **A Multi-Agent AI Research System** — Four specialized AI agents collaborate (Search, Read, Write, Critique) to deliver polished, citation-backed research reports on any topic.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54+-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1.4+-green.svg)](https://langchain.com)
[![Google Gemini](https://img.shields.io/badge/AI-Gemini%203.1%20Flash-blue)](https://ai.google.dev)

---

## 📖 Overview

ResearchMind is an autonomous research pipeline built with **LangChain** and **Streamlit** that orchestrates four specialized AI agents to conduct deep research on any topic:

| Agent | Role | Tools |
|-------|------|-------|
| **Search Agent** | Finds recent, reliable web sources | Tavily Search API |
| **Reader Agent** | Scrapes & extracts deep content from top URLs | BeautifulSoup + Requests |
| **Writer Chain** | Synthesizes research into structured reports | Google Gemini (gemini-3.1-flash-lite) |
| **Critic Chain** | Reviews, scores & improves the report | Google Gemini (gemini-3.1-flash-lite) |

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
- 🔍 **Real-time Web Search** — Uses Tavily API for current, reliable information
- 📄 **Deep Content Extraction** — Scrapes full article content for nuanced understanding
- 🎨 **Beautiful Streamlit UI** — Dark theme, animated pipeline visualization, expandable raw outputs
- 📥 **Export Reports** — Download finished reports as Markdown files
- ⚡ **Fast Inference** — Powered by Google Gemini for responsive LLM responses
- 🐳 **Dev Container Ready** — One-click GitHub Codespaces / VS Code Remote Containers support

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RESEARCHMIND PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  USER INPUT: "Quantum computing breakthroughs 2025"             │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ SEARCH AGENT │───▶│ READER AGENT │───▶│ WRITER CHAIN │      │
│  │              │    │              │    │               │       │
│  │ • Tavily     │    │ • Tavily     │    │ • Synthesizes │      │
│  │   Search     │    │   Search     │    │   research    │      │
│  │ • Top 5      │    │ • Scrapes    │    │ • Structures  │      │
│  │   results    │    │   best URL   │    │   report      │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘       │
│                                                  │              │
│                                                  ▼              │
│  ┌──────────────┐    ◀───────────────────────────┤              |
│  │ CRITIC CHAIN │                                 │             │
│  │              │                                 │             │
│  │ • Scores     │                                 │             │
│  │   (X/10)     │                                 │             │
│  │ • Strengths  │                                 │             │
│  │ • Improvements                   FINISHED REPORT             │
│  └──────────────┘                                 & FEEDBACK    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
researchmind/
├── .devcontainer/              # Dev container configuration
│   └── devcontainer.json       # VS Code / Codespaces setup
├── .venv/                      # Python virtual environment (gitignored)
├── __pycache__/                # Python bytecode cache (gitignored)
├── .env                        # Environment variables (gitignored - create from .env.example)
├── .env.example                # Template for environment variables
├── .gitignore                  # Git ignore rules
├── app.py                      # 🎨 Streamlit web application (main entry point)
├── agents.py                   # 🤖 Agent & chain definitions (Search, Reader, Writer, Critic)
├── pipeline.py                 # 🔄 CLI pipeline runner (run_research_pipeline function)
├── tools.py                    # 🔧 LangChain tools (web_search, scrape_url)
└── requirements.txt            # 📦 Python dependencies
```

### File Descriptions

| File | Purpose |
|------|---------|
| **app.py** | Main Streamlit web app with custom CSS, pipeline visualization, and results display |
| **agents.py** | Defines 4 agents/chains: `build_search_agent()`, `build_reader_agent()`, `writer_chain`, `critic_chain` |
| **pipeline.py** | CLI version of the pipeline — `run_research_pipeline(topic)` returns full state dict |
| **tools.py** | LangChain `@tool` functions: `web_search(query)` via Tavily, `scrape_url(url)` via BeautifulSoup |
| **requirements.txt** | All Python dependencies (LangChain, Streamlit, Tavily, BeautifulSoup, etc.) |

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

# 5. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys (see Configuration below)
```

### Configuration

Edit `.env` with your API keys:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxx
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | Google API key for Gemini LLM inference (gemini-3.1-flash-lite) |
| `TAVILY_API_KEY` | ✅ Yes | Tavily API key for web search |

---

## ▶️ Running the Application

### Option 1: Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` — Beautiful UI with live pipeline visualization.

### Option 2: CLI Pipeline

```bash
python pipeline.py
```

Prompts for a topic, runs the full pipeline, and prints the final report + critic feedback to console.

### Option 3: Using the Pipeline Programmatically

```python
from pipeline import run_research_pipeline

result = run_research_pipeline("Your research topic here")

# Access individual outputs
print(result["search_results"])   # Raw search results
print(result["scraped_content"])  # Deep scraped content
print(result["report"])           # Final research report
print(result["feedback"])         # Critic feedback with score
```

---

## Usage Examples

### Web UI (Streamlit)

1. Enter a topic: `"LLM agents 2025"`, `"CRISPR gene editing"`, `"Fusion energy progress"`
2. Click **⚡ Run Research Pipeline**
3. Watch the 4-step pipeline execute in real-time
4. View/expand raw outputs from each agent
5. Read the final report + critic feedback
6. Download as `.md` file

### CLI

```bash
$ python pipeline.py

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

### Adding New Tools/Agents

1. **New Tool** → Add to `tools.py` with `@tool` decorator
2. **New Agent** → Add to `agents.py` using `create_agent(model, tools=[...])`
3. **New Chain** → Add prompt template + `llm | StrOutputParser()` chain
4. **Integrate** → Update `pipeline.py` and/or `app.py` to use new component

---

## 📦 Dependencies

### Core (requirements.txt)

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥1.4.0 | Agent orchestration framework |
| `langchain-core` | ≥1.6.0 | Core LangChain abstractions |
| `langchain-community` | ≥0.4.0 | Community integrations |
| `langchain-google-genai` | ≥4.4.0 | Google Gemini LLM integration |
| `streamlit` | ≥1.54.0 | Web UI framework |
| `tavily-python` | ≥0.8.0 | Web search API client |
| `beautifulsoup4` | ≥4.14.0 | HTML parsing for scraping |
| `requests` | ≥2.32.0 | HTTP requests |
| `lxml` | ≥6.1.0 | Fast XML/HTML parser |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |
| `rich` | ≥15.0.0 | Rich terminal output (CLI) |
| `pydantic` | ≥2.13.0 | Data validation |
| `tenacity` | ≥9.1.0 | Retry logic for API calls |

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

---

## Acknowledgments

- **LangChain** — Agent orchestration framework
- **Google Gemini** — LLM inference for writer and critic chains
- **Tavily** — Real-time web search API
- **Streamlit** — Beautiful data apps in pure Python
- **BeautifulSoup** — HTML parsing made easy
