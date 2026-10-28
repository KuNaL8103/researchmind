# ResearchMind — Architecture & Interview Guide

> Four-stage LangChain research pipeline with ChromaDB vector RAG, citation-groundedness evaluation, quantitative telemetry, and a FastAPI/Docker deployment path.

---

## 1. 30-second pitch

ResearchMind takes a topic, searches the web with Tavily, scrapes the most relevant pages with BeautifulSoup, chunks them into an ephemeral ChromaDB vector index, retrieves the top-5 semantically relevant chunks, and gives only that evidence to a Gemini writer. A critic then reviews the report. Every run is measured: latency, tokens saved versus a full-context baseline, critic score, and whether the report's citations map to real retrieved URLs.

The key upgrade is **retrieval instead of context stuffing**: the writer sees ~5 targeted chunks rather than every scraped page. On the bundled benchmark that reduced the scraped evidence passed to the writer from **19,862 tokens to 1,192 tokens (94.0%)** while preserving source URLs for citations.

---

## 2. Before vs. after

| Area | Before | After |
|---|---|---|
| Reader output | One scraped page, truncated to 3,000 chars | Top-N pages scraped concurrently, full text retained for indexing |
| Writer context | Raw search text + reader prose | Search brief + **top-5 ChromaDB chunks** with URL metadata |
| Retrieval | None | Gemini embeddings (default) or local HuggingFace embeddings |
| Observability | `print()` only | Per-stage timings, token accounting, critic score, groundedness check, JSONL logs |
| API | CLI / Streamlit only | `POST /research` FastAPI endpoint with Pydantic schemas |
| Deployment | Local | `Dockerfile` with healthcheck and non-root user |
| Grounding | Prompt instruction only | Automatic citation verification against retrieved URLs |

---

## 3. Architecture

```text
                              ┌─────────────────────┐
USER TOPIC ──────────────────▶│  1. SEARCH AGENT    │
                              │  Tavily (top 5)     │
                              └──────────┬──────────┘
                                         │ titles + URLs + snippets
                                         ▼
                              ┌─────────────────────┐
                              │  2. READER AGENT    │
                              │  selects best URL   │
                              │  + BeautifulSoup    │
                              │  scrapes top-N URLs │
                              └──────────┬──────────┘
                                         │ raw page text + URL/title
                                         ▼
                    ┌────────────────────────────────────────┐
                    │  VECTOR RAG LAYER (src/rag.py)             │
                    │  RecursiveCharacterTextSplitter        │
                    │      chunk_size=1000, overlap=150      │
                    │  Gemini / HuggingFace embeddings       │
                    │  ephemeral ChromaDB collection          │
                    │  semantic search → top_k=5              │
                    └──────────────────┬─────────────────────┘
                                       │ 5 chunks + source URLs
                                       ▼
                              ┌─────────────────────┐
                              │  3. WRITER CHAIN    │
                              │  Gemini report with │
                              │  [1] … [n] citations│
                              └──────────┬──────────┘
                                         │ report
                                         ▼
                              ┌─────────────────────┐
                              │  4. CRITIC CHAIN    │
                              │  score 1–10 + review│
                              └──────────┬──────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────────┐
                    │  EVALUATION + TELEMETRY                │
                    │  latency · tokens saved · critic score │
                    │  citation precision · claim coverage   │
                    │  JSONL run log                        │
                    └────────────────────────────────────────┘
```

### File map

The RAG layer is deliberately **not a fifth LLM agent**. It is a deterministic tool layer between Read and Write: it cannot hallucinate, it costs only embeddings, and it gives the writer provenance-tagged evidence.

| File | Responsibility |
|---|---|
| `src/agents.py` | Lazy Gemini model, Search/Reader agents, Writer/Critic chains and prompts |
| `src/tools.py` | Tavily search, BeautifulSoup scraping, URL extraction, concurrent source scraping |
| `src/rag.py` | Chunking, embedding selection, ephemeral Chroma index, top-k retrieval, context formatting |
| `src/pipeline.py` | Orchestrates the four stages, builds metrics, runs evaluation, logs the run |
| `src/evaluation.py` | Critic score parser, groundedness/citation verification, telemetry, token counting, content normalization |
| `app.py` | Streamlit UI, live pipeline status, metrics row, sources and retrieved-chunk views |
| `api.py` | FastAPI service + merged Pydantic request/response contracts |
| `benchmarks/benchmark.py` | Offline token benchmark and full end-to-end benchmark harness |
| `tests/` | Offline tests for chunking/retrieval, evaluation, and pipeline orchestration |

---

## 4. How the ChromaDB RAG layer works

### 4.1 Scrape with provenance

The reader stage does two things:

1. The Reader agent selects and summarizes the most relevant URL (preserves the original agentic behavior).
2. The pipeline deterministically extracts every URL from the search output, puts the agent-selected URL first, and scrapes up to `max_sources` pages concurrently with `requests` + BeautifulSoup.

`fetch_page()` removes scripts, styles, navigation, footers, forms, and iframes, then normalizes whitespace. Each page keeps:

```python
{"url": "...", "title": "...", "text": "...", "ok": True}
```

`scrape_url` still returns the first 3,000 characters to the agent, so existing agent behavior is unchanged. The RAG layer uses the full cleaned text (up to `MAX_SCRAPE_CHARS`, default 20,000).

### 4.2 Chunking

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

- **1000 characters** keeps a chunk small enough for precise retrieval but large enough to contain a complete argument or finding.
- **150-character overlap** protects against a key sentence being split across a boundary.
- Recursive separators prefer paragraph breaks, then sentence boundaries, then spaces. This keeps chunks semantically coherent.
- Exact duplicate chunks (common with repeated nav/footer text) are removed with a SHA-1 fingerprint.

Every chunk becomes a LangChain `Document` with metadata:

```python
{
    "source": page_url,
    "url": page_url,
    "title": page_title,
    "chunk_index": index_within_page,
    "source_id": 1,
}
```

### 4.3 Embeddings

`EMBEDDING_PROVIDER` controls the backend:

| Provider | Model | Notes |
|---|---|---|
| `gemini` (default) | `models/gemini-embedding-001` | Uses separate `retrieval_document` / `retrieval_query` task types |
| `huggingface` | `sentence-transformers/all-MiniLM-L6-v2` | Fully local/free; uncomment the optional packages in `requirements.txt` |

If Gemini is selected but the key is missing or the API call fails, the code automatically tries HuggingFace. If both fail, the pipeline uses a labelled full-context fallback instead of crashing, and records `rag_enabled=false` plus `rag_error` in metrics.

### 4.4 Ephemeral Chroma collection

Each run creates a unique in-memory collection:

```python
Chroma(
    collection_name=f"research-{uuid4().hex[:12]}",
    embedding_function=document_embedder,
    client_settings=Settings(anonymized_telemetry=False),
)
```

Why ephemeral?

- Research is a one-shot workload; a run's pages should not leak into the next run's retrieval.
- It needs no disk, no migration, and no cleanup in Streamlit Cloud or a container.
- A persistent Chroma directory is a one-line change if you later want a reusable corpus.

### 4.4b Embedding batching and indexing latency

`BatchedEmbeddings` wraps the document embedder and splits chunks into `EMBEDDING_BATCH_SIZE` batches (default 32). Multiple batches run concurrently up to `EMBEDDING_CONCURRENCY` (default 4 for Gemini); local HuggingFace embeddings stay single-threaded. The wrapper records `embedding_batches` and `embedding_seconds`, which appear in telemetry and the Streamlit details panel. This makes `rag_index` latency attributable to embedding vs. Chroma insertion instead of one opaque number.

### 4.5 Source diversity, retrieval, and prompt assembly

The query is the user's topic. Retrieval uses the query embedder (Gemini `retrieval_query`) and fetches a candidate pool (`RAG_FETCH_K`, default 20), then selects `top_k=5` with a per-URL cap (`RAG_MAX_CHUNKS_PER_SOURCE=2`) so one long page cannot monopolize the context. `RAG_RETRIEVAL_MODE=mmr` is also supported; `similarity` disables diversification.

The source collector reads URLs from **every** search-agent message, including tool outputs, and falls back to a direct Tavily call if the pool is thin. It then scrapes up to 4 pages and retries additional candidates until at least 3 distinct pages succeed.

Chunks are grouped by URL before prompting, so each unique page gets exactly one `[SOURCE N]` number:

```text
[SOURCE 1] Page title
URL: https://example.com/article
Relevance: 0.842
Excerpt 1: <chunk text>
Excerpt 2: <another chunk from the same URL>

---

[SOURCE 2] Another independent source
URL: https://another.example/page
...
```

The prompt instructs the writer to cite `[N]` where `N` matches `[SOURCE N]`, and to copy URLs exactly. Because multiple chunks from one page share a source number, references `[1]`–`[N]` map to `N` distinct URLs rather than duplicate chunks.

### 4.6 Token accounting

Two contexts are measured with `tiktoken` (`cl100k_base`; Gemini uses SentencePiece, so treat this as a consistent approximation):

```text
baseline  = search results + reader brief + all raw scraped page text
optimized = search results + reader brief + top-5 retrieved chunks
saved     = max(0, baseline_tokens - optimized_tokens)
reduction = saved / baseline_tokens * 100
```

The baseline is deliberately the “context stuffing” alternative. It is not the old pipeline's 3,000-character truncation, so the percentage is not inflated by comparing against a already-truncated prompt.

---

## 5. Metrics and evaluation definitions

### 5.1 Latency

`PipelineTelemetry` records wall-clock seconds for:

| Stage | What it covers |
|---|---|
| `search` | Tavily search agent |
| `read` | Reader agent + concurrent scraping |
| `rag_index` | Chunking + batched embeddings + Chroma insert (batch count/time tracked separately) |
| `rag_retrieve` | Query embedding + top-k similarity search |
| `write` | Gemini writer |
| `critic` | Gemini critic |

Total `latency_seconds` is end-to-end wall clock. Per-run metrics are appended to `logs/researchmind_runs.jsonl` and returned by the API/Streamlit UI.

### 5.2 Critic score

A regex extracts `Overall Score: X/10` from the critic output, accepting markdown variants such as `**Overall Score:** 7.5 / 10`. The score is clamped to 0–10 and returned as `critic_score`.

### 5.3 Groundedness / citation check

The evaluator does not ask an LLM whether the report is grounded. It checks structure:

1. Split the report at the `References` heading.
2. Parse numbered reference entries and extract their URLs.
3. Compare every reference URL against the URLs that were actually provided to the writer (scraped sources, search-result URLs, retrieved chunk URLs).
4. Extract inline `[n]` markers from the report body.
5. Count claim-bearing paragraphs **and bullet points** under Key Findings (including `- **Evidence:**`, `- **Interpretation:**`, `- **Implications:**`) that carry at least one inline citation. Heading-only and short label-only lines are ignored.

```text
citation_precision      = verified reference URLs / all reference URLs
claim_coverage          = cited claim units / all claim units
retrieved_url_coverage  = cited retrieved URLs / all retrieved URLs
groundedness            = 0.5 × citation_precision + 0.5 × claim_coverage
passed                  = groundedness ≥ 0.60
                          AND no unsupported URLs
                          AND (no retrieved URLs OR at least one retrieved URL cited)
```

Verdicts: `Strongly grounded` (≥ 0.80), `Partially grounded` (≥ 0.60), `Weakly grounded` (> 0), `No verifiable citations` (0).

**What this proves:** no invented URLs and reasonable citation coverage.
**What it does not prove:** that each citation semantically supports the sentence. That would require an entailment/NLI check or a human/LLM judge. This limitation is stated explicitly in the code and should be stated in an interview.

---

## 6. Benchmarks

### 6.1 Measured: chunking / token reduction

Command:

```bash
python benchmarks/benchmark.py --token-only --runs 3 --topic "large language model agents"
```

Measured on **2026-10-28**, on 4 public Wikipedia pages, with production chunk settings:

| Metric | Value |
|---|---|
| Pages scraped | 4 |
| Raw scraped text | 19,862 tokens |
| RAG writer context (top-5) | 1,192 tokens |
| **Token reduction** | **94.0%** |
| Chunks indexed | 95 |
| Retrieved chunks | 5 |
| Chunk size / overlap | 1,000 / 150 |
| Local chunk + embed + retrieve overhead | 0.976 s (average of 3 runs) |

Raw data: `benchmarks/token_benchmark.json`.

**Honest caveat:** the offline benchmark uses deterministic hashing embeddings to select the top-5 chunks, so the 94.0% number measures *token accounting and chunking*, not Gemini retrieval quality. The chunk size, overlap, top-k, and token counter are the production ones. Local overhead is not end-to-end LLM latency.

### 6.2 End-to-end latency (run this with your keys)

This environment had no API keys, so end-to-end latency was **not** measured here. Run:

```bash
python benchmarks/benchmark.py --runs 3 --topic "LLM agents 2026"
```

It prints and writes `benchmarks/latest.json` with:

- mean / median / min / max end-to-end latency
- average token reduction
- average tokens saved
- average critic score
- average groundedness score
- per-run stage timings

Use those measured numbers on your resume. Do not claim a latency figure that you have not measured.

### 6.3 Resume bullets (measured parts are real)

> Built a four-stage LangChain research pipeline (Search → Read → Write → Critic) that cut scraped evidence passed to the writer by **94.0%** (19,862 → 1,192 tokens across four sources) using `RecursiveCharacterTextSplitter` (1,000/150) and ChromaDB top-5 semantic retrieval with source-URL metadata.

> Enforced multi-source grounding by collecting URLs from agent tool outputs, retrying scrapes until 3+ distinct pages succeeded, and applying per-URL retrieval diversification so top-5 evidence spans independent sources with one reference number per URL.

> Added per-run telemetry and an automated citation-groundedness evaluator that verifies reference URLs against retrieved evidence and computes citation precision + claim coverage, logging latency and quality metrics as JSONL.

> Exposed the pipeline through a Pydantic-typed FastAPI `POST /research` endpoint and containerized it with a non-root Docker image and healthcheck.

After running the full benchmark, add a sentence such as:

> Average end-to-end latency: **X.X s** over 3 runs (Gemini 3.1 Flash-Lite, Tavily, 3 sources, top-5 retrieval).

---

## 7. API and Docker

### Run the API

```bash
uvicorn api:app --reload --port 8000
```

Docs: <http://localhost:8000/docs>

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "quantum error correction 2026", "max_sources": 3, "top_k": 5}'
```

Response shape:

```json
{
  "run_id": "a1b2c3d4e5f6",
  "topic": "quantum error correction 2026",
  "report": "# ...",
  "feedback": "Overall Score: 8/10 ...",
  "sources": [{"url": "...", "title": "...", "char_count": 18342, "ok": true}],
  "retrieved_chunks": [{"content": "...", "url": "...", "title": "...", "score": 0.84}],
  "evaluation": {
    "critic_score": 8.0,
    "groundedness_score": 0.86,
    "citation_precision": 1.0,
    "claim_coverage": 0.72,
    "retrieved_url_coverage": 0.8,
    "unsupported_urls": [],
    "verdict": "Strongly grounded",
    "passed": true
  },
  "metrics": {
    "latency_seconds": 31.4,
    "stage_latency_seconds": {"search": 6.1, "read": 4.2, "rag_index": 3.8, "rag_retrieve": 0.9, "write": 12.8, "critic": 3.6},
    "token_reduction_pct": 87.5,
    "tokens_saved": 14210,
    "chunks_indexed": 61,
    "retrieved_chunks": 5,
    "retrieved_sources": 4,
    "embedding_batches": 2,
    "embedding_seconds": 3.1,
    "rag_enabled": true
  }
}
```

The endpoint returns **503** if required keys are missing, **422** for an invalid topic, and **500** for a pipeline failure. `/health` works without keys so container orchestration can probe liveness.

### Docker

```bash
docker build -t researchmind-api .
docker run --rm -p 8000:8000 --env-file .env researchmind-api
```

The image runs as UID 10001, exposes port 8000, and has an HTTP healthcheck. `requirements.txt` includes the API and RAG dependencies; optional local HuggingFace embedding packages are commented at the bottom to keep the default image smaller.

To run the Streamlit UI in the same image:

```bash
docker run --rm -p 8501:8501 --env-file .env researchmind-api \
  streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

---

## 8. Design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Ephemeral Chroma | Isolated per run, zero ops, no stale data | No cross-run corpus reuse; re-embedding each run |
| Chunk 1000 / overlap 150 | Coherent evidence, bounded prompt | Very long tables may split awkwardly |
| top-k = 5 | Small writer context, enough evidence for a report | May miss wide-ranging topics; configurable |
| Separate Gemini document/query task types | Better asymmetric retrieval | Two embedder objects |
| Deterministic scraping after agent selection | Reliable provenance for citations | Extra HTTP requests; some pages block bots |
| Full-context fallback | Pipeline survives embedding outages | Higher token usage, `rag_enabled=false` |
| Rule-based groundedness | Cheap, deterministic, no judge LLM | Does not verify semantic entailment |
| Sync FastAPI endpoint | Runs in FastAPI's threadpool; simple | For high throughput, switch to async/queue |
| `tiktoken` for token counts | Fast and consistent | Gemini tokenizer differs; treat as an estimate |

### Production hardening backlog

- Persistent Chroma or a managed vector DB for repeated topics.
- Redis/in-process cache for search + embeddings.
- Retry/backoff around Tavily/Gemini (tenacity is already a dependency).
- Structured observability (OpenTelemetry, LangSmith traces).
- Auth/rate limiting on `/research`; per-request cost budgets.
- Prompt-injection defenses: treat scraped text as untrusted data, strip instructions, allowlist domains.
- Async job queue for long reports and streaming progress (SSE/WebSocket).
- A labeled evaluation set to validate groundedness beyond structural checks.

---

## 9. Likely interview questions

**Why add RAG instead of increasing the context window?**
Long contexts cost more, add latency, and make it harder for the model to focus. Retrieval sends only evidence relevant to the topic, and each chunk keeps provenance so citations can be verified. The benchmark shows 94.0% fewer context tokens on four pages.

**Why ChromaDB?**
It is embedded, runs in-process, supports metadata filters, and needs no server for this workload. The code isolates it behind `ResearchRAG`, so swapping in pgvector, Qdrant, or Pinecone only changes `src/rag.py`.

**Why chunk 1,000 characters with 150 overlap?**
Large enough to hold a finding with context, small enough for precise similarity. Overlap prevents boundary loss. The splitter recurses on paragraph → sentence → space boundaries so chunks rarely cut mid-sentence.

**How do you stop hallucinations?**
Three layers: (1) the writer is instructed to use only retrieved evidence and copy URLs exactly; (2) it receives `[SOURCE n]` labels with URLs; (3) the evaluator rejects reference URLs that were never provided and measures uncited findings. This catches fabricated URLs and uncited claims, but not semantic mis-citation — state that limitation.

**What happens when embeddings fail?**
`get_embeddings()` tries Gemini, then local HuggingFace, then the pipeline falls back to full-context prompting. Metrics record `rag_enabled=false` and the error, so the failure is visible rather than silent.

**How would you scale to 1,000 requests/day?**
Keep the API stateless, add a queue for long runs, cache search/embedding results, use a shared vector store only if reuse is desired, add rate limits and cost budgets, and emit traces/metrics per run. The per-run ephemeral index keeps tenant data isolated.

**What is the most important limitation?**
The groundedness score is structural, not semantic. A report can cite the right URL while misrepresenting it. The next step is an entailment model or a judge with a labeled evaluation set.
