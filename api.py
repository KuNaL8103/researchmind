"""FastAPI wrapper around the ResearchMind pipeline.

Run locally:
    uvicorn api:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.pipeline import run_research_pipeline

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("researchmind.api")


# ── Pydantic request/response models ─────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Research question or topic.",
        examples=["Quantum computing error correction in 2026"],
    )
    max_sources: int = Field(
        4,
        ge=1,
        le=10,
        description="Number of search-result pages to scrape and index.",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of semantically relevant chunks retrieved for the writer.",
    )


class SourceModel(BaseModel):
    url: str = ""
    title: str = ""
    char_count: int = 0
    ok: bool = True
    error: str | None = None


class RetrievedChunkModel(BaseModel):
    content: str
    url: str = ""
    title: str = ""
    score: float = 0.0
    chunk_index: int = 0
    source_id: int = 0


class EvaluationModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    critic_score: float | None = None
    groundedness_score: float = 0.0
    citation_precision: float = 0.0
    claim_coverage: float = 0.0
    citations_found: int = 0
    reference_count: int = 0
    reference_urls: list[str] = Field(default_factory=list)
    verified_urls: list[str] = Field(default_factory=list)
    unsupported_urls: list[str] = Field(default_factory=list)
    retrieved_urls_cited: list[str] = Field(default_factory=list)
    retrieved_url_coverage: float = 0.0
    uncited_claim_count: int = 0
    verdict: str = "Not evaluated"
    passed: bool = False
    notes: list[str] = Field(default_factory=list)


class MetricsModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str = ""
    topic: str = ""
    timestamp: str = ""
    status: str = "success"
    latency_seconds: float = 0.0
    stage_latency_seconds: dict[str, float] = Field(default_factory=dict)

    sources_requested: int = 0
    sources_scraped: int = 0
    sources_failed: int = 0
    search_candidates: int = 0

    rag_enabled: bool = False
    rag_error: str | None = None
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    retrieval_mode: str = "diversified"
    max_chunks_per_source: int = 2
    chunks_indexed: int = 0
    chunks_deduplicated: int = 0
    retrieved_chunks: int = 0
    retrieved_sources: int = 0
    embedding_batches: int = 0
    embedding_seconds: float = 0.0

    raw_scraped_tokens: int = 0
    baseline_context_tokens: int = 0
    optimized_context_tokens: int = 0
    tokens_saved: int = 0
    token_reduction_pct: float = 0.0

    embedding_provider: str | None = None
    embedding_model: str | None = None

    critic_score: float | None = None
    groundedness_score: float | None = None
    citation_precision: float | None = None
    claim_coverage: float | None = None
    retrieved_url_coverage: float | None = None
    groundedness_passed: bool | None = None


class ResearchResponse(BaseModel):
    run_id: str
    topic: str
    report: str
    feedback: str
    sources: list[SourceModel] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkModel] = Field(default_factory=list)
    evaluation: EvaluationModel
    metrics: MetricsModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


app = FastAPI(
    title="ResearchMind API",
    version="2.0.0",
    description=(
        "Four-stage LangChain research pipeline (Search → Read → Write → Critic) "
        "with ChromaDB vector RAG, citation-groundedness evaluation, and telemetry."
    ),
)

origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "ResearchMind API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check that does not require API keys or network access."""

    return HealthResponse(status="ok", service="ResearchMind API", version=app.version)


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    """Run the full research pipeline and return report, sources, and metrics."""

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_API_KEY (or GEMINI_API_KEY) is not configured",
        )
    if not os.getenv("TAVILY_API_KEY"):
        raise HTTPException(status_code=503, detail="TAVILY_API_KEY is not configured")

    try:
        state = run_research_pipeline(
            request.topic,
            max_sources=request.max_sources,
            top_k=request.top_k,
            verbose=False,
            log_telemetry=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Research pipeline failed for topic=%r", request.topic)
        raise HTTPException(
            status_code=500, detail=f"Research pipeline failed: {type(exc).__name__}"
        ) from exc

    metrics = state["metrics"]
    return ResearchResponse(
        run_id=state["run_id"],
        topic=state["topic"],
        report=state["report"],
        feedback=state["feedback"],
        sources=[SourceModel(**source) for source in state.get("sources", [])],
        retrieved_chunks=[
            RetrievedChunkModel(**chunk) for chunk in state.get("retrieved_chunks", [])
        ],
        evaluation=EvaluationModel(**state["evaluation"]),
        metrics=MetricsModel(**metrics),
    )
