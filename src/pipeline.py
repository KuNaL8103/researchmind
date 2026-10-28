"""ResearchMind's four-stage pipeline with vector RAG and telemetry.

Stages: Search -> Read (scrape + Chroma index) -> Write (top-k retrieval) -> Critic.
"""

from __future__ import annotations

import logging
from typing import Callable

from .agents import (
    build_critic_chain,
    build_reader_agent,
    build_search_agent,
    build_writer_chain,
)
from .evaluation import (
    PipelineTelemetry,
    content_to_text,
    count_tokens,
    evaluate_report,
    log_run,
)
from .rag import RagConfig, ResearchRAG, format_retrieved_context
from .tools import (
    ScrapedPage,
    canonical_url,
    collect_urls_from_messages,
    extract_urls,
    scrape_urls,
    tavily_search,
)

logger = logging.getLogger(__name__)

StageCallback = Callable[[str, str, dict], None]
DEFAULT_MAX_SOURCES = 4
MIN_DISTINCT_SOURCES = 3
DEFAULT_TOP_K = 5


def _emit(callback: StageCallback | None, stage: str, status: str, detail: dict | None = None) -> None:
    if callback is None:
        return
    try:
        callback(stage, status, detail or {})
    except Exception as exc:  # a UI callback must never break the pipeline
        logger.warning("Stage callback failed for %s/%s: %s", stage, status, exc)


def _unique_urls(primary: list[str], secondary: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in primary + secondary:
        key = canonical_url(url)
        if key and key not in seen:
            seen.add(key)
            ordered.append(url)
    return ordered


def _fallback_pages(reader_output: str, selected_url: str) -> list[ScrapedPage]:
    """Keep the pipeline useful if every scrape fails."""

    if not reader_output:
        return []
    return [
        ScrapedPage(
            url=selected_url or "",
            title="Reader agent summary",
            text=reader_output,
        )
    ]


def _format_full_pages(pages: list[ScrapedPage]) -> str:
    blocks = []
    for index, page in enumerate(pages, start=1):
        blocks.append(
            f"[SOURCE {index}] {page.title}\nURL: {page.url}\n{page.text}"
        )
    return "\n\n---\n\n".join(blocks)


def _scrape_sources(
    candidates: list[str],
    *,
    max_sources: int,
    min_sources: int,
) -> list[ScrapedPage]:
    """Scrape up to max_sources pages, retrying extra candidates until min_sources succeed."""

    pages = scrape_urls(candidates, max_pages=max_sources)
    attempted = {canonical_url(page.url) for page in pages}
    successes = [page for page in pages if page.ok]
    remaining = [
        url for url in candidates if canonical_url(url) not in attempted
    ]

    while len(successes) < min_sources and remaining:
        needed = min_sources - len(successes)
        batch = scrape_urls(remaining, max_pages=needed)
        if not batch:
            break
        pages.extend(batch)
        attempted.update(canonical_url(page.url) for page in batch)
        remaining = [
            url for url in remaining if canonical_url(url) not in attempted
        ]
        successes = [page for page in pages if page.ok]

    return pages


def run_research_pipeline(
    topic: str,
    *,
    max_sources: int = DEFAULT_MAX_SOURCES,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = True,
    on_stage: StageCallback | None = None,
    log_telemetry: bool = True,
) -> dict:
    """Run Search -> Read -> Write -> Critic and return the full run state.

    The original state keys (``search_results``, ``scraped_content``,
    ``report``, ``feedback``) are preserved.  New keys add RAG sources,
    retrieved chunks, evaluation, and quantitative metrics.
    """

    topic = (topic or "").strip()
    if not topic:
        raise ValueError("Research topic must not be empty")

    max_sources = max(1, int(max_sources))
    top_k = max(1, int(top_k))

    telemetry = PipelineTelemetry(topic)
    state: dict = {"topic": topic, "run_id": telemetry.run_id}

    def log(*args) -> None:
        if verbose:
            print(*args)

    # ── Stage 1: Search ──────────────────────────────────────────────────────
    log("\n" + " =" * 50)
    log("step 1 - search agent is working ...")
    log("=" * 50)
    _emit(on_stage, "search", "running")

    with telemetry.stage("search"):
        search_agent = build_search_agent()
        search_result = search_agent.invoke(
            {
                "messages": [
                    ("user", f"Find recent, reliable and detailed information about: {topic}")
                ]
            }
        )
        search_messages = search_result["messages"]
        state["search_results"] = content_to_text(search_messages[-1].content)

        # Final agent prose often omits URLs. Collect them from every tool
        # message, then fall back to a direct Tavily search if the pool is thin.
        candidate_urls = collect_urls_from_messages(search_messages)
        if len(candidate_urls) < max_sources:
            fallback_results = tavily_search(
                topic, max_results=max(8, max_sources + 4)
            )
            candidate_urls = _unique_urls(
                candidate_urls, [result["url"] for result in fallback_results]
            )
        state["search_candidates"] = candidate_urls

    log("\n search result ", state["search_results"])
    log(f"\n source candidates ({len(state['search_candidates'])}): ", state["search_candidates"])
    _emit(
        on_stage,
        "search",
        "done",
        {"chars": len(state["search_results"]), "urls": len(state["search_candidates"])},
    )

    # ── Stage 2: Read (agent selection + deterministic scraping) ─────────────
    log("\n" + " =" * 50)
    log("step 2 - Reader agent is scraping top resources ...")
    log("=" * 50)
    _emit(on_stage, "read", "running")

    with telemetry.stage("read"):
        candidate_block = "\n".join(
            f"- {url}" for url in state["search_candidates"][:8]
        )
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"Based on the following search results about '{topic}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{state['search_results'][:1600]}\n\n"
                        f"Candidate URLs:\n{candidate_block}",
                    )
                ]
            }
        )
        reader_messages = reader_result["messages"]
        state["scraped_content"] = content_to_text(reader_messages[-1].content)

        # Prefer the URL the reader named in its final answer, but keep the full
        # candidate pool so we can scrape 3-5 independent sources.
        reader_urls = collect_urls_from_messages(reader_messages)
        selected_urls = extract_urls(state["scraped_content"])[:1] or reader_urls[:1]
        candidates = _unique_urls(
            selected_urls + reader_urls, state["search_candidates"]
        )
        min_sources = min(MIN_DISTINCT_SOURCES, max_sources)
        pages = _scrape_sources(
            candidates, max_sources=max_sources, min_sources=min_sources
        )
        if not any(page.ok for page in pages):
            pages = _fallback_pages(
                state["scraped_content"],
                selected_urls[0] if selected_urls else "",
            )

    state["sources"] = [
        {
            "url": page.url,
            "title": page.title,
            "char_count": page.char_count,
            "ok": page.ok,
            "error": page.error,
        }
        for page in pages
    ]
    state["source_texts"] = [page.text for page in pages if page.text]

    log("\nscraped content: \n", state["scraped_content"])
    _emit(
        on_stage,
        "read",
        "done",
        {
            "sources": sum(1 for page in pages if page.ok),
            "chars": sum(page.char_count for page in pages),
        },
    )

    # ── Stage 2b: Chunk + embed into ephemeral ChromaDB ──────────────────────
    rag_config = RagConfig.from_env(top_k=top_k)
    rag_stats = None
    rag_error = None
    retrieved = []
    rag_enabled = False
    _emit(on_stage, "rag", "running", {"top_k": top_k})

    try:
        with telemetry.stage("rag_index"):
            rag = ResearchRAG(rag_config)
            rag_stats = rag.index(
                [
                    {"url": page.url, "title": page.title, "text": page.text}
                    for page in pages
                    if page.ok and page.text
                ]
            )

        with telemetry.stage("rag_retrieve"):
            retrieved = rag.retrieve(topic, k=top_k)

        rag_enabled = bool(retrieved)
        _emit(
            on_stage,
            "rag",
            "done",
            {
                "chunks": rag_stats.chunks_indexed,
                "retrieved": len(retrieved),
                "provider": rag_stats.embedding_provider,
            },
        )
    except Exception as exc:
        rag_error = str(exc)
        logger.warning("RAG layer disabled: %s", exc)
        _emit(on_stage, "rag", "error", {"error": rag_error})

    # ── Stage 3: Write with retrieved evidence only ──────────────────────────
    log("\n" + " =" * 50)
    log("step 3 - Writer is drafting the report ...")
    log("=" * 50)
    _emit(on_stage, "write", "running")

    search_block = state["search_results"]
    reader_brief = state["scraped_content"]

    if retrieved:
        evidence_block = format_retrieved_context(retrieved)
        evidence_label = (
            f"RETRIEVED EVIDENCE (top-{len(retrieved)} semantically relevant chunks "
            f"from ChromaDB; cite with the SOURCE numbers and URLs below)"
        )
    else:
        evidence_block = _format_full_pages(pages)
        evidence_label = "FULL SCRAPED CONTENT (RAG unavailable; full-context fallback)"

    research_combined = (
        f"SEARCH RESULTS:\n{search_block}\n\n"
        f"READER BRIEF (unverified synthesis; cite only the evidence below):\n{reader_brief}\n\n"
        f"{evidence_label}:\n{evidence_block}"
    )

    with telemetry.stage("write"):
        state["report"] = content_to_text(
            build_writer_chain().invoke({"topic": topic, "research": research_combined})
        )

    log("\n Final Report\n", state["report"])
    _emit(on_stage, "write", "done", {"chars": len(state["report"])})

    # ── Stage 4: Critic ──────────────────────────────────────────────────────
    log("\n" + " =" * 50)
    log("step 4 - critic is reviewing the report ")
    log("=" * 50)
    _emit(on_stage, "critic", "running")

    with telemetry.stage("critic"):
        state["feedback"] = content_to_text(
            build_critic_chain().invoke({"report": state["report"]})
        )

    log("\n critic report \n", state["feedback"])
    _emit(on_stage, "critic", "done", {})

    # ── Quantitative evaluation ──────────────────────────────────────────────
    retrieved_urls = list(dict.fromkeys(chunk.url for chunk in retrieved if chunk.url))
    allowed_urls = _unique_urls(
        [source["url"] for source in state["sources"] if source["url"]],
        state["search_candidates"] + retrieved_urls,
    )
    evaluation = evaluate_report(
        report=state["report"],
        feedback=state["feedback"],
        allowed_urls=allowed_urls,
        retrieved_urls=retrieved_urls,
    )

    raw_scraped_text = "\n\n".join(page.text for page in pages if page.text)
    baseline_text = "\n\n".join([search_block, reader_brief, raw_scraped_text])
    baseline_tokens = count_tokens(baseline_text)
    optimized_tokens = count_tokens(research_combined)
    tokens_saved = max(0, baseline_tokens - optimized_tokens)

    latency_seconds = telemetry.latency_seconds
    metrics = telemetry.base_metrics()
    metrics.update(
        {
            "latency_seconds": latency_seconds,
            "status": "success",
            "sources_requested": max_sources,
            "sources_scraped": sum(1 for page in pages if page.ok),
            "sources_failed": sum(1 for page in pages if not page.ok),
            "search_candidates": len(state.get("search_candidates", [])),
            "rag_enabled": rag_enabled,
            "rag_error": rag_error,
            "chunk_size": rag_config.chunk_size,
            "chunk_overlap": rag_config.chunk_overlap,
            "top_k": top_k,
            "retrieval_mode": rag_config.retrieval_mode,
            "max_chunks_per_source": rag_config.max_chunks_per_source,
            "chunks_indexed": rag_stats.chunks_indexed if rag_stats else 0,
            "chunks_deduplicated": rag_stats.chunks_deduplicated if rag_stats else 0,
            "retrieved_chunks": len(retrieved),
            "retrieved_sources": len(retrieved_urls),
            "embedding_batches": rag_stats.embedding_batches if rag_stats else 0,
            "embedding_seconds": rag_stats.embedding_seconds if rag_stats else 0.0,
            "raw_scraped_tokens": count_tokens(raw_scraped_text),
            "baseline_context_tokens": baseline_tokens,
            "optimized_context_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "token_reduction_pct": (
                round(tokens_saved / baseline_tokens * 100, 1) if baseline_tokens else 0.0
            ),
            "embedding_provider": rag_stats.embedding_provider if rag_stats else None,
            "embedding_model": rag_stats.embedding_model if rag_stats else None,
            "critic_score": evaluation.get("critic_score"),
            "groundedness_score": evaluation.get("groundedness_score"),
            "citation_precision": evaluation.get("citation_precision"),
            "claim_coverage": evaluation.get("claim_coverage"),
            "retrieved_url_coverage": evaluation.get("retrieved_url_coverage"),
            "groundedness_passed": evaluation.get("passed"),
        }
    )

    state["retrieved_chunks"] = [chunk.to_dict() for chunk in retrieved]
    state["evaluation"] = evaluation
    state["metrics"] = metrics

    if log_telemetry:
        try:
            log_run(metrics)
        except Exception as exc:  # logging must never break a successful run
            logger.warning("Failed to persist telemetry: %s", exc)

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    result = run_research_pipeline(topic)
    print("\nMetrics:", result["metrics"])
    print("Evaluation:", result["evaluation"])
