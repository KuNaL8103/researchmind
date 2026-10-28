"""Benchmark ResearchMind's RAG token savings and end-to-end latency.

Examples
--------
Token-only (no API keys required; uses local deterministic embeddings):
    python benchmarks/benchmark.py --token-only

Full pipeline (requires GOOGLE_API_KEY and TAVILY_API_KEY):
    python benchmarks/benchmark.py --runs 3 --topic "LLM agents 2026"

Results are written next to this script and printed as a markdown table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import count_tokens  # noqa: E402
from src.rag import (  # noqa: E402
    EmbeddingBundle,
    RagConfig,
    ResearchRAG,
    format_retrieved_context,
)
from src.tools import scrape_urls  # noqa: E402

load_dotenv()

DEFAULT_TOPIC = "large language model agents"
DEFAULT_URLS = [
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Quantum_error_correction",
    "https://en.wikipedia.org/wiki/Fusion_energy",
    "https://en.wikipedia.org/wiki/CRISPR_gene_editing",
]
OUTPUT_DIR = Path(__file__).resolve().parent


class HashingEmbeddings(Embeddings):
    """Deterministic offline embeddings used only for benchmark retrieval."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.lower().split():
            digest = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16)
            vector[digest % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _write(name: str, payload: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_token_benchmark(topic: str, urls: list[str], top_k: int, runs: int) -> dict:
    """Measure chunking/token reduction without calling Gemini or Tavily."""

    print(f"Fetching {len(urls)} pages for token benchmark…")
    pages = [page for page in scrape_urls(urls, max_pages=len(urls)) if page.ok]
    if not pages:
        raise SystemExit("Could not fetch any benchmark pages (network required).")

    sources = [{"url": p.url, "title": p.title, "text": p.text} for p in pages]
    embedder = HashingEmbeddings()
    bundle = EmbeddingBundle(
        document=embedder, query=embedder, provider="hashing-benchmark", model="offline"
    )
    config = RagConfig(
        chunk_size=1000, chunk_overlap=150, top_k=top_k, provider="huggingface"
    )

    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        rag = ResearchRAG(config, embeddings=bundle)
        stats = rag.index(sources)
        chunks = rag.retrieve(topic, k=top_k)
        elapsed = round(time.perf_counter() - started, 3)

        raw_tokens = count_tokens("\n\n".join(source["text"] for source in sources))
        retrieved_tokens = count_tokens(format_retrieved_context(chunks))
        samples.append(
            {
                "local_rag_seconds": elapsed,
                "raw_tokens": raw_tokens,
                "retrieved_tokens": retrieved_tokens,
                "reduction_pct": round(
                    (raw_tokens - retrieved_tokens) / raw_tokens * 100, 1
                )
                if raw_tokens
                else 0.0,
                "chunks_indexed": stats.chunks_indexed,
                "retrieved_chunks": len(chunks),
            }
        )

    reductions = [sample["reduction_pct"] for sample in samples]
    raw_tokens = samples[0]["raw_tokens"]
    retrieved_tokens = samples[0]["retrieved_tokens"]
    payload = {
        "mode": "token-only",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "urls": [page.url for page in pages],
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "top_k": top_k,
        "runs": runs,
        "raw_tokens": raw_tokens,
        "retrieved_tokens": retrieved_tokens,
        "reduction_pct": round(statistics.mean(reductions), 1),
        "reduction_range_pct": [min(reductions), max(reductions)],
        "chunks_indexed": samples[0]["chunks_indexed"],
        "retrieved_chunks": samples[0]["retrieved_chunks"],
        "local_rag_seconds_avg": round(
            statistics.mean(s["local_rag_seconds"] for s in samples), 3
        ),
        "note": (
            "Offline deterministic embeddings are used to select top-k chunks; "
            "token counts use cl100k_base and production chunk settings (1000/150). "
            "Latency is local chunk+embed+retrieve overhead, not LLM/network latency."
        ),
    }

    path = _write("token_benchmark.json", payload)
    print(f"\nToken benchmark ({len(pages)} pages, top-{top_k}, {runs} run(s))")
    print(f"  Raw scraped tokens:     {raw_tokens:,}")
    print(f"  RAG context tokens:     {retrieved_tokens:,}")
    print(f"  Token reduction:        {payload['reduction_pct']:.1f}%")
    print(f"  Chunks indexed:         {payload['chunks_indexed']:,}")
    print(f"  Local RAG overhead:     {payload['local_rag_seconds_avg']:.3f}s")
    print(f"  Results:                {path}")
    return payload


def run_pipeline_benchmark(topic: str, runs: int, max_sources: int, top_k: int) -> dict:
    """Measure real end-to-end latency and quality metrics."""

    import os

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise SystemExit(
            "Full-pipeline benchmark needs GOOGLE_API_KEY (or GEMINI_API_KEY). "
            "Set it in .env, or run: python benchmark.py --token-only"
        )
    if not os.getenv("TAVILY_API_KEY"):
        raise SystemExit(
            "Full-pipeline benchmark needs TAVILY_API_KEY. "
            "Set it in .env, or run: python benchmark.py --token-only"
        )

    from src.pipeline import run_research_pipeline

    runs_data = []
    for index in range(1, runs + 1):
        print(f"Run {index}/{runs}: {topic}")
        state = run_research_pipeline(
            topic,
            max_sources=max_sources,
            top_k=top_k,
            verbose=False,
            log_telemetry=False,
        )
        metrics = state["metrics"]
        evaluation = state["evaluation"]
        runs_data.append(
            {
                "latency_seconds": metrics["latency_seconds"],
                "stage_latency_seconds": metrics["stage_latency_seconds"],
                "token_reduction_pct": metrics["token_reduction_pct"],
                "tokens_saved": metrics["tokens_saved"],
                "critic_score": evaluation.get("critic_score"),
                "groundedness_score": evaluation.get("groundedness_score"),
                "chunks_indexed": metrics["chunks_indexed"],
                "rag_enabled": metrics["rag_enabled"],
            }
        )

    latencies = [run["latency_seconds"] for run in runs_data]
    reductions = [run["token_reduction_pct"] for run in runs_data]
    critic_scores = [run["critic_score"] for run in runs_data if run["critic_score"] is not None]
    groundedness = [
        run["groundedness_score"]
        for run in runs_data
        if run["groundedness_score"] is not None
    ]

    payload = {
        "mode": "full-pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "runs": runs,
        "max_sources": max_sources,
        "top_k": top_k,
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 2),
            "median": round(statistics.median(latencies), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "token_reduction_pct": round(statistics.mean(reductions), 1),
        "tokens_saved": round(statistics.mean(run["tokens_saved"] for run in runs_data)),
        "critic_score_mean": round(statistics.mean(critic_scores), 2) if critic_scores else None,
        "groundedness_mean": round(statistics.mean(groundedness), 3) if groundedness else None,
        "rag_enabled_runs": sum(1 for run in runs_data if run["rag_enabled"]),
        "raw_runs": runs_data,
    }

    path = _write("latest.json", payload)
    print("\nEnd-to-end benchmark")
    print(f"  Average latency:        {payload['latency_seconds']['mean']:.2f}s")
    print(f"  Median latency:         {payload['latency_seconds']['median']:.2f}s")
    print(f"  Average token reduction:{payload['token_reduction_pct']:.1f}%")
    print(f"  Average critic score:   {payload['critic_score_mean']}/10")
    print(f"  Average groundedness:   {payload['groundedness_mean']}")
    print(f"  Results:                {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-only", action="store_true", help="Run offline token benchmark")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-sources", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--urls", nargs="*", default=DEFAULT_URLS)
    args = parser.parse_args()

    if args.token_only:
        run_token_benchmark(args.topic, args.urls, args.top_k, args.runs)
    else:
        run_pipeline_benchmark(args.topic, args.runs, args.max_sources, args.top_k)


if __name__ == "__main__":
    main()
