from types import SimpleNamespace

import src.pipeline as pipeline
from src.rag import RagIndexStats, RetrievedChunk
from src.tools import ScrapedPage

SEARCH_TEXT = (
    "Title: Quantum error correction\n"
    "URL: https://example.com/quantum\n"
    "Snippet: A below-threshold surface code was demonstrated."
)
READER_TEXT = "The reader selected https://example.com/quantum and summarized the result."
REPORT = """# Quantum Error Correction

## 4. Key Findings

### Finding 1: Below-threshold operation was demonstrated

**Evidence:** A surface-code experiment reported a logical error rate below threshold, using repeated syndrome measurements across many cycles [1]. This is a substantial improvement over earlier demonstrations.

**Interpretation:** The result supports the path toward fault-tolerant quantum computing [1].

## 7. References

[1] Quantum error correction experiment. Nature (2026). https://example.com/quantum
"""


class FakeAgent:
    def __init__(self, content):
        self.content = content

    def invoke(self, _payload):
        return {"messages": [SimpleNamespace(content=self.content)]}


class FakeChain:
    def __init__(self, content):
        self.content = content

    def invoke(self, _payload):
        return self.content


class FakeRAG:
    def __init__(self, config):
        self.config = config

    def index(self, sources):
        return RagIndexStats(
            chunks_indexed=12,
            chunks_deduplicated=1,
            source_count=len(sources),
            raw_chars=sum(len(source["text"]) for source in sources),
            chunk_chars=6000,
            embedding_provider="fake",
            embedding_model="fake",
        )

    def retrieve(self, query, k=None):
        return [
            RetrievedChunk(
                content="Below-threshold surface code. " * 30,
                url="https://example.com/quantum",
                title="Quantum error correction",
                score=0.91,
                source_id=1,
            )
        ]


def test_pipeline_orchestration_metrics_and_evaluation(monkeypatch):
    monkeypatch.setattr(pipeline, "build_search_agent", lambda: FakeAgent(SEARCH_TEXT))
    monkeypatch.setattr(pipeline, "build_reader_agent", lambda: FakeAgent(READER_TEXT))
    monkeypatch.setattr(pipeline, "build_writer_chain", lambda: FakeChain(REPORT))
    monkeypatch.setattr(
        pipeline, "build_critic_chain", lambda: FakeChain("Overall Score: 8/10")
    )
    monkeypatch.setattr(
        pipeline,
        "scrape_urls",
        lambda urls, max_pages: [
            ScrapedPage(
                url="https://example.com/quantum",
                title="Quantum error correction",
                text="A long page about quantum error correction. " * 400,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "ResearchRAG", FakeRAG)

    state = pipeline.run_research_pipeline(
        "quantum error correction",
        max_sources=1,
        top_k=5,
        verbose=False,
        log_telemetry=False,
    )

    assert state["report"] == REPORT
    assert state["sources"][0]["url"] == "https://example.com/quantum"
    assert state["retrieved_chunks"][0]["score"] == 0.91
    assert state["metrics"]["rag_enabled"] is True
    assert state["metrics"]["chunks_indexed"] == 12
    assert state["metrics"]["tokens_saved"] > 0
    assert state["metrics"]["token_reduction_pct"] > 0
    assert state["metrics"]["critic_score"] == 8.0
    assert state["evaluation"]["groundedness_score"] == 1.0
    assert state["evaluation"]["passed"] is True
    assert state["metrics"]["latency_seconds"] >= 0


def test_pipeline_normalizes_list_content_from_gemini(monkeypatch):

    monkeypatch.setattr(
        pipeline, "build_search_agent", lambda: FakeAgent([{"type": "text", "text": SEARCH_TEXT}])
    )
    monkeypatch.setattr(
        pipeline, "build_reader_agent", lambda: FakeAgent([{"type": "text", "text": READER_TEXT}])
    )
    monkeypatch.setattr(
        pipeline, "build_writer_chain", lambda: FakeChain([{"type": "text", "text": REPORT}])
    )
    monkeypatch.setattr(
        pipeline,
        "build_critic_chain",
        lambda: FakeChain([{"type": "text", "text": "Overall Score: 8/10"}]),
    )
    monkeypatch.setattr(
        pipeline,
        "scrape_urls",
        lambda urls, max_pages: [
            ScrapedPage(
                url="https://example.com/quantum",
                title="Quantum error correction",
                text="A long page about quantum error correction. " * 400,
            )
        ],
    )
    monkeypatch.setattr(pipeline, "ResearchRAG", FakeRAG)

    state = pipeline.run_research_pipeline(
        "quantum error correction",
        max_sources=1,
        top_k=5,
        verbose=False,
        log_telemetry=False,
    )

    assert isinstance(state["search_results"], str)
    assert isinstance(state["scraped_content"], str)
    assert isinstance(state["report"], str)
    assert isinstance(state["feedback"], str)
    assert state["report"] == REPORT.strip()
    assert state["metrics"]["critic_score"] == 8.0
    assert state["evaluation"]["groundedness_score"] == 1.0


def test_scrape_sources_retries_until_minimum_successes(monkeypatch):
    urls = [f"https://example.com/{index}" for index in range(1, 7)]
    calls = []

    def fake_scrape(candidates, max_pages):
        batch = list(candidates)[:max_pages]
        calls.append(batch)
        pages = []
        for url in batch:
            ok = url.endswith(("1", "5", "6"))
            pages.append(
                ScrapedPage(
                    url=url,
                    title=url,
                    text="page text" if ok else "",
                    ok=ok,
                    error=None if ok else "blocked",
                )
            )
        return pages

    monkeypatch.setattr(pipeline, "scrape_urls", fake_scrape)

    pages = pipeline._scrape_sources(urls, max_sources=4, min_sources=3)

    assert sum(1 for page in pages if page.ok) >= 3
    assert len(calls) >= 2
    assert calls[0] == urls[:4]
    assert calls[1] == urls[4:6]
