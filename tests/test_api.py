from fastapi.testclient import TestClient

import api


def fake_state():
    return {
        "run_id": "abc123",
        "topic": "quantum error correction",
        "report": "# Report\n\nFinding [1].\n\n## 7. References\n[1] Source. https://example.com/a",
        "feedback": "Overall Score: 8/10",
        "sources": [
            {
                "url": "https://example.com/a",
                "title": "Source A",
                "char_count": 1200,
                "ok": True,
                "error": None,
            }
        ],
        "retrieved_chunks": [
            {
                "content": "evidence",
                "url": "https://example.com/a",
                "title": "Source A",
                "score": 0.9,
                "chunk_index": 0,
                "source_id": 1,
            }
        ],
        "evaluation": {
            "critic_score": 8.0,
            "groundedness_score": 0.9,
            "citation_precision": 1.0,
            "claim_coverage": 0.8,
            "reference_urls": ["https://example.com/a"],
            "verified_urls": ["https://example.com/a"],
            "unsupported_urls": [],
            "passed": True,
            "verdict": "Strongly grounded",
        },
        "metrics": {
            "run_id": "abc123",
            "topic": "quantum error correction",
            "latency_seconds": 12.3,
            "stage_latency_seconds": {"search": 2.0, "write": 6.0},
            "token_reduction_pct": 80.0,
            "tokens_saved": 5000,
            "chunks_indexed": 10,
            "retrieved_chunks": 1,
            "rag_enabled": True,
        },
    }


def test_health_does_not_require_keys():
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_research_returns_structured_response(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setattr(api, "run_research_pipeline", lambda *args, **kwargs: fake_state())

    client = TestClient(api.app)
    response = client.post("/research", json={"topic": "quantum error correction"})

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "abc123"
    assert body["evaluation"]["critic_score"] == 8.0
    assert body["metrics"]["latency_seconds"] == 12.3
    assert body["sources"][0]["url"] == "https://example.com/a"


def test_research_rejects_missing_topic(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    client = TestClient(api.app)
    assert client.post("/research", json={"topic": "x"}).status_code == 422
