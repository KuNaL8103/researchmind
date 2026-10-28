import hashlib
import math
from collections import Counter

from langchain_core.embeddings import Embeddings

from src.rag import (
    BatchedEmbeddings,
    EmbeddingBundle,
    RagConfig,
    ResearchRAG,
    RetrievedChunk,
    format_retrieved_context,
)


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings for offline tests."""

    dim = 64

    def _embed(self, text):
        vector = [0.0] * self.dim
        for word in text.lower().split():
            digest = int(hashlib.sha1(word.encode()).hexdigest()[:8], 16)
            vector[digest % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def build_rag(top_k=3):
    embedder = FakeEmbeddings()
    bundle = EmbeddingBundle(
        document=embedder, query=embedder, provider="fake", model="fake"
    )
    return ResearchRAG(
        RagConfig(chunk_size=1000, chunk_overlap=150, top_k=top_k),
        embeddings=bundle,
    )


def test_chunking_and_retrieval_keep_source_metadata():
    sources = [
        {
            "url": "https://example.com/quantum",
            "title": "Quantum error correction",
            "text": (
                "Quantum error correction is essential for fault-tolerant quantum "
                "computing. Surface codes arrange physical qubits on a lattice and "
                "use parity checks. " * 30
            ),
        },
        {
            "url": "https://example.com/fusion",
            "title": "Fusion energy",
            "text": (
                "Fusion energy research reached new milestones. Tokamaks confine "
                "plasma with magnetic fields. " * 30
            ),
        },
    ]

    rag = build_rag(top_k=3)
    stats = rag.index(sources)
    chunks = rag.retrieve("surface code quantum error correction", k=3)

    assert stats.chunks_indexed > 0
    assert stats.raw_chars > stats.chunk_chars
    assert len(chunks) == 3
    assert all(chunk.url for chunk in chunks)
    assert chunks[0].url == "https://example.com/quantum"
    assert 0.0 <= chunks[0].score <= 1.0
    assert len({chunk.url for chunk in chunks}) >= 2
    assert max(Counter(chunk.url for chunk in chunks).values()) <= 2


def test_format_groups_chunks_by_unique_source_url():
    chunks = [
        RetrievedChunk("a1", "https://a.example", "Source A", 0.9, source_id=1),
        RetrievedChunk("a2", "https://a.example", "Source A", 0.8, source_id=1),
        RetrievedChunk("b1", "https://b.example", "Source B", 0.7, source_id=2),
    ]

    context = format_retrieved_context(chunks)

    assert context.count("[SOURCE 1]") == 1
    assert context.count("[SOURCE 2]") == 1
    assert "URL: https://a.example" in context
    assert "URL: https://b.example" in context
    assert "Excerpt 1: a1" in context
    assert "Excerpt 2: a2" in context


class CountingEmbeddings(Embeddings):
    def __init__(self):
        self.calls: list[int] = []

    def embed_documents(self, texts, batch_size=None):
        self.calls.append(len(texts))
        return [[float(len(text))] for text in texts]

    def embed_query(self, text):
        return [float(len(text))]


def test_batched_embeddings_uses_bounded_batches():
    base = CountingEmbeddings()
    wrapper = BatchedEmbeddings(base, batch_size=10, concurrency=2)
    texts = [f"text-{index}" for index in range(27)]

    vectors = wrapper.embed_documents(texts)

    assert len(vectors) == 27
    assert wrapper.last_batch_count == 3
    assert sorted(base.calls) == [7, 10, 10]
    assert wrapper.last_embedding_seconds >= 0
