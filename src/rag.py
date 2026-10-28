"""Ephemeral ChromaDB vector RAG layer for ResearchMind.

Flow:
    scraped pages -> RecursiveCharacterTextSplitter(1000/150) -> Gemini or
    HuggingFace embeddings -> in-memory Chroma collection -> top-k retrieval.

Every chunk keeps its source URL and title in metadata so the writer can cite
retrieved evidence rather than receiving entire raw pages in its prompt.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5


@dataclass
class RagConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    fetch_k: int = 20
    max_chunks_per_source: int = 2
    retrieval_mode: str = "diversified"
    embedding_batch_size: int = 32
    embedding_concurrency: int = 4
    provider: str = "gemini"
    gemini_model: str = "models/gemini-embedding-001"
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimensions: int | None = None

    @classmethod
    def from_env(cls, top_k: int | None = None) -> "RagConfig":
        dimensions = os.getenv("EMBEDDING_DIMENSIONS", "").strip()
        return cls(
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)),
            top_k=top_k or int(os.getenv("RAG_TOP_K", DEFAULT_TOP_K)),
            fetch_k=int(os.getenv("RAG_FETCH_K", 20)),
            max_chunks_per_source=int(os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", 2)),
            retrieval_mode=os.getenv("RAG_RETRIEVAL_MODE", "diversified").strip().lower(),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", 32)),
            embedding_concurrency=int(os.getenv("EMBEDDING_CONCURRENCY", 4)),
            provider=os.getenv("EMBEDDING_PROVIDER", "gemini").strip().lower(),
            gemini_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
            hf_model=os.getenv(
                "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            dimensions=int(dimensions) if dimensions else None,
        )


@dataclass
class EmbeddingBundle:
    """Document and query embedders (Gemini uses different task types)."""

    document: Embeddings
    query: Embeddings
    provider: str
    model: str


class BatchedEmbeddings(Embeddings):
    """Wrap an embedder with explicit, bounded batches.

    Chroma asks LangChain for all document embeddings in one call. This wrapper
    keeps that single call semantics but splits it into efficient batches,
    optionally issuing multiple batches concurrently. It also records timing so
    ``rag_index`` latency is observable instead of a black box.
    """

    def __init__(
        self,
        base: Embeddings,
        *,
        batch_size: int = 32,
        concurrency: int = 4,
    ) -> None:
        self.base = base
        self.batch_size = max(1, batch_size)
        self.concurrency = max(1, concurrency)
        self.last_batch_count = 0
        self.last_embedding_seconds = 0.0

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            # Gemini accepts an explicit batch_size; other embedders may not.
            return self.base.embed_documents(batch, batch_size=len(batch))
        except TypeError:
            return self.base.embed_documents(batch)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        texts = list(texts)
        batches = [
            texts[index : index + self.batch_size]
            for index in range(0, len(texts), self.batch_size)
        ]
        self.last_batch_count = len(batches)

        if len(batches) <= 1:
            vectors = self._embed_batch(batches[0]) if batches else []
        else:
            workers = min(self.concurrency, len(batches))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(self._embed_batch, batches))
            vectors = [vector for batch in results for vector in batch]

        self.last_embedding_seconds = round(time.perf_counter() - started, 3)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(text)


@dataclass
class RetrievedChunk:
    content: str
    url: str
    title: str
    score: float
    chunk_index: int = 0
    source_id: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RagIndexStats:
    chunks_indexed: int
    chunks_deduplicated: int
    source_count: int
    raw_chars: int
    chunk_chars: int
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_batches: int = 0
    embedding_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _gemini_embeddings(config: RagConfig) -> EmbeddingBundle:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    common: dict[str, Any] = {
        "model": config.gemini_model,
        "google_api_key": api_key,
        "vertexai": False,
    }
    if config.dimensions:
        common["output_dimensionality"] = config.dimensions

    document_embedder = GoogleGenerativeAIEmbeddings(
        **common, task_type="retrieval_document"
    )
    query_embedder = GoogleGenerativeAIEmbeddings(
        **common, task_type="retrieval_query"
    )
    return EmbeddingBundle(
        document=document_embedder,
        query=query_embedder,
        provider="gemini",
        model=config.gemini_model,
    )


def _huggingface_embeddings(config: RagConfig) -> EmbeddingBundle:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:  # optional dependency
        raise RuntimeError(
            "Local embeddings require langchain-huggingface and sentence-transformers. "
            "Install them with: pip install -r requirements-local.txt"
        ) from exc

    embedder = HuggingFaceEmbeddings(
        model_name=config.hf_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return EmbeddingBundle(
        document=embedder,
        query=embedder,
        provider="huggingface",
        model=config.hf_model,
    )


def get_embeddings(config: RagConfig | None = None) -> EmbeddingBundle:
    """Return Gemini embeddings, falling back to local HuggingFace embeddings.

    Set ``EMBEDDING_PROVIDER=gemini`` (default) or ``huggingface``.  If Gemini
    is selected but unavailable (no key / API error), the HuggingFace fallback
    is attempted automatically so the pipeline can keep running offline.
    """

    config = config or RagConfig.from_env()
    provider = (config.provider or "gemini").lower()

    if provider in {"gemini", "google"}:
        try:
            return _gemini_embeddings(config)
        except Exception as exc:
            logger.warning("Gemini embeddings unavailable (%s); trying HuggingFace", exc)
            try:
                return _huggingface_embeddings(config)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Gemini embeddings failed ({exc}); HuggingFace fallback also "
                    f"failed ({fallback_exc})"
                ) from fallback_exc

    if provider in {"huggingface", "hf", "local", "sentence-transformers"}:
        return _huggingface_embeddings(config)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={config.provider!r}. "
        "Use 'gemini' or 'huggingface'."
    )


class ResearchRAG:
    """In-memory Chroma index for a single research run."""

    def __init__(
        self,
        config: RagConfig | None = None,
        embeddings: EmbeddingBundle | None = None,
    ) -> None:
        self.config = config or RagConfig.from_env()
        self.embeddings = embeddings or get_embeddings(self.config)
        concurrency = (
            self.config.embedding_concurrency
            if self.embeddings.provider == "gemini"
            else 1
        )
        self.document_embeddings = BatchedEmbeddings(
            self.embeddings.document,
            batch_size=self.config.embedding_batch_size,
            concurrency=concurrency,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self.store = None
        self.stats = RagIndexStats(
            chunks_indexed=0,
            chunks_deduplicated=0,
            source_count=0,
            raw_chars=0,
            chunk_chars=0,
            embedding_provider=self.embeddings.provider,
            embedding_model=self.embeddings.model,
            embedding_batches=0,
            embedding_seconds=0.0,
        )

    def index(self, sources: Sequence[dict]) -> RagIndexStats:
        """Chunk scraped pages and embed them into an ephemeral Chroma collection."""

        documents: list[Document] = []
        seen: set[str] = set()
        duplicates = 0
        raw_chars = 0

        for source_id, source in enumerate(sources, start=1):
            text = (source.get("text") or "").strip()
            if not text:
                continue
            raw_chars += len(text)
            url = source.get("url") or ""
            title = source.get("title") or url

            for chunk_index, chunk in enumerate(self.splitter.split_text(text)):
                fingerprint = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
                if fingerprint in seen:
                    duplicates += 1
                    continue
                seen.add(fingerprint)
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": url,
                            "url": url,
                            "title": title,
                            "chunk_index": chunk_index,
                            "source_id": source_id,
                        },
                    )
                )

        self.stats = RagIndexStats(
            chunks_indexed=len(documents),
            chunks_deduplicated=duplicates,
            source_count=len({s.get("url") for s in sources if s.get("text")}),
            raw_chars=raw_chars,
            chunk_chars=sum(len(doc.page_content) for doc in documents),
            embedding_provider=self.embeddings.provider,
            embedding_model=self.embeddings.model,
            embedding_batches=0,
            embedding_seconds=0.0,
        )

        if not documents:
            self.store = None
            return self.stats

        # Imported lazily so the rest of the app can run without Chroma.
        from chromadb.config import Settings
        from langchain_chroma import Chroma

        collection_name = f"research-{uuid.uuid4().hex[:12]}"
        self.store = Chroma(
            collection_name=collection_name,
            embedding_function=self.document_embeddings,
            client_settings=Settings(
                anonymized_telemetry=False, is_persistent=False
            ),
        )
        self.store.add_documents(documents)
        self.stats.embedding_batches = self.document_embeddings.last_batch_count
        self.stats.embedding_seconds = self.document_embeddings.last_embedding_seconds
        logger.info(
            "Indexed %s chunks from %s sources in %s (%s embedding batch(es), %.2fs)",
            len(documents),
            self.stats.source_count,
            collection_name,
            self.stats.embedding_batches,
            self.stats.embedding_seconds,
        )
        return self.stats

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """Semantic retrieval with per-URL diversity so citations span sources."""

        if not self.store or not query:
            return []

        k = min(k or self.config.top_k, self.stats.chunks_indexed)
        if k <= 0:
            return []

        query_vector = self.embeddings.query.embed_query(query)
        fetch_k = min(
            self.stats.chunks_indexed, max(self.config.fetch_k, k * 4)
        )

        mode = (self.config.retrieval_mode or "diversified").lower()
        if mode == "mmr":
            mmr_documents = self.store.max_marginal_relevance_search_by_vector(
                query_vector, k=fetch_k, fetch_k=fetch_k, lambda_mult=0.5
            )
            scored = self.store.similarity_search_by_vector_with_relevance_scores(
                query_vector, k=fetch_k
            )
            score_map = {_document_key(doc): score for doc, score in scored}
            candidates = [
                (doc, float(score_map.get(_document_key(doc), 0.0)))
                for doc in mmr_documents
            ]
        else:
            candidates = self.store.similarity_search_by_vector_with_relevance_scores(
                query_vector, k=fetch_k
            )

        if mode == "similarity":
            selected = candidates[:k]
        else:
            selected = self._diversify(candidates, k)

        chunks: list[RetrievedChunk] = []
        for document, score in selected:
            metadata = document.metadata or {}
            chunks.append(
                RetrievedChunk(
                    content=document.page_content,
                    url=metadata.get("url") or metadata.get("source") or "",
                    title=metadata.get("title") or metadata.get("url") or "",
                    score=round(float(score), 4),
                    chunk_index=int(metadata.get("chunk_index") or 0),
                    source_id=int(metadata.get("source_id") or 0),
                )
            )
        return chunks

    def _diversify(
        self,
        candidates: Sequence[tuple[Document, float]],
        k: int,
    ) -> list[tuple[Document, float]]:
        """Select top-k while limiting chunks from any single source URL."""

        cap = max(1, self.config.max_chunks_per_source)
        selected: list[tuple[Document, float]] = []
        selected_keys: set[str] = set()
        per_url: dict[str, int] = {}

        for document, score in candidates:
            key = _document_key(document)
            url = (document.metadata or {}).get("url") or (document.metadata or {}).get("source") or ""
            if key in selected_keys or per_url.get(url, 0) >= cap:
                continue
            selected.append((document, score))
            selected_keys.add(key)
            per_url[url] = per_url.get(url, 0) + 1
            if len(selected) >= k:
                return selected

        # If the corpus has fewer sources than k/cap, fill remaining slots
        # rather than returning fewer than k chunks.
        for document, score in candidates:
            key = _document_key(document)
            if key in selected_keys:
                continue
            selected.append((document, score))
            selected_keys.add(key)
            if len(selected) >= k:
                break

        return selected[:k]


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    url = metadata.get("url") or metadata.get("source") or ""
    return hashlib.sha1(f"{url}\n{document.page_content}".encode("utf-8")).hexdigest()


def format_retrieved_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Format evidence with one [SOURCE N] number per unique URL.

    Multiple chunks from the same page are grouped under the same source
    number, so the writer's references cannot point at duplicate URLs.
    """

    grouped: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.url, []).append(chunk)

    blocks = []
    for index, (url, group) in enumerate(grouped.items(), start=1):
        title = next((chunk.title for chunk in group if chunk.title), url)
        relevance = max(chunk.score for chunk in group)
        excerpts = "\n\n".join(
            f"Excerpt {position}: {chunk.content}"
            for position, chunk in enumerate(group, start=1)
        )
        blocks.append(
            f"[SOURCE {index}] {title}\n"
            f"URL: {url}\n"
            f"Relevance: {relevance:.3f}\n"
            f"{excerpts}"
        )
    return "\n\n---\n\n".join(blocks)
