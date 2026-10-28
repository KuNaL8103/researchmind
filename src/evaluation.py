"""Evaluation, groundedness checks, telemetry, and text normalization.

This module consolidates three closely related concerns:

* normalizing provider-specific LLM message content into plain text
* token accounting, stage timers, and JSONL run logging
* critic-score parsing plus citation/groundedness verification
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)


# ── Text normalization ────────────────────────────────────────────────────────

def content_to_text(content: Any) -> str:
    """Flatten common LangChain/LLM content shapes into plain text.

    LangChain providers do not all return ``AIMessage.content`` as a string.
    Gemini can return a list of content blocks such as::

        [{"type": "text", "text": "final answer"}]

    Passing that list into regex, tokenization, or URL extraction raises
    ``TypeError: expected string or bytes-like object, got 'list'``.
    """

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = [content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            if key in content:
                return content_to_text(content[key])
        return ""
    return str(content)


# ── Token accounting and telemetry ───────────────────────────────────────────

@lru_cache(maxsize=1)
def _token_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # offline / optional dependency
        return None


def count_tokens(text: str) -> int:
    """Count tokens with tiktoken, falling back to a ~4-chars/token estimate."""

    text = content_to_text(text)
    if not text:
        return 0
    encoder = _token_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, (len(text) + 3) // 4)


class PipelineTelemetry:
    """Tracks wall-clock latency per pipeline stage for one research run."""

    def __init__(self, topic: str) -> None:
        self.run_id = uuid.uuid4().hex[:12]
        self.topic = topic
        self.started_at = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round(time.perf_counter() - start, 3)

    @property
    def latency_seconds(self) -> float:
        return round(time.perf_counter() - self.started_at, 3)

    def base_metrics(self) -> dict:
        """Metrics that are known as soon as the run starts."""

        return {
            "run_id": self.run_id,
            "topic": self.topic,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage_latency_seconds": dict(self.stages),
        }


def log_run(metrics: dict, *, log_dir: str | None = None) -> Path | None:
    """Append one JSON line per run. Logging failures never break the pipeline."""

    payload = {"event": "research_run", **metrics}
    logger.info("research_run %s", json.dumps(payload, default=str))

    directory = Path(log_dir or os.getenv("TELEMETRY_LOG_DIR", "logs"))
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "researchmind_runs.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")
        return path
    except OSError as exc:
        logger.warning("Could not write telemetry log: %s", exc)
        return None


# ── Evaluation and groundedness ──────────────────────────────────────────────

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
CRITIC_SCORE_RE = re.compile(
    r"overall\s*score\s*:?\s*\**\s*(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*10",
    re.IGNORECASE,
)
SECTION_RE = r"(?im)^#{1,6}\s*(?:\d+[.)]?\s*)?{name}\s*$"
FINDINGS_HEADING_RE = re.compile(
    r"(?im)^#{1,6}\s*(?:\d+[.)]?\s*)?(?:key\s+findings|findings)\b[^\n]*$"
)
BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
TABLE_ROW_RE = re.compile(r"^\s*\|")
HEADING_LINE_RE = re.compile(r"^\s*#{1,6}\s+")
SEPARATOR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def _section_pattern(name: str) -> str:
    return SECTION_RE.replace("{name}", name)


@dataclass
class GroundednessResult:
    groundedness_score: float = 0.0
    citation_precision: float = 0.0
    claim_coverage: float = 0.0
    citations_found: int = 0
    reference_count: int = 0
    reference_urls: list[str] = field(default_factory=list)
    verified_urls: list[str] = field(default_factory=list)
    unsupported_urls: list[str] = field(default_factory=list)
    retrieved_urls_cited: list[str] = field(default_factory=list)
    retrieved_url_coverage: float = 0.0
    uncited_claim_count: int = 0
    critic_score: float | None = None
    verdict: str = "Not evaluated"
    passed: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_critic_score(feedback: str | None) -> float | None:
    """Extract ``Overall Score: 8/10`` from the critic output."""

    feedback = content_to_text(feedback)
    if not feedback:
        return None
    match = CRITIC_SCORE_RE.search(feedback)
    if not match:
        return None
    try:
        score = float(match.group(1))
    except ValueError:
        return None
    return max(0.0, min(10.0, score))


def _extract_markers(text: str) -> set[int]:
    markers: set[int] = set()
    for match in CITATION_RE.finditer(text):
        for part in match.group(1).split(","):
            try:
                markers.add(int(part.strip()))
            except ValueError:
                continue
    return markers


def _references_section(report: str) -> tuple[str, str]:
    """Split the report into (body, references) using the References heading."""

    match = re.search(_section_pattern(r"references"), report)
    if not match:
        return report, ""
    body = report[: match.start()]
    references = report[match.end() :]
    # Stop at the next top-level heading if the model added content after refs.
    next_heading = re.search(r"(?m)^#{1,2}\s+\S", references)
    if next_heading:
        references = references[: next_heading.start()]
    return body, references


def _reference_entries(references: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for line in references.splitlines():
        match = re.match(r"^\s*[-*]?\s*\[(\d+)\]\s*(.+)$", line)
        if match:
            entries.append((int(match.group(1)), match.group(2).strip()))
    return entries


def _is_claim_unit(text: str) -> bool:
    cleaned = BULLET_RE.sub("", text).strip()
    if len(cleaned) < 30:
        return False
    if re.fullmatch(r"[\W_]+", cleaned):
        return False
    lowered = cleaned.lower().strip("*: ")
    if lowered.startswith(("references", "table of contents", "contents")):
        return False
    if re.fullmatch(
        r"(?:confidence|source|reliability)\s*:?\s*(?:high|medium|low)?\.?",
        lowered,
    ):
        return False
    return True


def _claim_blocks(report: str) -> list[str]:
    """Return claim-bearing paragraphs and bullet points from Key Findings."""

    body = report
    findings = FINDINGS_HEADING_RE.search(report)
    if findings:
        rest = report[findings.end() :]
        next_section = re.search(r"(?m)^#{1,2}\s+\S", rest)
        body = rest[: next_section.start()] if next_section else rest
    else:
        body, _ = _references_section(report)

    units: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            unit = " ".join(buffer).strip()
            if _is_claim_unit(unit):
                units.append(unit)
            buffer.clear()

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or HEADING_LINE_RE.match(line) or SEPARATOR_RE.match(line):
            flush_buffer()
            continue
        if BULLET_RE.match(line) or TABLE_ROW_RE.match(line):
            flush_buffer()
            if _is_claim_unit(line):
                units.append(line)
            continue
        buffer.append(line)

    flush_buffer()
    return units


def evaluate_groundedness(
    report: str,
    allowed_urls: Iterable[str],
    retrieved_urls: Iterable[str] | None = None,
    critic_feedback: str | None = None,
) -> GroundednessResult:
    """Verify that the report cites real, provided sources.

    The score combines two signals:
      * citation precision - share of reference URLs that exist in the evidence
      * claim coverage - share of claim units carrying an inline [n]
    """

    # Imported lazily to avoid a tools <-> evaluation import cycle.
    from .tools import canonical_url, extract_urls

    result = GroundednessResult(critic_score=parse_critic_score(critic_feedback))
    report = content_to_text(report)
    body, references = _references_section(report or "")
    entries = _reference_entries(references)

    reference_urls: list[str] = extract_urls(references)

    allowed = {canonical_url(url) for url in allowed_urls if url}
    retrieved = {canonical_url(url) for url in (retrieved_urls or []) if url}

    verified: list[str] = []
    unsupported: list[str] = []
    for url in reference_urls:
        key = canonical_url(url)
        if key in allowed:
            verified.append(url)
            if key in retrieved and url not in result.retrieved_urls_cited:
                result.retrieved_urls_cited.append(url)
        else:
            unsupported.append(url)

    markers_body = _extract_markers(body)
    markers_refs = _extract_markers(references)
    all_markers = markers_body | markers_refs

    result.citations_found = len(markers_body)
    result.reference_count = len(entries)
    result.reference_urls = reference_urls
    result.verified_urls = verified
    result.unsupported_urls = unsupported
    result.citation_precision = (
        round(len(verified) / len(reference_urls), 4) if reference_urls else 0.0
    )
    if retrieved:
        result.retrieved_url_coverage = round(
            len(result.retrieved_urls_cited) / len(retrieved), 4
        )

    # Markers should map to a numbered reference entry (when entries exist).
    if entries:
        entry_numbers = {number for number, _ in entries}
        dangling = sorted(all_markers - entry_numbers)
        if dangling:
            result.notes.append(
                "Citation markers without a matching reference: "
                + ", ".join(f"[{n}]" for n in dangling)
            )

    blocks = _claim_blocks(report or "")
    cited_blocks = [block for block in blocks if CITATION_RE.search(block)]
    result.uncited_claim_count = len(blocks) - len(cited_blocks)
    result.claim_coverage = (
        round(len(cited_blocks) / len(blocks), 4) if blocks else 0.0
    )

    result.groundedness_score = round(
        0.5 * result.citation_precision + 0.5 * result.claim_coverage, 4
    )
    result.passed = (
        result.groundedness_score >= 0.6
        and not unsupported
        and (not retrieved or bool(result.retrieved_urls_cited))
    )

    if result.groundedness_score >= 0.8:
        result.verdict = "Strongly grounded"
    elif result.groundedness_score >= 0.6:
        result.verdict = "Partially grounded"
    elif result.groundedness_score > 0:
        result.verdict = "Weakly grounded"
    else:
        result.verdict = "No verifiable citations"

    if unsupported:
        result.notes.append(
            f"{len(unsupported)} reference URL(s) were not present in the retrieved evidence"
        )
    if retrieved and not result.retrieved_urls_cited:
        result.notes.append("No reference URL came from the top-k retrieved evidence")
    if result.uncited_claim_count:
        result.notes.append(
            f"{result.uncited_claim_count} findings block(s) have no inline citation"
        )
    if not reference_urls:
        result.notes.append("No URLs found in the References section")
    if not blocks:
        result.notes.append(
            "No substantive claim blocks were detected for coverage scoring"
        )

    return result


def evaluate_report(
    report: str,
    feedback: str | None,
    allowed_urls: Sequence[str],
    retrieved_urls: Sequence[str] | None = None,
) -> dict:
    """Convenience wrapper returning a JSON-serialisable evaluation dict."""

    return evaluate_groundedness(
        report=report,
        allowed_urls=allowed_urls,
        retrieved_urls=retrieved_urls,
        critic_feedback=feedback,
    ).to_dict()
