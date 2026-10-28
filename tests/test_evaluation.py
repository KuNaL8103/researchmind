from src.evaluation import (
    content_to_text,
    evaluate_groundedness,
    parse_critic_score,
)

REPORT = """# Test Topic

## 4. Key Findings

### Finding 1: Quantum error correction improved

**Evidence:** Researchers demonstrated a below-threshold surface code with a logical error rate of 0.001 per cycle, a meaningful improvement over prior experiments [1]. The experiment used 49 physical qubits and repeated syndrome measurements over many cycles.

**Interpretation:** This result indicates that scaling physical qubits can reduce logical errors, which is a prerequisite for fault-tolerant machines [1].

### Finding 2: Fusion milestones continue

**Evidence:** A fusion facility reported a net energy gain from an inertial confinement experiment, although the gain did not include facility power costs [2]. Independent groups have not yet reproduced the result at the same scale.

## 7. References

[1] Quantum error correction experiment. Nature (2023). https://example.com/quantum
[2] Fusion ignition result. Example News (2022). https://example.com/hallucinated
"""


def test_critic_score_parser_handles_markdown():
    assert parse_critic_score("Overall Score: 8/10") == 8.0
    assert parse_critic_score("**Overall Score:** 7.5 / 10") == 7.5
    assert parse_critic_score("no score here") is None


def test_groundedness_flags_unsupported_urls_and_measures_coverage():
    result = evaluate_groundedness(
        report=REPORT,
        allowed_urls=["https://example.com/quantum"],
        retrieved_urls=["https://example.com/quantum"],
        critic_feedback="Overall Score: 9/10",
    )

    assert result.critic_score == 9.0
    assert result.citation_precision == 0.5
    assert result.claim_coverage == 1.0
    assert result.groundedness_score == 0.75
    assert result.unsupported_urls == ["https://example.com/hallucinated"]
    assert result.passed is False
    assert result.retrieved_urls_cited == ["https://example.com/quantum"]
    assert result.retrieved_url_coverage == 1.0


def test_uncited_findings_reduce_coverage():
    report = REPORT.replace("[2]", "")
    result = evaluate_groundedness(
        report=report,
        allowed_urls=["https://example.com/quantum", "https://example.com/hallucinated"],
    )
    assert result.claim_coverage == 0.6667
    assert result.uncited_claim_count == 1


BULLET_REPORT = """# Bullet Report

## 4. Key Findings:

### Finding 1: Surface codes improved

- **Evidence:** A below-threshold surface code reduced logical errors across repeated measurement cycles [1].
- **Interpretation:** The result is a prerequisite for fault-tolerant quantum computing at useful scale [1].
- **Implications:** Hardware roadmaps can now prioritize logical-qubit demonstrations over raw qubit counts [1].

### Finding 2: Fusion experiments advanced

- **Evidence:** An inertial confinement experiment reported a net energy gain, though facility power costs were excluded [2].
- **Interpretation:** The result is a scientific milestone rather than a commercial power breakthrough [2].
- **Implications:** Grid-scale fusion still requires gains in repetition rate, target fabrication, and cost [2].

## 7. References

[1] Quantum error correction experiment. Nature (2026). https://example.com/quantum
[2] Fusion ignition result. Example News (2026). https://example.com/fusion
"""


def test_bullet_findings_are_parsed_as_claim_units():
    result = evaluate_groundedness(
        report=BULLET_REPORT,
        allowed_urls=["https://example.com/quantum", "https://example.com/fusion"],
        retrieved_urls=["https://example.com/quantum", "https://example.com/fusion"],
    )

    assert result.claim_coverage == 1.0
    assert result.uncited_claim_count == 0
    assert result.groundedness_score == 1.0
    assert result.passed is True
    assert result.citation_precision == 1.0
    assert result.retrieved_url_coverage == 1.0
    assert result.verdict == "Strongly grounded"
    assert result.notes == []


def test_bullet_findings_without_citations_reduce_coverage():
    report = BULLET_REPORT.replace("[2]", "")
    result = evaluate_groundedness(
        report=report,
        allowed_urls=["https://example.com/quantum", "https://example.com/fusion"],
    )

    assert 0 < result.claim_coverage < 1.0
    assert result.uncited_claim_count > 0


def test_content_to_text_handles_plain_strings_and_none():
    assert content_to_text("hello") == "hello"
    assert content_to_text(None) == ""
    assert content_to_text("") == ""


def test_content_to_text_flattens_langchain_content_blocks():
    content = [
        {"type": "text", "text": "first block"},
        {"type": "text", "text": "second block"},
    ]
    assert content_to_text(content) == "first block\nsecond block"


def test_content_to_text_handles_dict_values_and_nested_lists():
    assert content_to_text({"text": "hello"}) == "hello"
    assert content_to_text({"content": [{"type": "text", "text": "nested"}]}) == "nested"
    assert content_to_text(["plain", {"text": "part"}]) == "plain\npart"


def test_content_to_text_ignores_non_text_tool_blocks():
    assert content_to_text([{"type": "tool_use", "name": "search"}]) == ""
