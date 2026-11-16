"""Citation contract tests: no network access or LLM required."""
from research_assistant.verification.citations import (
    parse_evidence_citations,
    malformed_evidence_citations,
    validate_report_citations,
)


def test_single_and_multiple_evidence_ids():
    report = (
        "First finding [E:abc-123]. "
        "Second finding [E:abc-123, E:def-456]. "
        "Another [E:def-456] [E:ghi-789]."
    )
    assert parse_evidence_citations(report) == [
        "E:abc-123", "E:def-456", "E:ghi-789"
    ]
    assert malformed_evidence_citations(report) == []


def test_unknown_ids_are_detected_even_when_grouped():
    result = validate_report_citations(
        "This cites [E:real, E:invented].",
        {"E:real": {"text": "real passage"}},
    )
    assert result["invalid_ids"] == ["E:invented"]
    assert result["malformed_groups"] == []


def test_badly_formatted_evidence_groups_are_flagged():
    report = "Bad [E:] and [E:abc E:def] but good [E:valid]."
    assert malformed_evidence_citations(report) == ["[E:]", "[E:abc E:def]"]
    assert parse_evidence_citations(report) == ["E:abc", "E:def", "E:valid"]


def test_non_citation_brackets_are_ignored():
    assert validate_report_citations("Equation [1] and [Table 2].", {}) == {
        "cited_ids": [], "invalid_ids": [], "malformed_groups": [],
    }


def test_unbracketed_citation_detected():
    result = validate_report_citations("Finding E:chunk-123 without brackets", {})
    assert result["loose_ids"] == ["E:chunk-123"]
