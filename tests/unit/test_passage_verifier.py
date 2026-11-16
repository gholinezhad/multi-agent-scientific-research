"""Verify gating when citations or individual claims are not supported.

No external LLM calls; the scientific assessment is mocked, while the
actual passage_verifier_node and routing policy execute.
"""
from types import SimpleNamespace
from research_assistant.verification import nodes


def _state(report):
    return {
        "question": "Example question", "papers": [{
            "url": "https://arxiv.org/abs/example",
            "passages": [{"chunk_id": "real", "page": 2, "text": "Relevant passage"}],
        }],
        "report": report, "verification_attempts": 0,
        "verification_research_rounds": 1,
        "auto_revision_count": 2,
        "research_retry_used": True,
    }


def test_report_with_unknown_id_never_passes(monkeypatch):
    monkeypatch.setattr(nodes, "extract_claims", lambda _report: [
        nodes.ScientificClaim(claim="Finding", citation_ids=["E:real"])
    ])
    monkeypatch.setattr(nodes, "check_claim", lambda _claim, _registry: {
        "claim": "Finding", "verdict": "supported", "source_ids": ["E:real"],
    })
    out = nodes.passage_verifier_node(_state("Finding [E:real, E:invented]"))
    assert out["verification"]["passed"] is False
    assert out["verification"]["invalid_report_ids"] == ["E:invented"]
    assert out["verification"]["next_action"] == "publish"


def test_partial_support_is_retained_at_budget_exhaustion(monkeypatch):
    monkeypatch.setattr(nodes, "extract_claims", lambda _report: [
        nodes.ScientificClaim(claim="Too broad", citation_ids=["E:real"])
    ])
    monkeypatch.setattr(nodes, "check_claim", lambda _claim, _registry: {
        "claim": "Too broad", "verdict": "partially_supported",
        "explanation": "Scope exceeds evidence", "source_ids": ["E:real"],
    })
    out = nodes.passage_verifier_node(_state("Too broad [E:real]"))
    assert out["verification"]["passed"] is False
    assert len(out["verification"]["unresolved_checks"]) == 1
    assert out["verification"]["next_action"] == "publish"
