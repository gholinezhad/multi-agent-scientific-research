"""Pure, offline tests: suggestions never imply scientific support."""
from research_assistant.verification.selection import (
    suggest_evidence_for_claim, build_revision_plan,
)

REGISTRY = {
    "E:wrong": {"text": "The model describes flow geometry.", "page": 1},
    "E:right": {"text": "The measured fractal dimension ranged from 1.64 to 1.85 in radial fingering experiments.", "page": 5},
    "E:other": {"text": "Surface tension modifies bubble shape.", "page": 9},
}


def test_specific_passage_ranks_ahead_of_generic_passage():
    results = suggest_evidence_for_claim(
        "The fractal dimension in radial fingering ranges from 1.64 to 1.85", REGISTRY)
    assert results[0]["evidence_id"] == "E:right"
    assert "text" not in results[0]


def test_no_overlap_does_not_invent_evidence():
    assert suggest_evidence_for_claim("quantum entanglement Bell inequality", REGISTRY) == []


def test_empty_query_and_limits():
    assert suggest_evidence_for_claim("the and", REGISTRY) == []
    assert suggest_evidence_for_claim("fractal dimension", REGISTRY, top_k=0) == []


def test_revision_plan_only_includes_failed_claims():
    checks = [
        {"claim": "Fractal dimension 1.64 to 1.85", "verdict": "unsupported", "source_ids": ["E:wrong"]},
        {"claim": "Surface tension modifies bubble shape", "verdict": "supported"},
    ]
    plan = build_revision_plan(checks, REGISTRY)
    assert len(plan) == 1
    assert plan[0]["candidate_passages_not_yet_verified"][0]["evidence_id"] == "E:right"
    assert plan[0]["previous_citations"] == ["E:wrong"]
