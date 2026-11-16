"""Mocked writer/verifier tests - requires installed project dependencies, not API calls."""
from types import SimpleNamespace

from research_assistant.verification import nodes


def _state():
    return {
        "question": "What is known about fingering?",
        "papers": [], "research_notes": [],
        "report": "Old unsupported finding [E:unknown]",
        "verification": {
            "passed": False,
            "next_action": "revise",
            "checks": [
                {"claim": "The model is always universal", "verdict": "partially_supported",
                 "explanation": "Only studied a bounded parameter range"},
            ],
            "invalid_report_ids": ["E:unknown"],
            "malformed_report_citations": [],
        },
        "human_action": "", "human_feedback": "",
        "revision_count": 1, "auto_revision_count": 1,
    }


def test_writer_receives_failed_claim_and_repair_instructions(monkeypatch):
    prompts = []
    def fake_invoke(messages):
        prompts.extend(messages)
        return SimpleNamespace(content="Qualified report")
    monkeypatch.setattr(nodes, "llm", SimpleNamespace(invoke=fake_invoke))
    out = nodes.evidence_writer_node(_state())
    rendered = " ".join(str(m) for m in prompts)
    assert "The model is always universal" in rendered
    assert "remove the unsupported claim" in rendered
    assert "E:unknown" in rendered
    assert "revision_plan" in rendered
    assert "Candidate passages" in rendered
    assert out["auto_revision_count"] == 2


def test_human_revision_does_not_consume_automatic_budget(monkeypatch):
    monkeypatch.setattr(nodes, "llm", SimpleNamespace(
        invoke=lambda _msgs: SimpleNamespace(content="Requested revision")
    ))
    state = _state()
    state.update(human_action="revise", human_feedback="Discuss boundaries", auto_revision_count=2)
    out = nodes.evidence_writer_node(state)
    assert out["auto_revision_count"] == 2
    assert out["human_action"] == ""
    assert out["human_feedback"] == ""
