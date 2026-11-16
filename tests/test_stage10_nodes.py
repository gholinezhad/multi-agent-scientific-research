from types import SimpleNamespace

from research_assistant.verification import nodes as stage10_nodes


def _state(report="", revision_count=0):
    return {
        "question": "A research question",
        "papers": [],
        "research_notes": [],
        "report": report,
        "verification": {},
        "human_feedback": "",
        "revision_count": revision_count,
    }


def test_evidence_writer_logs_initial_generation(monkeypatch, capsys):
    monkeypatch.setattr(
        stage10_nodes,
        "llm",
        SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(
                content="Initial report"
            )
        ),
    )

    result = stage10_nodes.evidence_writer_node(_state())

    assert (
        "[WRITER] Generating evidence-linked report..."
        in capsys.readouterr().out
    )
    assert result["report"] == "Initial report"
    assert result["revision_count"] == 0


def test_evidence_writer_logs_revision(monkeypatch, capsys):
    monkeypatch.setattr(
        stage10_nodes,
        "llm",
        SimpleNamespace(
            invoke=lambda _messages: SimpleNamespace(
                content="Revised report"
            )
        ),
    )

    result = stage10_nodes.evidence_writer_node(
        _state(report="Previous report", revision_count=1)
    )

    assert "[WRITER] Revising report..." in capsys.readouterr().out
    assert result["report"] == "Revised report"
    assert result["revision_count"] == 2
