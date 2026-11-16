"""Actual review/publish nodes across SQLite checkpointer lifetimes.

No external LLM/Qdrant calls. Tests fresh test threads rather than migration of
checkpoints created by older graph versions.
"""
import pytest
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver
from research_assistant.graphs.workflow import (
    human_review_node, route_human_review, publish_node,
)


class ReviewTestState(TypedDict):
    report: str
    verification: dict
    human_revision_count: int
    human_action: str
    human_feedback: str
    report_status: str


def fake_writer(state):
    """Model-free stand-in for a human-requested report revision."""
    return {
        "report": "Revised report",
        "human_action": "",
        "human_feedback": "",
    }


def build_test_graph(saver):
    builder = StateGraph(ReviewTestState)
    builder.add_node("human_review", human_review_node)
    builder.add_node("writer", fake_writer)
    builder.add_node("publish", publish_node)
    builder.add_edge(START, "human_review")
    builder.add_conditional_edges("human_review", route_human_review, {
        "writer": "writer", "publish": "publish",
    })
    builder.add_edge("writer", "human_review")
    builder.add_edge("publish", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.parametrize("action,passed,expected", [
    ("approve", True, "human_approved_verified"),
    ("approve", False, "human_approved_with_unresolved_issues"),
    ("stop", False, "stopped_without_approval"),
])
def test_checkpoint_resume_after_saver_restart(tmp_path, action, passed, expected):
    database = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": f"trial-{action}-{passed}"}}
    initial = {
        "report": "A research report", "verification": {
            "passed": passed,
            "summary": "Requires review",
            "checks": [] if passed else [{"claim": "Unproven", "verdict": "partially_supported"}],
        },
        "human_revision_count": 0, "human_action": "", "human_feedback": "",
        "report_status": "",
    }
    with SqliteSaver.from_conn_string(database) as saver:
        graph = build_test_graph(saver)
        result = graph.invoke(initial, config=config)
        assert result.get("__interrupt__")
        assert graph.get_state(config).next == ("human_review",)
        assert graph.get_state(config).values["report"] == "A research report"
        if not passed:
            assert result["__interrupt__"][0].value["unresolved_claims"]

    # Reconstruct the real nodes in a new graph to simulate a process restart.
    with SqliteSaver.from_conn_string(database) as saver:
        graph = build_test_graph(saver)
        result = graph.invoke(Command(resume={"action": action}), config=config)
        assert not result.get("__interrupt__")
        assert graph.get_state(config).values["report_status"] == expected
        assert graph.get_state(config).next == ()


def test_human_revision_interrupts_again_and_then_approves(tmp_path):
    database = str(tmp_path / "revision.sqlite")
    config = {"configurable": {"thread_id": "revision-thread"}}
    initial = {
        "report": "Initial report", "verification": {"passed": False, "checks": []},
        "human_revision_count": 0, "human_action": "", "human_feedback": "",
        "report_status": "",
    }
    with SqliteSaver.from_conn_string(database) as saver:
        graph = build_test_graph(saver)
        graph.invoke(initial, config=config)
    with SqliteSaver.from_conn_string(database) as saver:
        graph = build_test_graph(saver)
        result = graph.invoke(Command(resume={"action": "revise", "feedback": "Qualify result"}), config=config)
        assert result.get("__interrupt__")
        assert graph.get_state(config).values["report"] == "Revised report"
        assert graph.get_state(config).values["human_revision_count"] == 1
        assert result["__interrupt__"][0].value["human_revisions_remaining"] == 0
        graph.invoke(Command(resume={"action": "approve"}), config=config)
        assert graph.get_state(config).values["report_status"] == "human_approved_with_unresolved_issues"
