
import argparse
import json
import os
import uuid

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

from research_assistant.agents.research import (
    ResearchState,
    planner_node,
    supervisor_node,
    parallel_worker,
    dispatch_tasks,
    finalizer_node,
    route_verification,
    targeted_research_node,
)

from research_assistant.verification.nodes import (
    evidence_writer_node,
    passage_verifier_node,
)

load_dotenv()

# Recommended for checkpoint deserialization safety.
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"


class Stage7State(ResearchState):
    human_action: str
    human_feedback: str
    human_revision_count: int


MAX_HUMAN_REVISIONS = 1


def human_review_node(state: Stage7State):
    print("\n[HUMAN REVIEW] Approval required.")

    verification = state["verification"]

    # This payload is surfaced to the caller.
    review = {
        "message": "Review the research report.",
        "report_preview": state["report"][:2000],
        "automated_verification_passed": (
            verification.get("passed", False)
        ),
        "verification_summary": (
            verification.get("summary", "")
        ),
        "claim_checks": verification.get("checks", []),
        "unresolved_claims": [
            check for check in verification.get("checks", [])
            if check.get("verdict") != "supported"
        ],
        "invalid_citation_ids": verification.get("invalid_report_ids", []),
        "malformed_citations": verification.get("malformed_report_citations", []),
        "loose_citations": verification.get("loose_report_citations", []),
        "verification_warning": (
            "Automated verification did not pass. Approval will "
            "be explicitly marked as having unresolved issues."
            if not verification.get("passed", False) else None
        ),
        "human_revisions_remaining": (
            MAX_HUMAN_REVISIONS
            - state["human_revision_count"]
        ),
        "allowed_actions": [
            "approve",
            "revise",
            "stop",
        ],
    }

    # Execution pauses here.
    response = interrupt(review)

    action = response["action"]
    feedback = response.get("feedback", "")

    if action not in {"approve", "revise", "stop"}:
        action = "stop"

    # Require meaningful feedback for revisions.
    if action == "revise" and not feedback.strip():
        action = "stop"

    if (
        action == "revise"
        and state["human_revision_count"]
        >= MAX_HUMAN_REVISIONS
    ):
        action = "stop"

    return {
        "human_action": action,
        "human_feedback": feedback,
        "human_revision_count": (
            state["human_revision_count"] + 1
            if action == "revise"
            else state["human_revision_count"]
        ),
    }


def route_human_review(state: Stage7State):
    if state["human_action"] == "revise":
        return "writer"

    return "publish"



def publish_node(state: Stage7State):
    action = state["human_action"]
    passed = state["verification"].get(
        "passed", False
    )

    if action == "stop":
        status = "stopped_without_approval"

    elif action == "approve" and passed:
        status = "human_approved_verified"

    elif action == "approve":
        status = "human_approved_with_unresolved_issues"

    else:
        status = "incomplete"

    print(f"\n[PUBLISH] {status}")

    return {"report_status": status}


def build_graph(checkpointer):
    builder = StateGraph(Stage7State)

    # Existing research nodes
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("parallel_worker", parallel_worker)
    builder.add_node("finalizer", finalizer_node)
    builder.add_node(
        "writer",
        evidence_writer_node,
    )

    builder.add_node(
        "verifier",
        passage_verifier_node,
    )
    builder.add_node("targeted_research", targeted_research_node)

    # New human-in-the-loop nodes
    builder.add_node("human_review", human_review_node)
    builder.add_node("publish", publish_node)

    # Research
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        dispatch_tasks,
        ["parallel_worker", "finalizer"],
    )

    builder.add_edge(
        "parallel_worker", "supervisor"
    )

    builder.add_edge("finalizer", "writer")
    builder.add_edge("writer", "verifier")

    # Automated verification
    builder.add_conditional_edges(
        "verifier",
        route_verification,
        {
            "publish": "human_review",
            "research": "targeted_research",
            "revise": "writer",
        },
    )

    builder.add_conditional_edges(
        "targeted_research",
        dispatch_tasks,
        ["parallel_worker", "finalizer"],
    )

    # Human approval
    builder.add_conditional_edges(
        "human_review",
        route_human_review,
        {
            "writer": "writer",
            "publish": "publish",
        },
    )

    builder.add_edge("publish", END)

    # New: persistent checkpointing
    return builder.compile(
        checkpointer=checkpointer
    )


def create_initial_state(question: str):
    return {
        "question": question,
        "plan": "",
        "tasks": [],
        "papers": [],
        "research_notes": [],
        "search_history": [],
        "rounds": 0,
        "findings": "",
        "report": "",
        "verification": {},
        "verification_attempts": 0,
        "revision_count": 0,
        "auto_revision_count": 0,
        "verification_research_rounds": 0,
        "research_retry_used": False,
        "report_status": "",
        # Stage 7 fields
        "human_action": "",
        "human_feedback": "",
        "human_revision_count": 0,
    }


def show_execution_result(graph, config, result):
    interrupts = result.get("__interrupt__", [])

    if interrupts:
        print("\n" + "=" * 60)
        print("EXECUTION PAUSED FOR HUMAN REVIEW")

        for item in interrupts:
            print(json.dumps(
                item.value,
                indent=2,
                ensure_ascii=False,
            ))

        print("\nUse the same thread ID to resume.")
        return

    snapshot = graph.get_state(config)
    values = snapshot.values

    if not snapshot.next:
        print("\n" + "=" * 60)
        print("EXECUTION COMPLETED")
        print("Status:", values.get("report_status"))
        print("\nFINAL REPORT\n")
        print(values.get("report", ""))


def inspect_saved_state(graph, config):
    snapshot = graph.get_state(config)

    if not snapshot.values:
        print("No saved state found for this thread.")
        return

    values = snapshot.values

    print("\nSAVED RESEARCH SESSION")
    print("Question:", values.get("question"))
    print("Research rounds:", values.get("rounds"))
    print("Papers:", len(values.get("papers", [])))
    print(
        "Verification attempts:",
        values.get("verification_attempts"),
    )
    print(
        "Next scheduled nodes:",
        snapshot.next,
    )
    print(
        "Report status:",
        values.get("report_status"),
    )
    print("\nREPORT\n")
    print(values.get("report", ""))

    if snapshot.next:
        print("\nThis session has pending execution.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["start", "status", "resume"],
    )

    parser.add_argument(
        "thread_id",
        nargs="?",
    )

    parser.add_argument(
        "--question",
        default=(
            "Viscous fingering patterns in "
            "immiscible Hele-Shaw flow."
        ),
    )

    args = parser.parse_args()

    # Persistent SQLite database.
    with SqliteSaver.from_conn_string(
        "research_checkpoints.sqlite"
    ) as checkpointer:

        graph = build_graph(checkpointer)

        if args.command == "start":
            thread_id = (
                args.thread_id
                or str(uuid.uuid4())
            )

            config = {
                "configurable": {
                    "thread_id": thread_id
                },
                "recursion_limit": 100,
            }

            print("Research thread:", thread_id)

            # Avoid accidentally restarting a saved thread.
            existing = graph.get_state(config)
            if existing.values:
                print(
                    "This thread already exists. "
                    "Choose a new thread ID."
                )
                return

            result = graph.invoke(
                create_initial_state(args.question),
                config=config,
            )

            show_execution_result(
                graph, config, result
            )
            return

        if not args.thread_id:
            parser.error(
                "A thread ID is required for "
                "status or resume."
            )

        config = {
            "configurable": {
                "thread_id": args.thread_id
            },
            "recursion_limit": 100,
        }

        if args.command == "status":
            inspect_saved_state(graph, config)
            return

        # Resume an existing interrupted thread.
        snapshot = graph.get_state(config)

        if not snapshot.values:
            print("Thread not found.")
            return

        pending_interrupts = [
            intr
            for task in snapshot.tasks
            for intr in task.interrupts
        ]

        if not pending_interrupts:
            print(
                "No pending human review interrupt "
                "for this thread."
            )
            inspect_saved_state(graph, config)
            return

        print("\nCURRENT REPORT\n")
        print(snapshot.values.get("report", ""))

        print("\nVerification summary:")
        print(
            snapshot.values.get(
                "verification", {}
            ).get("summary", "")
        )

        unresolved = [
            item for item in snapshot.values.get("verification", {}).get("checks", [])
            if item.get("verdict") != "supported"
        ]
        if unresolved:
            print(f"\nWARNING: {len(unresolved)} unresolved scientific claims:")
            for item in unresolved:
                print(f"- {item.get('verdict')}: {item.get('claim')}")

        action = input(
            "\nChoose approve / revise / stop: "
        ).strip().lower()

        if action not in {
            "approve", "revise", "stop"
        }:
            print("Invalid action.")
            return

        feedback = ""

        if action == "revise":
            if (
                snapshot.values.get(
                    "human_revision_count", 0
                ) >= MAX_HUMAN_REVISIONS
            ):
                print("Human revision limit reached.")
                return

            feedback = input(
                "Revision instructions: "
            ).strip()

            if not feedback:
                print("Revision feedback is required.")
                return

        result = graph.invoke(
            Command(
                resume={
                    "action": action,
                    "feedback": feedback,
                }
            ),
            config=config,
        )

        show_execution_result(
            graph, config, result
        )


if __name__ == "__main__":
    main()
