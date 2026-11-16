
import os
import threading
import uuid

from contextlib import ExitStack, asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from research_assistant.graphs.workflow import (
    build_graph,
    create_initial_state,
    MAX_HUMAN_REVISIONS,
)

load_dotenv()

os.environ.setdefault(
    "LANGGRAPH_STRICT_MSGPACK", "true"
)

# Local development: serialize SQLite graph access.
graph_lock = threading.Lock()


# ---------------------------------------
# Request schemas
# ---------------------------------------

class ResearchRequest(BaseModel):
    question: str = Field(
        min_length=10,
        max_length=4000,
    )


class ReviewRequest(BaseModel):
    action: Literal[
        "approve",
        "revise",
        "stop",
    ]

    feedback: str = ""


# ---------------------------------------
# Application lifecycle
# ---------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    with ExitStack() as stack:
        checkpointer = stack.enter_context(
            SqliteSaver.from_conn_string(
                "research_checkpoints.sqlite"
            )
        )

        app.state.graph = build_graph(checkpointer)

        yield


app = FastAPI(
    title="Multi-Agent AI Research Assistant",
    version="0.8.0",
    lifespan=lifespan,
)


# ---------------------------------------
# Helpers
# ---------------------------------------

def get_config(thread_id: str):
    return {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 100,
    }


def get_snapshot(thread_id: str):
    snapshot = app.state.graph.get_state(
        get_config(thread_id)
    )

    if not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail="Research thread not found",
        )

    return snapshot


def serialize_state(thread_id: str):
    snapshot = get_snapshot(thread_id)
    state = snapshot.values

    pending_interrupts = [
        intr
        for task in snapshot.tasks
        for intr in task.interrupts
    ]

    if pending_interrupts:
        status = "awaiting_human_review"
    elif snapshot.next:
        status = "pending"
    else:
        status = "completed"

    return {
        "thread_id": thread_id,
        "execution_status": status,
        "report_status": state.get("report_status"),
        "question": state.get("question"),
        "rounds": state.get("rounds", 0),
        "paper_count": len(
            state.get("papers", [])
        ),
        "verification": state.get(
            "verification", {}
        ),
        "report": state.get("report", ""),
        "human_revision_count": state.get(
            "human_revision_count", 0
        ),
    }


# ---------------------------------------
# API routes
# ---------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", status_code=201)
def start_research(request: ResearchRequest):
    thread_id = str(uuid.uuid4())
    config = get_config(thread_id)

    with graph_lock:
        app.state.graph.invoke(
            create_initial_state(request.question),
            config=config,
        )

        return serialize_state(thread_id)


@app.get("/research/{thread_id}")
def research_status(thread_id: str):
    with graph_lock:
        return serialize_state(thread_id)


@app.post("/research/{thread_id}/review")
def review_research(
    thread_id: str,
    request: ReviewRequest,
):
    config = get_config(thread_id)

    with graph_lock:
        snapshot = get_snapshot(thread_id)

        interrupts = [
            intr
            for task in snapshot.tasks
            for intr in task.interrupts
        ]

        if not interrupts:
            raise HTTPException(
                status_code=409,
                detail="Thread is not awaiting review",
            )

        if request.action == "revise":
            if not request.feedback.strip():
                raise HTTPException(
                    status_code=422,
                    detail="Revision requires feedback",
                )

            used = snapshot.values.get(
                "human_revision_count", 0
            )

            if used >= MAX_HUMAN_REVISIONS:
                raise HTTPException(
                    status_code=409,
                    detail="Human revision limit reached",
                )

        app.state.graph.invoke(
            Command(
                resume={
                    "action": request.action,
                    "feedback": request.feedback,
                }
            ),
            config=config,
        )

        return serialize_state(thread_id)
