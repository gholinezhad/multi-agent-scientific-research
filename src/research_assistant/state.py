"""Canonical graph state types. Runtime nodes use these directly."""
import operator
from typing import Annotated, Any, Literal, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class ResearchTask(TypedDict):
    agent: Literal["literature", "rag"]
    query: str

class ResearchState(TypedDict):
    question: str
    plan: str
    tasks: list[ResearchTask]
    papers: Annotated[list[dict[str, Any]], operator.add]
    research_notes: Annotated[list[str], operator.add]
    search_history: Annotated[list[str], operator.add]
    rounds: int
    findings: str
    report: str
    verification: dict[str, Any]
    verification_attempts: int
    revision_count: int  # migration compatibility
    auto_revision_count: int
    verification_research_rounds: int
    research_retry_used: bool
    report_status: str

class SpecialistState(TypedDict):
    query: str
    messages: Annotated[list[AnyMessage], add_messages]

class WorkerInput(TypedDict):
    question: str
    task: ResearchTask
