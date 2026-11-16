
import json
import operator
import os

from typing import Annotated, Any, Literal, TypedDict

import arxiv
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from langchain_core.tools import tool
from research_assistant.tools.qdrant import search_papers

load_dotenv()

MAX_RESEARCH_ROUNDS = 2
MAX_TOOL_BATCHES = 2

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0,
)


# ----------------------------------------
# Parent graph state
# ----------------------------------------

from research_assistant.state import (
    ResearchState, SpecialistState as AgentState, WorkerInput,
)


# ----------------------------------------
# Structured supervisor output
# ----------------------------------------

class ResearchTask(BaseModel):
    agent: Literal["literature", "rag"]
    query: str = Field(
        description="Focused query for the agent"
    )


class SupervisorDecision(BaseModel):
    finish: bool = Field(
        description="True if research should end"
    )

    tasks: list[ResearchTask] = Field(
        default_factory=list,
        description="Zero to two research tasks"
    )

    reasoning: str = Field(
        description="Reason for the decision"
    )



class ClaimCheck(BaseModel):
    claim: str = Field(
        description="A scientific claim from the report"
    )

    verdict: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
    ]

    source_urls: list[str] = Field(
        default_factory=list,
        description="URLs of abstracts supporting the claim"
    )

    explanation: str = Field(
        description="Explain the assessment using evidence"
    )


class VerificationResult(BaseModel):
    checks: list[ClaimCheck] = Field(
        description="Assess the important scientific claims"
    )

    summary: str = Field(
        description="Overall evidence-quality assessment"
    )

    research_queries: list[str] = Field(
        default_factory=list,
        description="Queries to resolve important gaps"
    )


verifier_llm = llm.with_structured_output(
    VerificationResult
)


supervisor_llm = llm.with_structured_output(
    SupervisorDecision
)


def build_specialist_graph(name, tool):
    """Create an isolated ReAct specialist."""

    tool_llm = llm.bind_tools([tool])
    required_llm = llm.bind_tools(
        [tool],
        tool_choice="required",
    )

    def agent_node(state: AgentState):
        messages = state["messages"]

        tool_batches = sum(
            bool(getattr(m, "tool_calls", None))
            for m in messages
        )

        system = SystemMessage(
            content=(
                f"You are the {name} scientific agent. "
                "Use your available search tool to gather "
                "evidence relevant to the assigned task. "
                "You may search again if necessary. "
                "Use focused queries. "
                "Do not fabricate papers or citations. "
                "Only describe information supported "
                "by retrieved abstracts. "
                "If searches fail, acknowledge this. "
                "When finished, provide a concise "
                "evidence-based synthesis with URLs."
            )
        )

        if tool_batches == 0:
            # First call must request the tool.
            model = required_llm

        elif tool_batches >= MAX_TOOL_BATCHES:
            # Finish without more tool calls.
            model = llm

        else:
            # LLM may search again or finish.
            model = tool_llm

        prompt_messages = [system] + messages

        if tool_batches >= MAX_TOOL_BATCHES:
            prompt_messages.append(
                HumanMessage(
                    content=(
                        "The search budget is exhausted. "
                        "Summarize the retrieved evidence. "
                        "Do not request more tools."
                    )
                )
            )

        response = model.invoke(prompt_messages)

        return {"messages": [response]}

    def route_agent(state: AgentState):
        last = state["messages"][-1]

        if getattr(last, "tool_calls", None):
            return "tools"

        return END

    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode([tool]))

    builder.add_edge(START, "agent")

    builder.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            END: END,
        },
    )

    builder.add_edge("tools", "agent")

    return builder.compile()


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for scientific papers.

    Args:
        query: An arXiv-compatible search query,
            such as all:"neural operator".

    Returns:
        JSON containing paper metadata and abstracts.
    """

    print(f"\n[TOOL] Searching arXiv: {query}")

    try:
        client = arxiv.Client(
            page_size=5,
            delay_seconds=3,
            num_retries=2,
        )

        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []

        for paper in client.results(search):
            papers.append(
                {
                    "title": paper.title,
                    "authors": [
                        author.name
                        for author in paper.authors
                    ],
                    "published": paper.published.date().isoformat(),
                    "abstract": paper.summary,
                    "url": paper.entry_id,
                    "pdf_url": paper.pdf_url,
                }
            )

        return json.dumps(
            {"papers": papers},
            ensure_ascii=False,
        )

    except Exception as exc:
        return json.dumps(
            {
                "papers": [],
                "error": str(exc),
            }
        )
    

literature_graph = build_specialist_graph(
    "Literature",
    search_arxiv,
)

rag_graph = build_specialist_graph(
    "RAG",
    search_papers,
)


def parallel_worker(state: WorkerInput):
    task = state["task"]
    agent_name = task["agent"]
    query = task["query"]

    print(f"\n[WORKER] Starting {agent_name}: {query}")

    specialist = (
        literature_graph
        if agent_name == "literature"
        else rag_graph
    )

    result = specialist.invoke(
        {
            "query": query,
            "messages": [
                HumanMessage(
                    content=(
                        f"Research question:\n"
                        f"{state['question']}\n\n"
                        f"Assigned task:\n{query}"
                    )
                )
            ],
        },
        config={"recursion_limit": 12},
    )

    papers = []
    errors = []

    for message in result["messages"]:
        if not isinstance(message, ToolMessage):
            continue

        try:
            payload = json.loads(message.content)
            papers.extend(payload.get("papers", []))

            if payload.get("error"):
                errors.append(payload["error"])

        except (TypeError, ValueError):
            errors.append("Invalid tool response")

    final_message = result["messages"][-1]

    summary = str(final_message.content)

    if errors:
        summary += (
            "\nTool errors: "
            + "; ".join(errors)
        )

    print(
        f"[WORKER] Finished {agent_name}; "
        f"collected {len(papers)} paper records."
    )

    return {
        "papers": papers,
        "research_notes": [
            f"[{agent_name}] {query}\n{summary}"
        ],
        "search_history": [
            f"{agent_name}: {query}"
        ],
    }


def planner_node(state: ResearchState):
    print("\n[PLANNER] Creating plan...")

    response = llm.invoke([
        (
            "system",
            "Create a concise scientific research plan "
            "identifying the main evidence requirements."
        ),
        ("human", state["question"]),
    ])

    return {"plan": str(response.content)}


def supervisor_node(state: ResearchState):
    print("\n[SUPERVISOR] Evaluating evidence...")

    if state["research_retry_used"]:
        print(
            "[SUPERVISOR] Verification research complete. "
            "Returning to report generation."
        )
        return {"tasks": []}
    
    if state["rounds"] >= MAX_RESEARCH_ROUNDS:
        print("[SUPERVISOR] Round limit reached.")
        return {"tasks": []}

    # Avoid sending entire abstracts to supervisor.
    notes = state["research_notes"][-6:]

    decision = supervisor_llm.invoke([
        (
            "system",
            "You coordinate scientific research.\n"
            "Choose up to TWO independent tasks "
            "that can run in parallel.\n"
            "Available specialists:\n"
            "- literature: searches arXiv\n"
            "- rag: searches indexed Qdrant papers\n"
            "Use arXiv-compatible syntax for literature "
            "queries, e.g. all:\"neural operator\".\n"
            "Use natural language for RAG queries.\n"
            "Review existing findings before deciding.\n"
            "Avoid duplicate research tasks.\n"
            "If sufficient evidence exists, finish=true.\n"
            "On the first round, select research tasks.\n"
            "If finishing, return an empty task list."
        ),
        (
            "human",
            f"Question:\n{state['question']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Completed rounds: {state['rounds']}\n"
            f"Previous searches:\n"
            f"{json.dumps(state['search_history'])}\n"
            f"Research notes:\n"
            f"{json.dumps(notes)}\n"
            f"Paper records: {len(state['papers'])}"
        ),
    ])

    if decision.finish and state["rounds"] > 0:
        tasks = []
    else:
        tasks = [
            task.model_dump()
            for task in decision.tasks[:2]
        ]

        # Ensure at least one task in first round.
        if not tasks and state["rounds"] == 0:
            tasks = [
                {
                    "agent": "rag",
                    "query": state["question"],
                }
            ]

    print(f"[SUPERVISOR] {len(tasks)} tasks selected")

    for task in tasks:
        print(f"  {task['agent']}: {task['query']}")

    return {
        "tasks": tasks,
        "rounds": (
            state["rounds"] + 1
            if tasks else state["rounds"]
        ),
    }


def dispatch_tasks(state: ResearchState):
    if not state["tasks"]:
        return "finalizer"

    return [
        Send(
            "parallel_worker",
            {
                "question": state["question"],
                "task": task,
            },
        )
        for task in state["tasks"]
    ]



def finalizer_node(state: ResearchState):
    print("\n[FINALIZER] Merging evidence...")

    grouped = {}

    for paper in state["papers"]:
        url = paper.get("url")

        if not url:
            continue

        if url not in grouped:
            grouped[url] = {
                **paper,
                "passages": [],
            }

        target = grouped[url]

        # Preserve richer metadata when available.
        for field in ("title", "authors", "published"):
            if not target.get(field) and paper.get(field):
                target[field] = paper[field]

        passages = paper.get("passages", [])

        if not passages and paper.get("abstract"):
            passages = [{
                "chunk_id": f"{url}:abstract",
                "page": None,
                "text": paper["abstract"],
            }]

        existing_ids = {
            p["chunk_id"]
            for p in target["passages"]
        }

        for passage in passages:
            if passage["chunk_id"] not in existing_ids:
                target["passages"].append(passage)
                existing_ids.add(passage["chunk_id"])

    evidence = []

    for paper in grouped.values():
        # Keep the final context bounded.
        selected = paper["passages"][:5]

        evidence.append({
            "title": paper.get("title"),
            "url": paper["url"],
            "passages": selected,
            # Compatibility with Stage 6 verifier.
            "abstract": "\n\n".join(
                f"[Page {p.get('page')}]\n{p['text']}"
                for p in selected
            ),
        })

    return {
        "findings": json.dumps(
            {
                "notes": state["research_notes"],
                "papers": evidence,
            },
            ensure_ascii=False,
        )
    }


def writer_node(state: ResearchState):
    is_revision = (
        state["verification_attempts"] > 0
    )

    if is_revision:
        print("\n[WRITER] Revising report...")
    else:
        print("\n[WRITER] Writing initial report...")

    feedback = state["verification"]

    response = llm.invoke([
        (
            "system",
            "You are a scientific research writer. "
            "Write a structured, evidence-grounded report. "
            "Use only supplied paper abstracts and notes. "
            "Cite actual retrieved paper URLs. "
            "Do not invent scientific findings. "
            "Do not claim to have read full papers. "
            "If revising, address verification feedback "
            "by correcting, qualifying, or removing "
            "unsupported claims. "
            "Do not force unsupported conclusions. "
            "Preserve important limitations."
        ),
        (
            "human",
            f"Question:\n{state['question']}\n\n"
            f"Research evidence:\n{state['findings']}\n\n"
            f"Previous report:\n"
            f"{state['report'] if is_revision else 'None'}\n\n"
            f"Verification feedback:\n"
            f"{json.dumps(feedback, ensure_ascii=False)}"
        ),
    ])

    return {
        "report": str(response.content),
        "revision_count": (
            state["revision_count"] + 1
            if is_revision
            else state["revision_count"]
        ),
    }


import re

MAX_REVISIONS = 2


def verifier_node(state: ResearchState):
    print("\n[VERIFIER] Evaluating scientific claims...")

    # Deduplicate retrieved papers.
    papers_by_url = {
        p["url"]: p
        for p in state["papers"]
        if p.get("url")
    }

    # Limit context size.
    evidence = [
        {
            "title": p.get("title"),
            "url": p["url"],
            "abstract": (p.get("abstract") or "")[:2500],
        }
        for p in list(papers_by_url.values())[:20]
    ]

    result = verifier_llm.invoke([
        (
            "system",
            "You are a rigorous scientific evidence verifier. "
            "Evaluate important factual and comparative "
            "claims in the report against the supplied "
            "paper abstracts only. "
            "Assess up to eight important claims. "
            "Include quantitative and strong claims. "
            "A claim is supported only if the supplied "
            "evidence explicitly justifies its scope. "
            "Do not assume cited sources support claims. "
            "Do not use outside knowledge as evidence. "
            "Return only source URLs from the evidence. "
            "Use unsupported if evidence is missing, "
            "partially_supported if the scope is too broad, "
            "and contradicted if evidence conflicts. "
            "Suggest focused research queries when needed."
        ),
        (
            "human",
            f"Research question:\n{state['question']}\n\n"
            f"Draft report:\n{state['report']}\n\n"
            f"Available abstracts:\n"
            f"{json.dumps(evidence, ensure_ascii=False)}"
        ),
    ])

    allowed_urls = set(papers_by_url)

    checks = []
    for check in result.checks:
        item = check.model_dump()

        # Deterministic source validation.
        invalid = [
            url for url in item["source_urls"]
            if url not in allowed_urls
        ]

        if invalid:
            item["verdict"] = "unsupported"
            item["explanation"] += (
                f" Invalid source URLs: {invalid}"
            )

        # A supported claim must identify evidence.
        if (
            item["verdict"] == "supported"
            and not item["source_urls"]
        ):
            item["verdict"] = "unsupported"
            item["explanation"] += (
                " No supporting source was identified."
            )

        checks.append(item)

    # Also reject URLs in the report that weren't retrieved.
    report_urls = set(
        url.rstrip(".,;:")
        for url in re.findall(
            r"https?://[^\s<>\]\)]+",
            state["report"],
        )
    )
    invalid_report_urls = sorted(
        report_urls - allowed_urls
    )

    passed = (
        len(checks) > 0
        and all(
            c["verdict"] == "supported"
            for c in checks
        )
        and not invalid_report_urls
    )

    if passed:
        next_action = "publish"

    elif (
        not state["research_retry_used"]
        and state["rounds"] < MAX_RESEARCH_ROUNDS
        and result.research_queries
    ):
        next_action = "research"

    elif state["revision_count"] < MAX_REVISIONS:
        next_action = "revise"

    else:
        next_action = "publish"

    print(
        f"[VERIFIER] Passed: {passed}; "
        f"next: {next_action}"
    )

    return {
        "verification": {
            "passed": passed,
            "checks": checks,
            "summary": result.summary,
            "research_queries": result.research_queries,
            "invalid_report_urls": invalid_report_urls,
            "next_action": next_action,
        },
        "verification_attempts": (
            state["verification_attempts"] + 1
        ),
        "research_retry_used": (
            state["research_retry_used"]
            or next_action == "research"
        ),
    }

def route_verification(state: ResearchState):
    return state["verification"]["next_action"]


def targeted_research_node(state: ResearchState):
    print("\n[RESEARCH] Investigating evidence gaps...")

    queries = state["verification"].get(
        "research_queries", []
    )

    tasks = [
        {
            "agent": "rag",
            "query": query,
        }
        for query in queries[:2]
        if query.strip()
    ]

    if not tasks:
        tasks = [
            {
                "agent": "rag",
                "query": state["question"],
            }
        ]

    return {
        "tasks": tasks,
        "verification_research_rounds": state.get("verification_research_rounds", 0) + 1,
    }


def publish_node(state: ResearchState):
    passed = state["verification"]["passed"]

    status = (
        "passed_automated_verification"
        if passed
        else "unresolved_verification_issues"
    )

    print(f"\n[PUBLISH] Status: {status}")

    return {"report_status": status}


builder = StateGraph(ResearchState)

# Existing Stage 5 nodes
builder.add_node("planner", planner_node)
builder.add_node("supervisor", supervisor_node)
builder.add_node("parallel_worker", parallel_worker)
builder.add_node("finalizer", finalizer_node)
builder.add_node("writer", writer_node)

# New Stage 6 nodes
builder.add_node("verifier", verifier_node)
builder.add_node("targeted_research", targeted_research_node)
builder.add_node("publish", publish_node)

# Initial research workflow
builder.add_edge(START, "planner")
builder.add_edge("planner", "supervisor")

builder.add_conditional_edges(
    "supervisor",
    dispatch_tasks,
    ["parallel_worker", "finalizer"],
)

builder.add_edge("parallel_worker", "supervisor")
builder.add_edge("finalizer", "writer")

# Verification feedback loop
builder.add_edge("writer", "verifier")

builder.add_conditional_edges(
    "verifier",
    route_verification,
    {
        "publish": "publish",
        "research": "targeted_research",
        "revise": "writer",
    },
)

# Reuse parallel workers for targeted research.
builder.add_conditional_edges(
    "targeted_research",
    dispatch_tasks,
    ["parallel_worker", "finalizer"],
)

builder.add_edge("publish", END)

graph = builder.compile()


if __name__ == "__main__":
    initial_state: ResearchState = {
        "question": (
            "Viscous fingering patterns in "
            "immiscible Hele-Shaw flow."
        ),
        "plan": "",
        "tasks": [],
        "papers": [],
        "research_notes": [],
        "search_history": [],
        "rounds": 0,
        "findings": "",
        "report": "",

        # Stage 6
        "verification": {},
        "verification_attempts": 0,
        "revision_count": 0,
        "research_retry_used": False,
        "report_status": "",
    }

    result = graph.invoke(
        initial_state,
        config={"recursion_limit": 80},
    )

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print(result["report"])

    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")

    for check in result["verification"]["checks"]:
        print(f"\nClaim: {check['claim']}")
        print(f"Verdict: {check['verdict']}")
        print(f"Reason: {check['explanation']}")

    print("\nExecution statistics:")
    print("Research rounds:", result["rounds"])
    print("Verification attempts:", result["verification_attempts"])
    print("Writer revisions:", result["revision_count"])
    print("Final status:", result["report_status"])
