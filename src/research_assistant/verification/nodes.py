
import json
import os
import re
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from research_assistant.verification.evidence import build_evidence_registry
from research_assistant.verification.routing import decide_next_action
from research_assistant.verification.selection import build_revision_plan
from research_assistant.verification.citations import (
    parse_evidence_citations, validate_report_citations,
)

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0,
)

MAX_REVISIONS = 2
MAX_RESEARCH_ROUNDS = 2


def evidence_writer_node(state):
    registry = build_evidence_registry(
        state["papers"]
    )

    is_revision = bool(
        state.get("report")
    )

    if is_revision:
        print("\n[WRITER] Revising report...")
    else:
        print(
            "\n[WRITER] Generating evidence-linked report..."
        )

    feedback = state.get("verification", {})
    failed_checks = [
        check for check in feedback.get("checks", [])
        if check.get("verdict") != "supported"
    ]
    feedback_package = {
        "failed_claims": failed_checks,
        "revision_plan": build_revision_plan(failed_checks, registry),
        "invalid_citations": feedback.get("invalid_report_ids", []),
        "malformed_citations": feedback.get("malformed_report_citations", []),
        "loose_citations": feedback.get("loose_report_citations", []),
        "next_action": feedback.get("next_action"),
    }

    response = llm.invoke([
        (
            "system",
            "You are an evidence-grounded scientific "
            "research writer.\n"
            "Write a structured scientific report.\n"
            "Every substantive scientific claim must "
            "cite one or more supplied evidence IDs.\n"
            "Cite each evidence ID in its own brackets, "
            "for example [E:chunk-1] [E:chunk-2].\n"
            "Copy the full evidence ID exactly.\n"
            "Do not cite an ID absent from the registry.\n"
            "Do not place citations solely in a references section; "
            "attach them directly to each substantive claim, including "
            "the abstract and conclusion when stating findings. "
            "Do not make claims beyond what the "
            "passages establish.\n"
            "If evidence is insufficient, state that "
            "explicitly instead of inventing support.\n"
            "Evidence comes from PDF-extracted text; "
            "equations and tables may be incomplete.\n"
            "When revising, preserve already-supported statements "
            "and their citations unless they conflict with corrections. "
            "Use the revision_plan to repair only failed claims. "
            "Candidate passages are lexical suggestions, NOT verified support; "
            "read each passage and cite it only when it truly supports the claim. "
            "When revising, resolve EACH failed claim: "
            "(1) cite stronger existing evidence, "
            "(2) narrow the wording to match the evidence, "
            "or (3) remove the unsupported claim and state "
            "the evidence gap. Never silently repeat a failed "
            "claim. Do not invent evidence. "
            "Honor human feedback separately."
        ),
        (
            "human",
            f"Question:\n{state['question']}\n\n"
            f"Research notes:\n"
            f"{str(state.get('research_notes', []))[:10000]}\n\n"
            f"Evidence registry:\n"
            f"{json.dumps(registry, ensure_ascii=False)}\n\n"
            f"Previous report:\n"
            f"{state.get('report', '') if is_revision else ''}\n\n"
            f"Verification feedback:\n"
            f"{json.dumps(feedback_package, ensure_ascii=False)}\n\n"
            f"Human instructions:\n"
            f"{state.get('human_feedback', '')}"
        ),
    ])

    # Human revisions and automatic verification revisions have separate budgets.
    human_requested = state.get("human_action") == "revise"
    return {
        "report": str(response.content),
        "revision_count": state.get("revision_count", 0) + int(is_revision),
        "auto_revision_count": state.get("auto_revision_count", 0)
            + int(is_revision and not human_requested),
        "human_action": "" if human_requested else state.get("human_action", ""),
        "human_feedback": "" if human_requested else state.get("human_feedback", ""),
    }


class ScientificClaim(BaseModel):
    claim: str = Field(
        description="One independently checkable scientific claim"
    )
    citation_ids: list[str] = Field(
        description="Exact evidence IDs cited for this claim"
    )


class ExtractedClaims(BaseModel):
    claims: list[ScientificClaim]


claim_extractor = llm.with_structured_output(
    ExtractedClaims
)


def extract_claims(
    report: str,
) -> list[ScientificClaim]:
    result = claim_extractor.invoke([
        (
            "system",
            "Extract up to 12 important, independently "
            "checkable scientific claims from the report. "
            "Prioritize comparative, quantitative, "
            "causal, and strong generalization claims. "
            "Include uncited claims with an empty "
            "citation_ids list. "
            "Preserve the claim's original scope. "
            "Copy citation IDs exactly as written. "
            "Do not invent citations or claims."
        ),
        ("human", report),
    ])

    return result.claims[:12]


class EvidenceAssessment(BaseModel):
    verdict: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
    ]

    explanation: str = Field(
        description="Justification based on supplied evidence"
    )

    missing_information: str = Field(
        description="Evidence needed to establish the claim"
    )


assessment_llm = llm.with_structured_output(
    EvidenceAssessment
)

def validate_citations(
    citation_ids: list[str],
    registry: dict,
):
    valid = [
        cid
        for cid in citation_ids
        if cid in registry
    ]

    invalid = [
        cid
        for cid in citation_ids
        if cid not in registry
    ]

    return valid, invalid


def check_claim(claim: ScientificClaim, registry: dict):
    # Accept both individual and comma-separated bracket groups from the extractor.
    normalized_ids = []
    for cid in claim.citation_ids:
        normalized_ids.extend(parse_evidence_citations(cid) if "[" in cid else [
            piece.strip() for piece in cid.split(",") if piece.strip()
        ])
    valid_ids, invalid_ids = validate_citations(normalized_ids, registry)

    if invalid_ids:
        return {
            "claim": claim.claim,
            "verdict": "unsupported",
            "source_ids": valid_ids,
            "invalid_ids": invalid_ids,
            "explanation": "Unknown cited evidence IDs.",
            "missing_information": (
                "Replace invalid references with "
                "retrieved supporting passages."
            ),
        }

    if not valid_ids:
        return {
            "claim": claim.claim,
            "verdict": "unsupported",
            "source_ids": [],
            "invalid_ids": [],
            "explanation": "No evidence cited.",
            "missing_information": (
                "A supporting passage is required."
            ),
        }

    passages = [
        registry[cid]
        for cid in valid_ids
    ]

    result = assessment_llm.invoke([
        (
            "system",
            "You are an independent scientific "
            "evidence verifier.\n"
            "Assess the claim using ONLY the "
            "provided passages.\n"
            "Supported: passages justify the entire "
            "claim, including qualifiers and scope.\n"
            "Partially supported: some but not all "
            "of the claim is established.\n"
            "Unsupported: evidence is missing or "
            "does not establish the claim.\n"
            "Contradicted: supplied evidence "
            "directly conflicts with the claim.\n"
            "Do not use outside knowledge. "
            "Do not infer universal conclusions "
            "from one benchmark. "
            "Explain your decision concisely."
        ),
        (
            "human",
            f"Claim:\n{claim.claim}\n\n"
            f"Cited passages:\n"
            f"{json.dumps(passages, ensure_ascii=False)}"
        ),
    ])

    return {
        "claim": claim.claim,
        "verdict": result.verdict,
        "source_ids": valid_ids,
        "invalid_ids": [],
        "explanation": result.explanation,
        "missing_information": result.missing_information,
    }


def passage_verifier_node(state):
    print("\n[VERIFIER] Checking passage-level evidence...")

    registry = build_evidence_registry(
        state["papers"]
    )

    report = state["report"]
    claims = extract_claims(report)

    checks = [
        check_claim(claim, registry)
        for claim in claims
    ]

    # Independently identify every literal evidence
    # citation in the generated report.
    report_citations = validate_report_citations(report, registry)
    cited_ids = set(report_citations["cited_ids"])
    invalid_report_ids = report_citations["invalid_ids"]
    malformed_report_citations = report_citations["malformed_groups"]
    loose_report_citations = report_citations.get("loose_ids", [])

    passed = (
        bool(checks)
        and all(
            check["verdict"] == "supported"
            for check in checks
        )
        and not invalid_report_ids
        and not malformed_report_citations
        and not loose_report_citations
        and bool(cited_ids)
        and bool(registry)
    )

    failed_checks = [
        c for c in checks
        if c["verdict"] != "supported"
    ]

    # Suggest a targeted search for missing evidence.
    research_queries = [
        c["claim"]
        for c in failed_checks
        if c["verdict"] in (
            "unsupported",
            "partially_supported",
        )
    ][:2]

    next_action = decide_next_action(
        passed=passed,
        has_research_queries=bool(research_queries),
        verification_research_rounds=state.get("verification_research_rounds", 0),
        auto_revision_count=state.get("auto_revision_count", 0),
    )

    verification = {
        "passed": passed,
        "checks": checks,
        "summary": (
            f"{sum(c['verdict'] == 'supported' for c in checks)}"
            f" of {len(checks)} extracted claims supported."
        ),
        "invalid_report_ids": invalid_report_ids,
        "malformed_report_citations": malformed_report_citations,
        "loose_report_citations": loose_report_citations,
        "unresolved_checks": failed_checks,
        "research_queries": research_queries,
        "next_action": next_action,
    }

    print(
        f"[VERIFIER] Claims: {len(checks)} | "
        f"Passed: {passed} | Next: {next_action}"
    )

    return {
        "verification": verification,
        "verification_attempts": (
            state["verification_attempts"] + 1
        ),
        "research_retry_used": (
            state["research_retry_used"]
            or next_action == "research"
        ),
    }

