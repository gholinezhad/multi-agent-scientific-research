"""Deterministic candidate selection from *already retrieved* evidence.

Lexical similarity is not scientific entailment: suggestions must still be
assessed by the evidence verifier and cannot make a claim supported.
"""
from __future__ import annotations

import re
from typing import Any

STOP_WORDS = frozenset("""a an and are as at be by can for from has have in into is it its of on or our that the their these this to was were which while with within about across approximately also both does such than through under using where""".split())


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower())
            if len(token) > 2 and token not in STOP_WORDS}


def suggest_evidence_for_claim(
    claim: str, registry: dict[str, dict[str, Any]],
    *, top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rank existing passages for revision guidance, without calling a model.

    Return IDs and metadata but not another copy of the passage text: the
    writer already receives the canonical evidence registry.
    """
    query = _tokens(claim)
    if not query or top_k <= 0:
        return []
    scores = []
    for evidence_id, record in registry.items():
        text_terms = _tokens(str(record.get("text", "")))
        if not text_terms:
            continue
        overlap = query & text_terms
        if not overlap:
            continue
        # Query coverage favors evidence addressing more of the claim.
        score = len(overlap) / len(query)
        scores.append((score, len(overlap), evidence_id, record))
    scores.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [{
        "evidence_id": evid,
        "lexical_coverage": round(score, 3),
        "paper_url": record.get("paper_url"),
        "page": record.get("page"),
    } for score, _, evid, record in scores[:top_k]]


def build_revision_plan(
    checks: list[dict[str, Any]], registry: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create actionable repair hints for failed claims only."""
    return [{
        "claim": check.get("claim", ""),
        "verdict": check.get("verdict", "unsupported"),
        "why_failed": check.get("explanation", ""),
        "missing_information": check.get("missing_information", ""),
        "previous_citations": check.get("source_ids", []),
        "candidate_passages_not_yet_verified": suggest_evidence_for_claim(
            check.get("claim", ""), registry
        ),
    } for check in checks if check.get("verdict") != "supported"]
