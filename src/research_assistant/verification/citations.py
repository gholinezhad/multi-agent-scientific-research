"""Deterministic, format-tolerant parsing for evidence-ID citations.

Accepted: [E:id], [E:id1, E:id2], [E:id1][E:id2].
Malformed E: references inside brackets are retained as invalid references.
"""
import re

BRACKET = re.compile(r"\[([^\[\]\n]+)\]")
EVIDENCE_ID = re.compile(r"E:[A-Za-z0-9][A-Za-z0-9_.:-]*")


def parse_evidence_citations(text: str) -> list[str]:
    """Return ordered, unique evidence identifiers from bracket citations."""
    seen: set[str] = set()
    found: list[str] = []
    for bracket in BRACKET.finditer(text):
        content = bracket.group(1)
        if 'E:' not in content:
            continue
        for raw in EVIDENCE_ID.findall(content):
            if raw not in seen:
                found.append(raw)
                seen.add(raw)
    return found


def malformed_evidence_citations(text: str) -> list[str]:
    """Return bracket groups that appear to cite evidence but contain invalid syntax.

    A group is valid only when it consists of one or more evidence IDs
    separated by commas (and optional whitespace).
    """
    invalid = []
    for match in BRACKET.finditer(text):
        content = match.group(1)
        if 'E:' not in content:
            continue
        parts = [part.strip() for part in content.split(',')]
        if not parts or any(not EVIDENCE_ID.fullmatch(part) for part in parts):
            invalid.append(match.group(0))
    return invalid


def validate_report_citations(report: str, registry: dict) -> dict:
    ids = parse_evidence_citations(report)
    loose = []
    ranges = [(m.start(), m.end()) for m in BRACKET.finditer(report)]
    for m in EVIDENCE_ID.finditer(report):
        if not any(a <= m.start() and m.end() <= b for a, b in ranges):
            loose.append(m.group(0))
    result = {
        'cited_ids': ids,
        'invalid_ids': sorted(set(ids) - set(registry)),
        'malformed_groups': malformed_evidence_citations(report),
    }
    if loose:
        result['loose_ids'] = sorted(set(loose))
    return result
