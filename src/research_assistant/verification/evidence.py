
from typing import Any


def build_evidence_registry(
    papers: list[dict[str, Any]],
    max_items: int = 40,
) -> dict[str, dict[str, Any]]:
    """Build a stable mapping of chunk ID to evidence.

    Only full-text, chunk-identified passages are
    accepted. Abstract-only records aren't treated
    as page-specific evidence.
    """
    registry = {}

    for paper in papers:
        for passage in paper.get("passages", []):
            chunk_id = passage.get("chunk_id")
            page = passage.get("page")
            text = passage.get("text")

            if not chunk_id or not text:
                continue

            evidence_id = f"E:{chunk_id}"

            if evidence_id in registry:
                continue

            registry[evidence_id] = {
                "evidence_id": evidence_id,
                "paper_url": paper.get("url"),
                "title": paper.get("title"),
                "page": page,
                "text": text,
            }

    return dict(list(registry.items())[:max_items])
