
from research_assistant.verification.evidence import build_evidence_registry
from research_assistant.verification.nodes import validate_citations


def test_evidence_registry_deduplication():
    papers = [
        {
            "title": "Example paper",
            "url": "https://arxiv.org/abs/1234.5678",
            "passages": [
                {
                    "chunk_id": "chunk-001",
                    "page": 3,
                    "text": "A scientific passage.",
                },
                {
                    "chunk_id": "chunk-001",
                    "page": 3,
                    "text": "A scientific passage.",
                },
            ],
        }
    ]

    registry = build_evidence_registry(papers)

    assert len(registry) == 1
    assert "E:chunk-001" in registry
    assert registry["E:chunk-001"]["page"] == 3


def test_invalid_citation():
    registry = {
        "E:valid": {
            "text": "Supporting evidence."
        }
    }

    valid, invalid = validate_citations(
        ["E:valid", "E:invented"],
        registry,
    )

    assert valid == ["E:valid"]
    assert invalid == ["E:invented"]


def test_abstract_without_chunk_id():
    papers = [
        {
            "url": "https://arxiv.org/abs/1234.5678",
            "abstract": "Abstract-only evidence",
        }
    ]

    registry = build_evidence_registry(papers)

    assert registry == {}
