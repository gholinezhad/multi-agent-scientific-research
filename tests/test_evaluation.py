
import pytest

from evaluation.evaluate_retrieval import retrieval_metrics
from evaluation.evaluate_report import evaluate_report


def test_perfect_retrieval():
    metrics = retrieval_metrics(
        retrieved=["A", "B", "C"],
        relevant={"A", "B", "C"},
        k=3,
    )

    assert metrics["recall"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg"] == 1.0


def test_retrieval_with_missing_paper():
    metrics = retrieval_metrics(
        retrieved=["X", "A", "Y"],
        relevant={"A", "B"},
        k=3,
    )

    assert metrics["recall"] == 0.5
    assert metrics["mrr"] == 0.5
    assert 0 < metrics["ndcg"] < 1


def test_citation_validation():
    state = {
        "report": (
            "See https://arxiv.org/abs/1234.5678 "
            "and https://example.com/fake"
        ),
        "papers": [
            {
                "url": "https://arxiv.org/abs/1234.5678"
            }
        ],
        "verification": {
            "checks": [
                {"verdict": "supported"},
                {"verdict": "unsupported"},
            ]
        },
    }

    result = evaluate_report(state)

    assert result["citation_validity"] == 0.5
    assert result["checked_claim_support_rate"] == 0.5
    assert len(result["invalid_citations"]) == 1
