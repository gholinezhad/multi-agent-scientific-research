
import json
import math
import statistics
from pathlib import Path

from research_assistant.retrieval.legacy import get_vector_store

K = 5


def normalize_url(url: str) -> str:
    return url.rstrip("/").replace(
        "http://", "https://"
    )


def retrieval_metrics(
    retrieved: list[str],
    relevant: set[str],
    k: int,
):
    top_k = retrieved[:k]

    # Binary relevance
    labels = [
        int(url in relevant)
        for url in top_k
    ]

    recall = sum(labels) / len(relevant)

    first_hit = next(
        (i + 1 for i, hit in enumerate(labels) if hit),
        None,
    )

    reciprocal_rank = (
        1 / first_hit if first_hit else 0.0
    )

    dcg = sum(
        hit / math.log2(i + 2)
        for i, hit in enumerate(labels)
    )

    ideal_hits = min(k, len(relevant))

    idcg = sum(
        1 / math.log2(i + 2)
        for i in range(ideal_hits)
    )

    ndcg = dcg / idcg if idcg else 0.0

    return {
        "recall": recall,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
    }


def main():
    examples = json.loads(
        Path("eval_dataset.json").read_text()
    )

    store = get_vector_store()
    scores = []

    for example in examples:
        query = example["query"]

        relevant = {
            normalize_url(url)
            for url in example["relevant_urls"]
            if not url.startswith("REPLACE_")
        }

        if not relevant:
            raise ValueError(
                f"Query has no gold labels: {query}"
            )

        results = store.similarity_search(
            query=query,
            k=K,
        )

        retrieved = [
            normalize_url(doc.metadata["url"])
            for doc in results
        ]

        metrics = retrieval_metrics(
            retrieved=retrieved,
            relevant=relevant,
            k=K,
        )

        scores.append(metrics)

        print(f"\nQuery: {query}")
        print("Retrieved:", retrieved)

        for name, value in metrics.items():
            print(f"{name}@{K}: {value:.3f}")

    print("\n" + "=" * 45)
    print("AGGREGATE RESULTS")

    for metric in ("recall", "mrr", "ndcg"):
        average = statistics.mean(
            row[metric] for row in scores
        )
        print(f"{metric}@{K}: {average:.3f}")


if __name__ == "__main__":
    main()
