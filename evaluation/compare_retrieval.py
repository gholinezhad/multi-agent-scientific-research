
import json
import statistics
from pathlib import Path

from evaluation.evaluate_retrieval import retrieval_metrics
from research_assistant.retrieval.hybrid import retrieve_candidates
from research_assistant.retrieval.rerank import retrieve_reranked

K = 5


def search(query, mode):
    if mode == "reranked":
        results = retrieve_reranked(
            query,
            top_k=K,
            candidate_k=20,
        )
    else:
        results = retrieve_candidates(
            query,
            k=K,
            mode=mode,
        )

    return [
        doc.metadata["chunk_id"]
        for doc, _ in results
    ]


def main():
    examples = json.loads(
        Path("examples/eval_passages.json").read_text()
    )

    for mode in ("dense", "hybrid", "reranked"):
        all_scores = []

        for example in examples:
            relevant = set(
                example["relevant_chunk_ids"]
            )

            if (
                not relevant
                or any(
                    item.startswith("REPLACE_")
                    for item in relevant
                )
            ):
                raise ValueError(
                    "Replace all placeholder chunk IDs"
                )

            retrieved = search(
                example["query"], mode
            )

            metrics = retrieval_metrics(
                retrieved=retrieved,
                relevant=relevant,
                k=K,
            )

            all_scores.append(metrics)

        print(f"\n{mode.upper()}")

        for name in ("recall", "mrr", "ndcg"):
            average = statistics.mean(
                score[name] for score in all_scores
            )
            print(f"{name}@{K}: {average:.3f}")


if __name__ == "__main__":
    main()
