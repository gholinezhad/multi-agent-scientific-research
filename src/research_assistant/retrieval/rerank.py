
import os

from functools import lru_cache

from dotenv import load_dotenv

from research_assistant.retrieval.hybrid import retrieve_candidates

load_dotenv()


from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(
        os.getenv(
            "RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
    )


def rerank(
    query: str,
    candidates,
    top_k: int = 5,
):
    if not candidates:
        return []

    documents = [
        doc for doc, _ in candidates
    ]

    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    scores = get_reranker().predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda pair: float(pair[1]),
        reverse=True,
    )

    return [
        (doc, float(score))
        for doc, score in ranked[:top_k]
    ]


def retrieve_reranked(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
):
    candidates = retrieve_candidates(
        query=query,
        k=candidate_k,
        mode="hybrid",
    )

    return rerank(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )


if __name__ == "__main__":
    query = (
        "Viscous fingering patterns in "
        "immiscible Hele-Shaw flow."
    )

    results = retrieve_reranked(
        query=query,
        top_k=5,
        candidate_k=20,
    )

    for doc, score in results:
        print(
            score,
            doc.metadata["title"],
            doc.metadata["page"],
        )
        print(doc.page_content[:200])
