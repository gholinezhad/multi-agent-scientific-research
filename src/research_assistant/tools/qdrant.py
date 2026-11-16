
import json

from langchain_core.tools import tool

from research_assistant.retrieval.rerank import retrieve_reranked


@tool
def search_papers(query: str, k: int = 5) -> str:
    """Search indexed full-text scientific papers.

    Performs dense + BM25 hybrid retrieval, followed
    by cross-encoder reranking.

    Args:
        query: Focused scientific research question.
        k: Number of top passages, between 1 and 10.

    Returns:
        JSON containing source papers and page-labeled
        supporting passages extracted from full text.
    """

    print(f"\n[TOOL] Hybrid Qdrant search: {query}")

    try:
        results = retrieve_reranked(
            query=query,
            top_k=max(1, min(k, 10)),
            candidate_k=20,
        )

        grouped = {}

        for doc, score in results:
            metadata = doc.metadata
            url = metadata["url"]

            if url not in grouped:
                grouped[url] = {
                    "title": metadata["title"],
                    "authors": [],
                    "published": None,
                    "url": url,
                    "source_type": "fulltext",
                    "passages": [],
                }

            grouped[url]["passages"].append({
                "chunk_id": metadata["chunk_id"],
                "page": metadata["page"],
                "text": doc.page_content,
                "rerank_score": score,
            })

        papers = []

        for paper in grouped.values():
            # Compatibility with the existing writer
            # and verifier, which expect "abstract".
            # The content is now full-text evidence,
            # not an actual publication abstract.
            paper["abstract"] = "\n\n".join(
                f"[Page {p['page']}, "
                f"chunk {p['chunk_id']}]\n"
                f"{p['text']}"
                for p in paper["passages"]
            )

            papers.append(paper)

        return json.dumps(
            {"papers": papers},
            ensure_ascii=False,
        )

    except Exception as exc:
        return json.dumps({
            "papers": [],
            "error": str(exc),
        })
