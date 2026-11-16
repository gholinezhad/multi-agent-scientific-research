
import os

from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient

load_dotenv()

URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv(
    "QDRANT_FULLTEXT_COLLECTION",
    "scientific_papers_fulltext_v1",
)


@lru_cache(maxsize=2)
def get_store(mode="hybrid"):
    dense = OpenAIEmbeddings(
        model=os.getenv(
            "DENSE_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
    )

    retrieval_mode = (
        RetrievalMode.HYBRID
        if mode == "hybrid"
        else RetrievalMode.DENSE
    )

    return QdrantVectorStore(
        client=QdrantClient(url=URL),
        collection_name=COLLECTION,
        embedding=dense,
        sparse_embedding=(
            FastEmbedSparse(
                model_name=os.getenv(
                    "SPARSE_EMBEDDING_MODEL",
                    "Qdrant/bm25",
                )
            )
            if mode == "hybrid"
            else None
        ),
        retrieval_mode=retrieval_mode,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


def retrieve_candidates(
    query: str,
    k: int = 20,
    mode: str = "hybrid",
):
    if mode not in ("hybrid", "dense"):
        raise ValueError("Mode must be hybrid or dense")

    return get_store(mode).similarity_search_with_score(
        query=query,
        k=k,
    )


if __name__ == "__main__":
    query = (
        "Viscous fingering patterns in "
        "immiscible Hele-Shaw flow."
    )

    results = retrieve_candidates(query)

    for doc, score in results[:10]:
        print(
            f"{score:.4f} | "
            f"{doc.metadata['title']} | "
            f"page {doc.metadata['page']}"
        )
