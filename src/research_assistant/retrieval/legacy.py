
import json
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

QDRANT_URL = os.getenv(
    "QDRANT_URL", "http://localhost:6333"
)
COLLECTION = os.getenv(
    "QDRANT_COLLECTION", "scientific_papers"
)


def get_vector_store():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION,
        url=QDRANT_URL,
    )


@tool
def search_papers(query: str, k: int = 5) -> str:
    """Search the local scientific paper knowledge base.

    Use this tool to retrieve relevant indexed paper
    abstracts and their bibliographic metadata.

    Args:
        query: Natural-language scientific search query.
        k: Number of retrieved papers, from 1 to 10.

    Returns:
        JSON containing paper abstracts, metadata,
        and vector similarity scores.
    """

    print(f"\n[TOOL] Searching Qdrant: {query}")

    try:
        vector_store = get_vector_store()

        results = vector_store.similarity_search_with_score(
            query=query,
            k=max(1, min(k, 10)),
        )

        papers = []

        for doc, score in results:
            papers.append(
                {
                    "title": doc.metadata["title"],
                    "authors": doc.metadata["authors"],
                    "published": doc.metadata["published"],
                    "url": doc.metadata["url"],
                    "pdf_url": doc.metadata["pdf_url"],
                    "abstract": doc.page_content,
                    "similarity_score": float(score),
                    "source_type": "qdrant",
                }
            )

        return json.dumps(
            {"papers": papers},
            ensure_ascii=False,
        )

    except Exception as exc:
        return json.dumps(
            {
                "papers": [],
                "error": str(exc),
            }
        )


if __name__ == "__main__":
    print(
        search_papers.invoke(
            {
                "query": (
                    "Viscous fingering patterns in "
                    "immiscible Hele-Shaw flow."
                ),
                "k": 3,
            }
        )
    )
