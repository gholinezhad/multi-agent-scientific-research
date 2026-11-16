
import os
import random
import time
import uuid

import arxiv
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

load_dotenv()

QDRANT_URL = os.getenv(
    "QDRANT_URL", "http://localhost:6333"
)
COLLECTION = os.getenv(
    "QDRANT_COLLECTION", "scientific_papers"
)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

SEARCH_QUERIES = [
    'all:"immiscible viscous fingering"',
    'all:"Hele-Shaw Cell"',
    'all:"simulation of viscous fingering"',
    'all:"Saffman-Taylor instability"',
]


def fetch_papers_with_retry(max_retries=4):
    """Fetch arXiv papers, retrying HTTP 429 responses with backoff."""
    client = arxiv.Client(
        page_size=20,
        delay_seconds=5,
        num_retries=2,
    )

    papers_by_url = {}

    for query in SEARCH_QUERIES:
        print(f"Searching arXiv: {query}")

        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        for attempt in range(max_retries):
            try:
                results = list(client.results(search))
                break
            except arxiv.HTTPError as error:
                if error.status != 429 or attempt == max_retries - 1:
                    raise

                wait = min(30 * (2**attempt), 120)
                wait += random.uniform(0, 5)
                print(
                    "arXiv rate limited (429). "
                    f"Retrying in {wait:.1f} seconds..."
                )
                time.sleep(wait)

        for paper in results:
            papers_by_url[paper.entry_id] = paper

    return list(papers_by_url.values())


def build_documents(papers):
    """Convert papers into LangChain Documents."""

    documents = []
    ids = []

    for paper in papers:
        document = Document(
            page_content=paper.summary,
            metadata={
                "title": paper.title,
                "authors": [
                    author.name
                    for author in paper.authors
                ],
                "published": (
                    paper.published.date().isoformat()
                ),
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "source_type": "arxiv_abstract",
            },
        )

        documents.append(document)

        # Deterministic ID prevents duplicate records
        # when we ingest the same paper again.
        ids.append(
            str(uuid.uuid5(uuid.NAMESPACE_URL, paper.entry_id))
        )

    return documents, ids


def main():
    papers = fetch_papers_with_retry()
    documents, ids = build_documents(papers)

    if not documents:
        print("No documents retrieved. Nothing to index.")
        return

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    client = QdrantClient(url=QDRANT_URL)

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
    )

    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    print(
        f"\nIndexed {len(documents)} unique papers "
        f"into Qdrant collection: {COLLECTION}"
    )


if __name__ == "__main__":
    main()
