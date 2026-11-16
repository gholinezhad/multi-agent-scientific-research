
import hashlib
import json
import os
import uuid
from pathlib import Path

import pymupdf as fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import (
    QdrantVectorStore,
    FastEmbedSparse,
    RetrievalMode,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient

load_dotenv()

URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv(
    "QDRANT_FULLTEXT_COLLECTION",
    "scientific_papers_fulltext_v1",
)

DENSE_MODEL = os.getenv(
    "DENSE_EMBEDDING_MODEL",
    "text-embedding-3-small",
)
SPARSE_MODEL = os.getenv(
    "SPARSE_EMBEDDING_MODEL",
    "Qdrant/bm25",
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_paper(entry):
    path = Path(entry["path"])

    if not path.is_file():
        raise FileNotFoundError(path)

    paper_id = entry["url"].rstrip("/").split("/")[-1]
    documents = []
    ids = []

    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if not text:
                print(
                    f"Warning: no extracted text on "
                    f"{path.name}, page {page_number}"
                )
                continue

            chunks = splitter.split_text(text)

            for chunk_index, chunk in enumerate(chunks):
                content_hash = hashlib.sha256(
                    chunk.encode("utf-8")
                ).hexdigest()[:16]

                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{entry['url']}:{page_number}:"
                        f"{chunk_index}:{content_hash}",
                    )
                )

                doc = Document(
                    page_content=chunk,
                    metadata={
                        "paper_id": paper_id,
                        "title": entry["title"],
                        "url": entry["url"],
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                        "source_type": "fulltext",
                    },
                )

                documents.append(doc)
                ids.append(chunk_id)

    return documents, ids


def main():
    manifest = json.loads(
        Path("examples/papers_manifest.json").read_text()
    )

    all_documents = []
    all_ids = []

    for entry in manifest:
        docs, ids = load_paper(entry)

        print(
            f"{entry['title']}: "
            f"{len(docs)} chunks"
        )

        all_documents.extend(docs)
        all_ids.extend(ids)

    if not all_documents:
        raise RuntimeError(
            "No extractable full-text documents found"
        )

    print(
        f"\nTotal chunks to index: "
        f"{len(all_documents)}"
    )

    dense = OpenAIEmbeddings(model=DENSE_MODEL)
    sparse = FastEmbedSparse(model_name=SPARSE_MODEL)

    client = QdrantClient(url=URL)

    if client.collection_exists(COLLECTION):
        # Existing hybrid collection. Upsert by ID.
        store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION,
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
    else:
        # Create the collection with both vector types.
        store = QdrantVectorStore.from_documents(
            documents=all_documents,
            embedding=dense,
            sparse_embedding=sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            url=URL,
            collection_name=COLLECTION,
            vector_name="dense",
            sparse_vector_name="sparse",
            ids=all_ids,
        )

        print("Created new hybrid collection.")
        return

    # Process in modest batches.
    for start in range(0, len(all_documents), 32):
        stop = start + 32

        store.add_documents(
            documents=all_documents[start:stop],
            ids=all_ids[start:stop],
        )

    print("Full-text hybrid indexing complete.")


if __name__ == "__main__":
    main()
