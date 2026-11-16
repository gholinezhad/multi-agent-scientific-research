# Technical design and interview discussion notes

## Why LangGraph rather than a simple chain?

The workflow makes routing and state transitions explicit. A supervisor delegates tasks, parallel ReAct specialists search independent sources, and `Send` passes isolated worker input. `Annotated[list, operator.add]` reducers merge contributions. Verification can request additional research or writing revisions, and `interrupt()` preserves control for human approval. A checkpointer stores the thread between CLI runs.

## Why hybrid RAG?

Dense retrieval captures semantic similarity. Sparse BM25 retrieval matches distinctive scientific terminology. Qdrant combines candidate lists using reciprocal rank fusion (RRF), after which a cross-encoder rescoring step selects the highest-ranked passages. Dense, hybrid, and reranked configurations can be benchmarked using Recall@K, MRR@K, and nDCG@K; no performance improvement should be claimed without measured results on labeled data.

## Why passage-level evidence rather than paper-level citations?

Chunk IDs, page numbers, and paper URLs permit exact citation existence checks. The writer receives actual retrieved passages and cites IDs. A separate model attempts to assess semantic support. A deterministic parser catches unknown and malformed citations. A pass applies only to the assessed subset, not every statement. Similarity-based evidence candidates are guidance, not proof.

## Failure handling

Initial research, verification-triggered research, automated revisions, and human revisions have distinct budget counters. At exhausted budgets, the graph routes to human review while retaining a failed verification status; `approve` after a failed verification is marked `human_approved_with_unresolved_issues`. Stopping produces `stopped_without_approval`.

## Trade-offs

- SQLite is appropriate for a single-process portfolio demo, not multi-user public access.
- Synchronous API can block while the workflow runs.
- Cross-encoder model loading may take significant memory on first invocation.
- Extracting text from scientific PDFs can scramble equations, references, and figures.
- arXiv discovery is not equivalent to comprehensive scientific literature search.
- The default evidence registry caps passages; rare or later-retrieved evidence can be missed.
- API keys stay in `.env`; local server binds to loopback only.

## Roadmap for a real service

Authentication + thread ownership, durable worker queues, PostgreSQL checkpointing, idempotent retries, budgets/rate limits, secret management, secure PDF ingestion, telemetry, backups, and verified provenance snapshots. These are explicitly out of scope for v0.1.2.
