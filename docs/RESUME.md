# Resume-ready descriptions

Use only after confirming the final local tests and full workflow. Do not claim measured gains in accuracy or that the app is deployed publicly.

## One-line project description

Built a LangGraph multi-agent scientific research assistant integrating parallel ReAct literature/RAG specialists, Qdrant hybrid retrieval, cross-encoder reranking, evidence-linked claim checking, durable checkpoints, and human-in-the-loop approval.

## Three-bullet resume entry

**Multi-Agent Scientific Research Assistant — Python, LangGraph, Qdrant, FastAPI**

- Architected a supervisor-driven multi-agent workflow with parallel ReAct specialists, structured task routing, LangGraph state reducers, and resumable human-in-the-loop review.
- Implemented full-text scientific retrieval using dense/sparse Qdrant hybrid search, reciprocal rank fusion, cross-encoder reranking, and stable page-level passage citations.
- Built a bounded claim-verification and revision loop with citation validation, SQLite checkpoints, a local FastAPI interface, and automated tests.

## Interview talking points

1. Why multi-agent routing instead of a single monolithic ReAct chain?
2. How LangGraph `Send` and additive reducers handle concurrent workers.
3. Why BM25 + dense retrieval + RRF + cross-encoder reranking are complementary.
4. Distinguish a real citation from a passage that *actually supports* a claim.
5. Explain bounded retries, human review, checkpoint persistence, and error handling.
6. Describe the prototype-to-production plan: authentication, multi-tenant isolation, async queue, PostgreSQL, budgets, observability.
