# Portfolio release notes — v0.1.2 candidate

## Highlights

- One canonical LangGraph workflow; supervisor-driven parallel ReAct specialists.
- Qdrant hybrid dense/sparse retrieval, RRF, and local cross-encoder reranking.
- Traceable full-text evidence IDs with page metadata and citation validation.
- Citation-aware revision: failed claims get ranked candidates from the **existing** evidence registry; candidates are explicitly not treated as verified support. Already-supported text should be preserved during revision.
- Explicit warnings for unresolved claims; approvals with issues cannot masquerade as verification success.
- Local CLI, FastAPI API bound to localhost, SQLite checkpoint restoration, human review.
- Offline tests plus a sanitized, correctly labeled example execution trace.

## Known limitations

- The claim extractor evaluates a capped set of important claims, not every assertion.
- Lexical candidate selection cannot prove entailment; verification remains probabilistic.
- Source extraction can lose figures/equations/tables, and the collection may have narrow paper coverage.
- Tool errors and external rate limits can affect quality. No public API security, user isolation, or cost ceilings.
- Local, single-process portfolio MVP **not appropriate for internet exposure**.
- Final end-to-end validation is required on the maintainer's Mac before tagging.

## Local verification

Run `uv sync --locked && uv run pytest -q`, then execute the complete CLI flow documented in `examples/demo/README.md`. Record observed pytest results and unresolved-claim counts; do not claim a verified report without independent review.
