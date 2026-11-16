# Reproducible portfolio demonstration

The included `sample_trace.txt` is a **condensed excerpt from an actual local execution**, not a canned claim of a successful scientific verification. The run reached human review with unresolved claims and was stopped by the user.

## Reproduce on your machine

1. `cp .env.example .env` and set `OPENAI_API_KEY`.
2. `uv sync --locked` and `docker compose up -d`.
3. Populate the full-text Qdrant collection with legally accessible PDF files described in `examples/papers_manifest.json` (see root README).
4. `uv run pytest -q`.
5. `uv run multi-agent-research start --question "Viscous fingering patterns in immiscible Hele-Shaw flow"`.
6. Copy the printed thread ID; run `uv run multi-agent-research status THREAD_ID`.
7. Run `uv run multi-agent-research resume THREAD_ID`; choose `stop`, `revise`, or `approve` after inspecting the report. For an unverified report, prefer `stop`.

Live OpenAI responses and external search results are nondeterministic. The trace and evaluation score need not match the example. The example was collected before the v0.1.2 candidate-selection update.

**Do not commit** the `.env`, full checkpoint database, private PDFs, cached models, or private LangSmith traces.
