# Stage 11A — Verification reliability patch

These changes are designed for the portfolio demo, not public deployment.

## Changes
1. `verification/citations.py`: Parse both `[E:id]` and grouped `[E:id1, E:id2]` citations. Detect malformed groups and unknown IDs independently of LLM output.
2. `verification/nodes.py`: The writer receives the complete list of failed claim assessments, with explicit instructions to strengthen, qualify, or remove each failed claim. The verifier exposes `unresolved_checks` and rejects malformed/unknown citation groups.
3. `graphs/workflow.py`: Human review displays unresolved claims and a prominent warning when the automated check did not pass. Approval of such a report preserves status `human_approved_with_unresolved_issues`.
4. Unit tests cover grouped and malformed citation IDs, mocked revision feedback, and verification failure gates.
5. Integration tests exercise the *real* human review/publish nodes with a fresh SQLite checkpoint, reconstructing the checkpointer before resuming approve/stop/revise paths. They do not run the full LLM-powered workflow.

## Validate locally

```bash
uv sync
uv run pytest -q
uv run multi-agent-research start
# In a different terminal/process after interruption:
uv run multi-agent-research status YOUR_THREAD_ID
uv run multi-agent-research resume YOUR_THREAD_ID
```

For a new demonstration, use a fresh thread ID. Checkpoints from previous graph releases may not have a compatible node/state schema. Do not expose the unauthenticated FastAPI server publicly.

## Test limitations

- In the isolated packaging environment, `python -m compileall -q src tests` and the four pure citation-parser tests passed.
- The dependency-complete test suite could not run there because offline dependency installation requires an uncached CUDA package. Run the full tests in your existing local `uv` environment.
- This patch does not guarantee that the LLM extracts every scientific claim or that a supported verdict is scientifically correct.
