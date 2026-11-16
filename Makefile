
.PHONY: test lint api
test:
	uv run pytest -q
api:
	uv run uvicorn research_assistant.api:app --host 127.0.0.1 --port 8000
