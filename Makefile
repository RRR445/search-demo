.PHONY: install dev run fmt lint test

install:
pip install -r requirements.txt

dev:
uvicorn app:app --reload --port 8080

run:
uvicorn app:app --host 0.0.0.0 --port 8080

fmt:
ruff check --fix . || true
black .

lint:
ruff check .

test:
python -m pytest -q
