.PHONY: setup check lint types test bank train serve replay evaluate report demo

setup:
	uv sync

check: lint types test

lint:
	uv run ruff check src tests

types:
	uv run mypy src

test:
	uv run pytest

bank:             ## a labelled synthetic bank: 3,000 customers, 120 days
	uv run qh simulate --out data/bank.csv.gz

train: bank       ## train on the first 60 days; writes models/ and the live traffic split
	uv run qh train --data data/bank.csv.gz

serve:            ## scoring API and analyst queue on :8100
	uv run qh serve

replay:           ## stream the last 60 days into the running server
	uv run qh replay --rate 2000

evaluate:         ## the benchmark behind docs/benchmark.md (about 10 minutes)
	uv run qh evaluate --seeds 0 1 2

report:
	uv run qh report

demo:             ## everything at once, smaller bank
	uv run qh demo
