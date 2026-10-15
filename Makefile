.PHONY: install-dev install-all prepare test lint baseline smoke train evaluate-base evaluate-adapter app

install-dev:
	uv sync --locked --extra dev

install-all:
	uv sync --locked --extra dev --extra model --extra train --extra demo

prepare:
	python -m qlora_lab.data.prepare --input data/raw/examples.jsonl --output data/processed --seed 42

test:
	pytest -q

lint:
	ruff check src tests app.py

baseline: prepare
	python -m qlora_lab.evaluate --config configs/small.yaml --split data/processed/test.jsonl --output outputs/eval-baseline

smoke: prepare
	python -m qlora_lab.train --config configs/small.yaml --train data/processed/train.jsonl --validation data/processed/validation.jsonl --smoke

train: prepare
	python -m qlora_lab.train --config configs/small.yaml --train data/processed/train.jsonl --validation data/processed/validation.jsonl

evaluate-base: prepare
	python -m qlora_lab.evaluate --config configs/small.yaml --split data/processed/test.jsonl --output outputs/eval-baseline

evaluate-adapter: prepare
	python -m qlora_lab.evaluate --config configs/small.yaml --split data/processed/test.jsonl --output outputs/eval-adapter --adapter outputs/adapter

app:
	python app.py
