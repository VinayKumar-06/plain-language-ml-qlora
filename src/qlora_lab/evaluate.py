"""Run the fixed held-out set and save predictions plus a human-review sheet."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

from qlora_lab.config import ExperimentConfig, config_from_mapping
from qlora_lab.data.io import read_jsonl
from qlora_lab.evaluation.metrics import save_human_review, save_predictions, summarize_predictions
from qlora_lab.modeling import load_generator
from qlora_lab.runtime import hardware_summary


def evaluate(
    config: ExperimentConfig, split_path: Path, output_dir: Path, adapter_path: str | None
) -> dict:
    examples = read_jsonl(split_path)
    generator = load_generator(config, adapter_path)
    predictions = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for example in examples:
        started = time.perf_counter()
        prediction = generator.generate(example["instruction"])
        predictions.append(
            {
                "id": example["id"],
                "prediction": prediction,
                "latency_seconds": round(time.perf_counter() - started, 4),
            }
        )
    metrics = summarize_predictions(examples, predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(output_dir / "predictions.jsonl", predictions)
    save_human_review(output_dir / "human-review.csv", examples, predictions)
    run = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_id": config.model_id,
        "adapter_path": adapter_path,
        "split_path": str(split_path),
        "decoding": {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.05,
        },
        "config": asdict(config),
        "hardware": hardware_summary(torch),
        "model_revision": generator.model_revision,
        "peak_gpu_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "peak_gpu_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        ),
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/small.yaml"))
    parser.add_argument("--split", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/eval-baseline"))
    parser.add_argument("--adapter", help="Optional local PEFT adapter directory.")
    args = parser.parse_args()
    config = config_from_mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    print(json.dumps(evaluate(config, args.split, args.output, args.adapter), indent=2))


if __name__ == "__main__":
    main()
