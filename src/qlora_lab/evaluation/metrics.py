"""Transparent output checks; these are not semantic correctness metrics."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_LABELS = ("Definition:", "Example:", "Caveat:")


def has_required_structure(text: str) -> bool:
    positions: list[int] = []
    for label in EXPECTED_LABELS:
        if text.count(label) != 1:
            return False
        positions.append(text.index(label))
    if positions != sorted(positions):
        return False
    for index, start in enumerate(positions):
        end = positions[index + 1] if index + 1 < len(positions) else len(text)
        if not text[start + len(EXPECTED_LABELS[index]) : end].strip():
            return False
    return True


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def summarize_predictions(
    examples: list[dict[str, str]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {prediction.get("id"): prediction for prediction in predictions}
    expected_ids = {example["id"] for example in examples}
    if len(by_id) != len(predictions):
        raise ValueError("predictions contain duplicate ids")
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        unexpected = sorted(set(by_id) - expected_ids)
        raise ValueError(
            "prediction ids do not match evaluation rows "
            f"(missing={missing}, unexpected={unexpected})"
        )

    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    for example in examples:
        prediction = by_id[example["id"]]
        row = {
            "id": example["id"],
            "category": example["category"],
            "prediction": str(prediction.get("prediction", "")),
            "latency_seconds": float(prediction.get("latency_seconds", 0.0)),
        }
        all_rows.append(row)
        category_rows[row["category"]].append(row)

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "examples": len(rows),
            "format_compliance": round(
                sum(has_required_structure(row["prediction"]) for row in rows) / len(rows), 4
            )
            if rows
            else 0.0,
            "mean_words": round(sum(_word_count(row["prediction"]) for row in rows) / len(rows), 2)
            if rows
            else 0.0,
            "mean_latency_seconds": round(
                sum(row["latency_seconds"] for row in rows) / len(rows), 4
            )
            if rows
            else 0.0,
        }

    return {
        "overall": aggregate(all_rows),
        "by_category": {
            category: aggregate(rows) for category, rows in sorted(category_rows.items())
        },
        "interpretation": (
            "Format compliance measures only the requested labels and order. It does not establish "
            "factual correctness, usefulness, or improvement; review paired outputs manually."
        ),
    }


def save_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
    )


def save_human_review(
    path: Path, examples: list[dict[str, str]], predictions: list[dict[str, Any]]
) -> None:
    by_id = {row["id"]: row for row in predictions}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "id",
                "category",
                "instruction",
                "reference",
                "prediction",
                "rating",
                "review_note",
            ),
        )
        writer.writeheader()
        for example in examples:
            writer.writerow(
                {
                    "id": example["id"],
                    "category": example["category"],
                    "instruction": example["instruction"],
                    "reference": example["response"],
                    "prediction": by_id[example["id"]]["prediction"],
                    "rating": "",
                    "review_note": "",
                }
            )
