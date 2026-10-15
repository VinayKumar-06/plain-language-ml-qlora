"""Validate instruction records, remove duplicates, and split by concept group."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

SPLIT_NAMES = ("train", "validation", "test")
REQUIRED_TEXT_FIELDS = ("id", "category", "concept", "instruction", "response")


@dataclass(frozen=True)
class Record:
    id: str
    category: str
    concept: str
    instruction: str
    response: str


@dataclass
class PreparationReport:
    input_rows: int
    kept_rows: int
    dropped: list[dict[str, Any]]
    exact_duplicates: list[str]
    split_counts: dict[str, int]
    category_counts: dict[str, dict[str, int]]
    near_duplicate_candidates: list[dict[str, Any]]


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    input_rows = sum(bool(line.strip()) for line in lines)
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            dropped.append({"line": line_number, "reason": f"invalid JSON: {exc.msg}"})
            continue
        if not isinstance(value, dict):
            dropped.append({"line": line_number, "reason": "record must be a JSON object"})
            continue
        rows.append({"_line": line_number, **value})
    return rows, dropped, input_rows


def _validate_rows(rows: list[dict[str, Any]], dropped: list[dict[str, Any]]) -> list[Record]:
    valid: list[Record] = []
    seen_ids: set[str] = set()
    for row in rows:
        line = row.pop("_line")
        problem = None
        for field in REQUIRED_TEXT_FIELDS:
            value = row.get(field)
            if not isinstance(value, str):
                problem = f"{field} must be a string"
                break
            if not value.strip():
                problem = f"{field} must not be empty"
                break
        if problem:
            dropped.append({"line": line, "reason": problem})
            continue
        record = Record(**{key: row[key].strip() for key in REQUIRED_TEXT_FIELDS})
        if record.id in seen_ids:
            dropped.append({"line": line, "id": record.id, "reason": "duplicate record id"})
            continue
        seen_ids.add(record.id)
        valid.append(record)
    return valid


def _remove_exact_duplicates(records: list[Record]) -> tuple[list[Record], list[str]]:
    unique: list[Record] = []
    seen: set[tuple[str, str]] = set()
    duplicates: list[str] = []
    for record in records:
        key = (_normalized(record.instruction), _normalized(record.response))
        if key in seen:
            duplicates.append(record.id)
            continue
        seen.add(key)
        unique.append(record)
    return unique, duplicates


def _split_group_counts(group_count: int) -> tuple[int, int, int]:
    if group_count < 3:
        raise ValueError(
            "each category needs at least 3 distinct concepts for train/validation/test"
        )
    train_count = max(1, int(group_count * 0.6))
    validation_count = max(1, int(group_count * 0.2))
    test_count = group_count - train_count - validation_count
    while test_count < 1:
        if train_count > 1:
            train_count -= 1
        else:
            validation_count -= 1
        test_count = group_count - train_count - validation_count
    return train_count, validation_count, test_count


def _assign_concept_splits(records: list[Record], seed: int) -> dict[tuple[str, str], str]:
    concepts_by_category: dict[str, set[str]] = defaultdict(set)
    for record in records:
        concepts_by_category[record.category].add(record.concept)

    assignment: dict[tuple[str, str], str] = {}
    for category, concepts in sorted(concepts_by_category.items()):
        ordered = sorted(
            concepts,
            key=lambda concept: hashlib.sha256(
                f"{seed}\0{category}\0{concept}".encode()
            ).hexdigest(),
        )
        train_count, validation_count, _ = _split_group_counts(len(ordered))
        for index, concept in enumerate(ordered):
            if index < train_count:
                split = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            assignment[(category, concept)] = split
    return assignment


def _find_near_duplicate_candidates(
    records: list[Record], assignment: dict[tuple[str, str], str]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for left_index, left in enumerate(records):
        left_text = _normalized(left.instruction + " " + left.response)
        for right in records[left_index + 1 :]:
            left_group = (left.category, left.concept)
            right_group = (right.category, right.concept)
            if assignment[left_group] == assignment[right_group]:
                continue
            right_text = _normalized(right.instruction + " " + right.response)
            similarity = SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
            if similarity >= 0.92:
                candidates.append(
                    {"left_id": left.id, "right_id": right.id, "similarity": round(similarity, 3)}
                )
    return candidates


def prepare_dataset(input_path: Path, output_dir: Path, seed: int = 42) -> PreparationReport:
    rows, dropped, input_rows = _read_rows(input_path)
    valid = _validate_rows(rows, dropped)
    unique, duplicates = _remove_exact_duplicates(valid)
    if not unique:
        raise ValueError("no valid unique records remain after validation")

    assignments = _assign_concept_splits(unique, seed)
    splits: dict[str, list[Record]] = {name: [] for name in SPLIT_NAMES}
    categories: dict[str, Counter[str]] = defaultdict(Counter)
    for record in unique:
        split = assignments[(record.category, record.concept)]
        splits[split].append(record)
        categories[record.category][split] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        destination = output_dir / f"{split_name}.jsonl"
        with destination.open("w", encoding="utf-8") as stream:
            for record in split_records:
                stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    report = PreparationReport(
        input_rows=input_rows,
        kept_rows=len(unique),
        dropped=dropped,
        exact_duplicates=duplicates,
        split_counts={name: len(items) for name, items in splits.items()},
        category_counts={category: dict(counts) for category, counts in sorted(categories.items())},
        near_duplicate_candidates=_find_near_duplicate_candidates(unique, assignments),
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/examples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare_dataset(args.input, args.output, args.seed)
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
