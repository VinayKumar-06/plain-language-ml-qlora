import json
from pathlib import Path

import pytest

from qlora_lab.data.formatting import SYSTEM_MESSAGE, to_prompt_completion
from qlora_lab.data.prepare import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data/raw/examples.jsonl"


def test_project_dataset_has_reproducible_concept_disjoint_splits(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = prepare_dataset(RAW_DATA, first_dir, seed=17)
    second = prepare_dataset(RAW_DATA, second_dir, seed=17)

    assert first == second
    assert first.split_counts == {"train": 28, "validation": 12, "test": 12}
    concepts = {}
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line) for line in (first_dir / f"{split}.jsonl").read_text().splitlines()
        ]
        for row in rows:
            concepts.setdefault((row["category"], row["concept"]), set()).add(split)
    assert all(len(splits) == 1 for splits in concepts.values())
    assert {category: counts["test"] for category, counts in first.category_counts.items()} == {
        "data-practice": 4,
        "evaluation": 4,
        "foundations": 4,
    }
    for split in ("train", "validation", "test"):
        assert (first_dir / f"{split}.jsonl").read_bytes() == (
            second_dir / f"{split}.jsonl"
        ).read_bytes()


def test_invalid_json_rows_and_exact_duplicates_are_reported(tmp_path):
    source = tmp_path / "raw.jsonl"
    records = [
        {
            "id": "a",
            "category": "basics",
            "concept": "one",
            "instruction": "Define x",
            "response": "Answer x",
        },
        {
            "id": "b",
            "category": "basics",
            "concept": "one",
            "instruction": " define X ",
            "response": " answer x ",
        },
        {
            "id": "bad",
            "category": "basics",
            "concept": "one",
            "instruction": " ",
            "response": "ok",
        },
        {
            "id": "c",
            "category": "basics",
            "concept": "two",
            "instruction": "Define y",
            "response": "Answer y",
        },
        {
            "id": "d",
            "category": "basics",
            "concept": "three",
            "instruction": "Define z",
            "response": "Answer z",
        },
    ]
    lines = [json.dumps(row) for row in records] + ["{broken"]
    source.write_text("\n".join(lines), encoding="utf-8")
    report = prepare_dataset(source, tmp_path / "splits", seed=3)
    assert report.input_rows == 6
    assert report.kept_rows == 3
    assert report.exact_duplicates == ["b"]
    assert len(report.dropped) == 2
    assert any("invalid JSON" in item["reason"] for item in report.dropped)
    assert any("must not be empty" in item["reason"] for item in report.dropped)


def test_dataset_rejects_category_with_too_few_concepts(tmp_path):
    source = tmp_path / "raw.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": str(i),
                    "category": "only",
                    "concept": f"c{i}",
                    "instruction": f"q{i}",
                    "response": f"a{i}",
                }
            )
            for i in range(2)
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least 3 distinct concepts"):
        prepare_dataset(source, tmp_path / "splits")


def test_chat_format_masks_context_as_prompt_and_response_as_completion():
    formatted = to_prompt_completion(
        {"instruction": "Explain recall", "response": "Definition: ..."}
    )
    assert formatted["prompt"][0] == {"role": "system", "content": SYSTEM_MESSAGE}
    assert formatted["prompt"][1]["content"] == "Explain recall"
    assert formatted["completion"] == [{"role": "assistant", "content": "Definition: ..."}]
    with pytest.raises(ValueError):
        to_prompt_completion({"instruction": " ", "response": "answer"})
