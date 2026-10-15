import pytest

from qlora_lab.evaluation.metrics import (
    has_required_structure,
    save_human_review,
    summarize_predictions,
)

GOOD = "Definition: A.\nExample: B.\nCaveat: C."


def test_format_check_requires_all_labels_once_and_in_order():
    assert has_required_structure(GOOD)
    assert not has_required_structure("Definition: A.\nCaveat: C.")
    assert not has_required_structure("Example: B.\nDefinition: A.\nCaveat: C.")
    assert not has_required_structure("Definition: A.\nExample: B.\nCaveat: C.\nCaveat: D.")


def test_metrics_are_reported_overall_and_per_category_without_claiming_correctness():
    examples = [
        {"id": "1", "category": "basics", "instruction": "q", "response": GOOD},
        {"id": "2", "category": "data", "instruction": "q", "response": GOOD},
    ]
    predictions = [
        {"id": "1", "prediction": GOOD, "latency_seconds": 0.5},
        {"id": "2", "prediction": "unstructured", "latency_seconds": 1.5},
    ]
    result = summarize_predictions(examples, predictions)
    assert result["overall"]["format_compliance"] == 0.5
    assert result["by_category"]["basics"]["format_compliance"] == 1.0
    assert "does not establish factual correctness" in result["interpretation"]


def test_metrics_reject_missing_or_unexpected_prediction_ids():
    examples = [{"id": "expected", "category": "basics", "instruction": "q", "response": GOOD}]
    with pytest.raises(ValueError, match="prediction ids do not match"):
        summarize_predictions(examples, [{"id": "other", "prediction": GOOD}])


def test_metrics_reject_duplicate_prediction_ids():
    example = {"id": "expected", "category": "basics", "instruction": "q", "response": GOOD}
    with pytest.raises(ValueError, match="duplicate ids"):
        summarize_predictions([example], [{"id": "expected"}, {"id": "expected"}])


def test_review_sheet_keeps_rating_blank_until_a_person_reviews(tmp_path):
    examples = [{"id": "1", "category": "basics", "instruction": "q", "response": GOOD}]
    predictions = [{"id": "1", "prediction": GOOD}]
    path = tmp_path / "review.csv"
    save_human_review(path, examples, predictions)
    assert ",," in path.read_text(encoding="utf-8")
