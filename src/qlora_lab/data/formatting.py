"""Convert validated rows into the conversational prompt/completion format."""

from typing import Any

SYSTEM_MESSAGE = (
    "You are a patient machine-learning tutor. Explain ideas accurately in plain language. "
    "Use exactly three labeled lines: Definition:, Example:, and Caveat:. State uncertainty "
    "when a detail depends on context."
)


def to_prompt_completion(record: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    instruction = record.get("instruction", "").strip()
    response = record.get("response", "").strip()
    if not instruction or not response:
        raise ValueError("instruction and response must both contain text")
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": instruction},
        ],
        "completion": [{"role": "assistant", "content": response}],
    }


def format_many(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [to_prompt_completion(record) for record in records]
