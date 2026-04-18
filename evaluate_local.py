from __future__ import annotations

import json
import re
from pathlib import Path

from inference import run

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def load_jsonl(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_tool_call(text: str):
    match = TOOL_CALL_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def score_example(expected: str, predicted: str) -> float:
    expected_call = parse_tool_call(expected)
    predicted_call = parse_tool_call(predicted)

    if expected_call is None:
        return 1.0 if predicted_call is None else -0.5
    if predicted_call is None:
        return 0.0
    if predicted_call == expected_call:
        return 1.0
    if predicted_call.get("tool") == expected_call.get("tool"):
        return 0.5
    return 0.0


def main() -> int:
    rows = load_jsonl("local_test_set.jsonl")
    total = 0.0
    results = []

    for index, row in enumerate(rows, start=1):
        predicted = run(row["prompt"], row["history"])
        score = score_example(row["expected"], predicted)
        total += score
        results.append(
            {
                "id": index,
                "prompt": row["prompt"],
                "history": row["history"],
                "expected": row["expected"],
                "predicted": predicted,
                "score": score,
            }
        )
        print(f"[{index:02d}] score={score:.1f} prompt={row['prompt']}")

    mean_score = total / len(rows) if rows else 0.0
    Path("outputs").mkdir(exist_ok=True)
    with Path("outputs/local_eval_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=True)

    print(f"\nExamples: {len(rows)}")
    print(f"Mean score: {mean_score:.3f}")
    print("Detailed results: outputs/local_eval_results.json")
    return 0


if __name__ == "__main__":
    main()
