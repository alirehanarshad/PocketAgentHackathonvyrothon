from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = """You are Pocket-Agent, an offline mobile assistant that either:
1. emits exactly one valid tool call wrapped in <tool_call>...</tool_call>
2. or replies in plain natural language when the user should be refused

You must follow this tool schema exactly:
{"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}
{"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}
{"tool": "convert",  "args": {"value": "number", "from_unit": "string", "to_unit": "string"}}
{"tool": "currency", "args": {"amount": "number", "from": "ISO3", "to": "ISO3"}}
{"tool": "sql",      "args": {"query": "string"}}
"""


def tool_call(tool: str, args: dict) -> str:
    payload = json.dumps({"tool": tool, "args": args}, ensure_ascii=True, separators=(",", ":"))
    return f"<tool_call>{payload}</tool_call>"


def format_sft(messages: list[dict], answer: str) -> str:
    chunks = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"]
    for message in messages:
        chunks.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
    chunks.append(f"<|im_start|>assistant\n{answer}<|im_end|>")
    return "\n".join(chunks)


def weather_examples() -> list[dict]:
    rows = []
    templates = [
        ("what's the weather in {location} in {unit}", lambda x: x),
        ("weather for {location} use {unit}", lambda x: x),
        ("temp in {location} please, {unit}", lambda x: x),
        ("mosam {location} {unit} me batao", lambda x: x),
    ]
    cities = ["Lahore", "Karachi", "Dubai", "Tokyo", "Berlin", "Toronto", "Islamabad"]
    for city in cities:
        for unit in ["C", "F"]:
            for template, _ in templates:
                prompt = template.format(location=city, unit=unit)
                rows.append(
                    {"messages": [{"role": "user", "content": prompt}], "answer": tool_call("weather", {"location": city, "unit": unit})}
                )
    return rows


def calendar_examples() -> list[dict]:
    rows = []
    dates = ["2026-04-18", "2026-05-01", "2026-06-10", "2026-12-31"]
    titles = ["Team sync", "Dentist", "Hackathon demo", "Flight to Dubai"]
    for date in dates:
        rows.append(
            {"messages": [{"role": "user", "content": f"list my calendar for {date}"}], "answer": tool_call("calendar", {"action": "list", "date": date})}
        )
    for date in dates:
        for title in titles:
            rows.append(
                {
                    "messages": [{"role": "user", "content": f"create a calendar event on {date} called {title}"}],
                    "answer": tool_call("calendar", {"action": "create", "date": date, "title": title}),
                }
            )
            rows.append(
                {
                    "messages": [{"role": "user", "content": f"schedule {title} for {date}"}],
                    "answer": tool_call("calendar", {"action": "create", "date": date, "title": title}),
                }
            )
    return rows


def convert_examples() -> list[dict]:
    rows = []
    pairs = [
        (3.5, "kilograms", "pounds"),
        (10, "miles", "kilometers"),
        (32, "F", "C"),
        (2, "liters", "milliliters"),
        (5, "feet", "meters"),
    ]
    for value, from_unit, to_unit in pairs:
        rows.append(
            {
                "messages": [{"role": "user", "content": f"convert {value} {from_unit} to {to_unit}"}],
                "answer": tool_call("convert", {"value": value, "from_unit": from_unit, "to_unit": to_unit}),
            }
        )
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"convert {value} {from_unit} to {to_unit}"},
                    {"role": "assistant", "content": tool_call("convert", {"value": value, "from_unit": from_unit, "to_unit": to_unit})},
                    {"role": "user", "content": f"now convert that to {to_unit} again"},
                ],
                "answer": tool_call("convert", {"value": value, "from_unit": from_unit, "to_unit": to_unit}),
            }
        )
    return rows


def currency_examples() -> list[dict]:
    rows = []
    examples = [
        (50, "USD", "EUR"),
        (1200, "PKR", "USD"),
        (99.99, "AED", "PKR"),
        (300, "GBP", "USD"),
        (70, "EUR", "JPY"),
    ]
    for amount, src, dst in examples:
        rows.append(
            {
                "messages": [{"role": "user", "content": f"convert {amount} {src} to {dst}"}],
                "answer": tool_call("currency", {"amount": amount, "from": src, "to": dst}),
            }
        )
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"i have {amount} {src}"},
                    {"role": "assistant", "content": "How can I help with that?"},
                    {"role": "user", "content": f"convert that to {dst}"},
                ],
                "answer": tool_call("currency", {"amount": amount, "from": src, "to": dst}),
            }
        )
    return rows


def sql_examples() -> list[dict]:
    rows = []
    queries = [
        "SELECT * FROM orders LIMIT 5",
        "SELECT name, price FROM products WHERE price > 100",
        "SELECT COUNT(*) FROM users",
        "SELECT city, AVG(salary) FROM employees GROUP BY city",
        "SELECT * FROM tasks WHERE status = 'open'",
    ]
    for query in queries:
        rows.append(
            {
                "messages": [{"role": "user", "content": f"run this sql query: {query}"}],
                "answer": tool_call("sql", {"query": query}),
            }
        )
        rows.append(
            {
                "messages": [{"role": "user", "content": f"sql: {query}"}],
                "answer": tool_call("sql", {"query": query}),
            }
        )
    return rows


def refusal_examples() -> list[dict]:
    prompts = [
        "tell me a joke",
        "who are you",
        "set an alarm for 7 am",
        "book me an uber",
        "convert that please",
        "call my mom",
        "send an email to ali",
        "play some music",
    ]
    answer = "I cannot help with that using the available tools."
    return [{"messages": [{"role": "user", "content": prompt}], "answer": answer} for prompt in prompts]


def build_dataset(seed: int) -> list[dict]:
    random.seed(seed)
    rows = []
    for fn in [
        weather_examples,
        calendar_examples,
        convert_examples,
        currency_examples,
        sql_examples,
        refusal_examples,
    ]:
        rows.extend(fn())
    random.shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--eval-size", type=int, default=24)
    args = parser.parse_args()

    rows = build_dataset(args.seed)
    eval_size = min(args.eval_size, len(rows) // 4)
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]

    train_sft = [{"text": format_sft(row["messages"], row["answer"])} for row in train_rows]
    eval_sft = [{"text": format_sft(row["messages"], row["answer"])} for row in eval_rows]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train_raw.jsonl", train_rows)
    write_jsonl(output_dir / "eval_raw.jsonl", eval_rows)
    write_jsonl(output_dir / "train_sft.jsonl", train_sft)
    write_jsonl(output_dir / "eval_sft.jsonl", eval_sft)

    print(f"Wrote {len(train_rows)} training rows and {len(eval_rows)} eval rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
