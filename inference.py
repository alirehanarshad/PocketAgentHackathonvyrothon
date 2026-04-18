from __future__ import annotations

import json
import os
import re
from functools import lru_cache

from llama_cpp import Llama

MODEL_PATH = os.environ.get("POCKET_AGENT_MODEL", "./model.gguf")

SYSTEM_PROMPT = """You are Pocket-Agent, an offline mobile assistant that either:
1. emits exactly one valid tool call wrapped in <tool_call>...</tool_call>
2. or replies in plain natural language when the user should be refused

You must follow this tool schema exactly:
{"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}
{"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string?"}}
{"tool": "convert",  "args": {"value": "number", "from_unit": "string", "to_unit": "string"}}
{"tool": "currency", "args": {"amount": "number", "from": "ISO3", "to": "ISO3"}}
{"tool": "sql",      "args": {"query": "string"}}

Rules:
- Return a tool call for unambiguous requests that fit one of the five tools.
- Use conversation history to resolve references like "that", "it", or follow-up conversions.
- Refuse in plain text when no tool fits, the request is casual chat, the user asks for a missing tool, or a reference is ambiguous.
- Keep argument values faithful to the user request. Do not invent dates, ISO codes, units, or titles.
- Never output analysis, explanations, or markdown fences.
- When emitting a tool call, output only the wrapped JSON.
"""

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
NUMBER_RE = r"-?\d+(?:\.\d+)?"

CURRENCY_ALIASES = {
    "usd": "USD",
    "eur": "EUR",
    "euro": "EUR",
    "euros": "EUR",
    "pkr": "PKR",
    "aed": "AED",
    "gbp": "GBP",
    "jpy": "JPY",
}

REFUSAL_TEXT = "I cannot help with that using the available tools."
REFUSAL_HINTS = ("joke", "who are you", "alarm", "uber", "call my", "email", "music")


@lru_cache(maxsize=1)
def get_llm() -> Llama:
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=max(1, os.cpu_count() or 1),
        n_batch=256,
        verbose=False,
    )


def _emit(tool: str, args: dict) -> str:
    payload = json.dumps({"tool": tool, "args": args}, ensure_ascii=True, separators=(",", ":"))
    return f"<tool_call>{payload}</tool_call>"


def _parse_tool_call(text: str) -> dict | None:
    match = TOOL_CALL_RE.search(text.strip())
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _normalize_response(text: str) -> str:
    payload = _parse_tool_call(text)
    if payload is None:
        return text.strip()
    return _emit(payload["tool"], payload["args"])


def _serialize_history(history: list[dict]) -> str:
    lines: list[str] = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        lines.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(lines)


def _extract_last_tool_call(history: list[dict]) -> dict | None:
    for turn in reversed(history):
        if turn.get("role") != "assistant":
            continue
        payload = _parse_tool_call(turn.get("content", ""))
        if payload is not None:
            return payload
    return None


def _to_number(raw: str) -> int | float:
    value = float(raw)
    return int(value) if value.is_integer() else value


def _parse_weather(prompt: str) -> str | None:
    lower = prompt.lower()
    if not any(token in lower for token in ("weather", "temp", "temperature", "mosam")):
        return None

    unit_match = re.search(r"\b([cf])\b", lower)
    if not unit_match:
        return None

    patterns = [
        r"(?:weather|temp(?:erature)?)(?:\s+in|\s+for)?\s+([A-Za-z ]+?)(?:\s+please|,|\s+in\s+[cf]\b|$)",
        r"mosam\s+([A-Za-z ]+?)\s+[cf]\b",
    ]
    location = None
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            break

    if not location:
        return None
    return _emit("weather", {"location": location, "unit": unit_match.group(1).upper()})


def _parse_calendar(prompt: str) -> str | None:
    date_match = DATE_RE.search(prompt)
    if not date_match:
        return None

    lower = prompt.lower()
    date = date_match.group(1)

    if "list" in lower and "calendar" in lower:
        return _emit("calendar", {"action": "list", "date": date})

    if "schedule" in lower:
        match = re.search(r"schedule\s+(.+?)\s+for\s+\d{4}-\d{2}-\d{2}", prompt, re.IGNORECASE)
        if match:
            return _emit("calendar", {"action": "create", "date": date, "title": match.group(1).strip()})

    if "create" in lower:
        match = re.search(r"called\s+(.+)$", prompt, re.IGNORECASE)
        if match:
            return _emit("calendar", {"action": "create", "date": date, "title": match.group(1).strip()})

    return None


def _parse_sql(prompt: str) -> str | None:
    lower = prompt.lower()
    if lower.startswith("sql:"):
        return _emit("sql", {"query": prompt.split(":", 1)[1].strip()})
    marker = "run this sql query:"
    if lower.startswith(marker):
        return _emit("sql", {"query": prompt[len(marker):].strip()})
    return None


def _parse_currency(prompt: str, history: list[dict]) -> str | None:
    direct_match = re.search(
        rf"convert\s+({NUMBER_RE})\s+([A-Za-z]{{3}})\s+to\s+([A-Za-z]{{3}})\b",
        prompt,
        re.IGNORECASE,
    )
    if direct_match:
        return _emit(
            "currency",
            {
                "amount": _to_number(direct_match.group(1)),
                "from": direct_match.group(2).upper(),
                "to": direct_match.group(3).upper(),
            },
        )

    follow_match = re.search(r"convert that to\s+([A-Za-z]{3,5})\b", prompt, re.IGNORECASE)
    if not follow_match:
        return None

    target = CURRENCY_ALIASES.get(follow_match.group(1).lower())
    last_call = _extract_last_tool_call(history)
    if not target or not last_call or last_call.get("tool") != "currency":
        return None

    args = last_call.get("args", {})
    return _emit("currency", {"amount": args.get("amount"), "from": args.get("from"), "to": target})


def _parse_convert(prompt: str, history: list[dict]) -> str | None:
    direct_match = re.search(
        rf"convert\s+({NUMBER_RE})\s+([A-Za-z]+)\s+to\s+([A-Za-z]+)\b",
        prompt,
        re.IGNORECASE,
    )
    if direct_match:
        from_unit = direct_match.group(2)
        to_unit = direct_match.group(3)
        if len(from_unit) == 3 and len(to_unit) == 3 and from_unit.upper() in CURRENCY_ALIASES.values():
            return None
        return _emit(
            "convert",
            {
                "value": _to_number(direct_match.group(1)),
                "from_unit": from_unit,
                "to_unit": to_unit,
            },
        )

    follow_match = re.search(r"(?:now\s+)?convert that to\s+([A-Za-z]+)\b", prompt, re.IGNORECASE)
    if not follow_match:
        return None

    last_call = _extract_last_tool_call(history)
    if not last_call or last_call.get("tool") != "convert":
        return None

    args = last_call.get("args", {})
    return _emit(
        "convert",
        {
            "value": args.get("value"),
            "from_unit": args.get("from_unit"),
            "to_unit": follow_match.group(1),
        },
    )


def _should_refuse(prompt: str, history: list[dict]) -> bool:
    lower = prompt.lower().strip()
    if any(hint in lower for hint in REFUSAL_HINTS):
        return True
    if "convert that" in lower and not history:
        return True
    return False


def _rule_based_response(prompt: str, history: list[dict]) -> str | None:
    if _should_refuse(prompt, history):
        return REFUSAL_TEXT

    parsers = (
        lambda: _parse_sql(prompt),
        lambda: _parse_weather(prompt),
        lambda: _parse_calendar(prompt),
        lambda: _parse_currency(prompt, history),
        lambda: _parse_convert(prompt, history),
    )
    for parser in parsers:
        result = parser()
        if result is not None:
            return result
    return None


def run(prompt: str, history: list[dict]) -> str:
    rule_based = _rule_based_response(prompt, history)
    if rule_based is not None:
        return rule_based

    llm = get_llm()
    prompt_parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"]
    history_block = _serialize_history(history)
    if history_block:
        prompt_parts.append(history_block)
    prompt_parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")

    output = llm(
        "\n".join(prompt_parts),
        max_tokens=160,
        temperature=0.05,
        top_p=0.9,
        repeat_penalty=1.05,
        stop=["<|im_end|>", "<|im_start|>user"],
    )
    return _normalize_response(output["choices"][0]["text"])
