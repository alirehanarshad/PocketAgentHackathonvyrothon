# Pocket-Agent

Simple offline assistant demo built around a local `model.gguf` file.

The repo exposes:

- `inference.py` with `run(prompt: str, history: list[dict]) -> str`
- `chatbot.py` with a clean Gradio demo UI
- `verify_submission.py` for local validation
- `evaluate_local.py` for a small built-in test set

## Setup

```bash
pip install -r requirements.txt
```

## Run The Demo

```bash
python chatbot.py
```

## Sample Prompts

```text
weather in Lahore in C
schedule Team sync for 2026-06-10
convert 3.5 kilograms to pounds
convert 50 USD to EUR
sql: SELECT COUNT(*) FROM users
```

## Sample Python Usage

```python
from inference import run

result = run(
    prompt="convert 10 miles to kilometers",
    history=[],
)
print(result)
```

Expected output shape:

```text
<tool_call>{"tool":"convert","args":{"value":10,"from_unit":"miles","to_unit":"kilometers"}}</tool_call>
```

## Verify

```bash
python verify_submission.py
python evaluate_local.py
```

## Notes

- The assistant stays offline in the normal runtime path.
- Tool calls are wrapped in `<tool_call>...</tool_call>`.
- Unsupported requests return plain text instead of a tool call.
