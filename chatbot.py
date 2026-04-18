from __future__ import annotations

import json
import re

import gradio as gr

from inference import MODEL_PATH, run

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
SAMPLE_PROMPTS = [
    "weather in Lahore in C",
    "schedule Team sync for 2026-06-10",
    "convert 3.5 kilograms to pounds",
    "convert 50 USD to EUR",
    "sql: SELECT COUNT(*) FROM users",
]

CSS = """
.gradio-container {
  background: #f6f7fb;
  color: #18212f;
  font-family: "Segoe UI", sans-serif;
}

.app-shell {
  max-width: 920px;
  margin: 0 auto;
  padding: 24px 16px 32px;
}

.app-card {
  background: #ffffff;
  border: 1px solid #d8deea;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.app-title {
  margin: 0 0 8px;
  font-size: 32px;
}

.app-subtitle {
  margin: 0;
  color: #526074;
  line-height: 1.5;
}

.meta-note {
  color: #526074;
  font-size: 14px;
}

.sample-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 14px 0 4px;
}

.sample-chip {
  background: #eef3ff;
  border: 1px solid #d6e0ff;
  border-radius: 999px;
  color: #23408e;
  padding: 6px 10px;
  font-size: 13px;
}
"""

HEADER_HTML = """
<div class="app-shell">
  <div class="app-card">
    <h1 class="app-title">Pocket-Agent</h1>
    <p class="app-subtitle">
      Simple offline demo for structured tool calls. Ask for weather, calendar actions,
      conversions, currencies, or SQL and inspect the raw response below.
    </p>
    <div class="sample-row">
      <span class="sample-chip">weather in Lahore in C</span>
      <span class="sample-chip">schedule Team sync for 2026-06-10</span>
      <span class="sample-chip">convert 50 USD to EUR</span>
    </div>
    <p class="meta-note">Model path: <code>%s</code></p>
  </div>
</div>
""" % MODEL_PATH


def _history_to_turns(chat_history: list[dict]) -> list[dict]:
    turns: list[dict] = []
    for item in chat_history:
        user_text = item.get("user", "")
        assistant_text = item.get("assistant", "")
        if user_text:
            turns.append({"role": "user", "content": user_text})
        if assistant_text:
            turns.append({"role": "assistant", "content": assistant_text})
    return turns


def _format_tool_output(response: str) -> str:
    match = TOOL_CALL_RE.search(response)
    if not match:
        return "No tool call emitted on the last turn."
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return response
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _submit(message: str, chat_history: list[dict]):
    clean_message = message.strip()
    if not clean_message:
        display = [(item["user"], item["assistant"]) for item in (chat_history or [])]
        return "", display, chat_history or [], "No tool call emitted on the last turn."

    history = chat_history or []
    turns = _history_to_turns(history)
    reply = run(clean_message, turns)
    updated_history = history + [{"user": clean_message, "assistant": reply}]
    display = [(item["user"], item["assistant"]) for item in updated_history]
    return "", display, updated_history, _format_tool_output(reply)


def _load_sample(prompt: str):
    return prompt


def _clear():
    return [], [], "No tool call emitted on the last turn."


with gr.Blocks(css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Column(elem_classes=["app-shell"]):
        chat_state = gr.State([])

        with gr.Column(elem_classes=["app-card"]):
            gr.Markdown("### Chat")
            chatbot = gr.Chatbot(height=420, show_label=False)
            message = gr.Textbox(
                placeholder="Try one of the sample prompts or write your own request.",
                show_label=False,
            )
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")

            gr.Examples(
                examples=[[prompt] for prompt in SAMPLE_PROMPTS],
                inputs=message,
                fn=_load_sample,
                outputs=message,
                cache_examples=False,
            )

        with gr.Column(elem_classes=["app-card"]):
            gr.Markdown("### Last Raw Tool Output")
            tool_output = gr.Code(
                value="No tool call emitted on the last turn.",
                language="json",
            )

    send.click(
        _submit,
        inputs=[message, chat_state],
        outputs=[message, chatbot, chat_state, tool_output],
    )
    message.submit(
        _submit,
        inputs=[message, chat_state],
        outputs=[message, chatbot, chat_state, tool_output],
    )
    clear.click(_clear, outputs=[chatbot, chat_state, tool_output])


if __name__ == "__main__":
    demo.launch()
