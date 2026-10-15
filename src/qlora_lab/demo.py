"""Build and launch the Gradio interface for the base model or a local adapter."""

from __future__ import annotations

import os

import gradio as gr

from qlora_lab.config import ExperimentConfig
from qlora_lab.modeling import Generator, load_generator


def build_demo(generator: Generator) -> gr.Blocks:
    with gr.Blocks(title="Plain-language ML tutor") as demo:
        gr.Markdown(
            "# Plain-language ML tutor\n"
            "Ask about a machine-learning concept. Compare the answer with trusted "
            "learning material; this small demonstration model can make mistakes."
        )
        question = gr.Textbox(
            label="Question",
            placeholder="What is data leakage?",
            lines=3,
            max_lines=5,
        )
        answer = gr.Textbox(label="Answer", lines=8, interactive=False)
        submit = gr.Button("Explain", variant="primary")
        submit.click(fn=generator.generate, inputs=question, outputs=answer)
        question.submit(fn=generator.generate, inputs=question, outputs=answer)
        gr.Examples(
            examples=[
                ["Why can accuracy be misleading when classes are imbalanced?"],
                ["Explain the difference between precision and recall."],
                ["What does a learning rate change during training?"],
            ],
            inputs=question,
        )
    return demo


def main() -> None:
    config = ExperimentConfig()
    adapter_path = os.getenv("ADAPTER_PATH") or None
    generator = load_generator(config, adapter_path)
    build_demo(generator).launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
