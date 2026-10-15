import pytest


def test_gradio_demo_can_be_built_without_loading_model_weights():
    gr = pytest.importorskip("gradio")
    from qlora_lab.demo import build_demo

    class ExampleGenerator:
        def generate(self, prompt: str) -> str:
            return f"You asked: {prompt}"

    demo = build_demo(ExampleGenerator())
    assert isinstance(demo, gr.Blocks)
