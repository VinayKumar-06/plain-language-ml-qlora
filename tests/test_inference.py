import pytest


def test_generator_checks_question_limits_and_decodes_only_new_tokens():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from qlora_lab.config import ExperimentConfig
    from qlora_lab.modeling import Generator

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 3]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, tokens, skip_special_tokens):
            assert tokens.tolist() == [4, 5]
            assert skip_special_tokens is True
            return "Definition: short answer"

    class Model:
        device = torch.device("cpu")

        def generate(self, input_ids, **kwargs):
            assert kwargs["max_new_tokens"] == 12
            return torch.tensor([[1, 3, 4, 5]])

    generator = Generator(
        Model(),
        Tokenizer(),
        ExperimentConfig(max_seq_length=2, max_new_tokens=12),
    )
    assert generator.generate("What is a short concept?") == "Definition: short answer"
    with pytest.raises(ValueError, match="Enter a question"):
        generator.generate("  ")
    with pytest.raises(ValueError, match="8,000 characters"):
        generator.generate("x" * 8_001)
