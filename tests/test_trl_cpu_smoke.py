"""Train a tiny random model to exercise the installed TRL/PEFT integration on CPU."""

import math

import pytest

pytestmark = pytest.mark.integration


def test_one_step_tiny_cpu_sft_uses_completion_only_loss(tmp_path):
    pytest.importorskip("torch")
    datasets = pytest.importorskip("datasets")
    peft = pytest.importorskip("peft")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    trl = pytest.importorskip("trl")

    from qlora_lab.config import ExperimentConfig
    from qlora_lab.data.formatting import format_many
    from qlora_lab.train import build_training_arguments

    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    tokenizer_impl = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<' + message['role'] + '> ' + message['content'] + eos_token }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    )
    model_config = transformers.Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    model = transformers.Qwen2ForCausalLM(model_config)
    example = {
        "instruction": "What is recall?",
        "response": "Definition: fraction found. Example: screening. Caveat: compare precision.",
    }
    dataset = datasets.Dataset.from_list(format_many([example]))
    args = build_training_arguments(
        ExperimentConfig(
            max_seq_length=128,
            batch_size=1,
            gradient_accumulation_steps=1,
            output_dir=str(tmp_path / "tiny-adapter"),
        ),
        tmp_path / "tiny-adapter",
        smoke=True,
        bf16=False,
    )
    trainer = trl.SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft.LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )

    labels = trainer.train_dataset[0]["labels"]
    assert -100 in labels
    assert any(label != -100 for label in labels)
    assert any(parameter.requires_grad for parameter in trainer.model.parameters())
    result = trainer.train()
    assert result.global_step == 1
    assert math.isfinite(result.training_loss)
