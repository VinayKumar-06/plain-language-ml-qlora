"""Shared base-model and adapter loading for evaluation and the Gradio app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qlora_lab.config import ExperimentConfig
from qlora_lab.data.formatting import SYSTEM_MESSAGE


def load_generator(config: ExperimentConfig, adapter_path: str | None = None) -> Generator:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization_type,
            bnb_4bit_use_double_quant=config.double_quantization,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            quantization_config=quantization,
            device_map="auto",
            dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_id, dtype=torch.float32)
        if torch.backends.mps.is_available():
            model = model.to("mps")
        else:
            model = model.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_path:
        adapter = Path(adapter_path)
        if not adapter.is_dir():
            raise FileNotFoundError(f"adapter directory does not exist: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    return Generator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        model_revision=getattr(model.config, "_commit_hash", None),
    )


class Generator:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: ExperimentConfig,
        model_revision: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_revision = model_revision

    @torch.inference_mode()
    def generate(self, instruction: str) -> str:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Enter a question before generating an answer.")
        if len(instruction) > 8_000:
            raise ValueError("Question is too long; limit it to 8,000 characters.")
        messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {"role": "user", "content": instruction.strip()},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        if input_ids.shape[-1] > self.config.max_seq_length:
            raise ValueError(
                f"Question is too long for this model setting "
                f"({self.config.max_seq_length} tokens maximum)."
            )
        device = getattr(self.model, "device", input_ids.device)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        output = self.model.generate(
            **encoded,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        completion = output[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()
