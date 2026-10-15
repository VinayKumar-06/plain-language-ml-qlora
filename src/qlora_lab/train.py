"""Run QLoRA supervised fine-tuning; actual training requires a CUDA GPU."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from qlora_lab.config import ExperimentConfig, config_from_mapping
from qlora_lab.data.formatting import format_many
from qlora_lab.data.io import read_jsonl
from qlora_lab.runtime import hardware_summary, require_cuda_for_qlora


def build_training_arguments(
    config: ExperimentConfig,
    output_dir: Path,
    smoke: bool,
    bf16: bool,
) -> SFTConfig:
    """Keep TRL settings in one place so the configuration can be checked without a GPU."""
    return SFTConfig(
        output_dir=str(output_dir),
        max_length=config.max_seq_length,
        completion_only_loss=True,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        logging_steps=1,
        eval_strategy="steps" if smoke else "epoch",
        eval_steps=1 if smoke else None,
        save_strategy="steps" if smoke else "epoch",
        save_steps=1 if smoke else None,
        save_total_limit=2,
        load_best_model_at_end=not smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=not bf16,
        max_steps=1 if smoke else -1,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
        dataloader_pin_memory=not smoke,
        save_only_model=False,
    )


def validate_output_directory(output_dir: Path, resume_from_checkpoint: str | None) -> None:
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return
    checkpoint = Path(resume_from_checkpoint).resolve() if resume_from_checkpoint else None
    if (
        checkpoint is None
        or not checkpoint.is_dir()
        or checkpoint.parent.resolve() != output_dir.resolve()
    ):
        raise FileExistsError(
            f"training output already exists: {output_dir}; choose a new output_dir or resume "
            "from a checkpoint inside that directory"
        )


def run_training(
    config: ExperimentConfig,
    train_path: Path,
    validation_path: Path,
    smoke: bool = False,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    require_cuda_for_qlora(torch)
    output_dir = Path(config.output_dir)
    validate_output_directory(output_dir, resume_from_checkpoint)
    if smoke:
        train_rows = read_jsonl(train_path)[:4]
        validation_rows = read_jsonl(validation_path)[:2]
    else:
        train_rows = read_jsonl(train_path)
        validation_rows = read_jsonl(validation_path)
    if len(train_rows) < 1 or len(validation_rows) < 1:
        raise ValueError("training and validation splits must both contain examples")

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quantization_type,
        bnb_4bit_use_double_quant=config.double_quantization,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=quantization,
        device_map={"": torch.cuda.current_device()},
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    train_dataset = Dataset.from_list(format_many(train_rows))
    validation_dataset = Dataset.from_list(format_many(validation_rows))
    use_bf16 = torch.cuda.is_bf16_supported()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = build_training_arguments(config, output_dir, smoke, use_bf16)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[] if smoke else [EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainable = [
        (name, parameter.numel())
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or not any("lora_" in name for name, _ in trainable):
        raise RuntimeError("LoRA adapter parameters were not made trainable")
    trainable_count = sum(count for _, count in trainable)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "CUDA ran out of memory during training. Reduce max_seq_length, batch_size, or "
            "LoRA rank in the config, then rerun the smoke test."
        ) from exc
    elapsed = time.perf_counter() - started
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("training did not produce finite loss values")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    run_info = {
        "config": asdict(config),
        "hardware": hardware_summary(torch),
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "trainable_parameter_count": trainable_count,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": round(elapsed, 3),
        "training_loss_logs": losses,
        "eval_loss_logs": [
            entry["eval_loss"] for entry in trainer.state.log_history if "eval_loss" in entry
        ],
        "smoke_run": smoke,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "global_step": trainer.state.global_step,
        "train_loss": result.training_loss,
    }
    (output_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    return run_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/small.yaml"))
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/processed/validation.jsonl"))
    parser.add_argument(
        "--smoke", action="store_true", help="Run one optimizer step on up to four records."
    )
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    import yaml

    config = config_from_mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    result = run_training(
        config, args.train, args.validation, args.smoke, args.resume_from_checkpoint
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
