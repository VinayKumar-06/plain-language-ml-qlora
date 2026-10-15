# Design notes

## Why this task

The project teaches instruction fine-tuning through a small output contract: a plain-language definition, a concrete example, and a caveat. That contract can be checked automatically without pretending the check establishes correctness. A human review sheet keeps semantic judgments separate from easy-to-count formatting.

## Data flow

`data/raw/examples.jsonl` is the source. `qlora_lab.data.prepare` validates required text fields, rejects empty or wrongly typed fields, reports malformed JSON and duplicate IDs, removes exact duplicate instruction/response pairs, and assigns complete concept groups within each category to one split. Every category needs at least three distinct concepts so it can appear in train, validation, and test. The seeded split manifest and candidate near-duplicates make review decisions visible.

Responses and prompts become a conversational prompt/completion dataset. TRL computes loss on the completion only. This avoids teaching the model to reproduce the instruction text as its target.

## Training choices

- Qwen2.5-1.5B-Instruct is small enough for a modest GPU experiment and is listed under Apache-2.0.
- NF4, double quantization, and a LoRA adapter keep the training state smaller than full fine-tuning.
- CUDA is required by the current training entry point. It fails before downloading model weights when CUDA is missing.
- LoRA parameters are checked before training; smoke mode runs exactly one step and verifies finite logged loss.
- Validation loss selects checkpoints; early stopping has a patience of two validation checks.
- Gradient accumulation and checkpointing lower activation memory. The actual viable sequence length and batch settings must be checked on the selected GPU.
- The saved output is a PEFT adapter and tokenizer, not merged model weights.

## Evaluation boundary

Base and adapter runs consume the same held-out rows and deterministic decoding settings. Reported automatic checks are label/order compliance, word count, and observed latency, broken down by category. They do not measure factual correctness. A reviewer should compare paired outputs for factual accuracy, clarity, completeness, and usefulness of caveats. The fixed test split should not guide hyperparameter choices.

## Reproducibility and limitations

Direct library versions are pinned. `uv.lock` records the resolved transitive versions, while PyTorch wheel selection remains tied to the host's accelerator/runtime. Training metadata records config, package-independent hardware facts, model revision, elapsed time, logged losses, and trainable parameter count. GPU kernels may be nondeterministic even with fixed seeds.

The included dataset is intentionally small and has not been independently reviewed. No completed model baseline, adapter training, or human-rated comparison is included at project creation. Such results must be added after a real GPU run; none are inferred from code or training loss.
