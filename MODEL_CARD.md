# Model card: Plain-language ML tutor adapter

## Status

This card describes the intended artifact. No adapter has been trained or published yet. Populate the experiment results only from files produced by an actual run.

## Base model and method

- Base: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- Base model license: Apache-2.0 as stated by the model card; retain the base model's license and notices on redistribution.
- Intended tuning method: supervised instruction fine-tuning with LoRA adapters on a 4-bit NF4-quantized base model (QLoRA).
- Trainer: Hugging Face TRL `SFTTrainer` and PEFT.
- Default target modules: Qwen attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`); these can be changed in `configs/small.yaml`.
- Intended artifact: adapter weights and adapter configuration only; the base model is loaded separately.

## Data

The checked-in seed set contains 52 short instruction/response rows about 13 machine-learning concepts. Records are grouped by concept so variants of a concept stay in one split. The task-specific examples were prepared for this repository; they have not received independent expert review. The preparation manifest documents filtering, split sizes, and near-duplicate candidates after `make prepare`.

The test concepts are separate from training and validation concepts. Do not repeatedly use the test split for training decisions. Review all examples for factual accuracy before treating the tuned model as an educational resource.

## Intended use

Learning and demonstrating the mechanics of preparing a small instruction dataset, measuring a base model, applying QLoRA, comparing outputs, and serving an adapter. This is not a validated tutor or a source of authoritative technical guidance.

## Limitations and risks

- The seed set is small and narrow; it cannot support broad capability claims.
- The model can produce incorrect, incomplete, or overconfident explanations.
- Structural label compliance does not prove factual correctness.
- Fine-tuning may not improve held-out answers and can regress general ability.
- The project has no actual training or evaluation results until a CUDA run is completed.
- Inference resource use and latency depend on runtime hardware, library versions, and quantization support.

## Evaluation results

| Run | Held-out examples | Format compliance | Human-reviewed quality | Latency / hardware | Status |
|---|---:|---:|---:|---|---|
| Base model | — | — | — | — | Not run |
| Fine-tuned adapter | — | — | — | — | Not run |

Do not fill blank cells with estimates. Save run metadata and predictions under `outputs/`, then update this table from the actual metrics and review process.

## Reproduction

See the root [README](README.md) for exact environment setup, baseline, smoke training, full training, and evaluation commands. The checked-in config, dataset, seed, and software pins define the planned run. Record the actual model revision, CUDA driver, GPU, package lock, and command used in the run metadata.
