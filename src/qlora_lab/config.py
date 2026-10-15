"""Validated experiment settings shared by training and inference."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    seed: int = 42
    max_seq_length: int = 768
    max_new_tokens: int = 180
    learning_rate: float = 0.0001
    epochs: float = 8.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    quantization_type: str = "nf4"
    double_quantization: bool = True
    output_dir: str = "outputs/adapter"

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in (
            "max_seq_length",
            "max_new_tokens",
            "batch_size",
            "gradient_accumulation_steps",
            "lora_rank",
            "lora_alpha",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not self.lora_target_modules or any(
            not item.strip() for item in self.lora_target_modules
        ):
            raise ValueError("lora_target_modules must contain non-empty module names")
        if self.quantization_type not in {"nf4", "fp4"}:
            raise ValueError("quantization_type must be 'nf4' or 'fp4'")
        if not self.output_dir.strip():
            raise ValueError("output_dir must not be empty")


def config_from_mapping(values: dict) -> ExperimentConfig:
    """Convert YAML lists into immutable tuples and validate all config values."""
    normalized = dict(values)
    if "lora_target_modules" in normalized:
        normalized["lora_target_modules"] = tuple(normalized["lora_target_modules"])
    return ExperimentConfig(**normalized)
