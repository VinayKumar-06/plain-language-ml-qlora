"""Opt-in integration test; this performs a real one-step QLoRA training run on CUDA."""

import json
import os
from pathlib import Path

import pytest

from qlora_lab.config import ExperimentConfig
from qlora_lab.data.prepare import prepare_dataset

pytestmark = pytest.mark.gpu


def test_one_step_training_saves_a_finite_loss_adapter(tmp_path):
    torch = pytest.importorskip("torch", reason="GPU smoke test requires PyTorch")
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-enabled NVIDIA GPU")
    if os.getenv("RUN_GPU_TESTS") != "1":
        pytest.skip("set RUN_GPU_TESTS=1 to authorize the actual GPU smoke-training run")
    from qlora_lab.train import run_training

    root = Path(__file__).resolve().parents[1]
    prepared = tmp_path / "data"
    prepare_dataset(root / "data/raw/examples.jsonl", prepared)
    config = ExperimentConfig(output_dir=str(tmp_path / "adapter"), max_seq_length=512)
    result = run_training(
        config, prepared / "train.jsonl", prepared / "validation.jsonl", smoke=True
    )
    assert result["global_step"] == 1
    assert result["training_loss_logs"]
    assert all(
        value == value and abs(value) != float("inf") for value in result["training_loss_logs"]
    )
    assert (tmp_path / "adapter" / "adapter_config.json").is_file()
    assert json.loads((tmp_path / "adapter" / "run.json").read_text())["smoke_run"] is True
