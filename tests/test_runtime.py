from types import SimpleNamespace

import pytest

from qlora_lab.runtime import hardware_summary, require_cuda_for_qlora


def _fake_torch(cuda_available=False):
    return SimpleNamespace(
        __version__="test",
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )


def test_cpu_hardware_report_does_not_claim_gpu_details():
    report = hardware_summary(_fake_torch())
    assert not report["cuda_available"]
    assert not report["mps_available"]
    assert "gpu_name" not in report


def test_qlora_gate_explains_missing_cuda_instead_of_falling_back():
    with pytest.raises(RuntimeError, match="requires a CUDA-enabled NVIDIA GPU"):
        require_cuda_for_qlora(_fake_torch())
