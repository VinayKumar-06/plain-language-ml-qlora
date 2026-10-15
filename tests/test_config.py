import pytest

from qlora_lab.config import ExperimentConfig, config_from_mapping


def test_default_experiment_configuration_is_valid():
    config = ExperimentConfig()
    assert config.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config.lora_rank == 16


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", " "),
        ("seed", -1),
        ("max_seq_length", 0),
        ("learning_rate", 0),
        ("epochs", -1),
        ("lora_dropout", 1),
        ("output_dir", " "),
    ],
)
def test_invalid_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        ExperimentConfig(**{field: value})


def test_yaml_style_target_module_list_is_converted_and_configurable():
    config = config_from_mapping({"lora_target_modules": ["q_proj", "v_proj"]})
    assert config.lora_target_modules == ("q_proj", "v_proj")


@pytest.mark.parametrize(
    ("field", "value"),
    [("lora_target_modules", ()), ("quantization_type", "int8")],
)
def test_invalid_quantization_or_adapter_targets_are_rejected(field, value):
    with pytest.raises(ValueError):
        ExperimentConfig(**{field: value})
