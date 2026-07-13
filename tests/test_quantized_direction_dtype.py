import torch

from zo_vllm.core.param_metadata import ParamMetadata
from zo_vllm.training.direction import LOZODirectionProvider, TokenProbeBatch


def test_float8_param_metadata_samples_fp16_directions():
    metadata = {
        "layer.weight": ParamMetadata(
            name="layer.weight",
            shape=(3, 4),
            dtype=torch.float8_e4m3fn,
            device=torch.device("cpu"),
        )
    }
    provider = LOZODirectionProvider(
        param_metadata=metadata,
        rank=2,
        nu=1,
        random_device="cpu",
        direction_sampling="exact",
        direction_scale=1.0,
        seed=0,
    )

    sample = provider.next(TokenProbeBatch(token_id_groups=[]), step=1)
    direction = sample.directions["layer.weight"]

    assert sample.refreshed is True
    assert direction["U"].dtype is torch.float16
    assert direction["V"].dtype is torch.float16
    assert direction["V_T"].dtype is torch.float16


def test_float8_param_metadata_flat_sampling_uses_fp16():
    metadata = {
        "layer0.weight": ParamMetadata(
            name="layer0.weight",
            shape=(3, 4),
            dtype=torch.float8_e4m3fn,
            device=torch.device("cpu"),
        ),
        "layer1.weight": ParamMetadata(
            name="layer1.weight",
            shape=(5, 3),
            dtype=torch.float8_e4m3fn,
            device=torch.device("cpu"),
        ),
    }
    provider = LOZODirectionProvider(
        param_metadata=metadata,
        rank=2,
        nu=1,
        random_device="cpu",
        direction_sampling="flat",
        direction_scale=1.0,
        seed=0,
    )

    sample = provider.next(TokenProbeBatch(token_id_groups=[]), step=1)

    for direction in sample.directions.values():
        assert direction["U"].dtype is torch.float16
        assert direction["V"].dtype is torch.float16
        assert direction["V_T"].dtype is torch.float16
