import torch

from zo_vllm.core.lora_runtime.slot_writer import _direction_lora_pair
from zo_vllm.training.direction.lora_tensors import build_lora_runtime_pair_tensors


def test_embedding_direction_builds_lora_embedding_tensors():
    directions = {
        "model.decoder.embed_tokens.weight": {
            "U": torch.ones((4, 2), dtype=torch.float32),
            "V": torch.ones((8, 2), dtype=torch.float32),
            "scale": 1.0,
        }
    }

    plus_a, plus_b, minus_a, minus_b = build_lora_runtime_pair_tensors(
        directions,
        eps=0.5,
        output_device=None,
    )

    assert plus_a["model.decoder.embed_tokens.weight"].shape == (2, 8)
    assert plus_b["model.decoder.embed_tokens.weight"].shape == (4, 2)
    assert minus_a["model.decoder.embed_tokens.weight"].shape == (2, 8)
    assert torch.equal(
        minus_b["model.decoder.embed_tokens.weight"],
        -plus_b["model.decoder.embed_tokens.weight"],
    )


def test_output_projection_lora_pair_uses_tied_embedding_transpose_semantics():
    direction = {
        "U": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
        "V": torch.tensor(
            [
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
                [13.0, 14.0],
            ]
        ),
        "scale": 0.5,
    }

    lora_a, lora_b = _direction_lora_pair(
        direction,
        eps=0.25,
        sign=1.0,
        output_projection=True,
    )

    assert torch.equal(lora_a, direction["U"].T)
    assert torch.equal(lora_b, 0.125 * direction["V"])
