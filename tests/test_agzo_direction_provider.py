import pytest
import torch

from zo_vllm.core.param_metadata import ParamMetadata
from zo_vllm.training.direction import (
    AGZODirectionProvider,
    SubspaceTokenProbeBatch,
    SUAGZODirectionProvider,
    UAGZODirectionProvider,
    TokenProbeBatch,
)


def _agzo_metadata() -> dict[str, ParamMetadata]:
    return {
        "model.layers.0.fc1.weight": ParamMetadata(
            name="model.layers.0.fc1.weight",
            shape=(3, 2),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    }


class FakeAGZOEngine:
    def __init__(self):
        self.chunked_calls = 0
        self.direct_calls = 0

    def collect_agzo_directions_chunked(self, batches, **kwargs):
        self.chunked_calls += 1
        self.last_batches = batches
        rank = int(kwargs.get("agzo_rank", 1))
        return {
            "model.layers.0.fc1.weight": {
                "U": torch.zeros((3, rank), dtype=torch.float32),
                "V": torch.ones((2, rank), dtype=torch.float32),
            }
        }, {"basis_path": "chunked"}

    def collect_agzo_directions(self, token_id_groups, **kwargs):
        self.direct_calls += 1
        self.last_token_id_groups = token_id_groups
        rank = int(kwargs.get("agzo_rank", 1))
        return {
            "model.layers.0.fc1.weight": {
                "U": torch.zeros((3, rank), dtype=torch.float32),
                "V": torch.ones((2, rank), dtype=torch.float32),
            }
        }, {"basis_path": "direct"}


def test_agzo_subspace_factory_is_lazy_across_nu_reuse():
    engine = FakeAGZOEngine()
    provider = AGZODirectionProvider(
        engine=engine,
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=20000,
        power_iter_steps=1,
    )
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return [[[1, 2, 3]], [[4, 5, 6]]]

    batch = SubspaceTokenProbeBatch(
        token_id_groups=[[10, 11]],
        subspace_token_id_group_batch_factory=factory,
    )

    first = provider.next(batch, step=1)
    second = provider.next(batch, step=2)

    assert first.refreshed is True
    assert second.refreshed is False
    assert not torch.equal(
        first.directions["model.layers.0.fc1.weight"]["U"],
        torch.zeros((3, 1), dtype=torch.float32),
    )
    assert not torch.equal(
        second.directions["model.layers.0.fc1.weight"]["U"],
        torch.zeros((3, 1), dtype=torch.float32),
    )
    assert torch.equal(
        first.directions["model.layers.0.fc1.weight"]["V"],
        second.directions["model.layers.0.fc1.weight"]["V"],
    )
    assert first.info["subspace_num_chunks"] == 2
    assert first.info["v_provider"] == "queued_agzo"
    assert first.info["subspace_queue_active_slots"] == 1
    assert first.info["basis_path"] == "chunked"
    assert second.info["basis_reused"] is True
    assert second.info["u_provider"] == "gaussian"
    assert factory_calls == 1
    assert engine.chunked_calls == 1
    assert engine.direct_calls == 0


def test_agzo_checkpoint_restores_subspace_queue() -> None:
    batch = TokenProbeBatch(token_id_groups=[[10, 11]])
    continuous = AGZODirectionProvider(
        engine=FakeAGZOEngine(),
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=10,
        power_iter_steps=1,
    )
    continuous.next(batch, step=1)
    expected = continuous.next(batch, step=2)

    first = AGZODirectionProvider(
        engine=FakeAGZOEngine(),
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=10,
        power_iter_steps=1,
    )
    first.next(batch, step=1)
    resumed_engine = FakeAGZOEngine()
    resumed = AGZODirectionProvider(
        engine=resumed_engine,
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=10,
        power_iter_steps=1,
    )
    resumed.load_state_dict(first.state_dict())
    actual = resumed.next(batch, step=2)

    name = "model.layers.0.fc1.weight"
    assert actual.refreshed is False
    assert resumed_engine.direct_calls == 0
    assert torch.equal(actual.directions[name]["V"], expected.directions[name]["V"])
    assert torch.equal(actual.directions[name]["U"], expected.directions[name]["U"])


def test_uagzo_replaces_agzo_u_with_orthogonal_pool_sample():
    engine = FakeAGZOEngine()
    provider = UAGZODirectionProvider(
        engine=engine,
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=20000,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )
    batch = TokenProbeBatch(token_id_groups=[[10, 11]])

    sample = provider.next(batch, step=1)
    direction = sample.directions["model.layers.0.fc1.weight"]
    pool = provider._u_pool_for(
        "model.layers.0.fc1.weight",
        out_features=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert sample.info["direction_provider"] == "uagzo"
    assert sample.info["u_dim"] == 2
    assert torch.equal(direction["V"], torch.ones((2, 1), dtype=torch.float32))
    assert not torch.equal(direction["U"], torch.zeros((3, 1), dtype=torch.float32))
    assert torch.isclose(
        torch.dot(pool[:, 0], pool[:, 1]), torch.tensor(0.0), atol=1e-5
    )
    assert torch.isclose(
        direction["U"].norm(), torch.sqrt(torch.tensor(3.0)), atol=1e-5
    )
    assert torch.isclose(
        torch.tensor(direction["perturbation_normalization_scale"]),
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_uagzo_requires_pool_to_cover_rank():
    with pytest.raises(ValueError, match="u_dim"):
        UAGZODirectionProvider(
            engine=FakeAGZOEngine(),
            param_metadata=_agzo_metadata(),
            rank=2,
            nu=1,
            power_iter_steps=1,
            u_dim=1,
        )


def test_uagzo_queue_expands_v_without_requiring_larger_u_pool():
    engine = FakeAGZOEngine()
    provider = UAGZODirectionProvider(
        engine=engine,
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=1,
        power_iter_steps=1,
        subspace_queue_size=2,
        u_dim=1,
    )
    batch = TokenProbeBatch(token_id_groups=[[10, 11]])

    first = provider.next(batch, step=1)
    second = provider.next(batch, step=2)
    first_direction = first.directions["model.layers.0.fc1.weight"]
    second_direction = second.directions["model.layers.0.fc1.weight"]

    assert first_direction["U"].shape == (3, 2)
    assert first_direction["V"].shape == (2, 2)
    assert torch.equal(first_direction["V"][:, :1], torch.ones((2, 1)))
    assert torch.equal(first_direction["V"][:, 1:], torch.zeros((2, 1)))
    assert second_direction["U"].shape == (3, 2)
    assert torch.allclose(
        second_direction["V"],
        torch.ones((2, 2)),
    )
    assert torch.isclose(
        torch.tensor(first_direction["perturbation_normalization_scale"]),
        torch.tensor(1.0),
        atol=1e-6,
    )
    assert torch.isclose(
        torch.tensor(second_direction["perturbation_normalization_scale"]),
        1.0 / torch.sqrt(torch.tensor(2.0)),
        atol=1e-6,
    )
    assert second.info["v_provider"] == "queued_agzo"
    assert second.info["subspace_queue_active_slots"] == 2
    assert engine.direct_calls == 2


def test_suagzo_samples_continuous_u_subspace_with_normalized_coefficients():
    engine = FakeAGZOEngine()
    provider = SUAGZODirectionProvider(
        engine=engine,
        param_metadata=_agzo_metadata(),
        rank=2,
        nu=20000,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )
    batch = TokenProbeBatch(token_id_groups=[[10, 11]])

    sample = provider.next(batch, step=1)
    direction = sample.directions["model.layers.0.fc1.weight"]
    pool = provider._u_pool_for(
        "model.layers.0.fc1.weight",
        out_features=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.manual_seed(200000 + 1 + 7 * 1_000_000)
    expected_coefficients = torch.randn((2, 2), dtype=torch.float32) / torch.sqrt(
        torch.tensor(2.0)
    )
    expected_u = pool.matmul(expected_coefficients)

    assert sample.info["direction_provider"] == "suagzo"
    assert sample.info["u_provider"] == "subspace"
    assert sample.info["u_normalization"] == "coefficients/sqrt(u_dim)"
    assert sample.info["u_dim"] == 2
    assert torch.equal(direction["V"], torch.ones((2, 2), dtype=torch.float32))
    assert torch.allclose(direction["U"], expected_u)
    projected_coefficients = pool.T.matmul(direction["U"]) / 3.0
    assert torch.allclose(projected_coefficients, expected_coefficients, atol=1e-6)
    assert torch.count_nonzero(projected_coefficients.abs() > 1e-6) == 4


def test_suagzo_allows_subspace_dimension_below_rank():
    provider = SUAGZODirectionProvider(
        engine=FakeAGZOEngine(),
        param_metadata=_agzo_metadata(),
        rank=2,
        nu=1,
        power_iter_steps=1,
        u_dim=1,
    )
    sample = provider.next(TokenProbeBatch(token_id_groups=[[10, 11]]), step=1)
    direction = sample.directions["model.layers.0.fc1.weight"]

    assert direction["U"].shape == (3, 2)
    assert direction["V"].shape == (2, 2)
