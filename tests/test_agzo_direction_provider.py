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


def _two_target_metadata() -> dict[str, ParamMetadata]:
    return {
        **_agzo_metadata(),
        "model.layers.0.fc2.weight": ParamMetadata(
            name="model.layers.0.fc2.weight",
            shape=(4, 2),
            dtype=torch.float32,
            device=torch.device("cpu"),
        ),
    }


class FakeAGZOEngine:
    def __init__(self, metadata=None):
        self.chunked_calls = 0
        self.direct_calls = 0
        self.metadata = dict(metadata or _agzo_metadata())

    def collect_agzo_directions_chunked(self, batches, **kwargs):
        self.chunked_calls += 1
        self.last_batches = batches
        rank = int(kwargs.get("agzo_rank", 1))
        return self._directions(rank), {"basis_path": "chunked"}

    def collect_agzo_directions(self, token_id_groups, **kwargs):
        self.direct_calls += 1
        self.last_token_id_groups = token_id_groups
        rank = int(kwargs.get("agzo_rank", 1))
        return self._directions(rank), {"basis_path": "direct"}

    def _directions(self, rank):
        return {
            name: {
                "U": torch.zeros((metadata.shape[0], rank), dtype=torch.float32),
                "V": torch.ones((metadata.shape[1], rank), dtype=torch.float32),
            }
            for name, metadata in self.metadata.items()
        }


class FixedPerTargetSelector:
    def __init__(self, indices_by_name):
        self.indices_by_name = {
            str(name): int(index) for name, index in indices_by_name.items()
        }
        self.steps = []

    def select_indices(self, *, step: int, u_dim: int, targets):
        assert u_dim == 2
        assert all(int(target.rank) == 1 for target in targets)
        self.steps.append(int(step))
        return {target.name: (self.indices_by_name[target.name],) for target in targets}

    def info(self):
        return {"u_selector": "fixed_per_target"}

    def state_dict(self):
        return {
            "indices_by_name": dict(self.indices_by_name),
            "steps": list(self.steps),
        }

    def load_state_dict(self, state):
        self.indices_by_name = {
            str(name): int(index) for name, index in state["indices_by_name"].items()
        }
        self.steps = [int(step) for step in state["steps"]]


class RefreshingGlobalSelector(FixedPerTargetSelector):
    def __init__(self, index: int, *, refresh_interval: int = 2):
        super().__init__({name: index for name in _two_target_metadata()})
        self.refresh_interval = int(refresh_interval)

    def pool_generation_for_step(self, *, step: int) -> int:
        return (int(step) - 1) // self.refresh_interval


class ScalingSelector(FixedPerTargetSelector):
    def u_scale_for_selection(self, *, step, target_name, indices):
        assert step == 1
        assert indices == (self.indices_by_name[target_name],)
        return 2.0 if target_name.endswith("fc1.weight") else 0.5


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


def test_uagzo_selector_applies_independent_local_actions_across_targets():
    metadata = _two_target_metadata()
    indices = {
        "model.layers.0.fc1.weight": 1,
        "model.layers.0.fc2.weight": 0,
    }
    selector = FixedPerTargetSelector(indices)
    provider = UAGZODirectionProvider(
        engine=FakeAGZOEngine(metadata),
        param_metadata=metadata,
        rank=1,
        nu=-1,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )
    provider.set_u_index_selector(selector)

    sample = provider.next(TokenProbeBatch(token_id_groups=[[10, 11]]), step=1)

    assert selector.steps == [1]
    assert sample.info["u_pool_action_num_targets"] == 2
    assert sample.info["u_pool_action_unique_indices"] == 2
    assert sample.info["u_pool_index_mode"] == "per_target_selector"
    for name, direction in sample.directions.items():
        pool = provider._u_pool_for(
            name,
            out_features=direction["U"].shape[0],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        index = indices[name]
        assert torch.equal(direction["U"], pool[:, index : index + 1])


def test_uagzo_selector_refreshes_complete_global_direction_pool():
    metadata = _two_target_metadata()
    selector = RefreshingGlobalSelector(1)
    provider = UAGZODirectionProvider(
        engine=FakeAGZOEngine(metadata),
        param_metadata=metadata,
        rank=1,
        nu=-1,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )
    provider.set_u_index_selector(selector)
    batch = TokenProbeBatch(token_id_groups=[[10, 11]])

    first = provider.next(batch, step=1)
    refreshed = provider.next(batch, step=3)

    assert first.info["u_pool_generation"] == 0
    assert refreshed.info["u_pool_generation"] == 1
    assert first.info["u_pool_action_unique_indices"] == 1
    assert refreshed.info["u_pool_action_unique_indices"] == 1
    for name in metadata:
        assert not torch.equal(
            first.directions[name]["U"], refreshed.directions[name]["U"]
        )

    state = provider.state_dict()
    resumed_selector = RefreshingGlobalSelector(0)
    resumed = UAGZODirectionProvider(
        engine=FakeAGZOEngine(metadata),
        param_metadata=metadata,
        rank=1,
        nu=-1,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )
    resumed.set_u_index_selector(resumed_selector)
    resumed.load_state_dict(state)

    resumed_sample = resumed.next(batch, step=4)
    assert resumed_sample.info["u_pool_generation"] == 1
    for name in metadata:
        assert torch.equal(
            refreshed.directions[name]["U"], resumed_sample.directions[name]["U"]
        )


def test_uagzo_selector_applies_per_target_importance_scales():
    metadata = _two_target_metadata()
    indices = {name: 1 for name in metadata}
    selector = ScalingSelector(indices)
    provider = UAGZODirectionProvider(
        engine=FakeAGZOEngine(metadata),
        param_metadata=metadata,
        rank=1,
        nu=-1,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )
    provider.set_u_index_selector(selector)

    sample = provider.next(TokenProbeBatch(token_id_groups=[[10, 11]]), step=1)

    assert sample.info["u_pool_importance_scale_mean"] == 1.25
    assert sample.info["u_pool_importance_scale_max"] == 2.0
    for name, direction in sample.directions.items():
        pool = provider._u_pool_for(
            name,
            out_features=direction["U"].shape[0],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        scale = 2.0 if name.endswith("fc1.weight") else 0.5
        assert torch.equal(direction["U"], pool[:, 1:2] * scale)


def test_uagzo_preinitialized_v_skips_step_one_refresh():
    selector = FixedPerTargetSelector({"model.layers.0.fc1.weight": 1})
    engine = FakeAGZOEngine()
    provider = UAGZODirectionProvider(
        engine=engine,
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=-1,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )
    provider.set_u_index_selector(selector)
    bootstrap = SubspaceTokenProbeBatch(
        token_id_groups=[[10, 11]],
        subspace_token_id_group_batches=[[[1, 2]], [[3, 4]]],
        subspace_num_rows=512,
    )

    info = provider.prime_v(bootstrap, step=1)
    sample = provider.next(TokenProbeBatch(token_id_groups=[[20, 21]]), step=1)

    assert info["basis_preinitialized"] is True
    assert engine.chunked_calls == 1
    assert engine.direct_calls == 0
    assert sample.refreshed is False
    assert sample.info["basis_preinitialized"] is True
    assert sample.info["basis_reused"] is True
    assert sample.info["u_pool_action_num_targets"] == 1
    assert selector.steps == [1]


def test_uagzo_checkpoint_restores_pending_preinitialized_v():
    first = UAGZODirectionProvider(
        engine=FakeAGZOEngine(),
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=-1,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )
    first.prime_v(
        SubspaceTokenProbeBatch(
            token_id_groups=[[10, 11]],
            subspace_token_id_group_batches=[[[1, 2]]],
            subspace_num_rows=512,
        ),
        step=1,
    )
    state = first.state_dict()
    resumed_engine = FakeAGZOEngine()
    resumed = UAGZODirectionProvider(
        engine=resumed_engine,
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=-1,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )

    resumed.load_state_dict(state)
    sample = resumed.next(TokenProbeBatch(token_id_groups=[[20, 21]]), step=1)

    assert sample.refreshed is False
    assert sample.info["basis_preinitialized"] is True
    assert resumed_engine.chunked_calls == 0
    assert resumed_engine.direct_calls == 0


def test_uagzo_checkpoint_restores_selector_state():
    selector = FixedPerTargetSelector({"model.layers.0.fc1.weight": 1})
    provider = UAGZODirectionProvider(
        engine=FakeAGZOEngine(),
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=-1,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )
    provider.set_u_index_selector(selector)
    batch = TokenProbeBatch(token_id_groups=[[10, 11]])
    provider.next(batch, step=1)
    state = provider.state_dict()

    resumed_selector = FixedPerTargetSelector({"model.layers.0.fc1.weight": 0})
    resumed = UAGZODirectionProvider(
        engine=FakeAGZOEngine(),
        param_metadata=_agzo_metadata(),
        rank=1,
        nu=-1,
        power_iter_steps=1,
        u_dim=2,
        seed=7,
    )
    resumed.set_u_index_selector(resumed_selector)
    resumed.load_state_dict(state)

    assert resumed_selector.indices_by_name == {"model.layers.0.fc1.weight": 1}
    assert resumed_selector.steps == [1]
    assert resumed.next(batch, step=2).info["u_pool_action_num_targets"] == 1


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
