from __future__ import annotations

import unittest

import torch

from zo_vllm.core.param_metadata import ParamMetadata
from zo_vllm.training.direction import LOZOFastDirectionProvider, TokenProbeBatch


def _metadata() -> dict[str, ParamMetadata]:
    return {
        "model.decoder.layers.0.fc1.weight": ParamMetadata(
            name="model.decoder.layers.0.fc1.weight",
            shape=(8, 6),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    }


class _LifecycleSeeds:
    def __init__(self, seeds: dict[int, int]) -> None:
        self.seeds = dict(seeds)
        self.calls: list[int] = []

    def __call__(self, step: int) -> int:
        self.calls.append(int(step))
        return int(self.seeds[int(step)])

    def state_dict(self):
        return {"calls": list(self.calls)}

    def load_state_dict(self, state):
        self.calls = [int(step) for step in state["calls"]]


class LOZOFastDirectionProviderTest(unittest.TestCase):
    def test_lifecycle_seed_reuses_complete_u_v_until_seed_changes(self) -> None:
        batch = TokenProbeBatch(token_id_groups=[[1, 2]])
        seeds = _LifecycleSeeds({1: 101, 2: 101, 3: 202})
        provider = LOZOFastDirectionProvider(
            param_metadata=_metadata(),
            rank=1,
            nu=-1,
            random_device="cpu",
            seed_sampler=seeds,
        )

        self.assertTrue(provider.will_refresh(step=1))
        first = provider.next(batch, step=1)
        self.assertFalse(provider.will_refresh(step=2))
        continued = provider.next(batch, step=2)
        self.assertTrue(provider.will_refresh(step=3))
        replaced = provider.next(batch, step=3)

        name = "model.decoder.layers.0.fc1.weight"
        self.assertTrue(first.refreshed)
        self.assertFalse(continued.refreshed)
        self.assertTrue(replaced.refreshed)
        self.assertTrue(
            torch.equal(first.directions[name]["U"], continued.directions[name]["U"])
        )
        self.assertTrue(
            torch.equal(first.directions[name]["V"], continued.directions[name]["V"])
        )
        self.assertFalse(
            torch.equal(first.directions[name]["U"], replaced.directions[name]["U"])
        )
        self.assertFalse(
            torch.equal(first.directions[name]["V"], replaced.directions[name]["V"])
        )

    def test_lifecycle_seed_checkpoint_restores_sampler_and_direction(self) -> None:
        batch = TokenProbeBatch(token_id_groups=[[1, 2]])
        first_seeds = _LifecycleSeeds({1: 303, 2: 303})
        first = LOZOFastDirectionProvider(
            param_metadata=_metadata(),
            rank=1,
            nu=-1,
            random_device="cpu",
            seed_sampler=first_seeds,
        )
        expected = first.next(batch, step=1)
        state = first.state_dict()

        resumed_seeds = _LifecycleSeeds({1: 303, 2: 303})
        resumed = LOZOFastDirectionProvider(
            param_metadata=_metadata(),
            rank=1,
            nu=-1,
            random_device="cpu",
            seed_sampler=resumed_seeds,
        )
        resumed.load_state_dict(state)
        actual = resumed.next(batch, step=2)

        name = "model.decoder.layers.0.fc1.weight"
        self.assertFalse(actual.refreshed)
        self.assertEqual(resumed_seeds.calls, first_seeds.calls + [2])
        self.assertTrue(
            torch.equal(expected.directions[name]["U"], actual.directions[name]["U"])
        )
        self.assertTrue(
            torch.equal(expected.directions[name]["V"], actual.directions[name]["V"])
        )

    def test_checkpoint_restores_reused_v_basis(self) -> None:
        batch = TokenProbeBatch(token_id_groups=[[1, 2]])
        continuous = LOZOFastDirectionProvider(
            param_metadata=_metadata(),
            rank=1,
            nu=10,
            random_device="cpu",
        )
        continuous.next(batch, step=1)
        expected = continuous.next(batch, step=2)

        first = LOZOFastDirectionProvider(
            param_metadata=_metadata(),
            rank=1,
            nu=10,
            random_device="cpu",
        )
        first.next(batch, step=1)
        state = first.state_dict()
        resumed = LOZOFastDirectionProvider(
            param_metadata=_metadata(),
            rank=1,
            nu=10,
            random_device="cpu",
        )
        resumed.load_state_dict(state)
        actual = resumed.next(batch, step=2)

        name = "model.decoder.layers.0.fc1.weight"
        self.assertFalse(actual.refreshed)
        self.assertTrue(
            torch.equal(actual.directions[name]["V"], expected.directions[name]["V"])
        )
        self.assertTrue(
            torch.equal(actual.directions[name]["U"], expected.directions[name]["U"])
        )

    def test_unit_v_normalization_scales_rank1_columns(self) -> None:
        provider = LOZOFastDirectionProvider(
            param_metadata=_metadata(),
            rank=1,
            nu=10,
            random_device="cpu",
            v_normalization="unit",
        )

        directions = provider.sample_direction(123)
        v = directions["model.decoder.layers.0.fc1.weight"]["V"].float()

        self.assertTrue(
            torch.allclose(v.norm(dim=0), torch.ones(1), atol=1e-6, rtol=1e-6),
            msg=str(v.norm(dim=0)),
        )

    def test_default_v_normalization_preserves_gaussian_scale(self) -> None:
        provider = LOZOFastDirectionProvider(
            param_metadata=_metadata(),
            rank=1,
            nu=10,
            random_device="cpu",
        )

        directions = provider.sample_direction(123)
        v = directions["model.decoder.layers.0.fc1.weight"]["V"].float()

        self.assertGreater(float(v.norm()), 1.0)

    def test_default_perturbation_normalization_uses_analytic_rank_scale(self) -> None:
        scales = []
        for rank in (1, 4):
            provider = LOZOFastDirectionProvider(
                param_metadata=_metadata(),
                rank=rank,
                nu=10,
                random_device="cpu",
            )
            direction = provider.sample_direction(123)[
                "model.decoder.layers.0.fc1.weight"
            ]
            scales.append(float(direction["perturbation_normalization_scale"]))

        self.assertTrue(torch.isclose(torch.tensor(scales[0]), torch.tensor(1.0)))
        self.assertTrue(torch.isclose(torch.tensor(scales[1]), torch.tensor(0.5)))

    def test_perturbation_normalization_none_preserves_raw_lowrank_energy(self) -> None:
        provider = LOZOFastDirectionProvider(
            param_metadata=_metadata(),
            rank=4,
            nu=10,
            random_device="cpu",
            perturbation_normalization="none",
        )

        direction = provider.sample_direction(123)["model.decoder.layers.0.fc1.weight"]

        self.assertEqual(direction["perturbation_normalization"], "none")
        self.assertEqual(float(direction["perturbation_normalization_scale"]), 1.0)
        self.assertEqual(float(direction["scale"]), 1.0)


if __name__ == "__main__":
    unittest.main()
