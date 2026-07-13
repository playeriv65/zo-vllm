from __future__ import annotations

import math
import unittest

import torch

from zo_vllm.analysis import (
    select_subspace_pairs,
    summarize_subspace_pairs,
    weighted_subspace_similarity,
)


def _make_v_maps(rank: int, num_records: int = 5) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234 + rank)
    v_maps: list[dict[str, torch.Tensor]] = []
    for _ in range(num_records):
        v_maps.append(
            {
                "layers.0.mlp.fc1.weight": torch.randn(
                    17, rank, generator=generator
                ),
                "layers.1.self_attn.q_proj.weight": torch.randn(
                    19, rank, generator=generator
                ),
            }
        )
    return v_maps


class SubspaceSimilarityTest(unittest.TestCase):
    def test_rank1_fast_path_matches_generic_svd_path(self) -> None:
        v_maps = _make_v_maps(rank=1, num_records=2)

        fast = weighted_subspace_similarity(
            v_maps[0],
            v_maps[1],
            device="cpu",
            force_svd=False,
        )
        generic = weighted_subspace_similarity(
            v_maps[0],
            v_maps[1],
            device="cpu",
            force_svd=True,
        )

        self.assertEqual(fast.keys(), generic.keys())
        for key in fast:
            self.assertTrue(
                math.isclose(fast[key], generic[key], rel_tol=1e-6, abs_tol=1e-6),
                msg=f"{key}: {fast[key]} != {generic[key]}",
            )

    def test_batched_summary_matches_per_pair_summary(self) -> None:
        for rank in (1, 2):
            with self.subTest(rank=rank):
                v_maps = _make_v_maps(rank=rank, num_records=5)
                pairs = [(0, 1), (0, 3), (2, 4), (1, 4)]

                batched = summarize_subspace_pairs(
                    v_maps,
                    pairs,
                    tag="batched",
                    device="cpu",
                    batched=True,
                    include_pair_details=True,
                    force_svd=True,
                )
                per_pair = summarize_subspace_pairs(
                    v_maps,
                    pairs,
                    tag="per_pair",
                    device="cpu",
                    batched=False,
                    include_pair_details=True,
                    force_svd=True,
                )

                self.assertEqual(batched.num_pairs, per_pair.num_pairs)
                self.assertTrue(
                    math.isclose(
                        batched.mean_weighted_subspace_similarity,
                        per_pair.mean_weighted_subspace_similarity,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    )
                )
                for left, right in zip(batched.pairs, per_pair.pairs, strict=True):
                    self.assertEqual(left["left"], right["left"])
                    self.assertEqual(left["right"], right["right"])
                    self.assertTrue(
                        math.isclose(
                            left["weighted_subspace_similarity"],
                            right["weighted_subspace_similarity"],
                            rel_tol=1e-6,
                            abs_tol=1e-6,
                        )
                    )

    def test_pair_selection_modes_are_deterministic(self) -> None:
        default = select_subspace_pairs(
            6,
            mode="default",
            random_pairs=4,
            seed=7,
            gaps=(2, 4),
        )
        self.assertEqual(
            [tag for tag, _pairs in default],
            ["adjacent", "random", "gap_2", "gap_4"],
        )
        self.assertEqual(len(default[0][1]), 5)
        self.assertEqual(len(default[1][1]), 4)
        self.assertEqual(len(default[2][1]), 4)
        self.assertEqual(len(default[3][1]), 2)

        random_once = select_subspace_pairs(
            6,
            mode="random_nonoverlap",
            random_pairs=4,
            seed=11,
        )
        random_twice = select_subspace_pairs(
            6,
            mode="random_nonoverlap",
            random_pairs=4,
            seed=11,
        )
        self.assertEqual(random_once, random_twice)

        all_pairs = select_subspace_pairs(
            6,
            mode="all_nonoverlap",
            random_pairs=999,
            seed=0,
        )
        self.assertEqual(len(all_pairs[0][1]), 15)


if __name__ == "__main__":
    unittest.main()
