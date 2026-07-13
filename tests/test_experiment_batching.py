import torch

from zo_vllm.experiment.infra.batching import (
    batches_per_epoch,
    epoch_interval_to_steps,
    make_train_dataloader,
)


def _epoch_batches(loader):
    return [batch for batch in loader]


def test_sequential_dataloader_repeats_order_each_epoch():
    loader = make_train_dataloader(list(range(10)), 3, sampler="sequential", seed=123)

    assert _epoch_batches(loader) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert _epoch_batches(loader) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


def test_dataloader_drop_last_is_explicit():
    loader = make_train_dataloader(
        list(range(10)),
        3,
        sampler="sequential",
        seed=123,
        drop_last=True,
    )

    assert _epoch_batches(loader) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_hf_random_dataloader_reshuffles_each_epoch_and_is_reproducible():
    items = list(range(10))
    loader = make_train_dataloader(items, 3, sampler="hf_random", seed=123)
    repeat_loader = make_train_dataloader(items, 3, sampler="hf_random", seed=123)

    first_epoch = _epoch_batches(loader)
    second_epoch = _epoch_batches(loader)

    assert first_epoch != second_epoch
    assert first_epoch == _epoch_batches(repeat_loader)
    assert second_epoch == _epoch_batches(repeat_loader)

    expected_first = torch.randperm(10, generator=torch.Generator().manual_seed(123))
    expected_second = torch.randperm(10, generator=torch.Generator().manual_seed(124))
    assert [item for batch in first_epoch for item in batch] == expected_first.tolist()
    assert [item for batch in second_epoch for item in batch] == expected_second.tolist()


def test_epoch_interval_to_steps_respects_drop_last():
    assert batches_per_epoch(10, 3, drop_last=False) == 4
    assert batches_per_epoch(10, 3, drop_last=True) == 3
    assert (
        epoch_interval_to_steps(
            2,
            num_items=10,
            batch_size=3,
            drop_last=False,
        )
        == 8
    )
    assert (
        epoch_interval_to_steps(
            2,
            num_items=10,
            batch_size=3,
            drop_last=True,
        )
        == 6
    )
