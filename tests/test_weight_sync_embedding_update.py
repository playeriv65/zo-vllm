import torch

from zo_vllm.core.weight_sync import apply_embedding_lowrank_update_to_weight_


def test_embedding_lowrank_update_handles_hidden_first_unpadded_vocab():
    target = torch.zeros((3, 4), dtype=torch.float32)
    U = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    V = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    apply_embedding_lowrank_update_to_weight_(
        target,
        U,
        V,
        c=2.0,
        lr=0.1,
        weight_decay=0.0,
        precision="param",
    )

    expected = -0.2 * V @ U[:4, :].T
    torch.testing.assert_close(target, expected)


def test_embedding_lowrank_update_handles_vocab_first_unpadded_vocab():
    target = torch.zeros((4, 3), dtype=torch.float32)
    U = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    V = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    apply_embedding_lowrank_update_to_weight_(
        target,
        U,
        V,
        c=2.0,
        lr=0.1,
        weight_decay=0.0,
        precision="param",
    )

    expected = -0.2 * U[:4, :] @ V.T
    torch.testing.assert_close(target, expected)


def test_embedding_lowrank_update_handles_padded_vocab_first_target():
    target = torch.zeros((5, 3), dtype=torch.float32)
    U = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    V = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    apply_embedding_lowrank_update_to_weight_(
        target,
        U,
        V,
        c=2.0,
        lr=0.1,
        weight_decay=0.0,
        precision="param",
    )

    expected = torch.zeros((5, 3), dtype=torch.float32)
    expected[:4, :] = -0.2 * V @ U.T
    torch.testing.assert_close(target, expected)


def test_embedding_lowrank_update_handles_padded_hidden_first_target():
    target = torch.zeros((3, 5), dtype=torch.float32)
    U = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    V = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    apply_embedding_lowrank_update_to_weight_(
        target,
        U,
        V,
        c=2.0,
        lr=0.1,
        weight_decay=0.0,
        precision="param",
    )

    expected = torch.zeros((3, 5), dtype=torch.float32)
    expected[:, :4] = -0.2 * U @ V.T
    torch.testing.assert_close(target, expected)
