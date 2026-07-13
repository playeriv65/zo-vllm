from zo_vllm.training.task_batches import split_rows_for_agzo_subspace


def test_agzo_subspace_rows_chunk_at_sixteen():
    rows = list(range(512))
    chunks = split_rows_for_agzo_subspace(rows, chunk_size=16)

    assert len(chunks) == 32
    assert all(len(chunk) == 16 for chunk in chunks)
    assert chunks[0] == list(range(16))
    assert chunks[-1] == list(range(496, 512))


def test_agzo_subspace_rows_under_threshold_stay_single_chunk():
    rows = list(range(16))
    assert split_rows_for_agzo_subspace(rows, chunk_size=16) == [rows]
