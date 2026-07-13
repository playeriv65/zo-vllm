import pytest

from zo_vllm.core.token_groups import (
    borrow_or_copy_int_list,
    borrow_or_copy_token_group_batches,
    borrow_or_copy_token_groups,
)
from zo_vllm.engine import ZOVLLMEngine


def test_validate_score_inputs_accepts_hf_like_labels():
    ZOVLLMEngine._validate_score_inputs(
        [[1, 2, 3, 4]],
        None,
        [[-100, -100, 3, 4]],
        None,
    )


def test_validate_score_inputs_rejects_label_shape_mismatch():
    with pytest.raises(ValueError, match="labels rows must match"):
        ZOVLLMEngine._validate_score_inputs(
            [[1, 2, 3]],
            None,
            [[-100, 2]],
            None,
        )


def test_validate_score_inputs_rejects_empty_label_mask():
    with pytest.raises(ValueError, match="at least one active loss token"):
        ZOVLLMEngine._validate_score_inputs(
            [[1, 2, 3]],
            None,
            [[-100, -100, -100]],
            None,
        )


def test_normalized_token_containers_are_borrowed_without_copying():
    groups = [[1, 2], [3, 4]]
    values = [5, 6]
    batches = [groups]

    assert borrow_or_copy_token_groups(groups) is groups
    assert borrow_or_copy_int_list(values) is values
    assert borrow_or_copy_token_group_batches(batches) is batches


def test_external_token_sequences_are_normalized_once():
    groups = ((1, 2), (3, 4))
    values = (5, 6)
    batches = (groups,)

    normalized_groups = borrow_or_copy_token_groups(groups)
    normalized_values = borrow_or_copy_int_list(values)
    normalized_batches = borrow_or_copy_token_group_batches(batches)

    assert normalized_groups == [[1, 2], [3, 4]]
    assert normalized_groups is not groups
    assert normalized_values == [5, 6]
    assert normalized_batches == [[[1, 2], [3, 4]]]


@pytest.mark.parametrize(
    ("executor_name", "expected"),
    [("UniProcExecutor", True), ("MultiprocExecutor", False)],
)
def test_device_tensor_rpc_is_limited_to_uniproc_executor(executor_name, expected):
    executor_type = type(
        executor_name,
        (),
        {"supports_device_tensor_rpc": executor_name == "UniProcExecutor"},
    )
    engine = type(
        "EngineStub",
        (),
        {
            "llm": type(
                "LLMStub",
                (),
                {
                    "llm_engine": type(
                        "LLMEngineStub",
                        (),
                        {"model_executor": executor_type()},
                    )()
                },
            )()
        },
    )()

    assert ZOVLLMEngine._can_return_device_tensors(engine) is expected
