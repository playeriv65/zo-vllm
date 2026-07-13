from dataclasses import dataclass

from zo_vllm.tasks import TaskConfig, get_task
from zo_vllm.tasks.squad import SquadRow
from zo_vllm.tasks.sst2 import SST2Row
from zo_vllm.training.task_encoding import ZOTaskDataCollator, ZOTaskEncodingConfig


@dataclass
class TinyTokenizer:
    bos_token: str = "</s>"

    def encode(self, text: str, add_special_tokens: bool = True):
        tokens = [abs(hash(item)) % 1000 + 3 for item in text.split()]
        if add_special_tokens:
            return [2, *tokens]
        return tokens


def test_task_collator_builds_binary_zo_batch():
    rows = [SST2Row(sentence="good movie", label=1)]
    collator = ZOTaskDataCollator(
        tokenizer=TinyTokenizer(),
        config=ZOTaskEncodingConfig(objective_name="sst2_classification"),
    )

    batch = collator(rows)

    assert len(batch.token_id_groups) == 2
    assert batch.labels is None
    assert batch.loss_token_lens == [1, 1]
    assert batch.objective is not None


def test_task_adapter_data_collator_accepts_hf_rows():
    collator = get_task("sst2").data_collator(
        TinyTokenizer(),
        TaskConfig(name="sst2", num_train=1, num_dev=0, num_eval=0, data_seed=0),
    )

    batch = collator([{"sentence": "good movie", "label": 1, "idx": 0}])

    assert len(batch.token_id_groups) == 2
    assert batch.labels is None
    assert batch.loss_token_lens == [1, 1]
    assert batch.objective is not None


def test_squad_task_adapter_data_collator_uses_masked_lm_objective():
    collator = get_task("squad").data_collator(
        TinyTokenizer(),
        TaskConfig(
            name="squad",
            num_train=1,
            num_dev=0,
            num_eval=0,
            data_seed=0,
            max_length=64,
            max_new_tokens=4,
        ),
    )

    batch = collator(
        [
            SquadRow(
                title="T",
                context="short context",
                question="What?",
                answers=("answer",),
            )
        ]
    )

    assert len(batch.token_id_groups) == 1
    assert batch.labels is None
    assert batch.loss_token_lens == [1]
    assert batch.objective is not None


def test_task_collator_attaches_agzo_subspace_factory():
    rows = [SST2Row(sentence="good movie", label=1)]
    step = {"value": 2}
    collator = ZOTaskDataCollator(
        tokenizer=TinyTokenizer(),
        config=ZOTaskEncodingConfig(
            objective_name="sst2_classification",
            direction_provider="agzo",
            agzo_kappa=2,
            agzo_subspace_chunk_size=1,
        ),
        subspace_rows=[
            SST2Row(sentence="row zero", label=0),
            SST2Row(sentence="row one", label=1),
            SST2Row(sentence="row two", label=0),
        ],
        step_getter=lambda: step["value"],
    )

    batch = collator(rows)
    assert batch.subspace_num_rows == 2
    assert batch.subspace_token_id_group_batch_factory is not None

    subspace_batches = batch.subspace_token_id_group_batch_factory()

    assert len(subspace_batches) == 2
    assert all(len(chunk) == 2 for chunk in subspace_batches)
