import math
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from zo_vllm.core.binary_option_objective import (
    accuracy_from_multi_option_nll,
    classification_loss_from_multi_option_nll,
    predictions_from_multi_option_nll,
    regroup_flat_option_values,
)
from zo_vllm.engine import TokenScoreResult
from zo_vllm.training.task_batches import (
    BinaryClassificationScoringBatch,
    OptionClassificationScoringBatch,
    resolve_objective_name,
)
from zo_vllm.experiment.runners.args import build_vllm_zo_task_arg_parser
from zo_vllm.tasks import TaskConfig
from zo_vllm.tasks import get_task
import zo_vllm.tasks.superglue.base as superglue_base
import zo_vllm.tasks.superglue.boolq as superglue_boolq
from zo_vllm.tasks.superglue import (
    SUPERGLUE_OBJECTIVE_TO_TASK,
    SUPERGLUE_TASK_SPECS,
    dataset_to_superglue_rows,
    encode_superglue_option_prompts,
    superglue_prediction_metric,
)


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=True):
        tokens = str(text).replace("\n", " \n ").split()
        ids = [len(token) for token in tokens]
        return ([0] if add_special_tokens else []) + ids


def test_superglue_registry_exposes_all_tasks():
    expected = {
        "superglue_boolq",
        "superglue_cb",
        "superglue_copa",
        "superglue_multirc",
        "superglue_record",
        "superglue_rte",
        "superglue_wic",
        "superglue_wsc",
    }

    assert set(SUPERGLUE_TASK_SPECS) == expected
    for name in expected:
        assert get_task(name).name == name
        with pytest.raises(KeyError, match="unsupported task"):
            get_task(name.replace("_", "/"))


def test_superglue_tasks_are_split_by_trainable_task():
    task_dir = Path(__file__).resolve().parents[1] / "zo_vllm" / "tasks" / "superglue"
    for filename in (
        "boolq.py",
        "cb.py",
        "copa.py",
        "multirc.py",
        "record.py",
        "rte.py",
        "wic.py",
        "wsc.py",
    ):
        assert (task_dir / filename).exists(), filename
    assert not (task_dir.parent / "superglue.py").exists()


def test_superglue_objectives_are_cli_choices():
    parser = build_vllm_zo_task_arg_parser()
    for objective in SUPERGLUE_OBJECTIVE_TO_TASK:
        args = parser.parse_args(["--train-objective", objective])
        assert resolve_objective_name(args.train_objective) == objective


def test_hf_like_task_flags_resolve_to_internal_objectives():
    assert (
        resolve_objective_name(dataset_name="glue", dataset_config_name="sst2")
        == "sst2_classification"
    )
    assert (
        resolve_objective_name(dataset_name="super_glue", dataset_config_name="copa")
        == "superglue_copa_classification"
    )
    assert (
        resolve_objective_name(dataset_name="super_glue", dataset_config_name="record")
        == "superglue_record_nll"
    )
    assert resolve_objective_name(task_name="boolq") == "boolq_classification"
    assert (
        resolve_objective_name("superglue_wic_classification")
        == "superglue_wic_classification"
    )
    with pytest.raises(ValueError, match="unsupported objective_name"):
        resolve_objective_name("wic")


def test_superglue_small_train_split_uses_tail_dev_without_overlap(monkeypatch):
    train_rows = [
        {
            "premise": f"p{i}",
            "choice1": "a",
            "choice2": "b",
            "question": "cause",
            "label": i % 2,
        }
        for i in range(400)
    ]
    validation_rows = train_rows[:10]

    def fake_load_dataset(name, config):
        assert name == "super_glue"
        assert config == "copa"
        return DatasetDict(
            {
                "train": Dataset.from_list(train_rows),
                "validation": Dataset.from_list(validation_rows),
            }
        )

    monkeypatch.setattr(superglue_base, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        superglue_base,
        "shuffled_select",
        lambda dataset, *, seed, num: (
            dataset.select(range(min(int(num), len(dataset))))
            if num is not None
            else dataset
        ),
    )

    splits = get_task("superglue_copa").load_splits(
        TaskConfig(
            name="superglue_copa",
            num_train=1000,
            num_dev=100,
            num_eval=10,
            data_seed=42,
        )
    )

    assert len(splits.train) == 300
    assert len(splits.dev) == 100
    assert splits.train[-1]["premise"] == "p299"
    assert splits.dev[0]["premise"] == "p300"


def test_superglue_boolq_uses_standalone_boolq_dataset(monkeypatch):
    calls = []

    def fake_load_dataset(name):
        calls.append(name)
        return DatasetDict(
            {
                "train": Dataset.from_list(
                    [
                        {"passage": f"p{i}", "question": f"q{i}", "answer": bool(i % 2)}
                        for i in range(4)
                    ]
                ),
                "validation": Dataset.from_list(
                    [{"passage": "vp", "question": "vq", "answer": True}]
                ),
            }
        )

    monkeypatch.setattr(superglue_boolq, "load_dataset", fake_load_dataset)

    splits = get_task("superglue_boolq").load_splits(
        TaskConfig(
            name="superglue_boolq",
            num_train=2,
            num_dev=1,
            num_eval=1,
            data_seed=0,
        )
    )

    assert calls == ["boolq"]
    assert "title" not in splits.train[0]


def test_shuffled_select_shuffle_impl_contract(monkeypatch):
    rows = Dataset.from_list([{"x": i} for i in range(12)])

    monkeypatch.delenv("ZO_TASK_SHUFFLE_IMPL", raising=False)
    default_rows = list(superglue_base.shuffled_select(rows, seed=42, num=6)["x"])

    monkeypatch.setenv("ZO_TASK_SHUFFLE_IMPL", "hf")
    hf_rows = list(superglue_base.shuffled_select(rows, seed=42, num=6)["x"])

    monkeypatch.setenv("ZO_TASK_SHUFFLE_IMPL", "bad")
    with pytest.raises(ValueError, match="ZO_TASK_SHUFFLE_IMPL"):
        superglue_base.shuffled_select(rows, seed=42, num=6)

    assert default_rows == [10, 9, 0, 8, 5, 2]
    assert hf_rows == [0, 7, 6, 9, 11, 3]


def test_cb_prompt_encoding_is_three_way_option_classification():
    rows = dataset_to_superglue_rows(
        [
            {
                "premise": "The sky is blue.",
                "hypothesis": "The sky has a color.",
                "label": 0,
                "idx": 1,
            }
        ],
        task_name="superglue_cb",
    )

    encoded = encode_superglue_option_prompts(rows, WhitespaceTokenizer())

    assert encoded.labels == [0]
    assert encoded.label_sets == [(0,)]
    assert encoded.option_counts == [3]
    assert len(encoded.flat_option_ids) == 3
    assert min(encoded.flat_option_lens) > 0


def test_copa_option_len_matches_lozo_stripped_stem():
    rows = dataset_to_superglue_rows(
        [
            {
                "premise": "The man slipped.",
                "choice1": "He fell.",
                "choice2": "He smiled.",
                "question": "effect",
                "label": 0,
                "idx": 1,
            }
        ],
        task_name="superglue_copa",
    )
    tokenizer = WhitespaceTokenizer()
    encoded = encode_superglue_option_prompts(rows, tokenizer)
    spec = SUPERGLUE_TASK_SPECS["superglue_copa"]
    row = rows[0]

    stem_len = len(tokenizer.encode(spec.stem(row).strip(" ")))
    expected = [
        len(tokenizer.encode(spec.verbalized(row, candidate).strip(" "))) - stem_len
        for candidate in row.candidates
    ]

    assert encoded.option_lens == [expected]


def test_rte_prompt_matches_lozo_template_verbalizer_order():
    rows = dataset_to_superglue_rows(
        [
            {
                "premise": "A dog is running.",
                "hypothesis": "An animal is moving.",
                "label": 0,
                "idx": 1,
            }
        ],
        task_name="superglue_rte",
    )
    spec = SUPERGLUE_TASK_SPECS["superglue_rte"]

    stem = 'A dog is running.\nDoes this mean that "An animal is moving." is true? Yes or No?\n'
    assert spec.stem(rows[0]) == stem
    assert spec.verbalized(rows[0], 0) == stem + "Yes"
    assert spec.verbalized(rows[0], 1) == stem + "No"


def test_wsc_prompt_matches_lozo_template_verbalizer_order():
    rows = dataset_to_superglue_rows(
        [
            {
                "text": "Alice thanked Bob because she was grateful.",
                "span1_text": "Alice",
                "span2_text": "she",
                "label": 1,
                "idx": 1,
            }
        ],
        task_name="superglue_wsc",
    )
    spec = SUPERGLUE_TASK_SPECS["superglue_wsc"]

    stem = (
        "Alice thanked Bob because she was grateful.\n"
        'In the previous sentence, does the pronoun "she" refer to Alice? Yes or No?\n'
    )
    assert spec.stem(rows[0]) == stem
    assert spec.verbalized(rows[0], 0) == stem + "No"
    assert spec.verbalized(rows[0], 1) == stem + "Yes"


def test_multi_option_loss_and_accuracy_support_variable_width_rows():
    option_nll = [[0.2, 1.0, 2.0], [3.0, 0.7]]
    labels = [0, 1]

    assert predictions_from_multi_option_nll(option_nll) == [0, 1]
    assert accuracy_from_multi_option_nll(option_nll, labels) == 1.0
    assert accuracy_from_multi_option_nll(option_nll, [(1, 2), (1,)]) == 0.5
    assert regroup_flat_option_values([0.2, 1.0, 2.0, 3.0, 0.7], [3, 2]) == option_nll
    loss = classification_loss_from_multi_option_nll(option_nll, labels)
    expected = (
        math.log(math.exp(-0.2) + math.exp(-1.0) + math.exp(-2.0))
        + 0.2
        + math.log(math.exp(-3.0) + math.exp(-0.7))
        + 0.7
    ) / 2
    assert abs(loss - expected) < 1e-12


def test_option_classification_scoring_batch_uses_request_mean_nll():
    batch = OptionClassificationScoringBatch(
        token_id_groups=[[1, 2], [1, 2, 3, 4]],
        labels=[1],
        label_sets=[(1,)],
        option_counts=[2],
        loss_token_lens=[1, 3],
    )
    score = TokenScoreResult(
        loss=0.0,
        nll_sum=0.0,
        num_tokens=4,
        request_nll=[2.0, 1.0],
        request_num_tokens=[1, 3],
        detail={},
        raw={},
    )

    assert batch.accuracy(score) == 1.0
    assert batch.loss(score) == classification_loss_from_multi_option_nll(
        [[2.0, 1.0]],
        [1],
    )
    assert batch.row_request_nlls(score) == [[2.0, 1.0]]
    assert batch.row_request_num_tokens(score) == [[1, 3]]
    assert batch.per_row_losses(score) == [
        classification_loss_from_multi_option_nll([[2.0, 1.0]], [1])
    ]


def test_binary_scoring_batch_reports_per_row_option_losses():
    batch = BinaryClassificationScoringBatch(
        token_id_groups=[[1], [2], [3], [4]],
        labels=[0, 1],
        loss_token_lens=[1, 1, 1, 1],
    )
    score = TokenScoreResult(
        loss=0.0,
        nll_sum=0.0,
        num_tokens=4,
        request_nll=[0.3, 2.0, 1.7, 0.4],
        request_num_tokens=[1, 2, 3, 4],
        detail={},
        raw={},
    )

    assert batch.row_request_nlls(score) == [[0.3, 1.7], [2.0, 0.4]]
    assert batch.row_request_num_tokens(score) == [[1, 3], [2, 4]]
    expected = [
        classification_loss_from_multi_option_nll([[0.3, 1.7]], [0]),
        classification_loss_from_multi_option_nll([[2.0, 0.4]], [1]),
    ]
    assert batch.per_row_losses(score) == expected


def test_record_rows_use_entity_candidates_and_answer_metric():
    rows = dataset_to_superglue_rows(
        [
            {
                "passage": "@highlight\nAlice visited Paris.",
                "query": "@placeholder visited Paris.",
                "entities": ["Alice!", "Bob"],
                "answers": ["Alice!", "The Alice!"],
                "idx": {"query": 0, "passage": 0},
            }
        ],
        task_name="superglue_record",
    )
    encoded = encode_superglue_option_prompts(rows, WhitespaceTokenizer())

    assert encoded.labels == [0]
    assert encoded.label_sets == [(0,)]
    assert encoded.option_counts == [2]
    assert min(encoded.flat_option_lens) > 0
    assert superglue_prediction_metric(rows, [0]) == {"em": 1.0, "f1": 1.0}
    assert superglue_prediction_metric(rows, [1]) == {"em": 0.0, "f1": 0.0}
