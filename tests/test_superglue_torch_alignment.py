import torch

from zo_vllm.core.binary_option_objective import (
    classification_loss_from_multi_option_nll,
    predictions_from_multi_option_nll,
)
from zo_vllm.tasks import TaskConfig, get_task
from zo_vllm.tasks.superglue import (
    SUPERGLUE_OBJECTIVE_TO_TASK,
    dataset_to_superglue_rows,
    encode_superglue_option_prompts,
    superglue_prediction_metric,
)


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=True):
        tokens = str(text).replace("\n", " \n ").split()
        ids = [len(token) for token in tokens]
        return ([0] if add_special_tokens else []) + ids


def _synthetic_option_nll(option_counts, labels):
    rows = []
    for row_index, (count, label) in enumerate(zip(option_counts, labels)):
        row = [2.0 + row_index * 0.13 + option * 0.17 for option in range(count)]
        row[int(label)] = 0.2 + row_index * 0.11
        rows.append(row)
    return rows


def test_all_superglue_tasks_align_with_torch_cross_entropy():
    tokenizer = WhitespaceTokenizer()
    for objective_name, task_name in sorted(SUPERGLUE_OBJECTIVE_TO_TASK.items()):
        task = get_task(task_name)
        splits = task.load_splits(
            TaskConfig(
                name=task_name,
                num_train=2,
                num_dev=2,
                num_eval=2,
                data_seed=7,
            )
        )
        train_rows = dataset_to_superglue_rows(splits.train, task_name=task_name)
        dev_rows = dataset_to_superglue_rows(splits.dev, task_name=task_name)
        eval_rows = dataset_to_superglue_rows(splits.eval, task_name=task_name)
        assert train_rows, objective_name
        assert dev_rows, objective_name
        assert eval_rows, objective_name

        encoded = encode_superglue_option_prompts(train_rows, tokenizer)
        assert encoded.flat_option_ids, objective_name
        assert min(encoded.flat_option_lens) > 0, objective_name

        option_nll = _synthetic_option_nll(encoded.option_counts, encoded.labels)
        actual_loss = classification_loss_from_multi_option_nll(
            option_nll,
            encoded.labels,
        )
        max_options = max(encoded.option_counts)
        logits = torch.full((len(option_nll), max_options), -1.0e9)
        for row_index, row_nll in enumerate(option_nll):
            logits[row_index, : len(row_nll)] = -torch.tensor(row_nll)
        target = torch.tensor(encoded.labels, dtype=torch.long)
        expected_loss = torch.nn.functional.cross_entropy(logits, target).item()
        assert abs(actual_loss - expected_loss) < 1e-6, objective_name

        predictions = predictions_from_multi_option_nll(option_nll)
        metrics = superglue_prediction_metric(train_rows, predictions)
        if task_name == "superglue_record":
            assert set(metrics) == {"em", "f1"}, objective_name
        else:
            assert set(metrics) == {"accuracy"}, objective_name
