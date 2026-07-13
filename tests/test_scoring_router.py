from __future__ import annotations

from zo_vllm.training import objective_router as router
from zo_vllm.tasks.superglue.record import OBJECTIVE_NAME as RECORD_NLL_OBJECTIVE


class FakeObjectiveScore:
    def __init__(self, loss=1.0, accuracy=0.75, predictions=None):
        self.loss = loss
        self.accuracy = accuracy
        self.score = object()
        self._predictions = predictions or [1]


class FakeBatch:
    def __init__(self, predictions=None):
        self.predictions_value = predictions or [1]

    def predictions(self, _score):
        return list(self.predictions_value)


def test_score_objective_loss_routes_without_legacy_scoring_wrappers(monkeypatch):
    calls = []

    def fake_build(rows, tokenizer, *, objective_name, **_kwargs):
        calls.append(("build", objective_name, len(rows)))
        return FakeBatch()

    def fake_score(engine, batch, **kwargs):
        calls.append(("score", kwargs.get("lora_id")))
        return FakeObjectiveScore(loss=1.25)

    monkeypatch.setattr(router, "build_objective_batch", fake_build)
    monkeypatch.setattr(router, "score_clean_objective", fake_score)

    common = {
        "llm": object(),
        "rows": [{"sentence": "x"}],
        "tokenizer": object(),
        "max_logits_tokens": 128,
        "loss_impl": "logprobs",
        "max_length": 256,
        "max_new_tokens": 32,
        "lora_id": 7,
    }

    assert router.score_objective_loss("sst2_classification", **common) == 1.25
    assert router.score_objective_loss(RECORD_NLL_OBJECTIVE, **common) == 1.25
    assert router.score_objective_loss("squad_nll", **common) == 1.25
    assert router.score_objective_loss("prompt_nll", **common) is None
    assert calls == [
        ("build", "sst2_classification", 1),
        ("score", 7),
        ("build", RECORD_NLL_OBJECTIVE, 1),
        ("score", 7),
        ("build", "squad_nll", 1),
        ("score", 7),
    ]


def test_current_batch_train_loss_routes_prompt_nll(monkeypatch):
    calls = []

    def fake_detailed(*args, **kwargs):
        calls.append((len(args[1]), kwargs.get("lora_ids")))
        return 4.0, {"score_direct_worker_s": 0.125}

    monkeypatch.setattr(router, "score_direct_worker_detailed", fake_detailed)

    loss, score_s = router.score_current_batch_train_loss(
        "prompt_nll",
        object(),
        [{"prompt": "a"}, {"prompt": "b"}],
        object(),
        max_logits_tokens=128,
        loss_impl="logprobs",
        max_length=256,
        max_new_tokens=32,
        lora_id=11,
    )

    assert loss == 4.0
    assert score_s == 0.125
    assert calls == [(2, [11, 11])]


def test_current_batch_train_loss_rejects_unscorable_objective(monkeypatch):
    monkeypatch.setattr(router, "score_objective_loss", lambda *args, **kwargs: None)

    try:
        router.score_current_batch_train_loss(
            "unknown_objective",
            object(),
            [],
            object(),
            max_logits_tokens=128,
            loss_impl="logprobs",
            max_length=256,
            max_new_tokens=32,
        )
    except ValueError as exc:
        assert "cannot score train loss" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_eval_objective_metrics_uses_objective_predictions(monkeypatch):
    calls = []

    def fake_build(rows, tokenizer, *, objective_name, **_kwargs):
        calls.append(("build", objective_name, len(rows)))
        return FakeBatch(predictions=[1] * len(rows))

    def fake_score(engine, batch, **kwargs):
        calls.append(("score", kwargs.get("lora_id")))
        return FakeObjectiveScore(loss=1.0, accuracy=0.8)

    monkeypatch.setattr(router, "build_objective_batch", fake_build)
    monkeypatch.setattr(router, "score_clean_objective", fake_score)

    common = {
        "llm": object(),
        "tokenizer": object(),
        "dev_rows": [{"label": 1}],
        "valid_rows": [{"label": 1}, {"label": 1}],
        "max_logits_tokens": 128,
        "loss_impl": "logprobs",
        "max_length": 256,
        "max_new_tokens": 32,
        "lora_id": 9,
    }

    metrics = router.eval_objective_metrics("sst2_classification", **common)
    assert metrics is not None
    assert metrics.primary_name == "accuracy"
    assert metrics.dev_value == 0.8
    assert metrics.valid_value == 0.8
    assert metrics.dev_metrics == {"accuracy": 0.8}
    assert calls == [
        ("build", "sst2_classification", 1),
        ("score", 9),
        ("build", "sst2_classification", 2),
        ("score", 9),
    ]


def test_periodic_eval_routes_classification_metrics(monkeypatch):
    calls = []

    def fake_loss(*args, **kwargs):
        calls.append(("loss", args[0], len(args[2]), kwargs.get("lora_id")))
        return 6.0

    def fake_metrics(*args, **kwargs):
        calls.append(("metrics", args[0], len(args[3]), len(args[4])))
        return router.ObjectiveEvalMetrics(
            primary_name="accuracy",
            dev_value=0.81,
            valid_value=0.82,
            dev_metrics={"accuracy": 0.81},
            valid_metrics={"accuracy": 0.82},
        )

    monkeypatch.setattr(router, "score_objective_loss", fake_loss)
    monkeypatch.setattr(router, "eval_objective_metrics", fake_metrics)

    result = router.score_periodic_eval(
        "sst2_classification",
        object(),
        object(),
        [{"id": 1}],
        [{"id": 2}, {"id": 3}],
        accuracy_eval_mode="full",
        max_logits_tokens=128,
        loss_impl="logprobs",
        max_length=256,
        max_new_tokens=32,
        lora_id=13,
    )

    assert result is not None
    assert result.loss == 6.0
    assert result.primary_name == "accuracy"
    assert result.dev_value == 0.81
    assert result.valid_value == 0.82
    assert calls == [
        ("loss", "sst2_classification", 1, 13),
        ("metrics", "sst2_classification", 1, 2),
    ]


def test_periodic_eval_routes_squad_dev_metric_only(monkeypatch):
    calls = []

    def fake_loss(*args, **kwargs):
        calls.append(("loss", args[0], len(args[2]), kwargs.get("lora_id")))
        return 7.0

    def fake_squad_eval(*args, **kwargs):
        calls.append(("squad_eval", len(args[2]), kwargs.get("lora_id")))
        return {"f1": 0.66, "em": 0.44}

    monkeypatch.setattr(router, "score_objective_loss", fake_loss)
    monkeypatch.setattr(router, "eval_squad_f1", fake_squad_eval)

    result = router.score_periodic_eval(
        "squad_nll",
        object(),
        object(),
        [{"id": 1}],
        [{"id": 2}],
        accuracy_eval_mode="full",
        max_logits_tokens=128,
        loss_impl="logprobs",
        max_length=256,
        max_new_tokens=32,
        lora_id=14,
    )

    assert result is not None
    assert result.loss == 7.0
    assert result.primary_name == "f1"
    assert result.dev_value == 0.66
    assert result.valid_value is None
    assert result.dev_metrics == {"f1": 0.66, "em": 0.44}
    assert calls == [("loss", "squad_nll", 1, 14), ("squad_eval", 1, 14)]
