import gc
import threading
import weakref

import torch
import pytest

from zo_trainer.optimizer import ZOSGDOptimizer
from zo_vllm.engine import (
    PlusMinusScoreResult,
    RuntimeProbeSlots,
    TokenScoreResult,
    ZOVLLMEngine,
)
from zo_vllm.training import (
    EvolutionStrategyEstimator,
    MultiQueryZOEstimator,
    TokenProbeBatch,
    ZOEstimate,
    ZOGradientEstimate,
    ZOEstimatorConfig,
    ZOStepCallback,
    ZOStepConfig,
    ZOStepControl,
    ZOStepper,
)
from zo_vllm.core.probe_results import ProbeLossResult, ProbeTiming
from zo_vllm.training.direction import DirectionSample, DirectionSpec, RolloutProbeBatch


class FakeDirectionProvider:
    def will_refresh(self, *, step: int) -> bool:
        return False

    def direction_specs(self):
        return [
            DirectionSpec(
                name="layer.weight",
                out_features=1,
                in_features=1,
                rank=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        ]

    def next(self, batch, *, step: int):
        return DirectionSample(
            directions={"layer.weight": {"U": torch.ones(1, 1), "V": torch.ones(1, 1)}},
            refreshed=True,
            info={"perturb_seed": 123, "perturbation_effective_rms": 1.0},
        )


class FakeUpdateState:
    def __init__(self) -> None:
        self.events = []
        self.last_projected_grad = None
        self.applied_directions = []

    def prepare_for_score(self, directions):
        self.events.append("prepare")
        return directions

    def apply(
        self,
        directions,
        *,
        projected_grad: float,
        learning_rate: float,
        weight_decay: float,
        step: int,
    ):
        self.events.append("apply")
        self.last_projected_grad = float(projected_grad)
        self.applied_directions.append(directions)
        return {"applied_step": int(step)}

    def fold(self, *, step: int):
        self.events.append("fold")
        return 0.25


class FakeEngine:
    def __init__(self) -> None:
        self.score_calls = 0

    def score_plus_minus_directions(self, *args, **kwargs):
        self.score_calls += 1
        token_result = TokenScoreResult(
            loss=1.0,
            nll_sum=1.0,
            num_tokens=1,
            request_nll=[1.0],
            request_num_tokens=[1],
            detail={},
            raw={},
        )
        return PlusMinusScoreResult(
            loss_plus=1.2,
            loss_minus=0.8,
            projected_grad=200.0,
            plus=token_result,
            minus=token_result,
            update_info={"scored": True},
        )


class FakeOneSidedEngine:
    plus_id = 11
    minus_id = 12
    score_probe_slots_with_scorer = ZOVLLMEngine.score_probe_slots_with_scorer

    def __init__(self) -> None:
        self.slot_writes = []
        self.score_lora_ids = []

    def set_slot_direction(self, lora_id, directions, *, eps, sign=1.0):
        self.slot_writes.append((int(lora_id), float(eps), float(sign)))
        return {"slot": {"lora_id": int(lora_id)}}

    @property
    def probe_slot_capacity(self):
        return 2

    def set_probe_directions(self, directions, *, eps, sign=1.0):
        ids = (self.plus_id, self.minus_id)[: len(directions)]
        infos = [
            self.set_slot_direction(slot_id, direction, eps=eps, sign=sign)
            for slot_id, direction in zip(ids, directions)
        ]
        return RuntimeProbeSlots(ids), infos

    def score_probe_slots(
        self,
        slots,
        token_id_groups,
        *,
        loss_token_lens=None,
        labels=None,
        max_logits_tokens=8192,
        loss_impl="logprobs",
    ):
        count = len(slots._lora_ids)
        rows = list(token_id_groups)
        return self.score_token_groups(
            rows * count,
            loss_token_lens=(
                None if loss_token_lens is None else list(loss_token_lens) * count
            ),
            labels=None if labels is None else list(labels) * count,
            lora_ids=[slot_id for slot_id in slots._lora_ids for _ in range(len(rows))],
            max_logits_tokens=max_logits_tokens,
            loss_impl=loss_impl,
        )

    def score_token_groups(
        self,
        token_id_groups,
        *,
        loss_token_lens=None,
        labels=None,
        lora_ids=None,
        max_logits_tokens=8192,
        loss_impl="logprobs",
    ):
        del loss_token_lens, labels, max_logits_tokens, loss_impl
        if lora_ids is None:
            request_nll = [1.0 for _ in token_id_groups]
        else:
            self.score_lora_ids.extend(int(item) for item in lora_ids)
            request_nll = [1.2 + 0.2 * i for i, _ in enumerate(token_id_groups)]
        return TokenScoreResult(
            loss=sum(request_nll) / len(request_nll),
            nll_sum=sum(request_nll),
            num_tokens=len(request_nll),
            request_nll=request_nll,
            request_num_tokens=[1 for _ in request_nll],
            detail={},
            raw={},
        )


class FakeLossBackedBaseEngine:
    plus_id = 11
    minus_id = 12

    def __init__(self) -> None:
        self.slot_writes = []
        self.plus_minus_writes = []
        self._call_lock = threading.RLock()

    score_plus_minus_directions_with_scorer = (
        ZOVLLMEngine.score_plus_minus_directions_with_scorer
    )
    score_probe_slots_with_scorer = ZOVLLMEngine.score_probe_slots_with_scorer

    def _ensure_open(self) -> None:
        return None

    _validate_eps = staticmethod(ZOVLLMEngine._validate_eps)
    _validate_score_inputs = staticmethod(ZOVLLMEngine._validate_score_inputs)

    def set_plus_minus_directions(self, directions, *, eps, step=0):
        del directions
        self.plus_minus_writes.append((float(eps), int(step)))
        return {"plus_minus": True}

    def set_slot_direction(self, lora_id, directions, *, eps, sign=1.0):
        del directions
        self.slot_writes.append((int(lora_id), float(eps), float(sign)))
        return {"slot": {"lora_id": int(lora_id)}}

    @property
    def probe_slot_capacity(self):
        return 2

    def set_probe_directions(self, directions, *, eps, sign=1.0):
        ids = (self.plus_id, self.minus_id)[: len(directions)]
        infos = [
            self.set_slot_direction(slot_id, direction, eps=eps, sign=sign)
            for slot_id, direction in zip(ids, directions)
        ]
        return RuntimeProbeSlots(ids), infos


def _token_score(request_nll: list[float]) -> TokenScoreResult:
    return TokenScoreResult(
        loss=sum(request_nll) / len(request_nll),
        nll_sum=sum(request_nll),
        num_tokens=len(request_nll),
        request_nll=request_nll,
        request_num_tokens=[1] * len(request_nll),
        detail={},
        raw={
            "loss": sum(request_nll) / len(request_nll),
            "nll_sum": sum(request_nll),
            "num_tokens": len(request_nll),
            "request_nll": request_nll,
            "request_num_tokens": [1] * len(request_nll),
        },
    )


class FakeRolloutEngine(FakeOneSidedEngine):
    llm = None


class FakeEstimator:
    def __init__(self) -> None:
        self.config = None

    def estimate(
        self, *, engine, batch, direction_sampler, config, step: int, scorer=None
    ):
        del scorer
        self.config = config
        direction = direction_sampler.sample()
        token_result = TokenScoreResult(
            loss=2.0,
            nll_sum=2.0,
            num_tokens=1,
            request_nll=[2.0],
            request_num_tokens=[1],
            detail={},
            raw={},
        )
        score = PlusMinusScoreResult(
            loss_plus=2.5,
            loss_minus=1.5,
            projected_grad=500.0,
            plus=token_result,
            minus=token_result,
            update_info={"custom_scored": True},
        )
        return ZOEstimate(
            reported_loss=(score.loss_plus + score.loss_minus) / 2.0,
            loss_plus=score.loss_plus,
            loss_minus=score.loss_minus,
            projected_grad=score.projected_grad,
            gradient=ZOGradientEstimate(
                directions=direction.directions,
                scale=score.projected_grad,
            ),
            probe_metrics=score.update_info,
            estimator_metrics={"estimator": "fake", "query_count": 4},
        )


def test_zo_stepper_callbacks_bracket_update():
    events = []

    class RecordingCallback(ZOStepCallback):
        def on_score_end(self, *, score, control, **kwargs):
            events.append(("score_end", score.projected_grad))

        def on_update_begin(self, *, learning_rate, control, **kwargs):
            events.append(("update_begin", learning_rate))

        def on_update_end(self, *, update_info, control, **kwargs):
            events.append(("update_end", update_info["applied_step"]))

    update_state = FakeUpdateState()
    stepper = ZOStepper(
        engine=FakeEngine(),
        direction_provider=FakeDirectionProvider(),
        update_state=update_state,
        config=ZOStepConfig(),
        callbacks=[RecordingCallback()],
    )

    result = stepper.estimate(
        TokenProbeBatch(token_id_groups=[[1, 2]]), step=3
    ).apply(learning_rate=0.5, weight_decay=0.0)

    assert update_state.events == ["prepare", "apply"]
    assert events == [("score_end", 200.0), ("update_begin", 0.5), ("update_end", 3)]
    assert result.update_info == {"applied_step": 3}
    assert result.estimator_metrics == {
        "estimator": "single_direction_antithetic",
        "query_count": 2,
    }


def test_invalidated_runtime_slots_force_one_v_sync_without_resampling():
    sampled = []

    class RecordingCallback(ZOStepCallback):
        def on_direction_sampled(self, *, sample, **kwargs):
            sampled.append(sample)

    stepper = ZOStepper(
        engine=FakeEngine(),
        direction_provider=FakeDirectionProvider(),
        update_state=FakeUpdateState(),
        config=ZOStepConfig(),
        callbacks=[RecordingCallback()],
    )
    stepper.invalidate_direction_slot_state()

    stepper.estimate(TokenProbeBatch(token_id_groups=[[1, 2]]), step=1).apply(
        learning_rate=1e-6, weight_decay=0.0
    )
    stepper.estimate(TokenProbeBatch(token_id_groups=[[1, 2]]), step=2).apply(
        learning_rate=1e-6, weight_decay=0.0
    )

    assert sampled[0].directions["layer.weight"]["v_refreshed"] is True
    assert "v_refreshed" not in sampled[1].directions["layer.weight"]


def test_zo_stepper_callback_can_skip_update():
    class SkipUpdateCallback(ZOStepCallback):
        def on_update_begin(
            self,
            *,
            control: ZOStepControl,
            **kwargs,
        ):
            control.should_skip_update = True
            return control

    update_state = FakeUpdateState()
    stepper = ZOStepper(
        engine=FakeEngine(),
        direction_provider=FakeDirectionProvider(),
        update_state=update_state,
        config=ZOStepConfig(),
        callbacks=[SkipUpdateCallback()],
    )

    result = stepper.estimate(
        TokenProbeBatch(token_id_groups=[[1, 2]]), step=1
    ).apply(learning_rate=1e-6, weight_decay=0.0)

    assert update_state.events == ["prepare"]
    assert result.update_info == {"skipped_by_callback": True}


def test_zo_stepper_accepts_custom_estimator():
    estimator = FakeEstimator()
    stepper = ZOStepper(
        engine=FakeEngine(),
        direction_provider=FakeDirectionProvider(),
        update_state=FakeUpdateState(),
        config=ZOStepConfig(eps=0.25, score_chunk_size=3),
        estimator=estimator,
    )

    result = stepper.estimate(
        TokenProbeBatch(token_id_groups=[[1, 2]]), step=2
    ).apply(learning_rate=1e-6, weight_decay=0.0)

    assert isinstance(estimator.config, ZOEstimatorConfig)
    assert estimator.config.eps == 0.25
    assert estimator.config.score_chunk_size == 3
    assert result.projected_grad == 500.0
    assert result.probe_metrics["custom_scored"] is True
    assert result.profile_s
    assert result.estimator_metrics == {"estimator": "fake", "query_count": 4}


def test_multi_query_one_sided_batches_queries_into_lora_slots():
    engine = FakeOneSidedEngine()
    estimator = MultiQueryZOEstimator(
        num_queries=2,
        perturbation_sides="one_sided",
        query_microbatch_size=2,
        seed=7,
    )
    stepper = ZOStepper(
        engine=engine,
        direction_provider=FakeDirectionProvider(),
        update_state=FakeUpdateState(),
        config=ZOStepConfig(eps=0.001),
        estimator=estimator,
    )

    result = stepper.estimate(
        TokenProbeBatch(token_id_groups=[[1, 2]]), step=1
    ).apply(learning_rate=0.1, weight_decay=0.0)

    assert engine.slot_writes == [(11, 0.001, 1.0), (12, 0.001, 1.0)]
    assert engine.score_lora_ids == [11, 12]
    assert result.loss_minus == 1.0
    assert result.reported_loss == 1.0
    assert result.estimator_metrics["estimator"] == "multi_query"
    assert result.estimator_metrics["perturbation_sides"] == "one_sided"
    assert result.estimator_metrics["num_queries"] == 2
    assert result.projected_grad == pytest.approx(300.0)
    assert result.update_scale == 1.0
    assert result.probe_metrics["query_projected_grads"] == pytest.approx(
        [200.0, 400.0]
    )
    assert "aggregate_directions" not in result.probe_metrics


def test_estimate_with_score_fn_reuses_one_sided_multi_query_estimator():
    engine = FakeLossBackedBaseEngine()
    estimator = MultiQueryZOEstimator(
        num_queries=2,
        perturbation_sides="one_sided",
        query_microbatch_size=2,
        seed=7,
    )
    stepper = ZOStepper(
        engine=engine,  # type: ignore[arg-type]
        direction_provider=FakeDirectionProvider(),
        update_state=FakeUpdateState(),
        config=ZOStepConfig(eps=0.001),
        estimator=estimator,
    )
    score_calls = []

    def score_fn(*, token_id_groups, labels=None, lora_ids=None, **kwargs):
        del labels, kwargs
        score_calls.append(None if lora_ids is None else [int(x) for x in lora_ids])
        if lora_ids is None:
            return _token_score([1.0 for _ in token_id_groups])
        return _token_score(
            [1.2 if int(lora_id) == engine.plus_id else 1.4 for lora_id in lora_ids]
        )

    pending = stepper.estimate_with_score_fn(
        TokenProbeBatch(token_id_groups=[[1, 2]]),
        step=1,
        score_fn=score_fn,
    )
    result = pending.apply(learning_rate=0.25, weight_decay=0.0)

    assert engine.slot_writes == [(11, 0.001, 1.0), (12, 0.001, 1.0)]
    assert score_calls == [None, [11, 12]]
    assert result.estimator_metrics["estimator"] == "multi_query"
    assert result.estimator_metrics["perturbation_sides"] == "one_sided"
    assert result.estimator_metrics["num_queries"] == 2
    assert result.learning_rate == 0.25
    assert result.probe_metrics["query_projected_grads"] == pytest.approx(
        [200.0, 400.0]
    )


def test_estimate_with_score_fn_reuses_engine_plus_minus_orchestration():
    engine = FakeLossBackedBaseEngine()
    update_state = FakeUpdateState()
    stepper = ZOStepper(
        engine=engine,  # type: ignore[arg-type]
        direction_provider=FakeDirectionProvider(),
        update_state=update_state,
        config=ZOStepConfig(eps=0.001),
    )
    score_calls = []

    def score_fn(*, token_id_groups, labels=None, lora_ids=None, **kwargs):
        del labels, kwargs
        score_calls.append(None if lora_ids is None else [int(x) for x in lora_ids])
        assert len(token_id_groups) == 2
        score = ProbeLossResult(
            group_losses=tuple(
                1.2 if int(lora_id) == engine.plus_id else 0.8 for lora_id in lora_ids
            ),
            requests_per_group=1,
            timing=ProbeTiming(
                backend_profile_s=(("total_worker", 0.5),),
                engine_call_s=0.25,
            ),
        )
        return score

    pending = stepper.estimate_with_score_fn(
        TokenProbeBatch(token_id_groups=[[1, 2]]),
        step=2,
        score_fn=score_fn,
    )
    result = pending.apply(learning_rate=0.25, weight_decay=0.0)

    assert engine.plus_minus_writes == [(0.001, 2)]
    assert score_calls == [[11, 12]]
    assert result.projected_grad == pytest.approx(200.0)
    assert result.learning_rate == 0.25
    assert result.profile_s["scorer_engine_call"] == 0.25
    assert result.profile_s["worker_total_worker"] == 0.5
    assert update_state.last_projected_grad == pytest.approx(200.0)


def test_multi_query_two_sided_scores_multiple_queries_before_one_update():
    engine = FakeEngine()
    update_state = FakeUpdateState()
    estimator = MultiQueryZOEstimator(num_queries=3, seed=7)
    stepper = ZOStepper(
        engine=engine,
        direction_provider=FakeDirectionProvider(),
        update_state=update_state,
        config=ZOStepConfig(eps=0.001),
        estimator=estimator,
    )

    result = stepper.estimate(
        TokenProbeBatch(token_id_groups=[[1, 2]]), step=1
    ).apply(learning_rate=0.1, weight_decay=0.0)

    assert engine.score_calls == 3
    assert update_state.events == ["prepare", "prepare", "prepare", "apply"]
    assert update_state.last_projected_grad == 1.0
    assert result.estimator_metrics["perturbation_sides"] == "two_sided"
    assert result.estimator_metrics["num_queries"] == 3
    assert result.estimator_metrics["query_count"] == 6
    assert result.reported_loss == pytest.approx(
        (result.loss_plus + result.loss_minus) / 2.0
    )
    assert result.projected_grad == pytest.approx(200.0)
    assert result.update_scale == 1.0
    assert "aggregate_directions" not in result.probe_metrics


def test_evolution_strategy_batches_population_and_maximizes_reward(monkeypatch):
    rewards_by_chunk = iter([([1.0, 2.0], [4.0, 5.0]), ([3.0], [6.0])])

    def fake_generate_slot_rewards(**kwargs):
        assert kwargs["prompts"] == ["p"]
        assert kwargs["targets"] == [{"answer": 1}]
        assert kwargs["max_tokens"] == 8
        rewards, lengths = next(rewards_by_chunk)
        return rewards, lengths, {"generate": 0.1, "reward_postprocess": 0.01}

    monkeypatch.setattr(
        EvolutionStrategyEstimator,
        "_generate_slot_rewards",
        staticmethod(fake_generate_slot_rewards),
    )
    engine = FakeRolloutEngine()
    update_state = FakeUpdateState()
    estimator = EvolutionStrategyEstimator(
        population_size=3,
        sigma=0.01,
        query_microbatch_size=2,
        seed=7,
    )
    stepper = ZOStepper(
        engine=engine,
        direction_provider=FakeDirectionProvider(),
        update_state=update_state,
        config=ZOStepConfig(eps=0.001),
        estimator=estimator,
    )
    batch = RolloutProbeBatch(
        rollout_prompts=["p"],
        rollout_targets=[{"answer": 1}],
        rollout_reward_fn=lambda text, target: 0.0,
        rollout_max_tokens=8,
    )

    pending = stepper.estimate(batch, step=1)
    optimizer = ZOSGDOptimizer(
        [torch.nn.Parameter(torch.zeros(()))],
        lr=0.5,
    )
    optimizer.stage(pending)
    optimizer.step()
    result = optimizer.last_step_result

    assert result is not None
    assert engine.slot_writes == [(11, 0.01, 1.0), (12, 0.01, 1.0), (11, 0.01, 1.0)]
    assert update_state.events == [
        "prepare",
        "prepare",
        "prepare",
        "apply",
        "fold",
        "apply",
        "fold",
        "apply",
        "fold",
    ]
    assert update_state.last_projected_grad == -1.0
    assert result.loss_plus == pytest.approx(-2.0)
    assert result.reported_loss == pytest.approx(-2.0)
    assert result.projected_grad is None
    assert result.update_scale == -1.0
    assert result.estimator_metrics["estimator"] == "evolution_strategy"
    assert result.estimator_metrics["query_count"] == 3
    assert result.estimator_metrics["sigma"] == pytest.approx(0.01)
    assert result.estimator_metrics["reward_mean"] == pytest.approx(2.0)
    assert result.probe_metrics["query_rewards"] == pytest.approx([1.0, 2.0, 3.0])
    assert result.probe_metrics["query_coeffs"] == pytest.approx(
        [-1.2247448, 0.0, 1.2247448],
        abs=1e-6,
    )
    assert result.metrics()["profile_generate_s"] == pytest.approx(0.2)
    assert result.metrics()["profile_reward_postprocess_s"] == pytest.approx(0.02)
    assert result.metrics()["perturbation_sigma_effective_rms"] == pytest.approx(0.01)
    assert "aggregate_directions" not in result.probe_metrics


def test_evolution_strategy_independent_direction_mode_updates_each_query(monkeypatch):
    rewards_by_chunk = iter([([1.0, 2.0], [4.0, 5.0]), ([3.0], [6.0])])

    def fake_generate_slot_rewards(**kwargs):
        rewards, lengths = next(rewards_by_chunk)
        return rewards, lengths, {"generate": 0.1, "reward_postprocess": 0.01}

    monkeypatch.setattr(
        EvolutionStrategyEstimator,
        "_generate_slot_rewards",
        staticmethod(fake_generate_slot_rewards),
    )
    update_state = FakeUpdateState()
    estimator = EvolutionStrategyEstimator(
        population_size=3,
        sigma=0.01,
        query_microbatch_size=2,
        direction_mode="independent",
        direction_specs=[
            DirectionSpec(
                name="layer.weight",
                out_features=1,
                in_features=1,
                rank=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        ],
        seed=7,
    )
    stepper = ZOStepper(
        engine=FakeRolloutEngine(),
        direction_provider=FakeDirectionProvider(),
        update_state=update_state,
        config=ZOStepConfig(eps=0.001),
        estimator=estimator,
    )
    batch = RolloutProbeBatch(
        rollout_prompts=["p"],
        rollout_targets=[{"answer": 1}],
        rollout_reward_fn=lambda text, target: 0.0,
        rollout_max_tokens=8,
    )

    result = stepper.estimate(batch, step=1).apply(
        learning_rate=0.5, weight_decay=0.0
    )

    assert update_state.events == [
        "prepare",
        "prepare",
        "prepare",
        "apply",
        "fold",
        "apply",
        "fold",
        "apply",
        "fold",
    ]
    assert len(update_state.applied_directions) == 3
    assert result.update_info["mode"] == "multi_update"
    assert result.update_info["num_updates"] == 3
    assert result.update_info["fold_s"] == pytest.approx(0.75)
    assert result.estimator_metrics["direction_mode"] == "independent"
    assert result.probe_metrics["multi_query_direction_mode"] == "independent"
    v_values = [
        float(item["layer.weight"]["V"].item())
        for item in update_state.applied_directions
    ]
    assert len(set(v_values)) == 3


def test_step_direction_sampler_does_not_retain_every_seeded_query() -> None:
    tensor_refs = []

    class RepeatedSamplingEstimator:
        perturbation_normalization = "none"

        def estimate(
            self,
            *,
            direction_sampler,
            **kwargs,
        ):
            del kwargs
            for seed in (1, 2, 3):
                bundle = direction_sampler.sample(seed=seed)
                tensor_refs.append(
                    weakref.ref(bundle.directions["layer.weight"]["U"])
                )
                del bundle
            return ZOEstimate(
                reported_loss=0.0,
                loss_plus=0.0,
                loss_minus=0.0,
                projected_grad=0.0,
                gradient=ZOGradientEstimate(directions={}, scale=0.0),
            )

    stepper = ZOStepper(
        engine=FakeEngine(),
        direction_provider=FakeDirectionProvider(),
        update_state=FakeUpdateState(),
        config=ZOStepConfig(),
        estimator=RepeatedSamplingEstimator(),
    )

    pending = stepper.estimate(TokenProbeBatch(token_id_groups=[[1, 2]]), step=1)
    gc.collect()

    assert pending.step == 1
    assert tensor_refs[0]() is not None
    assert tensor_refs[1]() is None
    assert tensor_refs[2]() is None
