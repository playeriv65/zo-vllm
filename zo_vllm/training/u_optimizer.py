"""Optimizer over the U coefficients, shared by the update states.

The reference ZO implementations write their estimate into ``param.grad`` and
let a stock ``torch.optim`` take the step, which is where their momentum and
second-moment normalisation come from. The coefficients here are about 4e5
floats against 1.2e9 weights, so mirroring them as parameters costs nothing and
buys the real optimizers instead of a hand-written approximation of one.

Two things this module exists to get right:

* **fp32.** The bank stores U in the model dtype, which is fp16. Adam is
  unusable there: ``eps=1e-8`` underflows to exactly 0 and so does
  ``(1-beta2)*g^2`` for small ``g``, leaving a zero denominator that makes the
  first step +-inf, with 0/0 -> NaN wherever the gradient is exactly 0. That
  failure is independent of the learning rate, which is what makes it look
  like a tuning problem. The master copy is therefore fp32.

* **``adam_scalar``.** The per-step estimate is one scalar times one fixed
  direction, so a target's U entries are the coefficients of a single
  direction, not independent parameters. Element-wise Adam normalises them
  individually and turns a step along ``U`` into a step along ``sign(U)``.
  ``adam_scalar`` runs the Adam moments on the projected gradient itself and
  leaves the direction exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

U_OPTIMIZER_NAMES = ("plain", "sgd", "adam", "adamw", "adam_scalar", "zo_adamu")
U_OPTIMIZER_CODES = {name: code for code, name in enumerate(U_OPTIMIZER_NAMES)}


@dataclass
class UCoefficientOptimizer:
    """Momentum / second-moment handling for the U coefficients."""

    name: str = "sgd"
    momentum: float = 0.0
    beta2: float = 0.9
    eps: float = 1e-8
    params: dict[str, torch.nn.Parameter] = field(default_factory=dict)
    optimizer: torch.optim.Optimizer | None = None
    scalar_m: dict[str, float] = field(default_factory=dict)
    scalar_v: dict[str, float] = field(default_factory=dict)
    step_count: int = 0

    def __post_init__(self) -> None:
        if self.name not in U_OPTIMIZER_NAMES:
            raise ValueError(
                "u_optimizer must be one of " + ", ".join(U_OPTIMIZER_NAMES)
            )
        if not (0.0 <= float(self.momentum) < 1.0):
            raise ValueError("u_momentum must be in [0, 1)")
        if not (0.0 <= float(self.beta2) < 1.0):
            raise ValueError("u_beta2 must be in [0, 1)")

    @property
    def code(self) -> int:
        return U_OPTIMIZER_CODES[self.name]

    @property
    def uses_torch(self) -> bool:
        """Whether the step goes through ``torch.optim`` rather than an add."""

        return self.name in {"sgd", "adam", "adamw"}

    def begin_step(self) -> None:
        """Advance the bias-correction counter once per optimizer step."""

        if self.name in {"adam", "adam_scalar"}:
            self.step_count += 1

    def scalar_rate(self, name: str, g_hat: float) -> float:
        """Adam on the scalar coefficient, leaving the direction untouched."""

        b1, b2 = float(self.momentum), float(self.beta2)
        m = self.scalar_m.get(name, 0.0) * b1 + (1.0 - b1) * g_hat
        v = self.scalar_v.get(name, 0.0) * b2 + (1.0 - b2) * g_hat * g_hat
        self.scalar_m[name] = m
        self.scalar_v[name] = v
        t = max(1, int(self.step_count))
        m_hat = m / (1.0 - b1**t)
        v_hat = v / (1.0 - b2**t)
        return m_hat / (math.sqrt(v_hat) + float(self.eps))

    def _ensure(self, targets: dict[str, torch.Tensor]) -> None:
        fresh = False
        for key, target in targets.items():
            param = self.params.get(key)
            if param is None or param.shape != target.shape:
                self.params[key] = torch.nn.Parameter(
                    target.detach().to(torch.float32).clone(), requires_grad=True
                )
                fresh = True
        if self.optimizer is None or fresh:
            params = list(self.params.values())
            betas = (float(self.momentum), float(self.beta2))
            if self.name == "adam":
                self.optimizer = torch.optim.Adam(
                    params, lr=1.0, betas=betas, eps=float(self.eps)
                )
            elif self.name == "adamw":
                self.optimizer = torch.optim.AdamW(
                    params, lr=1.0, betas=betas, eps=float(self.eps), weight_decay=0.0
                )
            else:
                self.optimizer = torch.optim.SGD(
                    params, lr=1.0, momentum=float(self.momentum)
                )

    def step(
        self,
        grads: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        learning_rate: float,
    ) -> None:
        """Apply one optimizer step, writing the result back into ``targets``."""

        self._ensure(targets)
        assert self.optimizer is not None
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)
        for key, target in targets.items():
            # Re-seed the master from the target every step. The target is the
            # source of truth: the update state folds it into the weights and
            # zeroes it whenever the direction basis changes (every step under
            # random_full), and a master that outlived that fold re-applied the
            # already-folded U on top of the new basis each step, which is
            # lr-independent divergence. The step itself still runs in fp32,
            # which is all Adam needs to avoid the fp16 eps underflow.
            param = self.params[key]
            param.data.copy_(target.to(torch.float32))
            param.grad = grads[key].to(torch.float32)
        self.optimizer.step()
        for key, target in targets.items():
            updated = self.params[key].data
            if not bool(torch.isfinite(updated).all()):
                # Fail here rather than at the next forward pass, where the
                # only symptom is "objective returned non-finite values" and
                # the offending target is already lost.
                raise RuntimeError(
                    f"{self.name} produced non-finite u for {key}: "
                    f"lr={float(learning_rate):.3e} "
                    f"grad_absmax={float(grads[key].abs().max()):.3e} "
                    f"u_absmax_before={float(target.abs().max()):.3e}"
                )
            target.copy_(updated.to(target.dtype))
