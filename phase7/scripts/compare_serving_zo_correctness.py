#!/usr/bin/env python3
"""Compare serving-time ZO metrics against a pure-training reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TRAIN_FIELDS = (
    "loss_plus",
    "loss_minus",
    "projected_grad",
)
EVAL_FIELDS = (
    "eval_loss",
    "eval_accuracy",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _last_event(rows: list[dict[str, Any]], event: str) -> dict[str, Any] | None:
    for row in reversed(rows):
        if row.get("event") == event:
            return row
    return None


def _by_step(rows: list[dict[str, Any]], event: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("event") != event:
            continue
        step = row.get("step")
        if isinstance(step, int):
            result[step] = row
    return result


def _compare_last(
    reference: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in fields:
        ref_value = _finite_number(reference.get(field) if reference else None)
        cand_value = _finite_number(candidate.get(field) if candidate else None)
        if ref_value is None or cand_value is None:
            out[field] = {
                "reference": ref_value,
                "candidate": cand_value,
                "abs_diff": None,
                "rel_diff": None,
            }
            continue
        abs_diff = cand_value - ref_value
        denom = abs(ref_value) if ref_value != 0 else 1.0
        out[field] = {
            "reference": ref_value,
            "candidate": cand_value,
            "abs_diff": abs_diff,
            "rel_diff": abs_diff / denom,
        }
    return out


def _compare_by_step(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_steps = _by_step(reference_rows, "train_step")
    candidate_steps = _by_step(candidate_rows, "train_step")
    common_steps = sorted(set(reference_steps).intersection(candidate_steps))
    summary: dict[str, Any] = {
        "common_steps": len(common_steps),
        "first_common_step": common_steps[0] if common_steps else None,
        "last_common_step": common_steps[-1] if common_steps else None,
        "fields": {},
    }
    for field in TRAIN_FIELDS:
        diffs: list[float] = []
        max_entry: dict[str, Any] | None = None
        for step in common_steps:
            ref_value = _finite_number(reference_steps[step].get(field))
            cand_value = _finite_number(candidate_steps[step].get(field))
            if ref_value is None or cand_value is None:
                continue
            diff = cand_value - ref_value
            diffs.append(diff)
            if max_entry is None or abs(diff) > abs(max_entry["abs_diff"]):
                max_entry = {
                    "step": step,
                    "reference": ref_value,
                    "candidate": cand_value,
                    "abs_diff": diff,
                }
        summary["fields"][field] = {
            "count": len(diffs),
            "max_abs_diff": max((abs(v) for v in diffs), default=None),
            "mean_abs_diff": (
                sum(abs(v) for v in diffs) / len(diffs) if diffs else None
            ),
            "max_entry": max_entry,
        }
    return summary


def _status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if row.get("event") == "error"]
    stopped = _last_event(rows, "stopped")
    last_train = _last_event(rows, "train_step")
    last_eval = _last_event(rows, "eval")
    return {
        "num_rows": len(rows),
        "num_errors": len(errors),
        "last_error": errors[-1].get("error") if errors else None,
        "stopped_step": stopped.get("step") if stopped else None,
        "last_train_step": last_train.get("step") if last_train else None,
        "last_eval_step": last_eval.get("step") if last_eval else None,
    }


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Serving ZO Correctness Comparison",
        "",
        f"- Reference: `{summary['reference_path']}`",
        f"- Candidate: `{summary['candidate_path']}`",
        f"- Pass: `{summary['pass']}`",
        "",
        "## Status",
        "",
        f"- Reference: `{summary['reference_status']}`",
        f"- Candidate: `{summary['candidate_status']}`",
        "",
        "## Final Eval",
        "",
        "| metric | reference | candidate | abs diff | rel diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for field, data in summary["final_eval"].items():
        lines.append(
            "| {field} | {ref} | {cand} | {abs_diff} | {rel_diff} |".format(
                field=field,
                ref=data["reference"],
                cand=data["candidate"],
                abs_diff=data["abs_diff"],
                rel_diff=data["rel_diff"],
            )
        )
    lines.extend(
        [
            "",
            "## Final Train Step",
            "",
            "| metric | reference | candidate | abs diff | rel diff |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for field, data in summary["final_train"].items():
        lines.append(
            "| {field} | {ref} | {cand} | {abs_diff} | {rel_diff} |".format(
                field=field,
                ref=data["reference"],
                cand=data["candidate"],
                abs_diff=data["abs_diff"],
                rel_diff=data["rel_diff"],
            )
        )
    lines.extend(
        [
            "",
            "## Matched Steps",
            "",
            f"- Common steps: `{summary['stepwise']['common_steps']}`",
            f"- Last common step: `{summary['stepwise']['last_common_step']}`",
            "",
            "| metric | count | max abs diff | mean abs diff | max entry |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for field, data in summary["stepwise"]["fields"].items():
        lines.append(
            "| {field} | {count} | {max_abs} | {mean_abs} | `{entry}` |".format(
                field=field,
                count=data["count"],
                max_abs=data["max_abs_diff"],
                mean_abs=data["mean_abs_diff"],
                entry=data["max_entry"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--eval-loss-atol", type=float, default=5e-3)
    parser.add_argument("--eval-accuracy-atol", type=float, default=1e-6)
    args = parser.parse_args()

    reference_rows = _load_jsonl(args.reference)
    candidate_rows = _load_jsonl(args.candidate)
    final_eval = _compare_last(
        _last_event(reference_rows, "eval"),
        _last_event(candidate_rows, "eval"),
        EVAL_FIELDS,
    )
    final_train = _compare_last(
        _last_event(reference_rows, "train_step"),
        _last_event(candidate_rows, "train_step"),
        TRAIN_FIELDS,
    )
    stepwise = _compare_by_step(reference_rows, candidate_rows)
    ref_status = _status(reference_rows)
    cand_status = _status(candidate_rows)

    loss_diff = final_eval["eval_loss"]["abs_diff"]
    acc_diff = final_eval["eval_accuracy"]["abs_diff"]
    passed = (
        cand_status["num_errors"] == 0
        and ref_status["num_errors"] == 0
        and loss_diff is not None
        and abs(loss_diff) <= args.eval_loss_atol
        and acc_diff is not None
        and abs(acc_diff) <= args.eval_accuracy_atol
    )

    summary = {
        "reference_path": str(args.reference),
        "candidate_path": str(args.candidate),
        "pass": passed,
        "thresholds": {
            "eval_loss_atol": args.eval_loss_atol,
            "eval_accuracy_atol": args.eval_accuracy_atol,
        },
        "reference_status": ref_status,
        "candidate_status": cand_status,
        "final_eval": final_eval,
        "final_train": final_train,
        "stepwise": stepwise,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, args.out_md)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
