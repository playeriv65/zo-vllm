#!/usr/bin/env python3
"""Trace impacted benchmark requests back to mixed ZO scheduler batches."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _metric_list(data: dict[str, Any], *names: str) -> list[float]:
    for name in names:
        value = data.get(name)
        if isinstance(value, list):
            return [float(item) for item in value]
    return []


def _request_ids(data: dict[str, Any]) -> list[str]:
    request_ids = data.get("request_ids")
    if isinstance(request_ids, list):
        return [str(item) for item in request_ids]
    outputs = data.get("outputs")
    if isinstance(outputs, list):
        ids = []
        for output in outputs:
            if isinstance(output, dict):
                req_id = output.get("request_id") or output.get("id")
                if req_id is not None:
                    ids.append(str(req_id))
        if ids:
            return ids
    return []


def _bench_index_from_req_id(req_id: str) -> int | None:
    match = re.search(r"(?:^|-)bench-[^-]+-(\d+)-0-", req_id)
    if match:
        return int(match.group(1))
    return None


def _request_ids_from_trace(
    trace_rows: list[dict[str, Any]], num_requests: int
) -> list[str]:
    by_index: dict[int, str] = {}
    for row in trace_rows:
        if row.get("event") != "scheduler_batch":
            continue
        for request in row.get("requests") or []:
            req_id = str(request.get("req_id", ""))
            if request.get("kind") == "zo":
                continue
            index = _bench_index_from_req_id(req_id)
            if index is not None and index not in by_index:
                by_index[index] = req_id
    return [
        by_index.get(index, f"bench-index-{index}") for index in range(num_requests)
    ]


def _per_request_mean_itl(data: dict[str, Any]) -> list[float]:
    values = data.get("itls")
    if not isinstance(values, list):
        return _metric_list(data, "tpots", "tpot_ms")
    result = []
    for item in values:
        if isinstance(item, list) and item:
            result.append(sum(float(v) for v in item) / len(item))
        else:
            result.append(0.0)
    return result


def _per_request_max_itl(data: dict[str, Any]) -> list[float]:
    values = data.get("itls")
    if not isinstance(values, list):
        return _metric_list(data, "max_itls", "max_itl_ms")
    result = []
    for item in values:
        if isinstance(item, list) and item:
            result.append(max(float(v) for v in item))
        else:
            result.append(0.0)
    return result


def _zo_tag(req_id: str) -> str:
    parts = req_id.split("-")
    if len(parts) >= 4 and req_id.startswith("serving-zo-"):
        return "-".join(parts[:4])
    return req_id


def _kind_counts(requests: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for request in requests:
        kind = str(request.get("kind", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _compact_request(request: dict[str, Any]) -> str:
    req_id = str(request.get("req_id", ""))
    if len(req_id) > 48:
        req_id = req_id[:48]
    return (
        f"{req_id}[{request.get('kind')},tok={request.get('num_scheduled_tokens')},"
        f"phase={request.get('phase')},lora={request.get('lora_id')}]"
    )


def _build_request_batch_index(
    trace_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        if row.get("event") != "scheduler_batch":
            continue
        requests = list(row.get("requests") or [])
        for request in requests:
            req_id = request.get("req_id")
            if req_id is not None:
                index[str(req_id)].append(row)
    return index


def _summarize_request_batches(
    req_id: str,
    index: dict[str, list[dict[str, Any]]],
    *,
    max_batches: int,
) -> dict[str, Any]:
    batches = index.get(req_id, [])
    mixed_batches = []
    zo_token_total = 0
    zo_tags: Counter[str] = Counter()
    for batch in batches:
        requests = list(batch.get("requests") or [])
        counts = _kind_counts(requests)
        fg_count = sum(value for key, value in counts.items() if key != "zo")
        zo_count = counts.get("zo", 0)
        if not (fg_count and zo_count):
            continue
        batch_zo_tokens = 0
        for request in requests:
            if request.get("kind") != "zo":
                continue
            tokens = int(request.get("num_scheduled_tokens") or 0)
            batch_zo_tokens += tokens
            zo_tags[_zo_tag(str(request.get("req_id", "")))] += tokens
        zo_token_total += batch_zo_tokens
        mixed_batches.append(
            {
                "batch_id": batch.get("batch_id"),
                "total_num_scheduled_tokens": batch.get("total_num_scheduled_tokens"),
                "foreground_requests": fg_count,
                "zo_requests": zo_count,
                "zo_tokens": batch_zo_tokens,
                "first_requests": [
                    _compact_request(request) for request in requests[:8]
                ],
            }
        )
    mixed_batches.sort(key=lambda item: int(item["zo_tokens"]), reverse=True)
    return {
        "num_scheduler_batches": len(batches),
        "num_mixed_batches": len(mixed_batches),
        "num_foreground_only_batches": len(batches) - len(mixed_batches),
        "zo_tokens": zo_token_total,
        "zo_tags_by_tokens": dict(zo_tags.most_common(10)),
        "worst_mixed_batches": mixed_batches[:max_batches],
    }


def analyze(
    *,
    base: Path,
    candidate: Path,
    trace: Path,
    out_json: Path,
    out_md: Path,
    top_n: int,
    max_batches: int,
) -> dict[str, Any]:
    base_data = _load_json(base)
    candidate_data = _load_json(candidate)
    trace_rows = _load_jsonl(trace)
    request_index = _build_request_batch_index(trace_rows)

    base_ids = _request_ids(base_data)
    candidate_ids = _request_ids(candidate_data)
    base_ttft = _metric_list(base_data, "ttfts", "ttft_ms")
    cand_ttft = _metric_list(candidate_data, "ttfts", "ttft_ms")
    if not base_ids:
        base_ids = [f"bench-index-{index}" for index in range(len(base_ttft))]
    if not candidate_ids:
        candidate_ids = _request_ids_from_trace(trace_rows, len(cand_ttft))
    base_tpot = _per_request_mean_itl(base_data)
    cand_tpot = _per_request_mean_itl(candidate_data)
    base_max_itl = _per_request_max_itl(base_data)
    cand_max_itl = _per_request_max_itl(candidate_data)
    n = min(len(base_ids), len(candidate_ids))
    n = min(n, len(base_ttft), len(cand_ttft))

    rows = []
    for idx in range(n):
        req_id = candidate_ids[idx]
        impact = _summarize_request_batches(
            req_id, request_index, max_batches=max_batches
        )
        delta_ttft_ms = (cand_ttft[idx] - base_ttft[idx]) * 1000.0
        delta_tpot_ms = None
        if idx < len(base_tpot) and idx < len(cand_tpot):
            delta_tpot_ms = (cand_tpot[idx] - base_tpot[idx]) * 1000.0
        delta_max_itl_ms = None
        if idx < len(base_max_itl) and idx < len(cand_max_itl):
            delta_max_itl_ms = (cand_max_itl[idx] - base_max_itl[idx]) * 1000.0
        rows.append(
            {
                "idx": idx,
                "base_request_id": base_ids[idx],
                "candidate_request_id": req_id,
                "delta_ttft_ms": delta_ttft_ms,
                "delta_tpot_ms": delta_tpot_ms,
                "delta_max_itl_ms": delta_max_itl_ms,
                "base_ttft_ms": base_ttft[idx] * 1000.0,
                "candidate_ttft_ms": cand_ttft[idx] * 1000.0,
                **impact,
            }
        )

    def top_by(field: str) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                row[field] if isinstance(row.get(field), (int, float)) else -1e30
            ),
            reverse=True,
        )[:top_n]

    summary = {
        "base_path": str(base),
        "candidate_path": str(candidate),
        "trace_path": str(trace),
        "num_matched_requests": n,
        "top_delta_ttft": top_by("delta_ttft_ms"),
        "top_delta_tpot": top_by("delta_tpot_ms"),
        "top_delta_max_itl": top_by("delta_max_itl_ms"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_markdown(summary, out_md)
    return summary


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _write_table(lines: list[str], title: str, rows: list[dict[str, Any]]) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            "| idx | delta_ttft_ms | delta_tpot_ms | delta_max_itl_ms | mixed_batches | zo_tokens | main_zo_tags |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        tags = ", ".join(
            f"{key}:{value}" for key, value in row["zo_tags_by_tokens"].items()
        )
        lines.append(
            "| {idx} | {ttft} | {tpot} | {itl} | {mixed}/{total} | {tokens} | `{tags}` |".format(
                idx=row["idx"],
                ttft=_fmt(row["delta_ttft_ms"]),
                tpot=_fmt(row["delta_tpot_ms"]),
                itl=_fmt(row["delta_max_itl_ms"]),
                mixed=row["num_mixed_batches"],
                total=row["num_scheduler_batches"],
                tokens=row["zo_tokens"],
                tags=tags,
            )
        )
    lines.append("")


def _write_request_detail(lines: list[str], title: str, row: dict[str, Any]) -> None:
    lines.extend(
        [
            f"### Request {row['idx']} ({title})",
            "",
            f"- req_id: `{row['candidate_request_id']}`",
            f"- TTFT base/candidate/delta ms: `{_fmt(row['base_ttft_ms'])}` / "
            f"`{_fmt(row['candidate_ttft_ms'])}` / `{_fmt(row['delta_ttft_ms'])}`",
            f"- Scheduler batches: `{row['num_scheduler_batches']}`, "
            f"mixed: `{row['num_mixed_batches']}`, "
            f"foreground only: `{row['num_foreground_only_batches']}`",
            f"- ZO tokens in this request's batches: `{row['zo_tokens']}`",
            f"- ZO tags by tokens: `{row['zo_tags_by_tokens']}`",
            "",
            "Worst mixed batches:",
            "",
        ]
    )
    for batch in row["worst_mixed_batches"]:
        first_requests = "; ".join(batch["first_requests"])
        lines.extend(
            [
                f"- batch `{batch['batch_id']}`: "
                f"total_tokens={batch['total_num_scheduled_tokens']}, "
                f"fg={batch['foreground_requests']}, "
                f"zo={batch['zo_requests']}, "
                f"zo_tokens={batch['zo_tokens']}",
                f"  - first requests: `{first_requests}`",
            ]
        )
    lines.append("")


def _write_markdown(summary: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Request QoS Trace Impact",
        "",
        f"- Base: `{summary['base_path']}`",
        f"- Candidate: `{summary['candidate_path']}`",
        f"- Trace: `{summary['trace_path']}`",
        f"- Matched requests: `{summary['num_matched_requests']}`",
        "",
    ]
    sections = (
        ("Top Delta TTFT", "top_delta_ttft"),
        ("Top Delta TPOT", "top_delta_tpot"),
        ("Top Delta Max ITL", "top_delta_max_itl"),
    )
    for title, key in sections:
        rows = list(summary[key])
        _write_table(lines, title, rows)
        if rows:
            _write_request_detail(lines, title, rows[0])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=4)
    args = parser.parse_args()
    summary = analyze(
        base=args.base,
        candidate=args.candidate,
        trace=args.trace,
        out_json=args.out_json,
        out_md=args.out_md,
        top_n=args.top_n,
        max_batches=args.max_batches,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
