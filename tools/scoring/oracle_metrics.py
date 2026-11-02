#!/usr/bin/env python3
"""Score the dual-model Scale-2 oracle-state control with paired CIs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.scoring.structured_cot import (
    bootstrap_metrics,
    canonical_label,
    ci,
    distributions_from_sums,
    metrics_from_values,
    parse_prediction,
    read_jsonl,
    stable_seed,
    write_csv,
    write_jsonl,
)


CONDITIONS = ("pixel_only", "oracle_state_only", "pixel_plus_oracle_state")
METRICS = ("Acc_full", "Acc_nocue", "Acc_cf", "p_active", "IDR_raw", "IDR_cond")
VALUE_COLUMNS = ("full_success", "nocue_success", "cf_fail", "active", "idr")


def score_model(
    model: str,
    response_path: Path,
    requests: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    responses = read_jsonl(response_path)
    response_map = {str(row["uid"]): row for row in responses}
    missing = sorted(set(requests) - set(response_map))
    extra = sorted(set(response_map) - set(requests))
    rows = []
    for uid, request in requests.items():
        response = response_map.get(uid, {})
        exam = request["exam"]
        api_ok = bool(response.get("meta", {}).get("ok", False))
        raw = str(response.get("raw") or response.get("pred") or "")
        parsed = parse_prediction(raw, "direct_original", api_ok)
        gold = canonical_label(exam.get("answer") or exam.get("label"))
        rows.append(
            {
                "uid": uid,
                "source_uid": exam["source_uid"],
                "model": model,
                "env_task": exam["env_task"],
                "group_id": exam["group_id"],
                "variant": exam["variant"],
                "evidence_condition": exam["oracle_control_condition"],
                "gold": gold,
                "pred_norm": parsed["pred_norm"],
                "correct": parsed["pred_norm"] == gold,
                "parse_mode": parsed["parse_mode"],
                "strict_schema": parsed["strict_schema"],
                "raw": raw,
                "api_ok": api_ok,
                "api_error": response.get("meta", {}).get("error"),
                "latency_ms": response.get("meta", {}).get("latency_ms"),
                "finish_reason": response.get("meta", {}).get("finish_reason"),
            }
        )
    return rows, {
        "model": model,
        "requests": len(requests),
        "responses": len(responses),
        "missing": len(missing),
        "extra": len(extra),
        "duplicates": len(responses) - len(response_map),
        "api_errors": sum(row["parse_mode"] == "api_error" for row in rows),
        "unparsable": sum(row["parse_mode"] == "unparsable" for row in rows),
        "missing_uids": missing[:20],
        "extra_uids": extra[:20],
    }


def group_records(
    rows: list[dict[str, Any]],
    condition: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["evidence_condition"] != condition:
            continue
        key = (str(row["env_task"]), str(row["group_id"]))
        variant = str(row["variant"])
        if variant in grouped[key]:
            raise ValueError(f"duplicate {condition}/{key}/{variant}")
        grouped[key][variant] = row

    output = []
    for (environment, group_id), variants in sorted(grouped.items()):
        if set(variants) != {"full", "nocue", "cf"}:
            raise ValueError(f"incomplete {condition}/{environment}/{group_id}")
        full_success = variants["full"]["pred_norm"] == "Success"
        output.append(
            {
                "env_task": environment,
                "group_id": group_id,
                "full_success": float(full_success),
                "nocue_success": float(variants["nocue"]["pred_norm"] == "Success"),
                "cf_fail": float(variants["cf"]["pred_norm"] == "Fail"),
                "active": float(full_success),
                "idr": float(full_success and variants["cf"]["pred_norm"] == "Fail"),
            }
        )
    return output


def metric_rows(
    rows: list[dict[str, Any]],
    model: str,
    iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, list[str]]]]:
    output = []
    matrices: dict[str, tuple[np.ndarray, list[str]]] = {}
    for condition in CONDITIONS:
        records = group_records(rows, condition)
        values = np.asarray(
            [[record[column] for column in VALUE_COLUMNS] for record in records]
        )
        environments = [str(record["env_task"]) for record in records]
        matrices[condition] = (values, environments)
        point = metrics_from_values(values)
        group_dist, env_dist = bootstrap_metrics(
            values,
            environments,
            iterations,
            stable_seed(seed, model, condition),
        )
        for metric in METRICS:
            group_low, group_high, group_valid = ci(group_dist[metric])
            env_low, env_high, env_valid = ci(env_dist[metric])
            output.append(
                {
                    "model": model,
                    "evidence_condition": condition,
                    "n_groups": len(records),
                    "n_environments": len(set(environments)),
                    "metric": metric,
                    "estimate": point[metric],
                    "group_ci_low": group_low,
                    "group_ci_high": group_high,
                    "group_valid_replicates": group_valid,
                    "environment_ci_low": env_low,
                    "environment_ci_high": env_high,
                    "environment_valid_replicates": env_valid,
                    "bootstrap_iterations": iterations,
                }
            )
    return output, matrices


def delta_rows(
    matrices: dict[str, tuple[np.ndarray, list[str]]],
    model: str,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    baseline, environments = matrices["pixel_only"]
    for condition in ("oracle_state_only", "pixel_plus_oracle_state"):
        treatment, treatment_environments = matrices[condition]
        if environments != treatment_environments or len(baseline) != len(treatment):
            raise ValueError(f"unpaired conditions: {model}/{condition}")
        base_point = metrics_from_values(baseline)
        treatment_point = metrics_from_values(treatment)
        rng = np.random.default_rng(stable_seed(seed, model, condition, "group_delta"))
        sampled = rng.integers(0, len(baseline), size=(iterations, len(baseline)))
        base_group = distributions_from_sums(
            baseline[sampled].sum(axis=1), np.full(iterations, len(baseline))
        )
        treatment_group = distributions_from_sums(
            treatment[sampled].sum(axis=1), np.full(iterations, len(treatment))
        )

        names = sorted(set(environments))
        env_array = np.asarray(environments)
        base_env_sums = np.asarray(
            [baseline[np.flatnonzero(env_array == name)].sum(axis=0) for name in names]
        )
        treatment_env_sums = np.asarray(
            [treatment[np.flatnonzero(env_array == name)].sum(axis=0) for name in names]
        )
        env_sizes = np.asarray(
            [np.count_nonzero(env_array == name) for name in names]
        )
        env_rng = np.random.default_rng(
            stable_seed(seed, model, condition, "environment_delta")
        )
        sampled_envs = env_rng.integers(0, len(names), size=(iterations, len(names)))
        denominators = env_sizes[sampled_envs].sum(axis=1)
        base_env = distributions_from_sums(
            base_env_sums[sampled_envs].sum(axis=1), denominators
        )
        treatment_env = distributions_from_sums(
            treatment_env_sums[sampled_envs].sum(axis=1), denominators
        )
        for metric in METRICS:
            group_low, group_high, group_valid = ci(
                treatment_group[metric] - base_group[metric]
            )
            env_low, env_high, env_valid = ci(
                treatment_env[metric] - base_env[metric]
            )
            baseline_estimate = base_point[metric]
            treatment_estimate = treatment_point[metric]
            output.append(
                {
                    "model": model,
                    "comparison": f"{condition}-minus-pixel_only",
                    "n_groups": len(baseline),
                    "metric": metric,
                    "baseline_estimate": baseline_estimate,
                    "treatment_estimate": treatment_estimate,
                    "delta": (
                        treatment_estimate - baseline_estimate
                        if baseline_estimate is not None and treatment_estimate is not None
                        else None
                    ),
                    "paired_group_ci_low": group_low,
                    "paired_group_ci_high": group_high,
                    "group_valid_replicates": group_valid,
                    "paired_environment_ci_low": env_low,
                    "paired_environment_ci_high": env_high,
                    "environment_valid_replicates": env_valid,
                    "bootstrap_iterations": iterations,
                }
            )
    return output


def write_report(
    path: Path,
    metrics: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    parse_rows: list[dict[str, Any]],
) -> None:
    idr = [row for row in metrics if row["metric"] == "IDR_raw"]
    idr_delta = [row for row in deltas if row["metric"] == "IDR_raw"]
    lines = [
        "# Scale-2 Oracle-State Control Results",
        "",
        "## Run audit",
        "",
        "| Model | Requests | Responses | Missing | API errors | Unparsable |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in audits:
        lines.append(
            f"| {row['model']} | {row['requests']} | {row['responses']} | "
            f"{row['missing']} | {row['api_errors']} | {row['unparsable']} |"
        )
    lines.extend(
        [
            "",
            "## Paired IDR_raw",
            "",
            "| Model | Evidence | Groups | IDR_raw | Group 95% CI | Environment 95% CI |",
            "| --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in idr:
        lines.append(
            f"| {row['model']} | `{row['evidence_condition']}` | {row['n_groups']} | "
            f"{row['estimate']:.3f} | [{row['group_ci_low']:.3f}, "
            f"{row['group_ci_high']:.3f}] | [{row['environment_ci_low']:.3f}, "
            f"{row['environment_ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "## Paired IDR_raw deltas",
            "",
            "| Model | Comparison | Delta | Group 95% CI | Environment 95% CI |",
            "| --- | --- | ---: | --- | --- |",
        ]
    )
    for row in idr_delta:
        lines.append(
            f"| {row['model']} | `{row['comparison']}` | {row['delta']:.3f} | "
            f"[{row['paired_group_ci_low']:.3f}, {row['paired_group_ci_high']:.3f}] | "
            f"[{row['paired_environment_ci_low']:.3f}, "
            f"{row['paired_environment_ci_high']:.3f}] |"
        )
    lines.extend(
        [
            "",
            "All complete groups in the supplied request manifest remain in the denominator. API errors and",
            "unparsable outputs count as incorrect.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument(
        "--model-response",
        action="append",
        required=True,
        help="MODEL=/path/to/responses.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    request_rows = read_jsonl(args.requests)
    requests = {str(row["uid"]): row for row in request_rows}
    if len(requests) != len(request_rows):
        raise ValueError("duplicate request UID")

    all_rows = []
    all_metrics = []
    all_deltas = []
    audits = []
    parse_summary = []
    for spec in args.model_response:
        model, separator, path = spec.partition("=")
        if not separator:
            raise ValueError(f"invalid --model-response: {spec}")
        rows, audit = score_model(model, Path(path), requests)
        metrics, matrices = metric_rows(
            rows, model, args.bootstrap_iterations, args.seed
        )
        deltas = delta_rows(
            matrices, model, args.bootstrap_iterations, args.seed
        )
        all_rows.extend(rows)
        all_metrics.extend(metrics)
        all_deltas.extend(deltas)
        audits.append(audit)
        for (condition,), selected in sorted(
            group_by(rows, ("evidence_condition",)).items()
        ):
            modes = Counter(str(row["parse_mode"]) for row in selected)
            parse_summary.append(
                {
                    "model": model,
                    "evidence_condition": condition,
                    "n": len(selected),
                    "strict": modes["strict"],
                    "recoverable": modes["recoverable"],
                    "unparsable": modes["unparsable"],
                    "api_error": modes["api_error"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "scored_rows.jsonl", all_rows)
    write_csv(args.output_dir / "paired_metrics_with_ci.csv", all_metrics)
    write_csv(args.output_dir / "paired_condition_deltas.csv", all_deltas)
    write_csv(args.output_dir / "parse_mode_distribution.csv", parse_summary)
    write_csv(args.output_dir / "run_audit.csv", audits)
    write_report(
        args.output_dir / "oracle_state_results.md",
        all_metrics,
        all_deltas,
        audits,
        parse_summary,
    )
    print(
        json.dumps(
            {
                "models": len(audits),
                "scored_rows": len(all_rows),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


def group_by(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    output: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[tuple(str(row[key]) for key in keys)].append(row)
    return output


if __name__ == "__main__":
    main()
