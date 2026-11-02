#!/usr/bin/env python3
"""Score structured-CoT responses and compute paired prompt-effect metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("direct_original", "structured_direct", "structured_cot_v1")
METRICS = ("Acc_full", "Acc_nocue", "Acc_cf", "p_active", "IDR_raw", "IDR_cond")
VALUE_COLUMNS = ("full_success", "nocue_success", "cf_fail", "active", "idr")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def strip_markdown_fence(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    return (match.group(1).strip(), True) if match else (stripped, False)


def canonical_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"success", "yes"}:
        return "Success"
    if normalized in {"fail", "failure", "no"}:
        return "Fail"
    return None


def strict_structured_schema(arm: str, payload: dict[str, Any]) -> bool:
    if canonical_label(payload.get("verdict")) is None:
        return False
    if arm == "structured_direct":
        return set(payload) == {"verdict"}
    if arm == "structured_cot_v1":
        if "query_evidence" in payload:
            required = {
                "reference_goal", "query_evidence",
                "failed_requirement", "verdict",
            }
            evidence_key = "query_evidence"
        else:
            required = {
                "goal", "observed_evidence",
                "unmet_requirement", "verdict",
            }
            evidence_key = "observed_evidence"
        return (
            set(payload) == required
            and isinstance(payload[evidence_key], list)
            and 1 <= len(payload[evidence_key]) <= 2
            and all(isinstance(item, str) for item in payload[evidence_key])
            and all(isinstance(payload[key], str) for key in required - {evidence_key})
        )
    return False


def recover_unique_verdict(text: str) -> str | None:
    verdict_matches = re.findall(
        r"""["']?verdict["']?\s*[:=]\s*["']?(success|fail|failure|yes|no)\b""",
        text,
        flags=re.IGNORECASE,
    )
    labels = {canonical_label(match) for match in verdict_matches}
    labels.discard(None)
    if len(labels) == 1:
        return labels.pop()

    whole = canonical_label(text.strip().strip("\"'"))
    if whole:
        return whole

    final_match = re.search(
        r"(?:final\s+(?:answer|verdict)|answer)\s*[:=]\s*(success|fail|failure|yes|no)\b",
        text,
        flags=re.IGNORECASE,
    )
    if final_match:
        return canonical_label(final_match.group(1))

    # Some models append a bare verdict after an explanatory response. Recover
    # only an isolated terminal line so labels mentioned in the rationale cannot
    # be mistaken for the final answer.
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        terminal = re.fullmatch(
            r"(?:\*\*|__)?\s*(success|fail|failure|yes|no)\s*(?:\*\*|__)?[.!]?",
            nonempty_lines[-1],
            flags=re.IGNORECASE,
        )
        if terminal:
            return canonical_label(terminal.group(1))
    return None


def parse_prediction(raw: str, arm: str, api_ok: bool) -> dict[str, Any]:
    if not api_ok:
        return {
            "pred_norm": None,
            "parse_mode": "api_error",
            "strict_schema": False,
            "markdown_fence": False,
            "parsed_json": None,
        }
    text, fenced = strip_markdown_fence(raw)
    if arm == "direct_original":
        direct = canonical_label(text)
        if direct and text in {"Success", "Fail"}:
            mode = "strict"
        elif direct:
            mode = "recoverable"
        else:
            direct = recover_unique_verdict(text)
            mode = "recoverable" if direct else "unparsable"
        return {
            "pred_norm": direct,
            "parse_mode": mode,
            "strict_schema": mode == "strict",
            "markdown_fence": fenced,
            "parsed_json": None,
        }

    payload = None
    try:
        candidate = json.loads(text)
        if isinstance(candidate, dict):
            payload = candidate
    except json.JSONDecodeError:
        pass
    if payload is not None:
        verdict = canonical_label(payload.get("verdict"))
        strict = not fenced and strict_structured_schema(arm, payload)
        if verdict:
            return {
                "pred_norm": verdict,
                "parse_mode": "strict" if strict else "recoverable",
                "strict_schema": strict,
                "markdown_fence": fenced,
                "parsed_json": payload,
            }
    verdict = recover_unique_verdict(text)
    return {
        "pred_norm": verdict,
        "parse_mode": "recoverable" if verdict else "unparsable",
        "strict_schema": False,
        "markdown_fence": fenced,
        "parsed_json": payload,
    }


def score_model(
    model: str,
    response_path: Path,
    requests: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    responses = read_jsonl(response_path)
    response_map = {str(row["uid"]): row for row in responses}
    duplicate_count = len(responses) - len(response_map)
    missing = sorted(set(requests) - set(response_map))
    extra = sorted(set(response_map) - set(requests))
    rows = []
    for uid, request in requests.items():
        response = response_map.get(uid, {})
        exam = request["exam"]
        api_ok = bool(response.get("meta", {}).get("ok", False))
        raw = str(response.get("raw") or response.get("pred") or "")
        parsed = parse_prediction(
            raw,
            str(exam["prompt_ablation_arm"]),
            api_ok,
        )
        gold = canonical_label(exam.get("answer") or exam.get("label"))
        rows.append({
            "uid": uid,
            "source_uid": exam["source_uid"],
            "model": model,
            "task": exam["task"],
            "env_task": exam["env_task"],
            "group_id": exam["group_id"],
            "variant": exam["variant"],
            "prompt_arm": exam["prompt_ablation_arm"],
            "gold": gold,
            "pred_norm": parsed["pred_norm"],
            "correct": parsed["pred_norm"] == gold,
            "parse_mode": parsed["parse_mode"],
            "strict_schema": parsed["strict_schema"],
            "markdown_fence": parsed["markdown_fence"],
            "raw": raw,
            "api_ok": api_ok,
            "api_error": response.get("meta", {}).get("error"),
            "finish_reason": response.get("meta", {}).get("finish_reason"),
        })
    audit = {
        "model": model,
        "requests": len(requests),
        "responses": len(responses),
        "missing": len(missing),
        "extra": len(extra),
        "duplicates": duplicate_count,
        "api_errors": sum(row["parse_mode"] == "api_error" for row in rows),
        "missing_uids": missing[:20],
        "extra_uids": extra[:20],
    }
    return rows, audit


def group_records(
    rows: list[dict[str, Any]],
    arm: str,
    task: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["prompt_arm"] != arm or row["task"] != task:
            continue
        key = (row["env_task"], row["group_id"])
        if row["variant"] in grouped[key]:
            raise ValueError(f"duplicate group variant: {arm}/{task}/{key}/{row['variant']}")
        grouped[key][row["variant"]] = row
    out = []
    for (environment, group_id), variants in sorted(grouped.items()):
        if set(variants) != {"full", "nocue", "cf"}:
            raise ValueError(f"incomplete group: {arm}/{task}/{environment}/{group_id}")
        predictions = {
            variant: variants[variant]["pred_norm"]
            for variant in ("full", "nocue", "cf")
        }
        full_success = predictions["full"] == "Success"
        out.append({
            "env_task": environment,
            "group_id": group_id,
            "full_success": float(full_success),
            "nocue_success": float(predictions["nocue"] == "Success"),
            "cf_fail": float(predictions["cf"] == "Fail"),
            "active": float(full_success),
            "idr": float(full_success and predictions["cf"] == "Fail"),
        })
    return out


def metrics_from_values(values: np.ndarray) -> dict[str, float | None]:
    n = len(values)
    sums = values.sum(axis=0)
    return {
        "Acc_full": float(sums[0] / n),
        "Acc_nocue": float(sums[1] / n),
        "Acc_cf": float(sums[2] / n),
        "p_active": float(sums[3] / n),
        "IDR_raw": float(sums[4] / n),
        "IDR_cond": float(sums[4] / sums[3]) if sums[3] else None,
    }


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return (base_seed + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def bootstrap_metrics(
    values: np.ndarray,
    environments: list[str],
    iterations: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(iterations, len(values)))
    group_sums = values[sampled].sum(axis=1)
    group_dist = distributions_from_sums(
        group_sums,
        np.full(iterations, len(values)),
    )

    names = sorted(set(environments))
    env_array = np.asarray(environments)
    env_sums = np.asarray([
        values[np.flatnonzero(env_array == name)].sum(axis=0)
        for name in names
    ])
    env_sizes = np.asarray([
        np.count_nonzero(env_array == name)
        for name in names
    ])
    env_rng = np.random.default_rng(seed + 1)
    env_sampled = env_rng.integers(0, len(names), size=(iterations, len(names)))
    cluster_sums = env_sums[env_sampled].sum(axis=1)
    cluster_sizes = env_sizes[env_sampled].sum(axis=1)
    env_dist = distributions_from_sums(cluster_sums, cluster_sizes)
    return group_dist, env_dist


def distributions_from_sums(
    sums: np.ndarray,
    denominators: np.ndarray,
) -> dict[str, np.ndarray]:
    dist = {
        "Acc_full": sums[:, 0] / denominators,
        "Acc_nocue": sums[:, 1] / denominators,
        "Acc_cf": sums[:, 2] / denominators,
        "p_active": sums[:, 3] / denominators,
        "IDR_raw": sums[:, 4] / denominators,
    }
    dist["IDR_cond"] = np.divide(
        sums[:, 4],
        sums[:, 3],
        out=np.full(len(sums), np.nan),
        where=sums[:, 3] > 0,
    )
    return dist


def ci(values: np.ndarray) -> tuple[float | None, float | None, int]:
    valid = values[~np.isnan(values)]
    if not len(valid):
        return None, None, 0
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high), len(valid)


def paired_metric_rows(
    rows: list[dict[str, Any]],
    model: str,
    iterations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], tuple[np.ndarray, list[str]]]]:
    output = []
    matrices = {}
    for arm in ARMS:
        for task in ("C", "D"):
            records = group_records(rows, arm, task)
            values = np.asarray([
                [record[column] for column in VALUE_COLUMNS]
                for record in records
            ])
            environments = [record["env_task"] for record in records]
            matrices[(arm, task)] = (values, environments)
            point = metrics_from_values(values)
            group_dist, env_dist = bootstrap_metrics(
                values,
                environments,
                iterations,
                stable_seed(seed, model, arm, task),
            )
            for metric in METRICS:
                group_low, group_high, group_valid = ci(group_dist[metric])
                env_low, env_high, env_valid = ci(env_dist[metric])
                output.append({
                    "model": model,
                    "prompt_arm": arm,
                    "task": task,
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
                })
    return output, matrices


def paired_arm_delta_rows(
    matrices: dict[tuple[str, str], tuple[np.ndarray, list[str]]],
    model: str,
    baseline_arm: str,
    treatment_arm: str,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    out = []
    comparison = f"{treatment_arm}-minus-{baseline_arm}"
    seed_key = (
        treatment_arm
        if baseline_arm == "direct_original"
        else comparison
    )
    for task in ("C", "D"):
        baseline, environments = matrices[(baseline_arm, task)]
        treatment, treatment_envs = matrices[(treatment_arm, task)]
        if environments != treatment_envs or len(baseline) != len(treatment):
            raise ValueError(
                f"unpaired prompt matrices: {model}/{comparison}/{task}"
            )
        point_base = metrics_from_values(baseline)
        point_treatment = metrics_from_values(treatment)
        # Reuse identical sampled indices for a paired delta.
        rng = np.random.default_rng(
            stable_seed(seed, model, seed_key, task, "paired_group")
        )
        sampled = rng.integers(
            0, len(baseline), size=(iterations, len(baseline))
        )
        base_group = distributions_from_sums(
            baseline[sampled].sum(axis=1),
            np.full(iterations, len(baseline)),
        )
        treatment_group = distributions_from_sums(
            treatment[sampled].sum(axis=1),
            np.full(iterations, len(treatment)),
        )
        names = sorted(set(environments))
        env_array = np.asarray(environments)
        base_env_sums = np.asarray([
            baseline[np.flatnonzero(env_array == name)].sum(axis=0)
            for name in names
        ])
        treatment_env_sums = np.asarray([
            treatment[np.flatnonzero(env_array == name)].sum(axis=0)
            for name in names
        ])
        env_sizes = np.asarray([
            np.count_nonzero(env_array == name)
            for name in names
        ])
        env_rng = np.random.default_rng(
            stable_seed(seed, model, seed_key, task, "paired_env")
        )
        sampled_envs = env_rng.integers(
            0, len(names), size=(iterations, len(names))
        )
        denominators = env_sizes[sampled_envs].sum(axis=1)
        base_env = distributions_from_sums(
            base_env_sums[sampled_envs].sum(axis=1),
            denominators,
        )
        treatment_env = distributions_from_sums(
            treatment_env_sums[sampled_envs].sum(axis=1),
            denominators,
        )
        for metric in METRICS:
            group_delta = treatment_group[metric] - base_group[metric]
            env_delta = treatment_env[metric] - base_env[metric]
            group_low, group_high, group_valid = ci(group_delta)
            env_low, env_high, env_valid = ci(env_delta)
            base_value = point_base[metric]
            treatment_value = point_treatment[metric]
            delta = (
                treatment_value - base_value
                if base_value is not None and treatment_value is not None
                else None
            )
            out.append({
                "model": model,
                "comparison": comparison,
                "task": task,
                "n_groups": len(baseline),
                "metric": metric,
                "baseline_estimate": base_value,
                "treatment_estimate": treatment_value,
                "delta": delta,
                "paired_group_ci_low": group_low,
                "paired_group_ci_high": group_high,
                "group_valid_replicates": group_valid,
                "paired_environment_ci_low": env_low,
                "paired_environment_ci_high": env_high,
                "environment_valid_replicates": env_valid,
                "bootstrap_iterations": iterations,
            })
    return out


def prompt_delta_rows(
    matrices: dict[tuple[str, str], tuple[np.ndarray, list[str]]],
    model: str,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    out = []
    for treatment_arm in ("structured_direct", "structured_cot_v1"):
        out.extend(
            paired_arm_delta_rows(
                matrices,
                model,
                "direct_original",
                treatment_arm,
                iterations,
                seed,
            )
        )
    return out


def rationale_increment_rows(
    matrices: dict[tuple[str, str], tuple[np.ndarray, list[str]]],
    model: str,
    iterations: int,
    seed: int,
) -> list[dict[str, Any]]:
    return paired_arm_delta_rows(
        matrices,
        model,
        "structured_direct",
        "structured_cot_v1",
        iterations,
        seed,
    )


def summarize(
    all_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    rationale_rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    parse_rows = []
    accuracy_rows = []
    for (model, arm, task), selected in sorted(
        group_by(all_rows, ("model", "prompt_arm", "task")).items()
    ):
        modes = Counter(row["parse_mode"] for row in selected)
        parse_rows.append({
            "model": model,
            "prompt_arm": arm,
            "task": task,
            "n": len(selected),
            "strict": modes["strict"],
            "recoverable": modes["recoverable"],
            "unparsable": modes["unparsable"],
            "api_error": modes["api_error"],
            "strict_rate": modes["strict"] / len(selected),
        })
        accuracy_rows.append({
            "model": model,
            "prompt_arm": arm,
            "task": task,
            "n": len(selected),
            "accuracy": sum(row["correct"] for row in selected) / len(selected),
        })
    write_csv(output_dir / "parse_mode_distribution.csv", parse_rows)
    write_csv(output_dir / "task_accuracy.csv", accuracy_rows)
    write_csv(output_dir / "paired_metrics_with_ci.csv", paired_rows)
    write_csv(output_dir / "paired_prompt_deltas.csv", delta_rows)
    write_csv(
        output_dir / "paired_rationale_increment.csv",
        rationale_rows,
    )
    write_csv(output_dir / "run_audit.csv", audits)

    idr = {
        (row["model"], row["prompt_arm"], row["task"]): row
        for row in paired_rows if row["metric"] == "IDR_raw"
    }
    lines = [
        "# Structured-CoT Prompt Ablation Results",
        "",
        "## Run Audit",
        "",
        "| Model | Requests | Responses | Missing | Extra | Duplicates | API errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in audits:
        lines.append(
            f"| {row['model']} | {row['requests']} | {row['responses']} | "
            f"{row['missing']} | {row['extra']} | {row['duplicates']} | "
            f"{row['api_errors']} |"
        )
    lines += [
        "",
        "## Paired IDR_raw",
        "",
        "| Model | Prompt | Task | Groups | IDR_raw | Group 95% CI | Environment 95% CI |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for key in sorted(idr):
        row = idr[key]
        lines.append(
            f"| {row['model']} | `{row['prompt_arm']}` | {row['task']} | "
            f"{row['n_groups']} | {row['estimate']:.3f} | "
            f"[{row['group_ci_low']:.3f}, {row['group_ci_high']:.3f}] | "
            f"[{row['environment_ci_low']:.3f}, {row['environment_ci_high']:.3f}] |"
        )
    lines += [
        "",
        "Primary denominator policy: every complete Full/NoCue/CF group is kept;",
        "unparsable and API-error outputs count as incorrect.",
        "",
    ]
    (output_dir / "results_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def group_by(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument(
        "--response",
        action="append",
        required=True,
        help="MODEL=PATH; repeat once per model",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument('--per-row-only', action='store_true', help='Write parsed question scores only; no paired statistics or tables')
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    request_rows = read_jsonl(args.requests)
    requests = {str(row["uid"]): row for row in request_rows}
    if len(requests) != len(request_rows):
        raise SystemExit("duplicate request UIDs")

    all_rows: list[dict[str, Any]] = []
    audits = []
    paired_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    rationale_rows: list[dict[str, Any]] = []
    for value in args.response:
        model, path_text = value.split("=", 1)
        model_rows, audit = score_model(model, Path(path_text), requests)
        all_rows.extend(model_rows)
        audits.append(audit)
        if args.per_row_only:
            continue
        model_paired, matrices = paired_metric_rows(
            model_rows,
            model,
            args.iterations,
            args.seed,
        )
        paired_rows.extend(model_paired)
        delta_rows.extend(prompt_delta_rows(
            matrices,
            model,
            args.iterations,
            args.seed,
        ))
        rationale_rows.extend(rationale_increment_rows(
            matrices,
            model,
            args.iterations,
            args.seed,
        ))
        write_jsonl(args.output_dir / f"score_per_row_{model}.jsonl", model_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.per_row_only:
        write_jsonl(args.output_dir / 'scored_rows.jsonl', all_rows)
        failures = sum(bool(audit['missing'] or audit['extra'] or audit['duplicates']) for audit in audits)
        print(json.dumps(dict(scored_rows=len(all_rows), coverage_failures=failures, per_row_only=True), indent=2))
        if failures:
            raise SystemExit(1)
        return
    summarize(
        all_rows,
        paired_rows,
        delta_rows,
        rationale_rows,
        audits,
        args.output_dir,
    )
    failures = sum(
        audit["missing"] or audit["extra"] or audit["duplicates"]
        for audit in audits
    )
    print(json.dumps({
        "models": [audit["model"] for audit in audits],
        "scored_rows": len(all_rows),
        "coverage_failures": failures,
        "output_dir": str(args.output_dir),
    }, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
