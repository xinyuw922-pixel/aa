#!/usr/bin/env python3
"""Score hierarchical Oracle-State responses and paired verdict IDR."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils.oracle_task_types import TASK_TYPES
from tools.scoring.oracle_metrics import delta_rows, metric_rows
from tools.scoring.structured_cot import (
    canonical_label,
    read_jsonl,
    strip_markdown_fence,
    write_csv,
    write_jsonl,
)


FIELDS = ("TASK_TYPE", "INFERRED_OBJECTIVE", "VERDICT", "EXPLANATION")
STRICT_PATTERN = re.compile(
    r"""
    \ATASK_TYPE:\s*(?P<task_type>[A-Z_]+)\s*\n
    INFERRED_OBJECTIVE:\s*(?P<objective>[^\n]+?)\s*\n
    VERDICT:\s*(?P<verdict>Success|Fail)\s*\n
    EXPLANATION:\s*(?P<explanation>.+?)\s*\Z
    """,
    re.DOTALL | re.VERBOSE,
)


def normalize_recoverable_labels(text: str) -> str:
    for field in FIELDS:
        text = re.sub(
            rf"(?im)^\s*\*{{1,2}}{re.escape(field)}\s*:\s*\*{{1,2}}\s*",
            f"{field}: ",
            text,
        )
        text = re.sub(
            rf"(?im)^\s*\*{{1,2}}{re.escape(field)}\*{{1,2}}\s*:\s*",
            f"{field}: ",
            text,
        )
    return text


def recover_field(text: str, field: str) -> str | None:
    other_fields = "|".join(
        re.escape(candidate) for candidate in FIELDS if candidate != field
    )
    match = re.search(
        rf"(?ims)^\s*{re.escape(field)}\s*:\s*(.+?)"
        rf"(?=^\s*(?:{other_fields})\s*:|\Z)",
        text,
    )
    return match.group(1).strip() if match else None


def parse_hierarchical_response(raw: str, api_ok: bool) -> dict[str, Any]:
    if not api_ok:
        return {
            "parse_mode": "api_error",
            "strict_schema": False,
            "markdown_fence": False,
            "task_type": None,
            "objective": None,
            "verdict": None,
            "explanation": None,
        }
    text, fenced = strip_markdown_fence(raw)
    strict_match = STRICT_PATTERN.fullmatch(text)
    if strict_match:
        task_type = strict_match.group("task_type")
        verdict = canonical_label(strict_match.group("verdict"))
        if task_type in TASK_TYPES and verdict is not None and not fenced:
            return {
                "parse_mode": "strict",
                "strict_schema": True,
                "markdown_fence": False,
                "task_type": task_type,
                "objective": strict_match.group("objective").strip(),
                "verdict": verdict,
                "explanation": strict_match.group("explanation").strip(),
            }

    recoverable_text = normalize_recoverable_labels(text)
    task_type_raw = recover_field(recoverable_text, "TASK_TYPE")
    objective = recover_field(recoverable_text, "INFERRED_OBJECTIVE")
    verdict_raw = recover_field(recoverable_text, "VERDICT")
    explanation = recover_field(recoverable_text, "EXPLANATION")
    task_type = task_type_raw.upper().strip() if task_type_raw else None
    verdict = canonical_label(
        verdict_raw.strip().strip("`*_ .") if verdict_raw else None
    )
    complete = (
        task_type in TASK_TYPES
        and bool(objective)
        and verdict is not None
        and bool(explanation)
    )
    return {
        "parse_mode": "recoverable" if complete else "unparsable",
        "strict_schema": False,
        "markdown_fence": fenced,
        "task_type": task_type if task_type in TASK_TYPES else None,
        "objective": objective,
        "verdict": verdict if complete else None,
        "explanation": explanation,
    }


def contains_any(text: str, alternatives: tuple[str, ...]) -> bool:
    return any(token in text for token in alternatives)


def objective_concept_correct(
    objective: str | None,
    task_type: str,
    gold_objective: str,
) -> bool:
    if not objective:
        return False
    text = objective.lower()
    if task_type == "REACH_GOAL":
        return contains_any(
            text,
            ("goal", "terminal cell", "terminal tile", "destination"),
        )
    if task_type == "UNLOCK_AND_REACH":
        return (
            contains_any(text, ("key", "unlocking object"))
            and contains_any(text, ("door", "barrier"))
            and contains_any(
                text,
                ("goal", "terminal cell", "terminal tile", "destination"),
            )
        )
    if task_type == "PICK_UP_OBJECT":
        match = re.fullmatch(
            r"pick up the ([a-z]+) ([a-z]+)",
            gold_objective.strip().lower(),
        )
        return bool(
            match
            and contains_any(text, ("pick up", "pickup", "carry", "carrying"))
            and match.group(1) in text
            and match.group(2) in text
        )
    if task_type == "MATCH_CUE_OBJECT":
        return (
            contains_any(text, ("match", "matching", "same type", "correspond"))
            and contains_any(text, ("object", "endpoint", "end of"))
        )
    if task_type == "PUT_OBJECT_NEAR_OBJECT":
        match = re.fullmatch(
            r"put the ([a-z]+) ([a-z]+) near the ([a-z]+) ([a-z]+)",
            gold_objective.strip().lower(),
        )
        return bool(
            match
            and contains_any(text, ("near", "adjacent", "next to"))
            and all(match.group(index) in text for index in range(1, 5))
        )
    if task_type == "OPEN_DOORS_IN_ORDER":
        match = re.fullmatch(
            r"open the ([a-z]+) door then the ([a-z]+) door",
            gold_objective.strip().lower(),
        )
        if not match:
            return False
        first = text.find(match.group(1))
        second = text.find(match.group(2))
        return (
            contains_any(text, ("open", "opening"))
            and "door" in text
            and first >= 0
            and second > first
        )
    raise ValueError(f"unknown task type: {task_type}")


def explanation_flags(explanation: str | None) -> dict[str, bool]:
    text = (explanation or "").lower()
    reference = "reference" in text
    query = "query" in text
    return {
        "explanation_present": bool(text.strip()),
        "explanation_mentions_reference": reference,
        "explanation_mentions_query": query,
        "explanation_bilateral": reference and query,
    }


def score_model(
    model: str,
    response_path: Path,
    requests: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response_rows = read_jsonl(response_path)
    response_map = {str(row["uid"]): row for row in response_rows}
    output = []
    for uid, request in requests.items():
        exam = request["exam"]
        response = response_map.get(uid, {})
        api_ok = bool(response.get("meta", {}).get("ok", False))
        raw = str(response.get("raw") or response.get("pred") or "")
        parsed = parse_hierarchical_response(raw, api_ok)
        gold_verdict = canonical_label(exam.get("answer") or exam.get("label"))
        gold_task_type = str(exam["gold_task_type"])
        task_type_correct = parsed["task_type"] == gold_task_type
        objective_correct = objective_concept_correct(
            parsed["objective"],
            gold_task_type,
            str(exam["gold_specific_objective"]),
        )
        verdict_correct = parsed["verdict"] == gold_verdict
        flags = explanation_flags(parsed["explanation"])
        output.append(
            {
                "uid": uid,
                "source_uid": exam["source_uid"],
                "model": model,
                "env_task": exam["env_task"],
                "group_id": exam["group_id"],
                "variant": exam["variant"],
                "evidence_condition": exam["oracle_control_condition"],
                "gold_task_type": gold_task_type,
                "pred_task_type": parsed["task_type"],
                "task_type_correct": task_type_correct,
                "gold_specific_objective": exam["gold_specific_objective"],
                "pred_objective": parsed["objective"],
                "objective_concept_correct": objective_correct,
                "gold": gold_verdict,
                "pred_norm": parsed["verdict"],
                "verdict_correct": verdict_correct,
                "joint_type_objective_verdict_correct": (
                    task_type_correct and objective_correct and verdict_correct
                ),
                **flags,
                "parse_mode": parsed["parse_mode"],
                "strict_schema": parsed["strict_schema"],
                "markdown_fence": parsed["markdown_fence"],
                "raw": raw,
                "api_ok": api_ok,
                "api_error": response.get("meta", {}).get("error"),
                "latency_ms": response.get("meta", {}).get("latency_ms"),
                "finish_reason": response.get("meta", {}).get("finish_reason"),
            }
        )
    return output, {
        "model": model,
        "requests": len(requests),
        "responses": len(response_rows),
        "missing": len(set(requests) - set(response_map)),
        "extra": len(set(response_map) - set(requests)),
        "duplicates": len(response_rows) - len(response_map),
        "api_errors": sum(row["parse_mode"] == "api_error" for row in output),
        "unparsable": sum(row["parse_mode"] == "unparsable" for row in output),
    }


def hierarchy_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["evidence_condition"]))].append(row)
    output = []
    for (model, condition), selected in sorted(grouped.items()):
        n = len(selected)
        modes = Counter(row["parse_mode"] for row in selected)
        output.append(
            {
                "model": model,
                "evidence_condition": condition,
                "n": n,
                "strict_rate": modes["strict"] / n,
                "recoverable_rate": modes["recoverable"] / n,
                "unparsable_rate": modes["unparsable"] / n,
                "api_error_rate": modes["api_error"] / n,
                "task_type_acc": sum(
                    row["task_type_correct"] for row in selected
                )
                / n,
                "objective_concept_acc": sum(
                    row["objective_concept_correct"] for row in selected
                )
                / n,
                "verdict_acc": sum(row["verdict_correct"] for row in selected)
                / n,
                "joint_hierarchical_acc": sum(
                    row["joint_type_objective_verdict_correct"]
                    for row in selected
                )
                / n,
                "bilateral_explanation_rate": sum(
                    row["explanation_bilateral"] for row in selected
                )
                / n,
            }
        )
    return output


def run_self_test() -> None:
    strict = (
        "TASK_TYPE: PICK_UP_OBJECT\n"
        "INFERRED_OBJECTIVE: Pick up the grey ball.\n"
        "VERDICT: Success\n"
        "EXPLANATION: REFERENCE carries it; QUERY carries it."
    )
    assert parse_hierarchical_response(strict, True)["parse_mode"] == "strict"
    fenced = f"```\n{strict}\n```"
    assert parse_hierarchical_response(fenced, True)["parse_mode"] == "recoverable"
    bold = strict.replace("TASK_TYPE:", "**TASK_TYPE:**").replace(
        "INFERRED_OBJECTIVE:", "**INFERRED_OBJECTIVE:**"
    ).replace("VERDICT:", "**VERDICT:**").replace(
        "EXPLANATION:", "**EXPLANATION:**"
    )
    assert parse_hierarchical_response(bold, True)["parse_mode"] == "recoverable"
    block = bold.replace("**TASK_TYPE:** ", "**TASK_TYPE:**\n").replace(
        "**INFERRED_OBJECTIVE:** ", "**INFERRED_OBJECTIVE:**\n"
    ).replace("**VERDICT:** ", "**VERDICT:**\n").replace(
        "**EXPLANATION:** ", "**EXPLANATION:**\n"
    )
    assert parse_hierarchical_response(block, True)["parse_mode"] == "recoverable"
    missing = strict.rsplit("\n", 1)[0]
    assert parse_hierarchical_response(missing, True)["parse_mode"] == "unparsable"
    assert parse_hierarchical_response(strict, False)["parse_mode"] == "api_error"
    assert objective_concept_correct(
        "Pick up and carry the grey ball.",
        "PICK_UP_OBJECT",
        "pick up the grey ball",
    )
    assert objective_concept_correct(
        "Open the red door, then open the blue door.",
        "OPEN_DOORS_IN_ORDER",
        "open the red door then the blue door",
    )
    print(json.dumps({"self_test": "PASS"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument('--per-row-only', action='store_true', help='Write hierarchical row scores only; no bootstrap or summary tables')
    parser.add_argument("--requests", type=Path)
    parser.add_argument(
        "--model-response",
        action="append",
        help="MODEL=/path/to/responses.jsonl",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if not args.requests or not args.model_response or not args.output_dir:
        parser.error(
            "--requests, --model-response, and --output-dir are required"
        )

    request_rows = read_jsonl(args.requests)
    requests = {str(row["uid"]): row for row in request_rows}
    if len(requests) != len(request_rows):
        raise ValueError("duplicate request UID")

    all_rows = []
    all_metrics = []
    all_deltas = []
    audits = []
    for spec in args.model_response:
        model, separator, path = spec.partition("=")
        if not separator:
            raise ValueError(f"invalid --model-response: {spec}")
        rows, audit = score_model(model, Path(path), requests)
        if args.per_row_only:
            all_rows.extend(rows)
            audits.append(audit)
            continue
        metrics, matrices = metric_rows(
            rows,
            model,
            args.bootstrap_iterations,
            args.seed,
        )
        deltas = delta_rows(
            matrices,
            model,
            args.bootstrap_iterations,
            args.seed,
        )
        all_rows.extend(rows)
        all_metrics.extend(metrics)
        all_deltas.extend(deltas)
        audits.append(audit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "scored_rows.jsonl", all_rows)
    if args.per_row_only:
        failures = sum(bool(audit['missing'] or audit['extra'] or audit['duplicates']) for audit in audits)
        print(json.dumps(dict(models=len(audits), scored_rows=len(all_rows),
                              coverage_failures=failures, per_row_only=True), indent=2))
        if failures:
            raise SystemExit(1)
        return
    write_csv(
        args.output_dir / "hierarchical_metrics.csv",
        hierarchy_summary(all_rows),
    )
    write_csv(args.output_dir / "paired_metrics_with_ci.csv", all_metrics)
    write_csv(args.output_dir / "paired_condition_deltas.csv", all_deltas)
    write_csv(args.output_dir / "run_audit.csv", audits)
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


if __name__ == "__main__":
    main()
