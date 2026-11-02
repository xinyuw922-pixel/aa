#!/usr/bin/env python3
"""Build a balanced Scale-2 C/D structured-CoT prompt ablation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils.json_io import write_jsonl_utf8 as write_jsonl
from utils.minigrid_prompts import build_task_c_context

PROMPT_ARMS = ("direct_original", "structured_direct", "structured_cot_v1")
PROMPT_PROTOCOL = "task_context_matched_v2"
VARIANTS = ("full", "nocue", "cf")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def balanced_groups(
    requests: list[dict[str, Any]],
    task: str,
    cap_per_environment: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for request in requests:
        exam = request["exam"]
        if exam.get("task") != task:
            continue
        variant = str(exam.get("variant"))
        if variant not in VARIANTS:
            continue
        key = (str(exam["env_task"]), str(exam["group_id"]))
        if variant in grouped[key]:
            raise ValueError(f"duplicate {task}/{key}/{variant}")
        grouped[key][variant] = request

    selected: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    environment_counts: Counter[str] = Counter()
    for (environment, group_id), variants in sorted(grouped.items()):
        if set(variants) != set(VARIANTS):
            raise ValueError(f"incomplete {task}/{environment}/{group_id}")
        if environment_counts[environment] >= cap_per_environment:
            continue
        environment_counts[environment] += 1
        group_rows.append({
            "task": task,
            "env_task": environment,
            "group_id": group_id,
            "selection_rank_within_environment": environment_counts[environment],
        })
        selected.extend(variants[variant] for variant in VARIANTS)
    return selected, group_rows


def structured_direct_prompt(task: str, task_context: str = "") -> str:
    if task == "C":
        if not task_context:
            raise ValueError("Task C requires an explicit environment and mission")
        instruction = task_context
    elif task == "D":
        instruction = (
            "The reference trajectory is known successful. Use it only to "
            "identify the task goal, then judge whether the query trajectory "
            "completed that goal."
        )
    else:
        raise ValueError(task)
    return (
        f"{instruction}\n"
        "Answer with ONLY one JSON object, with no Markdown or extra text:\n"
        '{"verdict":"Success"}\n'
        'The verdict must be exactly "Success" or "Fail".'
    )


def structured_cot_prompt(task: str, task_context: str = "") -> str:
    if task == "C":
        if not task_context:
            raise ValueError("Task C requires an explicit environment and mission")
        instruction = (
            task_context + "\n"
            "Use the complete storyboard and distinguish observed completion "
            "from merely plausible progress."
        )
        schema = (
            '{"goal":"short phrase","observed_evidence":["item 1","item 2"],'
            '"unmet_requirement":"short phrase or none","verdict":"Success"}'
        )
    elif task == "D":
        instruction = (
            "The reference trajectory is known successful. Use it only to "
            "identify the task goal. Judge the query trajectory from its own "
            "observed outcome; it may be a counterfactual failure."
        )
        schema = (
            '{"reference_goal":"short phrase",'
            '"query_evidence":["item 1","item 2"],'
            '"failed_requirement":"short phrase or none","verdict":"Success"}'
        )
    else:
        raise ValueError(task)
    return (
        f"{instruction}\n"
        "Produce a brief structured evidence record. Use at most two evidence "
        "items and at most 12 words per text field.\n"
        "Answer with ONLY one JSON object, with no Markdown or extra text:\n"
        f"{schema}\n"
        'The verdict must be exactly "Success" or "Fail".'
    )


def prompt_for(request: dict[str, Any], arm: str) -> str:
    task = str(request["exam"]["task"])
    task_context = ""
    if task == "C":
        exam = request["exam"]
        task_context = build_task_c_context(
            str(exam.get("env_task") or ""), str(exam.get("mission") or ""),
        )
    if arm == "direct_original":
        if task == "C":
            return task_context + "\nAnswer with ONLY: Success or Fail."
        return str(request["prompt"])
    if arm == "structured_direct":
        return structured_direct_prompt(task, task_context)
    if arm == "structured_cot_v1":
        return structured_cot_prompt(task, task_context)
    raise ValueError(arm)


def ablation_request(request: dict[str, Any], arm: str) -> dict[str, Any]:
    source_uid = str(request["uid"])
    uid = f"{source_uid}.prompt_{arm}"
    exam = dict(request["exam"])
    exam.update({
        "uid": uid,
        "exam_id": uid,
        "source_uid": source_uid,
        "prompt_ablation_arm": arm,
        "prompt_ablation_protocol": PROMPT_PROTOCOL,
        "original_prompt": request["prompt"],
        "prompt": prompt_for(request, arm),
    })
    return {
        **request,
        "uid": uid,
        "exam_id": uid,
        "prompt": exam["prompt"],
        "exam": exam,
    }


def choose_smoke_groups(
    group_rows: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    for task in ("C", "D"):
        candidates = [row for row in group_rows if row["task"] == task]
        preferred = [
            row for row in candidates
            if row["env_task"] == "doorkey"
        ]
        row = (preferred or candidates)[0]
        selected.add((task, row["group_id"]))
    return selected


def validate(
    base_requests: list[dict[str, Any]],
    all_requests: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    smoke_requests: list[dict[str, Any]],
    expected_group_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, observed: Any) -> None:
        checks.append({
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "observed": observed,
        })

    check(
        "base UID uniqueness",
        len({row["uid"] for row in base_requests}) == len(base_requests),
        len(base_requests),
    )
    check(
        "ablation UID uniqueness",
        len({row["uid"] for row in all_requests}) == len(all_requests),
        len(all_requests),
    )
    check(
        "all image paths exist",
        all(Path(image).is_file() for row in all_requests for image in row["images"]),
        sum(len(row["images"]) for row in all_requests),
    )
    sources = {row["uid"]: row for row in base_requests}
    context_errors = []
    for row in all_requests:
        exam = row["exam"]
        source = sources.get(exam.get("source_uid"))
        try:
            aligned = (
                source is not None
                and row["images"] == source["images"]
                and exam.get("answer") == source["exam"].get("answer")
                and all(exam.get(key) == source["exam"].get(key)
                        for key in ("task", "env_task", "mission", "group_id", "variant"))
                and row["prompt"] == prompt_for(source, exam["prompt_ablation_arm"])
                and exam["prompt"] == row["prompt"]
                and exam.get("prompt_ablation_protocol") == PROMPT_PROTOCOL
            )
        except (KeyError, ValueError):
            aligned = False
        if not aligned:
            context_errors.append(row["uid"])
    check("task context, evidence and answers aligned across arms", not context_errors, context_errors)
    arm_counts = Counter(row["exam"]["prompt_ablation_arm"] for row in all_requests)
    check(
        "equal prompt-arm counts",
        len(set(arm_counts.values())) == 1 and set(arm_counts) == set(PROMPT_ARMS),
        dict(arm_counts),
    )
    task_group_counts = Counter(row["task"] for row in group_rows)
    check(
        "expected balanced group counts",
        task_group_counts == (expected_group_counts if expected_group_counts is not None else {"C": 101, "D": 98}),
        dict(task_group_counts),
    )
    environment_counts = Counter(
        f"{row['task']}:{row['env_task']}" for row in group_rows
    )
    check(
        "environment caps respected",
        all(count <= 12 for count in environment_counts.values()),
        dict(sorted(environment_counts.items())),
    )
    variant_groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in all_requests:
        exam = row["exam"]
        variant_groups[
            (
                str(exam["prompt_ablation_arm"]),
                str(exam["task"]),
                str(exam["group_id"]),
            )
        ].add(str(exam["variant"]))
    bad_groups = sum(set(variants) != set(VARIANTS) for variants in variant_groups.values())
    check("all ablation groups complete", bad_groups == 0, bad_groups)
    check(
        "smoke has two complete groups per arm",
        len(smoke_requests) == 18,
        len(smoke_requests),
    )
    failures = sum(row["status"] != "PASS" for row in checks)
    return {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "base_rows": len(base_requests),
        "full_requests_per_model": len(all_requests),
        "smoke_requests_per_model": len(smoke_requests),
        "groups": dict(task_group_counts),
        "environment_groups": dict(sorted(environment_counts.items())),
        "prompt_arms": list(PROMPT_ARMS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cap-per-environment", type=int, default=12)
    parser.add_argument('--expected-c-groups', type=int, default=101)
    parser.add_argument('--expected-d-groups', type=int, default=98)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error('Output already exists; use a fresh directory')
    if not 1 <= args.cap_per_environment <= 12:
        parser.error('--cap-per-environment must be between 1 and 12')

    requests = read_jsonl(args.source_requests)
    base_requests: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for task in ("C", "D"):
        selected, selected_groups = balanced_groups(
            requests,
            task,
            args.cap_per_environment,
        )
        base_requests.extend(selected)
        group_rows.extend(selected_groups)

    all_requests = [
        ablation_request(request, arm)
        for arm in PROMPT_ARMS
        for request in base_requests
    ]
    smoke_groups = choose_smoke_groups(group_rows)
    smoke_requests = [
        row for row in all_requests
        if (
            str(row["exam"]["task"]),
            str(row["exam"]["group_id"]),
        ) in smoke_groups
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "selected_groups.csv", group_rows)
    write_jsonl(args.output_dir / "requests_structured_cot_ablation.jsonl", all_requests)
    write_jsonl(args.output_dir / "smoke" / "requests_structured_cot_smoke.jsonl", smoke_requests)
    for arm in PROMPT_ARMS:
        write_jsonl(
            args.output_dir / "requests_by_arm" / f"requests_{arm}.jsonl",
            [
                row for row in all_requests
                if row["exam"]["prompt_ablation_arm"] == arm
            ],
        )
    for task in ("C", "D"):
        write_jsonl(
            args.output_dir / "exam" / f"task_{task.lower()}_exam.jsonl",
            [row["exam"] for row in all_requests if row["exam"]["task"] == task],
        )
        write_jsonl(
            args.output_dir / "smoke" / "exam" / f"task_{task.lower()}_exam.jsonl",
            [row["exam"] for row in smoke_requests if row["exam"]["task"] == task],
        )

    report = validate(
        base_requests,
        all_requests,
        group_rows,
        smoke_requests,
        expected_group_counts={'C': args.expected_c_groups, 'D': args.expected_d_groups},
    )
    report.update({
        "prompt_ablation_protocol": PROMPT_PROTOCOL,
        "source_requests": str(args.source_requests),
        "selection_policy": (
            "lexicographic group_id order; at most 12 complete groups per "
            "task/environment; selected before new model outputs"
        ),
    })
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Scale-2 Structured-CoT Manifest Validation",
        "",
        f"- Status: {report['status']}",
        f"- Base paired rows: {report['base_rows']}",
        f"- Requests per model: {report['full_requests_per_model']}",
        f"- Smoke requests per model: {report['smoke_requests_per_model']}",
        f"- Groups: {report['groups']}",
        f"- Prompt arms: {', '.join(PROMPT_ARMS)}",
        "",
        "| Check | Status | Observed |",
        "| --- | --- | --- |",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['check']} | {check['status']} | `{check['observed']}` |"
        )
    (args.output_dir / "manifest_validation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "groups": report["groups"],
        "full_requests_per_model": report["full_requests_per_model"],
        "smoke_requests_per_model": report["smoke_requests_per_model"],
    }, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
