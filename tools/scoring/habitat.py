#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from utils.formatting import format_percentage


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize(task: str, value: str) -> str | None:
    text = str(value or "").strip()
    if task == "A":
        matches = re.findall(r"\b([A-D])\b", text.upper())
        return matches[-1] if matches else None
    matches = re.findall(r"\b(Success|Fail|Failure)\b", text, re.I)
    if not matches:
        return None
    return "Success" if matches[-1].lower() == "success" else "Fail"


def accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    valid = sum(row["pred"] is not None for row in rows)
    return {
        "n": total,
        "valid": valid,
        "correct": correct,
        "accuracy": correct / total if total else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true', help='Mark all reports as synthetic pipeline checks')
    parser.add_argument('--per-row-only', action='store_true', help='Write scored_rows.jsonl without aggregate reports')
    parser.add_argument(
        "--exam-dir",
        type=Path,
        default=Path("Habitat/outputs/hssd-exam-en-550"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "Habitat/outputs/hssd-exam-en-550/model_runs/full"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Habitat/outputs/hssd-exam-en-550/model_results"
        ),
    )
    args = parser.parse_args()
    exam_dir = args.exam_dir.resolve()
    runs_root = args.runs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    answers = json.loads(
        (exam_dir / "answer_key.json").read_text(encoding="utf-8")
    )["answers"]
    gold = {row["question_id"]: row for row in answers}
    expected = set(gold)
    all_reports = {}

    for model_dir in sorted(
        path for path in runs_root.iterdir()
        if path.is_dir() and path.name != "logs"
    ):
        by_question: dict[str, dict[str, Any]] = {}
        duplicates = []
        retry_replacements = []
        for path in sorted(model_dir.rglob("*.jsonl")):
            for response in read_jsonl(path):
                question_id = (
                    response.get("meta", {})
                    .get("exam", {})
                    .get("question_id")
                )
                if not question_id:
                    uid = str(response.get("uid", ""))
                    question_id = uid.rsplit(".", 1)[-1]
                if question_id in by_question:
                    if "zz_retries" in path.parts:
                        retry_replacements.append(question_id)
                    else:
                        duplicates.append(question_id)
                        continue
                by_question[question_id] = response

        scored = []
        for question_id, gold_row in gold.items():
            response = by_question.get(question_id)
            raw = "" if response is None else str(response.get("raw", ""))
            pred = normalize(gold_row["task"], raw)
            ok = bool(
                response is not None
                and response.get("meta", {}).get("ok")
            )
            scored.append(
                {
                    "question_id": question_id,
                    "task": gold_row["task"],
                    "gold": gold_row["gold"],
                    "pred": pred,
                    "correct": pred == gold_row["gold"],
                    "api_ok": ok,
                    "condition": gold_row.get("condition"),
                    "prompt_style": gold_row.get("prompt_style"),
                    "action_sequence_label": gold_row.get(
                        "stratum", {}
                    ).get("action_sequence_label"),
                    "source_uid": gold_row["source_uid"],
                    "raw": raw,
                }
            )

        if args.per_row_only:
            from utils.json_io import write_jsonl_utf8
            write_jsonl_utf8(output_dir / 'scored_rows.jsonl', scored)
            continue
        by_task = {
            task: accuracy([row for row in scored if row["task"] == task])
            for task in ("A", "C", "D")
        }
        by_condition = {}
        for task in ("C", "D"):
            by_condition[task] = {
                condition: accuracy(
                    [
                        row for row in scored
                        if row["task"] == task
                        and row["condition"] == condition
                    ]
                )
                for condition in ("full", "cf")
            }
        by_d_style = {
            style: accuracy(
                [
                    row for row in scored
                    if row["task"] == "D"
                    and row["prompt_style"] == style
                ]
            )
            for style in sorted(
                {
                    str(row["prompt_style"])
                    for row in scored
                    if row["task"] == "D"
                }
            )
        }
        by_a_action = {
            action: accuracy(
                [
                    row for row in scored
                    if row["task"] == "A"
                    and row["action_sequence_label"] == action
                ]
            )
            for action in sorted(
                {
                    str(row["action_sequence_label"])
                    for row in scored
                    if row["task"] == "A"
                }
            )
        }
        prediction_distribution = {
            task: dict(
                Counter(
                    row["pred"]
                    for row in scored
                    if row["task"] == task
                )
            )
            for task in ("A", "C", "D")
        }
        missing = sorted(expected - set(by_question))
        extras = sorted(set(by_question) - expected)
        report = {
            "synthetic": args.synthetic,
            "notice": "SYNTHETIC / NOT MODEL RESULTS" if args.synthetic else "Observed model responses",
            "model": model_dir.name,
            "overall": accuracy(scored),
            "by_task": by_task,
            "by_condition": by_condition,
            "task_a_by_action": by_a_action,
            "task_d_by_prompt_style": by_d_style,
            "prediction_distribution": prediction_distribution,
            "response_rows": len(by_question),
            "api_error_count": sum(
                not row["api_ok"]
                for row in scored
                if row["question_id"] in by_question
            ),
            "invalid_prediction_count": sum(
                row["pred"] is None for row in scored
            ),
            "missing_count": len(missing),
            "missing_question_ids": missing,
            "duplicate_question_ids": sorted(set(duplicates)),
            "retry_replaced_question_ids": sorted(
                set(retry_replacements)
            ),
            "extra_question_ids": extras,
        }
        all_reports[model_dir.name] = report
        (output_dir / f"{model_dir.name}_score_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with (output_dir / f"{model_dir.name}_per_row.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            fieldnames = [
                "question_id",
                "task",
                "gold",
                "pred",
                "correct",
                "api_ok",
                "condition",
                "prompt_style",
                "action_sequence_label",
                "source_uid",
                "raw",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(scored)

    if args.per_row_only:
        return 0
    (output_dir / "score_summary.json").write_text(
        json.dumps(all_reports, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Habitat Exam Scoring" + (" - SYNTHETIC / NOT MODEL RESULTS" if args.synthetic else ""),
        "",
        "| Model | Overall | Task A | Task C | Task D-R |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, report in all_reports.items():
        lines.append(
            f"| {model} | {format_percentage(report['overall']['accuracy'])} | "
            f"{format_percentage(report['by_task']['A']['accuracy'])} | "
            f"{format_percentage(report['by_task']['C']['accuracy'])} | "
            f"{format_percentage(report['by_task']['D']['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Full/CF breakdown",
            "",
            "| Model | C Full | C CF | D Full | D CF |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model, report in all_reports.items():
        condition = report["by_condition"]
        lines.append(
            f"| {model} | "
            f"{format_percentage(condition['C']['full']['accuracy'])} | "
            f"{format_percentage(condition['C']['cf']['accuracy'])} | "
            f"{format_percentage(condition['D']['full']['accuracy'])} | "
            f"{format_percentage(condition['D']['cf']['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "Coverage, parsing and API status are reported separately for each run above.",
            "",
        ]
    )
    (output_dir / "comparison.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(all_reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
