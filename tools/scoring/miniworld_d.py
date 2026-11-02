#!/usr/bin/env python3
"""Score MiniWorld trajectory judgments against exam answers.

Supports strict and recoverable answers. Invoke through scripts/score.sh
with the matching exam JSONL and response JSONL."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


API_ERROR_PREFIX = "__GW_ERR__:"
AB_RE = re.compile(r"\b([AB])\b", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_responses(path: Path) -> dict[str, dict[str, Any]]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    out: dict[str, dict[str, Any]] = {}
    for fp in files:
        for row in load_jsonl(fp):
            uid = row.get("uid") or row.get("exam_id")
            if uid and uid not in out:
                out[uid] = row
    return out


def parse_answer(text: str) -> tuple[str | None, str]:
    t = (text or "").strip()
    if not t:
        return None, "fail"
    if t.upper() in {"A", "B"}:
        return t.upper(), "strict"
    m = AB_RE.search(t)
    if m:
        return m.group(1).upper(), "recoverable"
    return None, "fail"


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else n / d


def main() -> None:
    ap = argparse.ArgumentParser(description="Score D-MW Trajectory Judgment inference")
    ap.add_argument(
        "--exam_jsonl",
        required=True,
        help="Path to D-MW exam JSONL",
    )
    ap.add_argument(
        "--responses",
        required=True,
        help="Path to response JSONL file or directory",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output path for score report JSON. Default: <responses>/../score_report.json",
    )
    ap.add_argument(
        "--per_row_out",
        default=None,
        help="Output path for per-row JSONL. Default: <responses>/../score_per_row.jsonl",
    )
    ap.add_argument('--per-row-only', action='store_true', help='Write question scores without aggregate reports')
    args = ap.parse_args()

    exam_path = Path(args.exam_jsonl)
    responses_path = Path(args.responses)
    out_path = Path(args.out) if args.out else (responses_path.parent / "score_report.json")
    per_row_path = Path(args.per_row_out) if args.per_row_out else (responses_path.parent / "score_per_row.jsonl")

    gold_rows = load_jsonl(exam_path)
    responses = load_responses(responses_path)
    by_uid = {row["uid"]: row for row in gold_rows}

    parse_mode_counter: Counter[str] = Counter()
    failure_hist: Counter[str] = Counter()
    per_row: list[dict[str, Any]] = []
    correct = 0
    api_errors = 0
    missing = 0
    recoverable = 0
    strict = 0
    model_name: str | None = None

    for uid, gold in by_uid.items():
        resp = responses.get(uid)
        gold_answer = str(gold["correct_candidate"]).upper()
        success_dir = str(gold.get("success_dir", "?"))
        failure_dir = str(gold.get("failure_dir", "?"))

        row_result: dict[str, Any] = {
            "uid": uid,
            "gold": gold_answer,
            "success_dir": success_dir,
            "failure_dir": failure_dir,
            "planned_trajectory": gold.get("planned_trajectory", "?"),
            "pred": None,
            "parse_mode": "missing",
            "correct": False,
            "api_error": False,
            "raw": None,
        }

        if resp is None:
            missing += 1
            failure_hist["missing_response"] += 1
            per_row.append(row_result)
            continue

        meta = resp.get("meta", {}) if isinstance(resp.get("meta"), dict) else {}
        model_name = model_name or meta.get("model")
        raw = str(resp.get("raw", resp.get("pred", "")))
        row_result["raw"] = raw

        if raw.startswith(API_ERROR_PREFIX) or meta.get("ok") is False:
            api_errors += 1
            row_result["api_error"] = True
            failure_hist["api_error"] += 1

        pred, mode = parse_answer(str(resp.get("pred", raw)))
        row_result["pred"] = pred
        row_result["parse_mode"] = mode
        parse_mode_counter[mode] += 1

        if mode == "strict":
            strict += 1
        elif mode == "recoverable":
            recoverable += 1

        if pred is None:
            failure_hist["parse_fail"] += 1
        else:
            is_correct = pred == gold_answer
            row_result["correct"] = is_correct
            if is_correct:
                correct += 1

        per_row.append(row_result)

    # Write per-row
    per_row_path.parent.mkdir(parents=True, exist_ok=True)
    with per_row_path.open("w", encoding="utf-8") as f:
        for row in per_row:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(by_uid)
    answered = total - missing
    usable = answered - api_errors

    if args.per_row_only:
        return
    report = {
        "model": model_name,
        "exam": str(exam_path),
        "total_questions": total,
        "correct": correct,
        "accuracy": pct(correct, total),
        "accuracy_answered": pct(correct, answered) if answered > 0 else None,
        "accuracy_usable": pct(correct, usable) if usable > 0 else None,
        "class_balance": {
            "total": total,
            "missing": missing,
            "api_errors": api_errors,
            "answered": answered,
        },
        "parse_modes": dict(parse_mode_counter),
        "strict_vs_recoverable": {"strict": strict, "recoverable": recoverable},
        "failure_histogram": dict(failure_hist),
        "majority_baseline": {
            "A_count": sum(1 for r in gold_rows if r["correct_candidate"] == "A"),
            "B_count": sum(1 for r in gold_rows if r["correct_candidate"] == "B"),
            "majority_class": "A" if sum(1 for r in gold_rows if r["correct_candidate"] == "A") > sum(1 for r in gold_rows if r["correct_candidate"] == "B") else "B",
            "majority_baseline_accuracy": max(
                sum(1 for r in gold_rows if r["correct_candidate"] == "A"),
                sum(1 for r in gold_rows if r["correct_candidate"] == "B"),
            ) / total,
        },
        "per_row_relpath": str(per_row_path.relative_to(out_path.parent)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("=" * 60)
    print(f"D-MW Score Report  (model: {model_name or 'unknown'})")
    print("=" * 60)
    print(f"  Exam:            {exam_path.name}")
    print(f"  Total:           {total}")
    print(f"  Correct:          {correct} / {total}  = {pct(correct,total):.1%}")
    print(f"  Accuracy(answered): {pct(correct,answered):.1%}  (excl. missing={missing})")
    if usable > 0:
        print(f"  Accuracy(usable):  {pct(correct,usable):.1%}  (excl. API errors={api_errors})")
    mb = report["majority_baseline"]
    print(f"  Majority baseline: {mb['majority_baseline_accuracy']:.1%}  ({mb['majority_class']})")
    print(f"  Parse: strict={strict}  recoverable={recoverable}  missing={missing}")
    print(f"  Score report:     {out_path}")
    print(f"  Per-row:          {per_row_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
