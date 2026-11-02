#!/usr/bin/env python3
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
    ap = argparse.ArgumentParser(description="Score Task A-MW v2 inference outputs")
    ap.add_argument(
        "--exam_jsonl",
        default="experiments/generated/miniworld/taskA-MW-v2-preflight/taskA-MW-v2-prototype.jsonl",
    )
    ap.add_argument(
        "--responses",
        required=True,
        help="Path to response JSONL file or directory",
    )
    ap.add_argument(
        "--out",
        default="experiments/generated/miniworld/taskA-MW-v2-preflight/score_report.json",
    )
    ap.add_argument(
        "--per_row_out",
        default="experiments/generated/miniworld/taskA-MW-v2-preflight/score_per_row.jsonl",
    )
    ap.add_argument('--per-row-only', action='store_true', help='Write question scores without aggregate reports')
    args = ap.parse_args()

    gold_rows = load_jsonl(Path(args.exam_jsonl))
    responses = load_responses(Path(args.responses))
    by_uid = {row["uid"]: row for row in gold_rows}

    parse_mode_counter: Counter[str] = Counter()
    failure_hist: Counter[str] = Counter()
    bucket_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    strategy_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    per_row: list[dict[str, Any]] = []
    correct = 0
    api_errors = 0
    missing = 0
    recoverable = 0
    strict = 0
    model_name = None

    for uid, gold in by_uid.items():
        resp = responses.get(uid)
        gold_answer = str(gold["correct_candidate"]).upper()
        difficulty = str(gold.get("difficulty_bucket", "unknown"))
        strategy = str(gold.get("distractor_strategy", "unknown"))

        row_result = {
            "uid": uid,
            "gold": gold_answer,
            "difficulty_bucket": difficulty,
            "distractor_strategy": strategy,
            "pred": None,
            "parse_mode": "missing",
            "correct": False,
            "api_error": False,
            "raw": None,
        }

        bucket_stats[difficulty]["n"] += 1
        strategy_stats[strategy]["n"] += 1

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
                bucket_stats[difficulty]["correct"] += 1
                strategy_stats[strategy]["correct"] += 1
            else:
                failure_hist["wrong_answer"] += 1

        per_row.append(row_result)

    if args.per_row_only:
        from utils.json_io import write_jsonl_utf8
        write_jsonl_utf8(Path(args.per_row_out), per_row)
        return
    total = len(gold_rows)
    report = {
        "task": "A-MW-v2",
        "model": model_name,
        "exam_jsonl": str(Path(args.exam_jsonl)),
        "responses": str(Path(args.responses)),
        "coverage": {
            "n_gold": total,
            "n_responses_loaded": len(responses),
            "missing_responses": missing,
        },
        "overall": {
            "n": total,
            "correct": correct,
            "accuracy": pct(correct, total),
            "api_error_rate": pct(api_errors, total),
            "parse_mode": dict(parse_mode_counter),
            "strict_rate": pct(strict, total),
            "recoverable_rate": pct(recoverable, total),
            "failure_hist": dict(failure_hist),
        },
        "by_difficulty": {
            k: {
                "n": v["n"],
                "correct": v["correct"],
                "accuracy": pct(v["correct"], v["n"]),
            }
            for k, v in sorted(bucket_stats.items())
        },
        "by_distractor_strategy": {
            k: {
                "n": v["n"],
                "correct": v["correct"],
                "accuracy": pct(v["correct"], v["n"]),
            }
            for k, v in sorted(strategy_stats.items())
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    per_row_path = Path(args.per_row_out)
    with per_row_path.open("w", encoding="utf-8") as f:
        for row in per_row:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved score report to {out_path}")
    print(f"Saved per-row report to {per_row_path}")


if __name__ == "__main__":
    main()
