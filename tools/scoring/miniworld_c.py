#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from utils.exam_schema import parse_verdict


API_ERROR_PREFIX = "__GW_ERR__:"


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
            uid = str(row.get("uid") or row.get("exam_id") or "")
            if uid and uid not in out:
                out[uid] = row
    return out


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else n / d


def bucket_from_source(source_summary: str) -> str:
    source_summary = str(source_summary or "")
    if "migration_showcase" in source_summary:
        return "paired"
    if "structural_showcase" in source_summary:
        return "structural"
    if "highdisc_round1" in source_summary:
        return "highdisc_round1"
    if "highdisc_round2" in source_summary:
        return "highdisc_round2"
    if "highdisc_round3" in source_summary:
        return "highdisc_round3"
    return source_summary or "unknown"


def update_bucket(stats: dict[str, dict[str, int]], key: str, is_correct: bool, is_unknown: bool) -> None:
    bucket = stats.setdefault(key, {"n": 0, "correct": 0, "unknown": 0})
    bucket["n"] += 1
    bucket["correct"] += int(is_correct)
    bucket["unknown"] += int(is_unknown)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score Task C-MW inference outputs")
    ap.add_argument(
        "--exam_jsonl",
        required=True,
        help="Path to Task C-MW exam JSONL",
    )
    ap.add_argument(
        "--responses",
        required=True,
        help="Path to response JSONL file or directory",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output summary JSON path. Default: <responses>/../smoke_summary.json",
    )
    ap.add_argument(
        "--per_row_out",
        default=None,
        help="Output per-row JSONL path. Default: <responses>/../score_per_row.jsonl",
    )
    ap.add_argument('--per-row-only', action='store_true', help='Write question scores without aggregate reports')
    args = ap.parse_args()

    exam_path = Path(args.exam_jsonl)
    responses_path = Path(args.responses)
    out_path = Path(args.out) if args.out else (responses_path.parent / "smoke_summary.json")
    per_row_path = Path(args.per_row_out) if args.per_row_out else (responses_path.parent / "score_per_row.jsonl")

    gold_rows = load_jsonl(exam_path)
    responses = load_responses(responses_path)
    by_uid = {str(row["uid"]): row for row in gold_rows}

    total = len(by_uid)
    correct = 0
    unknown = 0
    model_name: str | None = None
    by_variant: dict[str, dict[str, int]] = {}
    by_family: dict[str, dict[str, int]] = {}
    by_difficulty: dict[str, dict[str, int]] = {}
    by_source: dict[str, dict[str, int]] = {}
    by_framing: dict[str, dict[str, int]] = {}
    per_row: list[dict[str, Any]] = []

    for uid, gold in by_uid.items():
        resp = responses.get(uid)
        gold_answer = str(gold["answer"])
        row_result: dict[str, Any] = {
            "uid": uid,
            "family": gold.get("family"),
            "template_id": gold.get("template_id"),
            "variant": gold.get("variant"),
            "difficulty": gold.get("difficulty"),
            "framing": gold.get("framing"),
            "source_bucket": bucket_from_source(str(gold.get("source_summary", ""))),
            "gold": gold_answer,
            "pred": None,
            "correct": False,
            "unknown": True,
            "raw": None,
        }

        if resp is not None:
            meta = resp.get("meta", {}) if isinstance(resp.get("meta"), dict) else {}
            model_name = model_name or meta.get("model")
            raw = str(resp.get("raw", resp.get("pred", "")))
            row_result["raw"] = raw
            pred, _ = parse_verdict(str(resp.get("pred", raw)))
            if raw.startswith(API_ERROR_PREFIX) or meta.get("ok") is False:
                pred = None
            row_result["pred"] = pred
            row_result["unknown"] = pred is None
            row_result["correct"] = pred == gold_answer

        if row_result["correct"]:
            correct += 1
        if row_result["unknown"]:
            unknown += 1

        update_bucket(by_variant, str(gold.get("variant")), row_result["correct"], row_result["unknown"])
        update_bucket(by_family, str(gold.get("family")), row_result["correct"], row_result["unknown"])
        update_bucket(by_difficulty, str(gold.get("difficulty")), row_result["correct"], row_result["unknown"])
        if gold.get("framing") is not None:
            update_bucket(by_framing, str(gold.get("framing")), row_result["correct"], row_result["unknown"])
        update_bucket(
            by_source,
            bucket_from_source(str(gold.get("source_summary", ""))),
            row_result["correct"],
            row_result["unknown"],
        )
        per_row.append(row_result)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with per_row_path.open("w", encoding="utf-8") as f:
        for row in per_row:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.per_row_only:
        return
    report = {
        "total": total,
        "correct": correct,
        "accuracy": pct(correct, total),
        "unknown": unknown,
        "by_variant": by_variant,
        "by_family": by_family,
        "by_difficulty": by_difficulty,
        "by_source": by_source,
        "by_framing": by_framing,
        "response_path": str(responses_path),
        "per_row_path": str(per_row_path),
        "model": model_name,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
