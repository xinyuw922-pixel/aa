#!/usr/bin/env python3
"""Balance Task A candidate labels without changing semantic branch metadata."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


from utils.miniworld_io import load_jsonl


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def swap_ab_presentation(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["candidate_a_rel_path"], out["candidate_b_rel_path"] = (
        row["candidate_b_rel_path"],
        row["candidate_a_rel_path"],
    )

    out["correct_candidate"] = "A" if row["correct_candidate"] == "B" else "B"
    out["ground_truth"] = out["correct_candidate"]
    out["label"] = 0 if out["correct_candidate"] == "A" else 1

    # Semantic correct/distractor poses do not depend on A/B presentation order.

    note = str(row.get("note", "")).strip()
    suffix = "ab_label_rebalanced_v1"
    out["note"] = f"{note};{suffix}" if note else suffix
    return out


def rewrite_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records_count"] = len(rows)
    manifest["ab_balance"] = dict(Counter(r["correct_candidate"] for r in rows))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebalance A/B label presentation for Task A-MW v2 exam set")
    ap.add_argument("--src_dir", required=True, help="Source exam directory")
    ap.add_argument("--dst_dir", required=True, help="Destination exam directory")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    jsonl_name = "taskA-MW-v2-prototype.jsonl"
    manifest_name = "taskA-MW-v2-prototype_manifest.json"

    if dst_dir.exists():
        raise SystemExit(f"Destination already exists: {dst_dir}")
    shutil.copytree(src_dir, dst_dir)

    jsonl_path = dst_dir / jsonl_name
    rows = load_jsonl(jsonl_path)
    counts = Counter(r["correct_candidate"] for r in rows)
    if not counts:
        raise SystemExit("No rows found")

    majority_label, majority_count = counts.most_common(1)[0]
    minority_label = "A" if majority_label == "B" else "B"
    minority_count = counts.get(minority_label, 0)
    if majority_count - minority_count <= 1:
        rewrite_manifest(dst_dir / manifest_name, rows)
        dump_jsonl(jsonl_path, rows)
        print(f"Already balanced enough: {counts}")
        return

    need_flip = (majority_count - minority_count) // 2
    flipped = 0
    balanced_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["correct_candidate"] == majority_label and flipped < need_flip:
            balanced_rows.append(swap_ab_presentation(row))
            flipped += 1
        else:
            balanced_rows.append(row)

    dump_jsonl(jsonl_path, balanced_rows)
    rewrite_manifest(dst_dir / manifest_name, balanced_rows)
    print(
        f"Rebalanced {dst_dir}: flipped={flipped}, "
        f"before={dict(counts)}, after={dict(Counter(r['correct_candidate'] for r in balanced_rows))}"
    )


if __name__ == "__main__":
    main()
