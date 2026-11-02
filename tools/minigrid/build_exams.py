#!/usr/bin/env python3
"""Build Task A/C/D manifests from a generated MiniGrid raw triplet bank."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any




from tools.minigrid.task_a import build_task_a
from tools.minigrid.task_c import build_base_task_c as build_task_c
from tools.minigrid.task_d import build_base_task_d as build_task_d

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "rebuttal_packet/10_minigrid_10x_expansion"
SCHEMA_VERSION = "gridwm.exam.v1"

from utils.minigrid_panels import FONT_PATHS
ACTION_NAMES = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


from utils.json_io import write_jsonl_utf8 as write_jsonl


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()




























def validate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "env_counts": dict(Counter(str(row.get("env_task")) for row in rows)),
        "answer_counts": dict(Counter(str(row.get("answer")) for row in rows)),
        "variant_counts": dict(Counter(str(row.get("variant", "")) for row in rows)),
        "unique_uids": len({row["uid"] for row in rows}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-dir", default="rebuttal_packet/10_minigrid_10x_expansion/manifests_scale1_v1")
    ap.add_argument("--task-a-target", type=int, default=1030)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    raw_root = resolve(args.raw_root)
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(raw_root / "triplets.jsonl")
    task_c = build_task_c(raw_root, rows, out_dir)
    task_d = build_task_d(task_c, out_dir)
    task_a = build_task_a(raw_root, rows, out_dir, target_rows=args.task_a_target)

    write_jsonl(out_dir / "taskA_expanded_manifest.jsonl", task_a)
    write_jsonl(out_dir / "taskC_expanded_manifest.jsonl", task_c)
    write_jsonl(out_dir / "taskD_expanded_manifest.jsonl", task_d)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "minigrid_10x_acd_from_raw_bank",
        "raw_root": str(raw_root),
        "out_dir": str(out_dir),
        "task_a_target": args.task_a_target,
        "tasks": {
            "A": validate_summary(task_a),
            "C": validate_summary(task_c),
            "D": validate_summary(task_d),
        },
        "notes": [
            "Task C uses one storyboard per raw full/nocue/cf row.",
            "Task D uses the first successful full storyboard per environment as reference.",
            "Task A uses adjacent full-trajectory frame transitions with same-env distractors.",
        ],
    }
    (out_dir / "manifest_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["tasks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
