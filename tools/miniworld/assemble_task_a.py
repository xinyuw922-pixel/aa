#!/usr/bin/env python3
"""Assemble environment-specific Task A packs into one cross-environment exam."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


from utils.miniworld_io import load_jsonl


def copy_image(src_root: Path, image_ref: str, dst_dir: Path, prefix: str) -> str:
    src = Path(image_ref)
    if not src.is_absolute():
        src = src_root / image_ref
    if not src.exists():
        raise FileNotFoundError(f"Missing image: {src}")
    dst_name = f"{prefix}_{src.name}"
    dst = dst_dir / dst_name
    shutil.copy2(src, dst)
    return f"images/{dst_name}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge A-MW-v2 env-specific packs into one crossenv pack")
    ap.add_argument(
        "--fourrooms_jsonl",
        default="experiments/generated/miniworld/taskA-MW-v2-candidate-50-balanced/taskA-MW-v2-prototype.jsonl",
    )
    ap.add_argument(
        "--pickupobjects_jsonl",
        default="experiments/generated/miniworld/taskA-MW-v2-PickupObjects/taskA-MW-v2-pickupobjects.jsonl",
    )
    ap.add_argument(
        "--putnext_jsonl",
        default="experiments/generated/miniworld/taskA-MW-v2-PutNext/taskA-MW-v2-putnext.jsonl",
    )
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskA-MW-v2-crossenv",
    )
    ap.add_argument("--shuffle_seed", type=int, default=20260408)
    args = ap.parse_args()

    sources = [
        ("fourrooms", Path(args.fourrooms_jsonl)),
        ("pickupobjects", Path(args.pickupobjects_jsonl)),
        ("putnext", Path(args.putnext_jsonl)),
    ]

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    for env_name, jsonl_path in sources:
        rows = load_jsonl(jsonl_path)
        src_root = jsonl_path.parent
        for row in rows:
            uid = str(row["uid"])
            prefix = uid.replace(".", "_")
            new_row = dict(row)
            new_row["source_pack_relpath"] = str(jsonl_path)
            new_row["current_image_rel_path"] = copy_image(
                src_root, str(row["current_image_rel_path"]), img_dir, f"{prefix}_current"
            )
            new_row["candidate_a_rel_path"] = copy_image(
                src_root, str(row["candidate_a_rel_path"]), img_dir, f"{prefix}_candA"
            )
            new_row["candidate_b_rel_path"] = copy_image(
                src_root, str(row["candidate_b_rel_path"]), img_dir, f"{prefix}_candB"
            )
            merged_rows.append(new_row)
            source_counts[env_name] += 1

    rng = random.Random(args.shuffle_seed)
    rng.shuffle(merged_rows)
    for idx, row in enumerate(merged_rows, start=1):
        row["crossenv_index"] = idx

    exam_path = out_dir / "taskA-MW-v2-crossenv.jsonl"
    with exam_path.open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    compat_path = out_dir / "task_a_exam.jsonl"
    shutil.copy2(exam_path, compat_path)

    manifest = {
        "schema_version": "gridwm.mw_exam.v2",
        "task": "A-MW-v2-crossenv",
        "source_jsonls": {name: str(path) for name, path in sources},
        "shuffle_seed": args.shuffle_seed,
        "records_count": len(merged_rows),
        "source_counts": dict(source_counts),
        "records_relpath": exam_path.name,
        "compat_exam_relpath": compat_path.name,
    }
    (out_dir / "taskA-MW-v2-crossenv_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
