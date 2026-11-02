#!/usr/bin/env python3
"""Audit duplicate and near-duplicate signals in generated MiniGrid triplets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any



ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("GRIDWM_REPORT_ROOT", str(ROOT / "rebuttal_packet/10_minigrid_10x_expansion"))).resolve()


from utils.minigrid_io import load_jsonl


from utils.minigrid_similarity import stable_hash


from utils.minigrid_similarity import image_ahash


from utils.minigrid_similarity import hamming


from utils.minigrid_similarity import levenshtein


from utils.minigrid_similarity import selected_frame_indices


def summarize_group(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {str(row.get("variant")): row for row in rows}
    full = by_variant.get("full") or rows[0]
    frames = list(full.get("frames", []))
    frame_hashes = []
    missing_frames = 0
    for idx in selected_frame_indices(frames):
        fp = root / frames[idx]
        ih = image_ahash(fp)
        if ih is None:
            missing_frames += 1
        else:
            frame_hashes.append(f"{ih:016x}")

    states = list(full.get("state_seq", []))
    state_signature = {
        "mission": full.get("mission", ""),
        "start": states[0] if states else None,
        "end": states[-1] if states else None,
        "objects": states[-1].get("objects", []) if states else [],
    }
    actions = tuple(int(x) for x in full.get("actions_id", []))
    return {
        "env_key": str(full.get("task") or root.name),
        "env_id": str(full.get("env_id", "")),
        "group_id": str(full.get("group_id", "")),
        "seed": full.get("seed"),
        "variants": sorted(by_variant),
        "complete_triplet": sorted(by_variant) == ["cf", "full", "nocue"],
        "action_len": len(actions),
        "action_signature": stable_hash(actions),
        "actions": actions,
        "state_signature": stable_hash(state_signature),
        "image_signature": stable_hash(frame_hashes),
        "selected_frame_hashes": frame_hashes,
        "missing_selected_frames": missing_frames,
    }


def duplicate_count(values: list[str]) -> int:
    counts = Counter(values)
    return sum(count - 1 for count in counts.values() if count > 1)


def audit_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_env[group["env_key"]].append(group)

    env_rows = []
    near_rows = []
    for env_key, env_groups in sorted(by_env.items()):
        exact_action_dups = duplicate_count([g["action_signature"] for g in env_groups])
        exact_state_dups = duplicate_count([g["state_signature"] for g in env_groups])
        exact_image_dups = duplicate_count([g["image_signature"] for g in env_groups])
        near_action_pairs = 0
        near_image_mean_pairs = 0
        min_action_distance = None
        min_image_mean_hamming = None
        for i, left in enumerate(env_groups):
            for right in env_groups[i + 1 :]:
                max_len = max(len(left["actions"]), len(right["actions"]), 1)
                action_dist = levenshtein(left["actions"], right["actions"]) / max_len
                min_action_distance = action_dist if min_action_distance is None else min(min_action_distance, action_dist)
                if action_dist < 0.15:
                    near_action_pairs += 1
                    near_rows.append({
                        "env_key": env_key,
                        "signal": "action",
                        "left_group": left["group_id"],
                        "right_group": right["group_id"],
                        "distance": f"{action_dist:.4f}",
                    })
                hams = []
                for lh, rh in zip(left["selected_frame_hashes"], right["selected_frame_hashes"]):
                    hams.append(hamming(int(lh, 16), int(rh, 16)))
                if hams:
                    img_dist = sum(hams) / len(hams)
                    min_image_mean_hamming = img_dist if min_image_mean_hamming is None else min(min_image_mean_hamming, img_dist)
                    if img_dist <= 4:
                        near_image_mean_pairs += 1
                        near_rows.append({
                            "env_key": env_key,
                            "signal": "image_ahash_mean",
                            "left_group": left["group_id"],
                            "right_group": right["group_id"],
                            "distance": f"{img_dist:.2f}",
                        })
        env_rows.append({
            "env_key": env_key,
            "groups": len(env_groups),
            "complete_groups": sum(1 for g in env_groups if g["complete_triplet"]),
            "exact_action_duplicates": exact_action_dups,
            "exact_state_duplicates": exact_state_dups,
            "exact_image_duplicates": exact_image_dups,
            "near_action_pairs_lt_0_15": near_action_pairs,
            "near_image_mean_pairs_le_4": near_image_mean_pairs,
            "min_action_distance": "" if min_action_distance is None else f"{min_action_distance:.4f}",
            "min_image_mean_hamming": "" if min_image_mean_hamming is None else f"{min_image_mean_hamming:.2f}",
            "missing_selected_frames": sum(int(g["missing_selected_frames"]) for g in env_groups),
        })
    return {"env_rows": env_rows, "near_rows": near_rows}


def render_md(report: dict[str, Any]) -> str:
    rows = [
        "| {env_key} | {groups} | {complete_groups} | {exact_action_duplicates} | {exact_state_duplicates} | {exact_image_duplicates} | {near_action_pairs_lt_0_15} | {near_image_mean_pairs_le_4} | {min_action_distance} | {min_image_mean_hamming} |".format(**row)
        for row in report["env_rows"]
    ]
    return "\n".join([
        "# MiniGrid Triplet Similarity Audit",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        f"- Input roots: `{', '.join(report['input_roots'])}`",
        f"- Total groups: `{report['total_groups']}`",
        f"- Total near-pair warnings: `{len(report['near_rows'])}`",
        "",
        "| Env | Groups | Complete | Exact action dups | Exact state dups | Exact image dups | Near action pairs | Near image mean pairs | Min action dist | Min image mean ham |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Near action threshold: normalized Levenshtein distance `<0.15`.",
        "Near image threshold: mean selected-frame aHash Hamming distance `<=4`.",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="Generated env roots or parent root containing env subdirs.")
    ap.add_argument("--report-stem", default="triplet_similarity_audit")
    args = ap.parse_args()

    roots = []
    for raw in args.roots:
        path = Path(raw)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        if (path / "triplets.jsonl").exists():
            roots.append(path)
        else:
            roots.extend(sorted(p for p in path.iterdir() if (p / "triplets.jsonl").exists()))

    groups = []
    for root in roots:
        rows = load_jsonl(root / "triplets.jsonl")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("group_id"))].append(row)
        for group_rows in grouped.values():
            groups.append(summarize_group(root, group_rows))

    audit = audit_groups(groups)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_roots": [str(p) for p in roots],
        "total_groups": len(groups),
        "groups": [
            {k: v for k, v in group.items() if k != "actions"}
            for group in groups
        ],
        **audit,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{args.report_stem}.json"
    md_path = OUT_DIR / f"{args.report_stem}.md"
    csv_path = OUT_DIR / f"{args.report_stem}.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "env_key",
            "groups",
            "complete_groups",
            "exact_action_duplicates",
            "exact_state_duplicates",
            "exact_image_duplicates",
            "near_action_pairs_lt_0_15",
            "near_image_mean_pairs_le_4",
            "min_action_distance",
            "min_image_mean_hamming",
            "missing_selected_frames",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["env_rows"])
    print(json.dumps({
        "groups": len(groups),
        "near_warnings": len(report["near_rows"]),
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
