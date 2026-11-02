#!/usr/bin/env python3
"""Select a low-duplicate MiniGrid triplet bank from a generated pool."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.minigrid_similarity import image_ahash, selected_frame_indices, stable_hash
from utils.json_io import load_jsonl


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("GRIDWM_REPORT_ROOT", str(ROOT / "rebuttal_packet/10_minigrid_10x_expansion"))).resolve()


from utils.minigrid_groups import group_rows


def summarize_candidate(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {str(row.get("variant")): row for row in rows}
    full = by_variant.get("full") or rows[0]
    frames = list(full.get("frames", []))
    frame_hashes = []
    for idx in selected_frame_indices(frames):
        fp = root / frames[idx]
        ih = image_ahash(fp)
        if ih is not None:
            frame_hashes.append(ih)
    states = list(full.get("state_seq", []))
    state_signature = {
        "mission": full.get("mission", ""),
        "start": states[0] if states else None,
        "end": states[-1] if states else None,
        "objects": states[-1].get("objects", []) if states else [],
    }
    actions = tuple(int(x) for x in full.get("actions_id", []))
    return {
        "env_key": str(full.get("task") or full.get("env_task") or "unknown"),
        "group_id": str(full.get("group_id", "")),
        "rows": rows,
        "actions": actions,
        "action_signature": stable_hash(actions),
        "state_signature": stable_hash(state_signature),
        "image_signature": stable_hash([f"{x:016x}" for x in frame_hashes]),
        "frame_hashes": frame_hashes,
        "complete_triplet": sorted(by_variant) == ["cf", "full", "nocue"],
    }


from utils.minigrid_similarity import normalized_action_distance


from utils.minigrid_similarity import mean_image_hamming


def select_env(
    candidates: list[dict[str, Any]],
    target: int,
    near_action_threshold: float,
    near_image_mean_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    rejected = []
    seen_action = set()
    seen_state = set()
    seen_image = set()
    for cand in candidates:
        reason = ""
        nearest_group = ""
        nearest_value = ""
        if not cand["complete_triplet"]:
            reason = "incomplete_triplet"
        elif cand["action_signature"] in seen_action:
            reason = "exact_action_duplicate"
        elif cand["state_signature"] in seen_state:
            reason = "exact_state_duplicate"
        elif cand["image_signature"] in seen_image:
            reason = "exact_image_duplicate"
        else:
            for prev in selected:
                action_dist = normalized_action_distance(cand, prev)
                if action_dist < near_action_threshold:
                    reason = "near_action_duplicate"
                    nearest_group = prev["group_id"]
                    nearest_value = f"{action_dist:.4f}"
                    break
                image_dist = mean_image_hamming(cand, prev)
                if image_dist is not None and image_dist <= near_image_mean_threshold:
                    reason = "near_image_duplicate"
                    nearest_group = prev["group_id"]
                    nearest_value = f"{image_dist:.2f}"
                    break
        if reason:
            rejected.append({
                "group_id": cand["group_id"],
                "reason": reason,
                "nearest_group": nearest_group,
                "nearest_value": nearest_value,
            })
            continue
        selected.append(cand)
        seen_action.add(cand["action_signature"])
        seen_state.add(cand["state_signature"])
        seen_image.add(cand["image_signature"])
        if len(selected) >= target:
            break
    return selected, rejected


def copy_group(src_root: Path, dst_root: Path, group_id: str) -> None:
    src_group = src_root / group_id
    dst_group = dst_root / group_id
    if src_group.exists():
        shutil.copytree(src_group, dst_group)


def render_md(report: dict[str, Any]) -> str:
    rows = []
    for item in report["envs"]:
        reason_counts = Counter(row["reason"] for row in item["rejected"])
        reason_text = ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))
        rows.append(
            f"| `{item['env_key']}` | {item['pool_groups']} | {item['selected_groups']} | {len(item['rejected'])} | {reason_text} |"
        )
    return "\n".join([
        "# MiniGrid Triplet Bank Selection Report",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        f"- Source root: `{report['source_root']}`",
        f"- Output root: `{report['output_root']}`",
        f"- Target per env: `{report['target_per_env']}`",
        f"- Selected total: `{report['selected_total']}`",
        "",
        "| Env | Pool groups | Selected groups | Rejected before/while selecting | Reject reasons |",
        "| --- | ---: | ---: | ---: | --- |",
        *rows,
        "",
        "Selection rejects exact action/state/image duplicates, near action duplicates, and near image duplicates.",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--target-per-env", type=int, default=10)
    ap.add_argument("--near-action-threshold", type=float, default=0.15)
    ap.add_argument("--near-image-mean-threshold", type=float, default=4.0)
    ap.add_argument("--report-stem", default="triplet_bank_selection_report")
    args = ap.parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    if not source_root.is_absolute():
        source_root = (ROOT / source_root).resolve()
    if not out_root.is_absolute():
        out_root = (ROOT / out_root).resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(source_root / "triplets.jsonl")
    candidates = [
        summarize_candidate(source_root, group)
        for group in group_rows(rows).values()
    ]
    by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        by_env[cand["env_key"]].append(cand)

    selected_all = []
    env_reports = []
    for env_key, env_candidates in sorted(by_env.items()):
        selected, rejected = select_env(
            env_candidates,
            target=args.target_per_env,
            near_action_threshold=args.near_action_threshold,
            near_image_mean_threshold=args.near_image_mean_threshold,
        )
        selected_all.extend(selected)
        env_reports.append({
            "env_key": env_key,
            "pool_groups": len(env_candidates),
            "selected_groups": len(selected),
            "selected_group_ids": [row["group_id"] for row in selected],
            "rejected": rejected,
        })

    with (out_root / "triplets.jsonl").open("w", encoding="utf-8") as f:
        for cand in selected_all:
            copy_group(source_root, out_root, cand["group_id"])
            for row in cand["rows"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(out_root),
        "target_per_env": args.target_per_env,
        "selected_total": len(selected_all),
        "envs": env_reports,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{args.report_stem}.json"
    md_path = OUT_DIR / f"{args.report_stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    print(json.dumps({
        "selected_total": len(selected_all),
        "json": str(json_path),
        "markdown": str(md_path),
        "out_root": str(out_root),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
