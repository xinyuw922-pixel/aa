#!/usr/bin/env python3
"""Build a filtered Tier-1 MiniGrid triplet bank from oversampled candidates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.minigrid_similarity import image_ahash, selected_frame_indices, stable_hash
from utils.json_io import load_jsonl
from tools.minigrid.environment_registry import TASKS


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("GRIDWM_REPORT_ROOT", str(ROOT / "rebuttal_packet/10_minigrid_10x_expansion"))).resolve()

STABLE_ENV_ID_POOLS: dict[str, list[str]] = {
    "memory": [
        "MiniGrid-MemoryS7-v0",
        "MiniGrid-MemoryS9-v0",
        "MiniGrid-MemoryS11-v0",
        "MiniGrid-MemoryS13-v0",
        "MiniGrid-MemoryS13Random-v0",
        "MiniGrid-MemoryS17Random-v0",
    ],
}

EXPERIMENTAL_ENV_ID_POOLS: dict[str, list[str]] = {
    # Existing LavaGap generator is validated on S7. S5/S6 render-smoke, but
    # candidate generation can stall because the current semantic gates are
    # tuned for S7.
    "lavagap": [
        "MiniGrid-LavaGapS5-v0",
        "MiniGrid-LavaGapS6-v0",
        "MiniGrid-LavaGapS7-v0",
    ],
}


def run_cmd(cmd: list[str], timeout_s: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": -1,
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "timeout": True,
        }


from utils.minigrid_groups import group_rows


def summarize_candidate(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {str(row.get("variant")): row for row in rows}
    full = by_variant.get("full") or rows[0]
    frames = list(full.get("frames", []))
    frame_hashes: list[int] = []
    for idx in selected_frame_indices(frames):
        ih = image_ahash(root / frames[idx])
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
        "group_id": str(full.get("group_id", "")),
        "root": root,
        "seed": full.get("seed"),
        "complete_triplet": sorted(by_variant) == ["cf", "full", "nocue"],
        "rows": rows,
        "actions": actions,
        "action_signature": stable_hash(actions),
        "state_signature": stable_hash(state_signature),
        "image_signature": stable_hash([f"{x:016x}" for x in frame_hashes]),
        "frame_hashes": frame_hashes,
    }


def with_env_id_args(gen_args: list[str], env_id: str) -> list[str]:
    args = list(gen_args)
    for flag in ("--env-id", "--env_id"):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                args[idx + 1] = env_id
                return args
    return [*args, "--env-id", env_id]


def env_slug(env_id: str) -> str:
    return (
        env_id.replace("MiniGrid-", "")
        .replace("-v0", "")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


from utils.minigrid_similarity import normalized_action_distance


from utils.minigrid_similarity import mean_image_hamming


def select_diverse(
    candidates: list[dict[str, Any]],
    target: int,
    near_action_threshold: float,
    near_image_mean_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
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
                "seed": cand["seed"],
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


def rewrite_row_group(row: dict[str, Any], old_group_id: str, new_group_id: str) -> dict[str, Any]:
    out = json.loads(json.dumps(row))
    out["group_id"] = new_group_id
    out["source_group_id"] = old_group_id
    frames = []
    for frame in out.get("frames", []):
        parts = str(frame).split("/", 1)
        if parts and parts[0] == old_group_id and len(parts) == 2:
            frames.append(f"{new_group_id}/{parts[1]}")
        else:
            frames.append(str(frame).replace(old_group_id, new_group_id, 1))
    out["frames"] = frames
    return out


def write_selected_env(dst_root: Path, selected: list[dict[str, Any]]) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    with (dst_root / "triplets.jsonl").open("w", encoding="utf-8") as f:
        for idx, cand in enumerate(selected):
            old_group_id = cand["group_id"]
            new_group_id = f"{cand['env_key']}_sel{idx:06d}"
            src_group = cand["root"] / old_group_id
            dst_group = dst_root / new_group_id
            if src_group.exists():
                shutil.copytree(src_group, dst_group)
            for row in cand["rows"]:
                f.write(json.dumps(rewrite_row_group(row, old_group_id, new_group_id), ensure_ascii=False) + "\n")


def summarize_output(root: Path) -> dict[str, Any]:
    rows = load_jsonl(root / "triplets.jsonl") if (root / "triplets.jsonl").exists() else []
    grouped = group_rows(rows)
    png_count = sum(1 for _ in root.rglob("*.png"))
    return {
        "rows": len(rows),
        "groups": len(grouped),
        "complete_groups": sum(
            1 for group_rows_ in grouped.values()
            if {str(row.get("variant")) for row in group_rows_} == {"full", "nocue", "cf"}
        ),
        "png_count": png_count,
    }


def render_md(report: dict[str, Any]) -> str:
    rows = []
    for item in report["tasks"]:
        reason_counts = Counter(row["reason"] for row in item["rejected"])
        reason_text = ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))
        rows.append(
            "| {env_key} | {pool_groups} | {selected_groups} | {complete_groups} | {validation} | {rejected_count} | {reasons} |".format(
                env_key=item["env_key"],
                pool_groups=item["pool_summary"]["groups"],
                selected_groups=item["output_summary"]["groups"],
                complete_groups=item["output_summary"]["complete_groups"],
                validation="PASS" if item["validation"]["returncode"] == 0 else "FAIL",
                rejected_count=len(item["rejected"]),
                reasons=reason_text,
            )
        )
    all_pass = all(
        item["validation"]["returncode"] == 0 and item["output_summary"]["groups"] >= report["target_per_env"]
        for item in report["tasks"]
    )
    return "\n".join([
        "# Tier-1 MiniGrid Diverse Bank Report",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        f"- Target per env: `{report['target_per_env']}`",
        f"- Pool per env: `{report['pool_per_env']}`",
        f"- Output root: `{report['output_root']}`",
        f"- Candidate pool root: `{report['pool_root']}`",
        f"- Overall status: `{'PASS' if all_pass else 'PARTIAL'}`",
        "",
        "| Env | Pool groups | Selected groups | Complete groups | Validate | Rejected before target | Main reject reasons |",
        "| --- | ---: | ---: | ---: | --- | ---: | --- |",
        *rows,
        "",
        "Selection rejects exact action/state/image duplicates, near action duplicates, and near image duplicates before accepting a group.",
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-per-env", type=int, default=10)
    ap.add_argument("--pool-per-env", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=200000)
    ap.add_argument("--pool-root", default="tmp/minigrid_10x_tier1_diverse_pool")
    ap.add_argument("--out-root", default="tmp/minigrid_10x_tier1_diverse_bank")
    ap.add_argument("--report-stem", default="tier1_diverse_bank_report")
    ap.add_argument("--near-action-threshold", type=float, default=0.15)
    ap.add_argument("--near-image-mean-threshold", type=float, default=4.0)
    ap.add_argument("--use-experimental-env-id-pools", action="store_true")
    ap.add_argument("--timeout-generate-s", type=int, default=1800)
    ap.add_argument("--timeout-validate-s", type=int, default=600)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_root = Path(args.pool_root)
    out_root = Path(args.out_root)
    if not pool_root.is_absolute():
        pool_root = ROOT / pool_root
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    for path in (pool_root, out_root):
        if path.exists():
            raise FileExistsError(f"Existing intermediate output is preserved: {path}; choose a fresh directory")
    for path in (pool_root, out_root):
        path.mkdir(parents=True, exist_ok=True)

    tasks = []
    for spec in TASKS:
        env_key = spec["env_key"]
        env_id_pools = dict(STABLE_ENV_ID_POOLS)
        if args.use_experimental_env_id_pools:
            env_id_pools.update(EXPERIMENTAL_ENV_ID_POOLS)
        env_ids = env_id_pools.get(env_key, [])
        if not env_ids:
            env_id = ""
            raw_args = list(spec["gen_args"])
            if "--env-id" in raw_args:
                env_id = raw_args[raw_args.index("--env-id") + 1]
            elif "--env_id" in raw_args:
                env_id = raw_args[raw_args.index("--env_id") + 1]
            env_ids = [env_id] if env_id else [env_key]
        dst_root = out_root / env_key
        candidates = []
        pool_generations = []
        per_env_counts = []
        base_count = args.pool_per_env // len(env_ids)
        remainder = args.pool_per_env % len(env_ids)
        for env_idx, env_id in enumerate(env_ids):
            count = base_count + (1 if env_idx < remainder else 0)
            if count <= 0:
                continue
            src_root = pool_root / env_key / env_slug(env_id)
            gen_cmd = [
                sys.executable,
                spec["module"],
                "generate",
                "--out-dir",
                str(src_root),
                "--seed-start",
                str(int(spec["seed_start"]) + int(args.seed_offset) + env_idx * 100000),
                "--num",
                str(count),
                *with_env_id_args(spec["gen_args"], env_id),
            ]
            pool_generation = run_cmd(gen_cmd, timeout_s=args.timeout_generate_s)
            pool_generations.append(pool_generation)
            rows = load_jsonl(src_root / "triplets.jsonl") if (src_root / "triplets.jsonl").exists() else []
            env_candidates = [
                summarize_candidate(src_root, group_rows_)
                for group_rows_ in group_rows(rows).values()
            ]
            for cand in env_candidates:
                cand["env_key"] = env_key
            candidates.extend(env_candidates)
            per_env_counts.append({"env_id": env_id, "requested": count, "groups": len(env_candidates)})
        selected, rejected = select_diverse(
            candidates,
            target=args.target_per_env,
            near_action_threshold=args.near_action_threshold,
            near_image_mean_threshold=args.near_image_mean_threshold,
        )
        write_selected_env(dst_root, selected)
        val_cmd = [
            sys.executable,
            spec["module"],
            "validate",
            "--root",
            str(dst_root),
            *spec["validate_args"],
        ]
        validation = run_cmd(val_cmd, timeout_s=args.timeout_validate_s) if selected else {
            "cmd": val_cmd,
            "returncode": -1,
            "stdout_tail": "",
            "stderr_tail": "skipped because no selected groups",
            "timeout": False,
        }
        tasks.append({
            "env_key": env_key,
            "pool_generations": pool_generations,
            "validation": validation,
            "pool_root": str(pool_root / env_key),
            "output_root": str(dst_root),
            "pool_summary": {
                "groups": len(candidates),
                "complete_groups": sum(1 for cand in candidates if cand["complete_triplet"]),
                "per_env_id": per_env_counts,
            },
            "output_summary": summarize_output(dst_root),
            "selected": [
                {"group_id": row["group_id"], "seed": row["seed"]}
                for row in selected
            ],
            "rejected": rejected,
        })

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "target_per_env": args.target_per_env,
        "pool_per_env": args.pool_per_env,
        "seed_offset": args.seed_offset,
        "near_action_threshold": args.near_action_threshold,
        "near_image_mean_threshold": args.near_image_mean_threshold,
        "pool_root": str(pool_root),
        "output_root": str(out_root),
        "tasks": tasks,
    }
    json_path = OUT_DIR / f"{args.report_stem}.json"
    md_path = OUT_DIR / f"{args.report_stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    print(json.dumps({
        "tasks": len(tasks),
        "target_per_env": args.target_per_env,
        "selected_total": sum(item["output_summary"]["groups"] for item in tasks),
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
