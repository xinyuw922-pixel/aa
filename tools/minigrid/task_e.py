#!/usr/bin/env python3
"""Build object identity/status questions at valid event checkpoints from
UnlockPickup and BlockedUnlockPickup trajectories."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys as _sys
_scripts_dir = Path(__file__).resolve().parent
_repo_root = _scripts_dir.parent.parent
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))
elif str(_scripts_dir) not in _sys.path:
    _sys.path.insert(0, str(_scripts_dir))



PROMOTED_ENVS = ("unlockpickup", "blockedunlockpickup")


import argparse


import json


import math


from pathlib import Path


from typing import List, Dict, Any, Optional, Tuple


from collections import defaultdict


import sys as _sys


_scripts_dir = Path(__file__).resolve().parent


_root = _scripts_dir.parent


from PIL import Image


SCHEMA_VERSION = "gridwm.exam.v1.2"


GENERATOR_VERSION = "task_e_v12_expansion"


IDENTITY_LABELS = [
    "key_yellow", "key_green", "key_purple",
    "key_red", "key_grey",
    "ball_yellow", "ball_green", "ball_blue", "ball_purple",
    "key_blue", "ball_grey", "ball_red",
]


STATUS_LABELS = [
    "holding",
    "dropped",
    "reacquired_and_holding",
]


HORIZON_SHORT = "short"    # 1–10


HORIZON_MEDIUM = "medium"  # 11–25


HORIZON_LONG = "long"     # 26+


HORIZON_BUCKETS = {
    (1, 10): HORIZON_SHORT,
    (11, 25): HORIZON_MEDIUM,
    (26, 9999): HORIZON_LONG,
}


IDENTITY_TEMPLATE = (
    "Below is a MiniGrid trajectory storyboard showing the trajectory up to the last shown frame only.\n"
    "Question: Earlier in the shown trajectory, what was the first object the agent picked up?\n"
    "Answer using JSON only: {{\"answer\":\"<one label from {label_list}>\"}}"
)


STATUS_TEMPLATE = (
    "Below is a MiniGrid trajectory storyboard showing the trajectory up to the last shown frame only.\n"
    "Question: By the last shown frame, what is the status of the first object the agent picked up earlier?\n"
    "Answer using JSON only: {{\"answer\":\"<one label from holding, dropped, reacquired_and_holding>\"}}"
)


def norm_carrying_label(carrying: Any) -> str:
    if carrying is None:
        return "none"
    if isinstance(carrying, dict):
        return f"{carrying.get('type', 'unknown')}_{carrying.get('color', 'unknown')}"
    return str(carrying)


def normalize_trace(state_seq: List[dict]) -> List[dict]:
    steps = []
    prev_carry = None
    for i, s in enumerate(state_seq):
        if not isinstance(s, dict):
            steps.append({"step": i, "raw": s})
            continue
        agent = s.get("agent", {})
        carrying = agent.get("carrying")
        carry_label = norm_carrying_label(carrying)
        steps.append({
            "step": i,
            "carrying": carrying,
            "carrying_label": carry_label,
            "carry_changed": (carrying != prev_carry),
        })
        prev_carry = carrying
    return steps


def extract_events(trace: List[dict]) -> Dict[str, Any]:
    pickups = []
    drops = []
    reacquires = []
    ever_held = set()
    prev_carry = None

    for st in trace:
        i = st["step"]
        carrying = st["carrying"]
        label = st["carrying_label"]

        if carrying is not None and prev_carry is None:
            if label in ever_held:
                reacquires.append((i, label))
            else:
                pickups.append((i, label))
                ever_held.add(label)
        elif carrying is None and prev_carry is not None:
            drops.append((i, norm_carrying_label(prev_carry)))
            ever_held.add(norm_carrying_label(prev_carry))

        prev_carry = carrying

    return {
        "pickups": pickups,
        "drops": drops,
        "reacquires": reacquires,
        "terminal_label": trace[-1]["carrying_label"] if trace else "none",
        "n_steps": len(trace),
    }


def classify_status(
    first_label: str,
    pickups: List[Tuple[int, str]],
    drops: List[Tuple[int, str]],
    reacquires: List[Tuple[int, str]],
    query_step: int,
) -> str:
    lp = [(s, l) for s, l in pickups if l == first_label]
    ld = [(s, l) for s, l in drops if l == first_label]
    lr = [(s, l) for s, l in reacquires if l == first_label]

    if not lp:
        return "dropped"

    last_pickup_before_q = max((s for s, l in lp if s <= query_step), default=None)
    last_drop_before_q = max((s for s, l in ld if s <= query_step), default=None)
    last_reacq_before_q = max((s for s, l in lr if s <= query_step), default=None)

    if last_reacq_before_q is not None:
        return "reacquired_and_holding"
    if last_pickup_before_q is not None:
        if last_drop_before_q is None or last_drop_before_q < last_pickup_before_q:
            return "holding"
        if last_drop_before_q > last_pickup_before_q:
            return "dropped"
    return "dropped"


def assign_horizon_bucket(gap: int) -> str:
    for (lo, hi), bucket in HORIZON_BUCKETS.items():
        if lo <= gap <= hi:
            return bucket
    return HORIZON_LONG


def pick_identity_steps(
    first_pickup_step: int,
    n_steps: int,
) -> Dict[str, Optional[int]]:
    """
    Pick canonical query_step for each identity role.
    Returns {role: query_step_or_None}.
    """
    # Short: earliest step in short bucket
    short_candidates = [
        qs for qs in range(first_pickup_step + 1, min(first_pickup_step + 11, n_steps))
        if qs - first_pickup_step <= 10
    ]
    # Medium: earliest step in medium bucket
    medium_candidates = [
        qs for qs in range(first_pickup_step + 11, min(first_pickup_step + 26, n_steps))
        if 11 <= qs - first_pickup_step <= 25
    ]
    # Long: earliest step in long bucket
    long_candidates = [
        qs for qs in range(first_pickup_step + 26, n_steps)
        if qs - first_pickup_step >= 26
    ]
    return {
        "identity_short": short_candidates[0] if short_candidates else None,
        "identity_medium": medium_candidates[0] if medium_candidates else None,
        "identity_long": long_candidates[0] if long_candidates else None,
    }


def pick_status_steps(
    first_pickup_step: int,
    first_drop_step: Optional[int],
    n_steps: int,
) -> Dict[str, Optional[int]]:
    """
    Pick canonical query_step for each status role.
    """
    results = {}

    # early_holding: 1-3 steps after first pickup
    early_cands = [qs for qs in range(first_pickup_step + 1, min(first_pickup_step + 4, n_steps))]
    results["status_early_holding"] = early_cands[0] if early_cands else None

    # mid_holding: roughly halfway between pickup and drop (or pickup+15 if no drop)
    if first_drop_step is not None and first_drop_step > first_pickup_step + 5:
        mid = (first_pickup_step + first_drop_step) // 2
        results["status_mid_holding"] = mid
    else:
        # No clear drop gap; pick around step +15
        results["status_mid_holding"] = min(first_pickup_step + 15, n_steps - 1)

    # first_post_drop: earliest step strictly after the drop
    if first_drop_step is not None:
        post_drop = first_drop_step + 1
        results["status_first_post_drop"] = post_drop if post_drop < n_steps else None
        # late_post_drop: use n-1 (last frame), but if n-1 == post_drop (rare),
        # use n-2 (second-to-last). n-1 always shows the final state.
        late = n_steps - 1
        if late == post_drop:
            late = n_steps - 2
        results["status_late_post_drop"] = late if late > first_drop_step and late < n_steps else None
    else:
        # Never dropped — no post-drop roles
        results["status_first_post_drop"] = None
        results["status_late_post_drop"] = None

    return results


def select_prefix_frames(
    query_step: int,
    first_pickup_step: int,
    k_min: int = 5,
    k_max: int = 12,
) -> List[int]:
    n = query_step + 1
    selected = set()
    selected.add(0)
    selected.add(query_step)
    if query_step >= 1:
        selected.add(query_step - 1)
    for dx in (-1, 0, 1):
        s = max(0, min(query_step, first_pickup_step + dx))
        selected.add(s)
    sorted_steps = sorted(selected)
    for i in range(len(sorted_steps) - 1):
        gap = sorted_steps[i + 1] - sorted_steps[i]
        if gap > 3:
            mid = (sorted_steps[i] + sorted_steps[i + 1]) // 2
            selected.add(mid)
    result = sorted(selected)
    if len(result) > k_max:
        return result  # caller must check and reject if > k_max
    if len(result) < k_min and n > len(result):
        candidates = []
        if result[0] > 0:
            candidates.append(result[0] - 1)
        if result[-1] < query_step:
            candidates.append(result[-1] + 1)
        for c in candidates:
            if len(result) < k_min and c not in result and 0 <= c <= query_step:
                result.append(c)
        result = sorted(result)
    return result


def compute_layout(n_frames: int) -> Dict[str, Any]:
    if n_frames <= 1:
        return {"rows": 1, "cols": 1, "n_cells": 1, "n_empty": 0}
    best_r, best_c = 1, n_frames
    best_score = float("inf")
    for r in range(1, n_frames + 1):
        c = (n_frames + r - 1) // r
        cells = r * c
        n_empty = cells - n_frames
        aspect_ratio = c / r if r > 0 else float("inf")
        score = abs(math.log(aspect_ratio)) * 2 + n_empty if aspect_ratio > 0 else float("inf")
        if score < best_score:
            best_score = score
            best_r, best_c = r, c
    if best_c < best_r:
        best_r, best_c = best_c, best_r
    return {
        "rows": best_r,
        "cols": best_c,
        "n_cells": best_r * best_c,
        "n_empty": best_r * best_c - n_frames,
    }


def load_frame(env_dir: Path, frame_rel: str) -> Image.Image:
    path = env_dir / frame_rel
    if not path.exists():
        raise FileNotFoundError(f"Missing frame: {path}")
    return Image.open(path).convert("RGB")


def render_montage(
    env_dir: Path,
    frames: List[str],
    selected_steps: List[int],
    layout: Dict[str, Any],
    out_path: Path,
) -> Image.Image:
    images = [load_frame(env_dir, frames[i]) for i in selected_steps]
    if not images:
        raise ValueError(f"No frames for montage: {out_path}")
    w, h = images[0].size
    rows, cols = layout.get("rows", 1), layout.get("cols", len(images))
    canvas = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
    for idx, img in enumerate(images):
        x = (idx % cols) * w
        y = (idx // cols) * h
        canvas.paste(img, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return canvas


def build_v12_row(
    env: str,
    gid: str,
    query_type: str,
    query_step: int,
    anchor_step: int,
    horizon_gap: int,
    horizon_bucket: str,
    answer: str,
    first_label: Optional[str],
    frames: List[str],
    env_dir: Path,
    out_dir: Path,
    row_idx: int,
    sampling_role: str,
) -> dict:
    qs_bucket_str = horizon_bucket[:3]

    if query_type == "identity_of_first_picked_object":
        label_space = IDENTITY_LABELS
        prompt = IDENTITY_TEMPLATE.format(label_list=", ".join(IDENTITY_LABELS))
        img_filename = f"identity_first_pick_{qs_bucket_str}_q{row_idx:03d}.png"
    else:
        label_space = STATUS_LABELS
        prompt = STATUS_TEMPLATE
        img_filename = f"status_first_pick_{qs_bucket_str}_q{row_idx:03d}.png"

    img_rel = f"images/taskE_v12/{env}/{gid}/{img_filename}"
    img_out = out_dir / img_rel

    selected_steps = select_prefix_frames(query_step, anchor_step, k_min=5, k_max=12)

    assert all(0 <= s <= query_step for s in selected_steps), \
        f"Frame step {selected_steps} exceeds query_step {query_step}"

    layout = compute_layout(len(selected_steps))
    render_montage(env_dir, frames, selected_steps, layout, img_out)

    exam_id = f"E_v12.{env}.{gid}.{sampling_role}.q{row_idx:03d}"
    uid = exam_id

    return {
        "schema_version": SCHEMA_VERSION,
        "task": "E",
        "query_type": query_type,
        "exam_id": exam_id,
        "uid": uid,
        "env_task": env,
        "group_id": gid,
        "image": img_rel,
        "prompt": prompt,
        "label_space": label_space,
        "answer": answer,
        "answer_json": {"answer": answer},
        "query_step": query_step,
        "history_end_step": query_step,
        "anchor_step": anchor_step,
        "horizon_gap": horizon_gap,
        "horizon_bucket": horizon_bucket,
        "selected_steps": selected_steps,
        "sampling_role": sampling_role,
        "support_steps": {
            "first_pickup": anchor_step,
            "query_step": query_step,
        },
        "first_picked_object_gold": first_label,
        "meta": {
            "query_semantics": "checkpoint_prefix_only",
            "generator_version": GENERATOR_VERSION,
            "schema": SCHEMA_VERSION,
        },
    }


def build_rows_for_env(
    env_name: str,
    pilot_rows: List[Dict[str, Any]],
    source_root: Path,
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    stats: Dict[str, int] = defaultdict(int)

    for source in pilot_rows:
        gid = source["group_id"]
        frames = source["frames"]
        trace = normalize_trace(source["state_seq"])
        events = extract_events(trace)
        pickups = [(s, l) for s, l in events["pickups"] if l != "none"]
        drops = list(events["drops"])
        n_steps = events["n_steps"]

        if not pickups:
            stats["no_pickup"] += 1
            continue

        first_pickup_step, first_label = pickups[0]
        first_drop_step = next((s for s, l in drops if l == first_label), None)

        identity_steps = pick_identity_steps(first_pickup_step, n_steps)
        for role, qs in identity_steps.items():
            if qs is None:
                stats[f"no_candidate_{role}"] += 1
                continue
            gap = qs - first_pickup_step
            bucket = assign_horizon_bucket(gap)
            selected = select_prefix_frames(qs, first_pickup_step)
            if len(selected) > 12 or qs >= n_steps:
                stats[f"reject_{role}"] += 1
                continue
            row = build_v12_row(
                env=env_name,
                gid=gid,
                query_type="identity_of_first_picked_object",
                query_step=qs,
                anchor_step=first_pickup_step,
                horizon_gap=gap,
                horizon_bucket=bucket,
                answer=first_label,
                first_label=first_label,
                frames=frames,
                env_dir=source_root,
                out_dir=out_dir,
                row_idx=len(rows),
                sampling_role=role,
            )
            row["source_seed"] = source["seed"]
            row["source_phase"] = "58"
            rows.append(row)
            stats[f"row_{role}"] += 1

        status_steps = pick_status_steps(first_pickup_step, first_drop_step, n_steps)
        used_status_qs = set()
        for role, qs in status_steps.items():
            if qs is None:
                stats[f"no_candidate_{role}"] += 1
                continue
            if qs in used_status_qs:
                stats[f"dup_status_qs_{role}"] += 1
                continue
            gap = qs - first_pickup_step
            bucket = assign_horizon_bucket(gap)
            status_answer = classify_status(first_label, pickups, drops, events["reacquires"], qs)
            if role.startswith("status_") and "holding" in role and status_answer != "holding":
                stats[f"status_not_holding_{role}"] += 1
                continue
            selected = select_prefix_frames(qs, first_pickup_step)
            if len(selected) > 12 or qs >= n_steps:
                stats[f"reject_{role}"] += 1
                continue
            row = build_v12_row(
                env=env_name,
                gid=gid,
                query_type="status_of_first_picked_object_at_checkpoint",
                query_step=qs,
                anchor_step=first_pickup_step,
                horizon_gap=gap,
                horizon_bucket=bucket,
                answer=status_answer,
                first_label=first_label,
                frames=frames,
                env_dir=source_root,
                out_dir=out_dir,
                row_idx=len(rows),
                sampling_role=role,
            )
            row["source_seed"] = source["seed"]
            row["source_phase"] = "58"
            rows.append(row)
            used_status_qs.add(qs)
            stats[f"row_{role}"] += 1

    return rows, dict(stats)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_env = Counter()
    by_role = Counter()
    by_query_type = Counter()
    by_horizon = Counter()
    by_env_role = defaultdict(Counter)
    for row in rows:
        by_env[row["env_task"]] += 1
        by_role[row["sampling_role"]] += 1
        by_query_type[row["query_type"]] += 1
        by_horizon[row["horizon_bucket"]] += 1
        by_env_role[row["env_task"]][row["sampling_role"]] += 1
    return {
        "n_rows": len(rows),
        "by_env": dict(by_env),
        "by_sampling_role": dict(by_role),
        "by_query_type": dict(by_query_type),
        "by_horizon_bucket": dict(by_horizon),
        "by_env_role": {k: dict(v) for k, v in by_env_role.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase58-root",
        type=Path,
        default=Path("runs/phase58_taske_env_screen_2026-03-31"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("datasets/exams_taske_v13_candidate"),
    )
    args = ap.parse_args()

    source_root = args.phase58_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    build_stats: Dict[str, Any] = {}

    for env_name in PROMOTED_ENVS:
        pilot_path = source_root / f"{env_name}_pilot_full.jsonl"
        pilot_rows = [
            json.loads(line)
            for line in pilot_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows, stats = build_rows_for_env(env_name, pilot_rows, source_root, out_dir)
        build_stats[env_name] = stats
        all_rows.extend(rows)

    exam_path = out_dir / "task_e_exam_v13.jsonl"
    with exam_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    review_rows = []
    for row in all_rows:
        review_rows.append(
            {
                "exam_id": row["exam_id"],
                "image_abs": str((out_dir / row["image"]).resolve()),
                "prompt": row["prompt"],
                "gold_answer": row["answer"],
                "sampling_role": row["sampling_role"],
                "env_task": row["env_task"],
                "horizon_bucket": row["horizon_bucket"],
                "query_type": row["query_type"],
            }
        )
    review_path = out_dir / "human_audit_package_v13.jsonl"
    with review_path.open("w", encoding="utf-8") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "task": "E",
        "version": "v1.3-candidate",
        "source_phase": "58",
        "promoted_envs": list(PROMOTED_ENVS),
        "summary": summarize(all_rows),
        "build_stats": build_stats,
        "files": {
            "exam": str(exam_path),
            "human_audit_package": str(review_path),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
