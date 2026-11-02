#!/usr/bin/env python3
"""Generate successor-choice questions in PickupObjects using navigation
sequences among colored objects."""
from __future__ import annotations

from utils.miniworld_successor import MACRO_ACTIONS, EN_ACTION_LABELS, MIN_SAME_DIR_POS_DELTA

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from tools.miniworld.common import concat_vertical_rgb, dump_json, ensure_output_dir, export_rgb, make_env, maybe_render_top_view, maybe_render_top_view_with_scale, runtime_context, try_import_stack
else:
    from tools.miniworld.common import concat_vertical_rgb, dump_json, ensure_output_dir, export_rgb, make_env, maybe_render_top_view, maybe_render_top_view_with_scale, runtime_context, try_import_stack


# Reuse the proven FourRooms action sequence pool
ACTION_SEQUENCE_POOL = [
    (["turn_left_30", "forward_5"], ["turn_right_30", "forward_5"]),
    (["turn_right_30", "forward_5"], ["turn_left_30", "forward_5"]),
    (["forward_5", "turn_left_30"], ["forward_5", "turn_right_30"]),
    (["forward_10"], ["turn_left_30", "forward_5"]),
    (["forward_5", "turn_left_30", "forward_5"], ["forward_5", "turn_right_30", "forward_5"]),
    (["turn_left_60", "forward_10"], ["turn_right_60", "forward_10"]),
    (["forward_5", "forward_5"], ["turn_left_30", "forward_10"]),
    (["turn_left_30", "forward_10"], ["turn_right_30", "forward_10"]),
    (["forward_10", "turn_left_30"], ["turn_right_30", "forward_10"]),
    (["turn_right_60", "forward_5"], ["turn_left_30", "forward_10"]),
    (["forward_10", "forward_5"], ["forward_5", "turn_left_60"]),
    (["turn_left_30", "forward_5", "forward_5"], ["turn_right_30", "forward_5", "forward_5"]),
    (["turn_right_30", "forward_5", "forward_5"], ["turn_left_30", "forward_5", "forward_5"]),
    (["turn_left_60", "forward_5"], ["turn_right_60", "forward_5"]),
    (["forward_5", "turn_left_30", "forward_10"], ["forward_5", "turn_right_30", "forward_10"]),
]

ACTION_MAP = {
    "turn_left": 0,
    "turn_right": 1,
    "move_forward": 2,
}

MAX_CANDIDATE_EGO_MATCH_RATE = 0.80
DEFAULT_EASY_RATIO = 0.5


from utils.miniworld_successor import expand_macro_sequence


from utils.miniworld_successor import get_agent_state


from utils.miniworld_successor import get_motion_contract


def get_room_index(env, pos_xz: list[float]) -> int | None:
    target = getattr(env, "unwrapped", env)
    rooms = getattr(target, "rooms", [])
    if not rooms:
        return None
    x, z = pos_xz
    for idx, room in enumerate(rooms):
        if hasattr(room, "min_x") and room.min_x <= x <= room.max_x and room.min_z <= z <= room.max_z:
            return idx
    return None


def render_state_images(env, obs, img_dir, prefix):
    top = maybe_render_top_view(env)
    if top is None:
        raise RuntimeError("Topview rendering failed for PickupObjects")
    concat = concat_vertical_rgb(obs, top)
    img_path = img_dir / f"{prefix}.png"
    export_rgb(img_path, concat)
    return img_path




from utils.miniworld_successor import en_seq


def build_prompt(sequence: list[str]) -> str:
    actions_str = en_seq(sequence)
    return (
        "Task: Predict the true successor state.\n"
        "You are given the agent's current state and two candidate successor views.\n"
        "Action units used in this task:\n"
        "- move forward by one unit = 5 primitive environment forward steps\n"
        "- move forward by two units = 10 primitive environment forward steps\n"
        "- turn left/right by one unit = 30 degrees\n"
        "- turn left/right by two units = 60 degrees\n"
        "The top-view grid uses the same distance scale: one grid cell equals one forward unit.\n"
        f"Starting from the current state, execute the following action sequence:\n[{actions_str}]\n"
        "Which candidate view, A or B, is the true successor state after the full sequence?\n"
        "Answer with a single letter: A or B."
    )


from utils.miniworld_successor import _infer_strategy


from utils.miniworld_successor import _make_distractor_source


from utils.miniworld_successor import _state_distance


from utils.miniworld_successor import _is_too_subtle


from utils.miniworld_successor import _ego_panels_identical


def _ego_match_rate(path_a: Path, path_b: Path) -> float:
    import numpy as np

    with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
        a = im_a.convert("RGB")
        b = im_b.convert("RGB")
        ego_h = min(a.height, b.height) // 2
        ego_a = a.crop((0, 0, a.width, ego_h))
        ego_b = b.crop((0, 0, b.width, ego_h))
        arr_a = np.array(ego_a, dtype=np.int16)
        arr_b = np.array(ego_b, dtype=np.int16)
        diff = np.abs(arr_a - arr_b).sum(axis=2)
        total = diff.shape[0] * diff.shape[1]
        return float((diff == 0).sum()) / total if total else 0.0


def make_question(
    env_id: str,
    seed: int,
    env_slug: str,
    obs_width: int,
    obs_height: int,
    correct_seq: list[str],
    distractor_seq: list[str],
    question_idx: int,
    img_dir: Path,
) -> dict[str, Any]:
    gym_obj, _ = try_import_stack()

    env0, obs0, _ = make_env(
        gym_obj, env_id, seed=seed,
        obs_width=obs_width, obs_height=obs_height,
        window_width=obs_width, window_height=obs_height,
    )
    current_img = render_state_images(env0, obs0, img_dir, f"q{question_idx:02d}_current")
    current_state = get_agent_state(env0)
    _, current_topview_scale = maybe_render_top_view_with_scale(env0)
    motion_contract = get_motion_contract(env0)

    primitive_correct = expand_macro_sequence(correct_seq)
    primitive_dist = expand_macro_sequence(distractor_seq)

    # Correct candidate
    env_c, obs_c, _ = make_env(
        gym_obj, env_id, seed=seed,
        obs_width=obs_width, obs_height=obs_height,
        window_width=obs_width, window_height=obs_height,
    )
    for macro in primitive_correct:
        obs_c, reward, terminated, truncated, info = env_c.step(ACTION_MAP[macro])
    correct_img = render_state_images(env_c, obs_c, img_dir, f"q{question_idx:02d}_candidate_correct")
    correct_final = get_agent_state(env_c)
    env_c.close()

    # Distractor candidate
    env_d, obs_d, _ = make_env(
        gym_obj, env_id, seed=seed,
        obs_width=obs_width, obs_height=obs_height,
        window_width=obs_width, window_height=obs_height,
    )
    for macro in primitive_dist:
        obs_d, reward, terminated, truncated, info = env_d.step(ACTION_MAP[macro])
    dist_img = render_state_images(env_d, obs_d, img_dir, f"q{question_idx:02d}_candidate_distractor")
    dist_final = get_agent_state(env_d)
    env_d.close()
    env0.close()

    # Degenerate guards
    if (correct_final["pos"] == dist_final["pos"]
            and correct_final["cardinal"] == dist_final["cardinal"]):
        raise ValueError(
            f"Degenerate: candidates identical pos={correct_final['pos']} "
            f"dir={correct_final['cardinal']} seed={seed}"
        )
    if (correct_final["pos"] == current_state["pos"]
            and correct_final["cardinal"] == current_state["cardinal"]):
        raise ValueError(f"Degenerate: correct lands at current state seed={seed}")
    if (dist_final["pos"] == current_state["pos"]
            and dist_final["cardinal"] == current_state["cardinal"]):
        raise ValueError(f"Degenerate: distractor lands at current state seed={seed}")
    if _is_too_subtle(correct_final, dist_final):
        raise ValueError(
            f"Subtle: states too close seed={seed} "
            f"dist={_state_distance(correct_final, dist_final):.3f}"
        )
    if _ego_panels_identical(current_img, correct_img):
        raise ValueError(f"Subtle: current ego identical to correct ego seed={seed}")
    if _ego_panels_identical(current_img, dist_img):
        raise ValueError(f"Subtle: current ego identical to distractor ego seed={seed}")
    cand_ego_match = _ego_match_rate(correct_img, dist_img)
    if cand_ego_match >= MAX_CANDIDATE_EGO_MATCH_RATE:
        raise ValueError(
            f"Subtle: candidate ego panels too similar seed={seed} match={cand_ego_match:.6f}"
        )

    # A/B label randomization
    if random.random() < 0.5:
        cand_a_path = correct_img.relative_to(img_dir.parent).as_posix()
        cand_b_path = dist_img.relative_to(img_dir.parent).as_posix()
        correct_label = "A"
        label_idx = 0
        correct_final_out = correct_final
        dist_final_out = dist_final
    else:
        cand_a_path = dist_img.relative_to(img_dir.parent).as_posix()
        cand_b_path = correct_img.relative_to(img_dir.parent).as_posix()
        correct_label = "B"
        label_idx = 1
        correct_final_out = correct_final
        dist_final_out = dist_final

    current_rel = current_img.relative_to(img_dir.parent).as_posix()
    uid = f"A-MW-v2.pickupobjects.s{seed:06d}.t{question_idx}"

    return {
        "exam_id": uid,
        "uid": uid,
        "schema_version": "gridwm.mw_exam.v2",
        "task": "A-MW-v2",
        "env_task": "pickupobjects",
        "subtask": "successor_choice",
        "seed": seed,
        "action_sequence": correct_seq,
        "distractor_sequence": distractor_seq,
        "primitive_action_sequence": primitive_correct,
        "primitive_distractor_sequence": primitive_dist,
        "ground_truth": correct_label,
        "ground_truth_type": "successor_choice",
        "current_image_rel_path": current_rel,
        "candidate_a_rel_path": cand_a_path,
        "candidate_b_rel_path": cand_b_path,
        "correct_candidate": correct_label,
        "distractor_source": _make_distractor_source(correct_seq, distractor_seq),
        "distractor_strategy": _infer_strategy(correct_seq, distractor_seq),
        "prompt": build_prompt(correct_seq),
        "candidates": [
            {"label": "A", "value": "Candidate A is the true successor"},
            {"label": "B", "value": "Candidate B is the true successor"},
        ],
        "label": label_idx,
        "note": "A-MW-v2_PickupObjects_slice",
        "current_agent_pos": current_state["pos"],
        "current_agent_cardinal": current_state["cardinal"],
        "current_topview_scale": current_topview_scale,
        "forward_step": round(motion_contract["forward_step"], 4),
        "turn_step_deg": round(motion_contract["turn_step_deg"], 4),
        # PickupObjects does not have red_box; fill with N/A values for schema compatibility
        "red_box_pos": None,
        "red_box_room_idx": None,
        "current_room_idx": None,
        "red_box_visible_in_current_view": None,
        "current_room_has_red_box": None,
        "difficulty_bucket": "easy",
        "correct_candidate_agent_pos": correct_final_out["pos"],
        "correct_candidate_agent_cardinal": correct_final_out["cardinal"],
        "distractor_candidate_agent_pos": dist_final_out["pos"],
        "distractor_candidate_agent_cardinal": dist_final_out["cardinal"],
        "correct_candidate_same_as_current": (
            correct_final["pos"] == current_state["pos"]
            and correct_final["cardinal"] == current_state["cardinal"]
        ),
        "distractor_candidate_same_as_current": (
            dist_final["pos"] == current_state["pos"]
            and dist_final["cardinal"] == current_state["cardinal"]
        ),
        "correct_distractor_states_identical": (
            correct_final["pos"] == dist_final["pos"]
            and correct_final["cardinal"] == dist_final["cardinal"]
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Task A-MW v2 generator for PickupObjects")
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskA-MW-v2-PickupObjects",
    )
    ap.add_argument("--env_id", default="MiniWorld-PickupObjects-v0")
    ap.add_argument("--env_slug", default="pickupobjects")
    ap.add_argument("--obs_width", type=int, default=640)
    ap.add_argument("--obs_height", type=int, default=480)
    ap.add_argument("--num_questions", type=int, default=15)
    ap.add_argument("--seed_start", type=int, default=101)
    ap.add_argument("--easy_ratio", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = ensure_output_dir(Path(args.out_dir))
    img_dir = ensure_output_dir(out_dir / "images")

    manifest = {
        "schema_version": "gridwm.mw_exam.v2",
        "task": "A-MW-v2",
        "env_id": args.env_id,
        "visual_contract": "ego_topview_concat",
        "render_size": {
            "obs_width": args.obs_width,
            "obs_height": args.obs_height,
            "concat_height": args.obs_height * 2,
            "concat_width": args.obs_width,
        },
        "runtime": runtime_context(),
        "num_questions": args.num_questions,
        "question_design": "successor_choice",
        "distractor_strategy": "mixed (alternative_prefix + alternative_suffix) — per-question field records exact strategy",
        "min_same_dir_pos_delta": MIN_SAME_DIR_POS_DELTA,
        "easy_ratio_target": args.easy_ratio,
        "records": [],
    }

    print(f"Generating {args.num_questions} Task A-MW v2 questions for {args.env_id}...")
    print(f"  Output: {out_dir}")

    easy_target = max(1, round(args.num_questions * args.easy_ratio))
    hard_target = args.num_questions - easy_target
    easy_count = 0
    hard_count = 0
    generated = 0
    base_seed = args.seed_start
    max_attempts = 500

    while generated < args.num_questions:
        idx = generated
        seq_pair = ACTION_SEQUENCE_POOL[idx % len(ACTION_SEQUENCE_POOL)]
        correct_seq, distractor_seq = seq_pair

        seed = base_seed + idx
        print(f"  [{idx+1:02d}/{args.num_questions}] seed={seed} seq={correct_seq}")
        try:
            record = make_question(
                env_id=args.env_id,
                seed=seed,
                env_slug=args.env_slug,
                obs_width=args.obs_width,
                obs_height=args.obs_height,
                correct_seq=correct_seq,
                distractor_seq=distractor_seq,
                question_idx=idx + 1,
                img_dir=img_dir,
            )
        except ValueError as exc:
            if "Degenerate" in str(exc) or "Subtle" in str(exc):
                print(f"    [WARN] seed={seed} rejected: {exc} — shifting seed")
                base_seed += 1
                continue
            raise

        # For PickupObjects, we can't easily determine easy/hard via red_box
        # (rooms don't apply to single-room env), so accept all as "easy"
        bucket = "easy"
        manifest["records"].append(record)
        easy_count += 1
        generated += 1

    a_count = sum(1 for r in manifest["records"] if r["ground_truth"] == "A")
    b_count = len(manifest["records"]) - a_count
    manifest["ab_balance"] = {"A": a_count, "B": b_count}

    jsonl_path = out_dir / "taskA-MW-v2-pickupobjects.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in manifest["records"]:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_summary = dict(manifest)
    manifest_summary.pop("records")
    manifest_summary["records_count"] = len(manifest["records"])
    manifest_summary["records_relpath"] = str(jsonl_path.relative_to(out_dir))
    manifest_path = out_dir / "taskA-MW-v2-pickupobjects_manifest.json"
    dump_json(manifest_path, manifest_summary)

    print(f"\nDone. Output:")
    print(f"  Manifest: {manifest_path}")
    print(f"  JSONL:    {jsonl_path}")
    print(f"  Images:   {img_dir}/")
    print(f"  Total:    {len(manifest['records'])} questions")
    print(f"  A/B:      {a_count}/{b_count}")


if __name__ == "__main__":
    main()
