#!/usr/bin/env python3
"""Shared trajectory rollout and storyboard rendering for KeyDoorStory tasks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from tools.miniworld.common import dump_json, ensure_output_dir, export_rgb, make_env, maybe_render_top_view, register_local_miniworld_envs, try_import_stack
else:
    from tools.miniworld.common import dump_json, ensure_output_dir, export_rgb, make_env, maybe_render_top_view, register_local_miniworld_envs, try_import_stack


ENV_ID = "MiniWorld-KeyDoorStory-v0"
DEFAULT_ACTION_NAMES = [
    "move_forward",
    "pickup",
    "move_forward",
    "move_forward",
    "move_forward",
    "move_forward",
    "move_forward",
    "move_forward",
    "toggle",
    "move_forward",
    "move_forward",
    "move_forward",
    "move_forward",
    "move_forward",
]


from utils.miniworld_records import get_action_id_map


def build_prompt(action_sequence: list[str]) -> str:
    return (
        "You are given a storyboard showing one continuous first-person trajectory in a 3D environment.\n"
        "The frames are arranged in temporal order from the beginning to the end of the trajectory.\n\n"
        f"The executed action sequence is:\n{action_sequence}\n\n"
        "Judge whether this trajectory successfully completes the task.\n\n"
        "Output rule:\n"
        "Your entire response must be exactly one word:\n"
        "SUCCESS\n"
        "or\n"
        "FAIL\n"
    )


from utils.miniworld_images import annotate_frame


from utils.miniworld_images import build_montage


from utils.miniworld_images import concat_vertical


def build_event_coverage_montage(
    frames_by_step: dict[int, np.ndarray],
    topviews_by_step: dict[int, np.ndarray],
    *,
    event_steps: dict[str, int],
    event_specs: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, int]]]:
    panels: list[np.ndarray] = []
    coverage_rows: list[dict[str, int]] = []
    for spec in event_specs:
        event_key = str(spec.get("key"))
        if not spec.get("coverage", True):
            continue
        if event_key not in event_steps:
            continue
        post_t = int(event_steps[event_key])
        pre_t = max(0, post_t - 1)
        event_label = str(spec.get("label", event_key))
        for phase, t in (("pre", pre_t), ("post", post_t)):
            ego = frames_by_step[t]
            top = topviews_by_step[t]
            panel = concat_vertical(ego, top)
            label = f"{phase}-{event_label}  t{t:02d}"
            panels.append(annotate_frame(panel, label))
            coverage_rows.append({"event": event_label, "phase": phase, "t": t})
    return build_montage(panels, cols=2), coverage_rows


def rollout_variant(
    gym: Any,
    *,
    seed: int,
    obs_width: int,
    obs_height: int,
    out_dir: Path,
    variant: str,
    intervention_type: str | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frames_dir = ensure_output_dir(out_dir / f"frames_{variant}")
    env, obs0, _ = make_env(
        gym,
        ENV_ID,
        seed=seed,
        obs_width=obs_width,
        obs_height=obs_height,
        window_width=obs_width,
        window_height=obs_height,
        env_kwargs=env_kwargs,
    )
    if intervention_type is not None:
        env.unwrapped.apply_intervention(intervention_type)
        obs0 = env.render()

    action_sequence = list(getattr(env.unwrapped, "get_story_action_sequence", lambda: DEFAULT_ACTION_NAMES)())
    event_specs = list(
        getattr(
            env.unwrapped,
            "get_story_event_specs",
            lambda: [
                {"key": "key_collected", "label": "pickup", "coverage": True},
                {"key": "gate_open", "label": "toggle", "coverage": True},
                {"key": "goal_reached", "label": "goal", "coverage": False},
            ],
        )()
    )
    tracked_event_keys = [str(spec.get("key")) for spec in event_specs]
    action_id_map = get_action_id_map(env, action_sequence)
    frames: list[np.ndarray] = []
    frames_by_step: dict[int, np.ndarray] = {}
    topviews_by_step: dict[int, np.ndarray] = {}
    step_records: list[dict[str, Any]] = []
    event_steps: dict[str, int] = {}

    initial_state = env.unwrapped.get_story_state()
    frame0 = annotate_frame(obs0, "t00 start")
    frames.append(frame0)
    frames_by_step[0] = obs0
    export_rgb(frames_dir / "t00_start.png", frame0)
    topviews_dir = ensure_output_dir(out_dir / f"topviews_{variant}")
    top0 = maybe_render_top_view(env)
    if top0 is not None:
        topviews_by_step[0] = top0
        export_rgb(topviews_dir / "t00_start_topview.png", top0)
    step_records.append({"t": 0, "action": None, "state": initial_state})

    prev_flags = {key: bool(initial_state.get(key, False)) for key in tracked_event_keys}
    terminated = False
    truncation = False
    final_reward = 0.0

    for step_idx, action_name in enumerate(action_sequence, start=1):
        obs, reward, terminated, truncation, _ = env.step(action_id_map[action_name])
        final_reward = float(reward)
        state = env.unwrapped.get_story_state()
        frame = annotate_frame(obs, f"t{step_idx:02d} {action_name}")
        frames.append(frame)
        frames_by_step[step_idx] = obs
        export_rgb(frames_dir / f"t{step_idx:02d}_{action_name}.png", frame)
        top = maybe_render_top_view(env)
        if top is not None:
            topviews_by_step[step_idx] = top
            export_rgb(topviews_dir / f"t{step_idx:02d}_{action_name}_topview.png", top)
        step_records.append(
            {
                "t": step_idx,
                "action": action_name,
                "reward": reward,
                "terminated": terminated,
                "truncation": truncation,
                "state": state,
            }
        )
        for key in tracked_event_keys:
            now = bool(state.get(key, False))
            if now and not prev_flags[key] and key not in event_steps:
                event_steps[key] = step_idx
            prev_flags[key] = now
        if terminated or truncation:
            break

    montage_relpath = f"D-MW_v2_keydoor_{variant}_storyboard.png"
    montage = build_montage(frames, cols=4)
    export_rgb(out_dir / montage_relpath, montage)

    topview = maybe_render_top_view(env)
    if topview is not None:
        export_rgb(out_dir / f"D-MW_v2_keydoor_topview_{variant}.png", topview)

    event_coverage_relpath = None
    event_coverage_rows: list[dict[str, int]] = []
    if variant == "full" and topviews_by_step:
        coverage_img, coverage_rows = build_event_coverage_montage(
            frames_by_step,
            topviews_by_step,
            event_steps=event_steps,
            event_specs=event_specs,
        )
        event_coverage_relpath = "D-MW_v2_keydoor_full_event_coverage.png"
        export_rgb(out_dir / event_coverage_relpath, coverage_img)
        event_coverage_rows = coverage_rows

    payload = {
        "env_id": ENV_ID,
        "variant": variant,
        "seed": seed,
        "task": "D-MW-v2",
        "subtask": "storyboard_success_fail",
        "uid": f"D-MW-v2.keydoor_story.{variant}.s{seed:06d}",
        "schema_version": "gridwm.mw_exam.v3_storyboard",
        "action_sequence": action_sequence,
        "ground_truth": "SUCCESS" if prev_flags["goal_reached"] else "FAIL",
        "success": bool(prev_flags["goal_reached"]),
        "terminated": terminated,
        "truncated": truncation,
        "reward": final_reward,
        "intervention_type": intervention_type,
        "event_steps": event_steps,
        "model_input_image_relpath": montage_relpath,
        "frames_dir_relpath": f"frames_{variant}",
        "topviews_dir_relpath": f"topviews_{variant}",
        "event_coverage_image_relpath": event_coverage_relpath,
        "event_coverage_rows": event_coverage_rows,
        "layout_metadata": env.unwrapped.get_layout_metadata(),
        "step_records": step_records,
    }
    env.close()
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate D-MW v2 storyboard prototype")
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskD-MW-v2-keydoor-prototype",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--obs_width", type=int, default=320)
    ap.add_argument("--obs_height", type=int, default=240)
    ap.add_argument("--force_lane_z", type=float, default=3.0)
    ap.add_argument("--template_id", default="four_stage_right_branch_room")
    args = ap.parse_args()

    out_dir = ensure_output_dir(Path(args.out_dir))
    gym, _ = try_import_stack()
    register_local_miniworld_envs(gym)

    full_payload = rollout_variant(
        gym,
        seed=args.seed,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
        out_dir=out_dir,
        variant="full",
        intervention_type=None,
        env_kwargs={"force_lane_z": args.force_lane_z, "force_template_id": args.template_id},
    )
    prompt = build_prompt(full_payload["action_sequence"])
    (out_dir / "storyboard_prompt.txt").write_text(prompt, encoding="utf-8")
    full_payload["prompt_relpath"] = "storyboard_prompt.txt"
    dump_json(out_dir / "storyboard_full_manifest.json", full_payload)

    cf_payload = rollout_variant(
        gym,
        seed=args.seed,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
        out_dir=out_dir,
        variant="cf",
        intervention_type="remove_key",
        env_kwargs={"force_lane_z": args.force_lane_z, "force_template_id": args.template_id},
    )
    cf_payload["prompt_relpath"] = "storyboard_prompt.txt"
    dump_json(out_dir / "storyboard_cf_manifest.json", cf_payload)

    pair_manifest = {
        "env_id": ENV_ID,
        "task": "D-MW-v2",
        "group_id": f"D-MW-v2.keydoor_story.s{args.seed:06d}",
        "seed": args.seed,
        "variants": {
            "full": "storyboard_full_manifest.json",
            "cf": "storyboard_cf_manifest.json",
        },
        "shared_action_sequence": full_payload["action_sequence"],
        "layout_metadata": full_payload.get("layout_metadata"),
        "event_steps": full_payload.get("event_steps"),
        "event_coverage_image_relpath": full_payload.get("event_coverage_image_relpath"),
    }
    dump_json(out_dir / "storyboard_pair_manifest.json", pair_manifest)
    print(f"Saved prototype storyboards to {out_dir}")


if __name__ == "__main__":
    main()
