#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from tools.miniworld.common import dump_json, ensure_output_dir, make_env, register_local_miniworld_envs, try_import_stack
    from tools.miniworld.trajectory_core import rollout_variant
else:
    from tools.miniworld.common import dump_json, ensure_output_dir, make_env, register_local_miniworld_envs, try_import_stack
    from tools.miniworld.trajectory_core import rollout_variant


FAMILY_SPECS = (
    {
        "family": "doorkey",
        "template_id": "migration_doorkey_canonical",
        "seed": 1701,
        "cf_intervention": "remove_key",
        "note": "DoorKey Task C-MW prototype",
    },
    {
        "family": "multiroom",
        "template_id": "migration_multiroom_canonical",
        "seed": 1702,
        "cf_intervention": "lock_gate2_forever",
        "note": "MultiRoom Task C-MW prototype",
    },
    {
        "family": "redblue",
        "template_id": "migration_redblue_canonical",
        "seed": 1703,
        "cf_intervention": "swap_redblue_order",
        "note": "RedBlue Task C-MW prototype",
    },
)

TASK_C_GOALS = {
    "doorkey": "pick up the key, open the door, and reach the goal tile.",
    "multiroom": "navigate through the rooms (opening doors as needed) and reach the final goal tile.",
    "redblue": "open the red door before the blue door, then reach the goal. The red door must be opened first.",
}

from utils.miniworld_images import FONT_PATHS


from utils.miniworld_images import load_font


from utils.miniworld_records import get_action_id_map


def find_frame_path(frames_dir: Path, t: int) -> Path:
    exact = sorted(frames_dir.glob(f"t{t:02d}_*.png"))
    if not exact:
        raise FileNotFoundError(f"Missing frame for t={t} in {frames_dir}")
    return exact[0]


from utils.miniworld_images import annotate_frame_with_step


from utils.miniworld_images import build_grid


def derive_taskc_event_rows(payload: dict[str, Any]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    seen: set[tuple[str, str, int]] = set()

    def add_row(event: str, phase: str, t: int) -> None:
        key = (event, phase, t)
        if key in seen:
            return
        seen.add(key)
        rows.append({"event": event, "phase": phase, "t": t})

    for row in payload.get("event_coverage_rows", []):
        add_row(str(row["event"]), str(row["phase"]), int(row["t"]))

    labeled_events = {
        "key_collected": "pickup",
        "gate_open": "toggle",
        "gate2_open": "toggle-2",
        "red_door_open": "toggle-red",
        "blue_door_open": "toggle-blue",
        "goal_reached": "goal",
    }
    covered_events = {str(row["event"]) for row in rows}
    for key, t in payload.get("event_steps", {}).items():
        label = labeled_events.get(str(key), str(key))
        if label in covered_events:
            continue
        post_t = int(t)
        pre_t = max(0, post_t - 1)
        add_row(label, "pre", pre_t)
        add_row(label, "post", post_t)
        covered_events.add(label)

    return sorted(rows, key=lambda row: (int(row["t"]), 0 if str(row["phase"]) == "pre" else 1, str(row["event"])))


def choose_keyframe_steps(
    payload: dict[str, Any],
    *,
    k_max: int = 12,
) -> list[int]:
    step_records = payload["step_records"]
    total_t = int(step_records[-1]["t"])

    # Pure Task C selector: keep only event anchors and terminal evidence.
    forced = {0, max(0, total_t - 1), total_t}
    for row in derive_taskc_event_rows(payload):
        forced.add(int(row["t"]))

    result = sorted(forced)
    if len(result) >= k_max:
        return result
    return result


def build_nocue_transform(
    family: str,
    *,
    selected_steps: list[int],
    task_c_event_rows: list[dict[str, int | str]],
    event_steps: dict[str, Any],
) -> dict[int, str]:
    transform_tags: dict[int, str] = {}

    if family == "doorkey":
        goal_t = int(event_steps.get("goal_reached", selected_steps[-1]))
        transform_tags[goal_t] = "hide-goal"
        return transform_tags

    if family == "multiroom":
        goal_t = int(event_steps.get("goal_reached", selected_steps[-1]))
        transform_tags[goal_t] = "hide-goal"
        return transform_tags

    if family == "redblue":
        neutralize_steps = {
            int(row["t"])
            for row in task_c_event_rows
            if str(row["event"]) in {"toggle-red", "toggle-blue"} and str(row["phase"]) == "pre"
        }
        for t in neutralize_steps:
            transform_tags[t] = "hide-doors"
        return transform_tags

    return transform_tags


def build_keyframe_montage(
    payload: dict[str, Any],
    base_dir: Path,
    *,
    variant: str,
    selected_steps: list[int] | None = None,
    step_transform: Any | None = None,
) -> tuple[str, list[int]]:
    frames_dir = base_dir / payload["frames_dir_relpath"]
    steps = selected_steps or choose_keyframe_steps(payload)
    frames: list[Image.Image] = []
    for t in steps:
        img = Image.open(find_frame_path(frames_dir, t)).convert("RGB")
        if step_transform is not None:
            img = step_transform(t, img)
        frames.append(annotate_frame_with_step(img, t))
    montage = build_grid(frames, cols=4)
    relpath = f"C-MW_{payload['layout_metadata']['task_family']}_{variant}_keyframes.png"
    montage.save(base_dir / relpath, "PNG")
    return relpath, steps


def rerender_nocue_montage(
    gym: Any,
    *,
    seed: int,
    obs_width: int,
    obs_height: int,
    family_dir: Path,
    template_id: str,
    force_lane_z: float,
    action_sequence: list[str],
    family: str,
    selected_steps: list[int],
    nocue_tags: dict[int, str],
) -> str:
    env, obs0, _info = make_env(
        gym,
        "MiniWorld-KeyDoorStory-v0",
        seed=seed,
        obs_width=obs_width,
        obs_height=obs_height,
        window_width=obs_width,
        window_height=obs_height,
        env_kwargs={"force_lane_z": force_lane_z, "force_template_id": template_id},
    )
    target = env.unwrapped
    action_id_map = get_action_id_map(env, action_sequence)

    def capture(step_idx: int, obs_frame: Any) -> Image.Image:
        override = nocue_tags.get(step_idx)
        if override is None:
            img = Image.fromarray(obs_frame).convert("RGB")
            return annotate_frame_with_step(img, step_idx)

        target.clear_taskc_visual_override()
        if hasattr(target, "_render_static"):
            target._render_static()
        if override is not None:
            target.apply_taskc_visual_override(override)
            if hasattr(target, "_render_static"):
                target._render_static()
        arr = env.render()
        img = Image.fromarray(arr).convert("RGB")
        target.clear_taskc_visual_override()
        if hasattr(target, "_render_static"):
            target._render_static()
        return annotate_frame_with_step(img, step_idx)

    selected_set = set(selected_steps)
    frames: list[Image.Image] = []
    if 0 in selected_set:
        frames.append(capture(0, obs0))
    for step_idx, action_name in enumerate(action_sequence, start=1):
        obs, _reward, _terminated, _truncation, _info = env.step(action_id_map[action_name])
        if step_idx in selected_set:
            frames.append(capture(step_idx, obs))

    montage = build_grid(frames, cols=4)
    relpath = f"C-MW_{family}_nocue_keyframes.png"
    montage.save(family_dir / relpath, "PNG")
    env.close()
    return relpath


def build_prompt(goal_clause: str) -> str:
    return (
        "Below is a storyboard from one MiniWorld trajectory.\n"
        f"Task goal: {goal_clause}\n"
        "Did the trajectory satisfy this goal?\n"
        "Answer with ONLY: Yes or No."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Task C-MW family prototypes")
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskC-MW-family-prototypes",
    )
    ap.add_argument("--obs_width", type=int, default=320)
    ap.add_argument("--obs_height", type=int, default=240)
    ap.add_argument("--force_lane_z", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = ensure_output_dir(Path(args.out_dir))
    gym, _ = try_import_stack()
    register_local_miniworld_envs(gym)

    rows: list[dict[str, Any]] = []

    for spec in FAMILY_SPECS:
        family = str(spec["family"])
        template_id = str(spec["template_id"])
        family_dir = ensure_output_dir(out_dir / family / template_id)
        seed = int(spec["seed"])
        goal_clause = TASK_C_GOALS[family]
        prompt = build_prompt(goal_clause)
        prompt_relpath = "task_c_mw_prompt.txt"
        (family_dir / prompt_relpath).write_text(prompt, encoding="utf-8")

        payload_full = rollout_variant(
            gym,
            seed=seed,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
            out_dir=family_dir,
            variant="full",
            intervention_type=None,
            env_kwargs={"force_lane_z": args.force_lane_z, "force_template_id": template_id},
        )
        payload_cf = rollout_variant(
            gym,
            seed=seed,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
            out_dir=family_dir,
            variant="cf",
            intervention_type=str(spec["cf_intervention"]),
            env_kwargs={"force_lane_z": args.force_lane_z, "force_template_id": template_id},
        )

        shared_selected_steps = choose_keyframe_steps(payload_full)
        full_event_rows = derive_taskc_event_rows(payload_full)
        nocue_tags = build_nocue_transform(
            family,
            selected_steps=shared_selected_steps,
            task_c_event_rows=full_event_rows,
            event_steps=payload_full.get("event_steps", {}),
        )

        variant_payloads: list[tuple[str, dict[str, Any], str, str, dict[str, Any] | None]] = [
            ("full", payload_full, "Success", "full", None),
            (
                "nocue",
                dict(payload_full),
                "Success",
                "full",
                {
                    "strategy": f"{family}_nocue_render_v2_hide",
                    "masked_steps": sorted(int(t) for t in nocue_tags),
                    "transform_tags": {str(k): v for k, v in sorted(nocue_tags.items())},
                    "render_mode": "env-level temporary invisibility",
                },
            ),
            ("cf", payload_cf, "Fail", "cf", None),
        ]

        for variant, payload, answer, payload_variant_source, nocue_meta in variant_payloads:
            payload = dict(payload)
            if variant == "nocue":
                keyframe_relpath = rerender_nocue_montage(
                    gym,
                    seed=seed,
                    obs_width=args.obs_width,
                    obs_height=args.obs_height,
                    family_dir=family_dir,
                    template_id=template_id,
                    force_lane_z=args.force_lane_z,
                    action_sequence=list(payload_full["action_sequence"]),
                    family=family,
                    selected_steps=shared_selected_steps,
                    nocue_tags=nocue_tags,
                )
                selected_steps = list(shared_selected_steps)
            else:
                keyframe_relpath, selected_steps = build_keyframe_montage(
                    payload,
                    family_dir,
                    variant=variant,
                    selected_steps=shared_selected_steps,
                )
            payload["task"] = "C-MW-prototype"
            payload["subtask"] = "sparse_keyframe_judgment"
            payload["variant"] = variant
            payload["keyframe_selector"] = "event-plus-tail-only-v1"
            payload["k_frame"] = len(selected_steps)
            payload["task_c_event_rows"] = derive_taskc_event_rows(payload)
            payload["selected_steps"] = selected_steps
            payload["model_input_image_relpath"] = keyframe_relpath
            payload["prompt_relpath"] = prompt_relpath
            payload["task_goal"] = f"Task goal: {goal_clause}"
            payload["success_criterion_text"] = goal_clause
            payload["prompt_version"] = "miniworld-taskc-minigrid-style-v1"
            payload["task_c_selected_from_variant"] = payload_variant_source
            if nocue_meta is not None:
                payload["nocue_meta"] = nocue_meta
            dump_json(family_dir / f"task_c_{variant}_manifest.json", payload)

            row = {
                "task": "C",
                "subtask": "C-MW-prototype",
                "family": family,
                "template_id": template_id,
                "variant": variant,
                "answer": answer,
                "image": f"{family}/{template_id}/{keyframe_relpath}",
                "prompt": prompt,
                "task_goal": f"Task goal: {goal_clause}",
                "success_criterion_text": goal_clause,
                "prompt_version": "miniworld-taskc-minigrid-style-v1",
                "k_frame": len(selected_steps),
                "selected_steps": selected_steps,
                "keyframe_selector": "event-plus-tail-only-v1",
                "task_c_event_rows": payload.get("task_c_event_rows", []),
                "event_steps": payload.get("event_steps", {}),
                "note": spec["note"],
                "source_manifest_relpath": f"{family}/{template_id}/task_c_{variant}_manifest.json",
                "task_c_selected_from_variant": payload_variant_source,
            }
            if nocue_meta is not None:
                row["nocue_meta"] = nocue_meta
            rows.append(row)

    exam_path = out_dir / "task_c_mw_family_prototypes.jsonl"
    with exam_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "num_records": len(rows),
        "families": {family: sum(1 for row in rows if row["family"] == family) for family in ("doorkey", "multiroom", "redblue")},
        "variants": ["full", "nocue", "cf"],
        "keyframe_selector": "event-plus-tail-only-v1",
        "records": rows,
    }
    dump_json(out_dir / "task_c_mw_family_prototypes_summary.json", summary)
    print(f"Saved Task C-MW family prototypes to {out_dir}")


if __name__ == "__main__":
    main()
