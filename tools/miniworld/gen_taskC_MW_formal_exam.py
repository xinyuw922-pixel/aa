#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from tools.miniworld.common import dump_json, ensure_output_dir, register_local_miniworld_envs, try_import_stack
    from tools.miniworld.taskc_scene_logic import TASK_C_GOALS, build_keyframe_montage, build_nocue_transform, build_prompt, choose_keyframe_steps, derive_taskc_event_rows, rerender_nocue_montage
    from tools.miniworld.trajectory_core import rollout_variant
else:
    from tools.miniworld.common import dump_json, ensure_output_dir, register_local_miniworld_envs, try_import_stack
    from tools.miniworld.taskc_scene_logic import TASK_C_GOALS, build_keyframe_montage, build_nocue_transform, build_prompt, choose_keyframe_steps, derive_taskc_event_rows, rerender_nocue_montage
    from tools.miniworld.trajectory_core import rollout_variant


SOURCE_SUMMARIES = (
    "experiments/generated/miniworld/taskD-MW-migration-showcase-families-paired/migration_showcase_summary.json",
    "experiments/generated/miniworld/taskD-MW-structural-showcase-v2/structural_showcase_summary.json",
    "experiments/generated/miniworld/taskD-MW-highdisc-expansion-round1-clean/highdisc_round1_summary.json",
    "experiments/generated/miniworld/taskD-MW-highdisc-expansion-round2/highdisc_round2_summary.json",
    "experiments/generated/miniworld/taskD-MW-highdisc-expansion-round3/highdisc_round3_summary.json",
)


def load_template_specs(summaries=None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for src_path in (summaries if summaries is not None else SOURCE_SUMMARIES):
        data = json.loads(Path(src_path).read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            if not row.get("pair_ok", False):
                continue
            template_id = str(row["template_id"])
            if template_id in seen:
                continue
            seen.add(template_id)
            layout_metadata = dict(row.get("layout_metadata", {}))
            specs.append(
                {
                    "family": str(row["family"]),
                    "template_id": template_id,
                    "seed": int(row["seed"]),
                    "cf_intervention": str(row["cf_intervention"]),
                    "note": str(row.get("note", "")),
                    "difficulty": str(layout_metadata.get("difficulty", "default")),
                    "force_lane_z": float(layout_metadata.get("lane_z", 3.0)),
                    "source_summary": str(Path(src_path).name),
                }
            )
    return specs


def build_group_id(family: str, template_id: str, seed: int) -> str:
    return f"{family}.{template_id}.s{seed:06d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build formal Task C-MW exam bank from stable MiniWorld templates")
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskC-MW-formal-bank",
    )
    ap.add_argument("--obs_width", type=int, default=320)
    ap.add_argument("--obs_height", type=int, default=240)
    ap.add_argument('--source-summary', action='append', help='Explicit newly generated pool summary for small regression runs')
    ap.add_argument('--max-groups', type=int, default=0)
    args = ap.parse_args()

    out_dir = ensure_output_dir(Path(args.out_dir))
    gym, _ = try_import_stack()
    register_local_miniworld_envs(gym)

    template_specs = load_template_specs(args.source_summary)
    if args.max_groups:
        if args.max_groups < 0:
            ap.error('--max-groups must be nonnegative')
        buckets = {family: [r for r in template_specs if r['family'] == family]
                   for family in sorted({r['family'] for r in template_specs})}
        ordered = []
        while any(buckets.values()):
            for family in buckets:
                if buckets[family]:
                    ordered.append(buckets[family].pop(0))
        template_specs = ordered[:args.max_groups]
    rows: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []

    for group_index, spec in enumerate(template_specs, start=1):
        family = str(spec["family"])
        template_id = str(spec["template_id"])
        seed = int(spec["seed"])
        family_dir = ensure_output_dir(out_dir / family / template_id)
        group_id = build_group_id(family, template_id, seed)
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
            env_kwargs={"force_lane_z": float(spec["force_lane_z"]), "force_template_id": template_id},
        )
        payload_cf = rollout_variant(
            gym,
            seed=seed,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
            out_dir=family_dir,
            variant="cf",
            intervention_type=str(spec["cf_intervention"]),
            env_kwargs={"force_lane_z": float(spec["force_lane_z"]), "force_template_id": template_id},
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
                    force_lane_z=float(spec["force_lane_z"]),
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

            payload["task"] = "C-MW-formal"
            payload["subtask"] = "sparse_keyframe_judgment"
            payload["variant"] = variant
            payload["group_id"] = group_id
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
            payload["template_note"] = str(spec["note"])
            payload["template_source_summary"] = str(spec["source_summary"])
            if nocue_meta is not None:
                payload["nocue_meta"] = nocue_meta
            dump_json(family_dir / f"task_c_{variant}_manifest.json", payload)

            row = {
                "exam_id": f"C-MW.{group_id}.{variant}",
                "uid": f"C-MW.{group_id}.{variant}",
                "schema_version": "gridwm.mw_exam.v1",
                "task": "C",
                "subtask": "C-MW-formal",
                "env_task": family,
                "env_id": "MiniWorld-KeyDoorStory-v0",
                "group_id": group_id,
                "family": family,
                "template_id": template_id,
                "variant": variant,
                "answer": answer,
                "label": 1 if answer == "Success" else 0,
                "image": f"{family}/{template_id}/{keyframe_relpath}",
                "prompt": prompt,
                "task_goal": f"Task goal: {goal_clause}",
                "success_criterion_text": goal_clause,
                "prompt_version": "miniworld-taskc-minigrid-style-v1",
                "difficulty": str(spec["difficulty"]),
                "seed": seed,
                "source_summary": str(spec["source_summary"]),
                "template_note": str(spec["note"]),
                "n_frames_selected": len(selected_steps),
                "n_frames": len(selected_steps),
                "selected_steps": selected_steps,
                "keyframe_selector": "event-plus-tail-only-v1",
                "task_c_event_rows": payload.get("task_c_event_rows", []),
                "event_steps": payload.get("event_steps", {}),
                "task_c_selected_from_variant": payload_variant_source,
                "source_manifest_relpath": f"{family}/{template_id}/task_c_{variant}_manifest.json",
                "group_index": group_index,
            }
            if nocue_meta is not None:
                row["nocue_meta"] = nocue_meta
            rows.append(row)

        group_records.append(
            {
                "group_index": group_index,
                "group_id": group_id,
                "family": family,
                "template_id": template_id,
                "difficulty": str(spec["difficulty"]),
                "seed": seed,
                "source_summary": str(spec["source_summary"]),
                "template_note": str(spec["note"]),
                "selected_steps": shared_selected_steps,
                "variants": {
                    "full": f"{family}/{template_id}/C-MW_{family}_full_keyframes.png",
                    "nocue": f"{family}/{template_id}/C-MW_{family}_nocue_keyframes.png",
                    "cf": f"{family}/{template_id}/C-MW_{family}_cf_keyframes.png",
                },
            }
        )

    exam_jsonl = out_dir / "task_c_mw_formal_exam.jsonl"
    with exam_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    compatibility_exam = out_dir / "task_c_exam.jsonl"
    compatibility_exam.write_text(exam_jsonl.read_text(encoding="utf-8"), encoding="utf-8")

    summary = {
        "task": "C-MW-formal",
        "num_groups": len(group_records),
        "num_records": len(rows),
        "families": {family: sum(1 for row in rows if row["family"] == family) for family in ("doorkey", "multiroom", "redblue")},
        "variants": ["full", "nocue", "cf"],
        "keyframe_selector": "event-plus-tail-only-v1",
        "prompt_version": "miniworld-taskc-minigrid-style-v1",
        "groups": group_records,
    }
    dump_json(out_dir / "task_c_mw_formal_summary.json", summary)
    print(f"Saved Task C-MW formal exam bank to {out_dir}")


if __name__ == "__main__":
    main()
