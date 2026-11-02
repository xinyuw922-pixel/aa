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
    from tools.miniworld.trajectory_core import build_prompt, rollout_variant
else:
    from tools.miniworld.common import dump_json, ensure_output_dir, register_local_miniworld_envs, try_import_stack
    from tools.miniworld.trajectory_core import build_prompt, rollout_variant


STRUCTURAL_SPECS = (
    {
        "family": "doorkey",
        "template_id": "migration_doorkey_structural",
        "seed": 201,
        "note": "结构增强：开门后还需拐入分支目标房。",
        "cf_intervention": "remove_key",
    },
    {
        "family": "multiroom",
        "template_id": "migration_multiroom_structural",
        "seed": 202,
        "note": "结构增强：两扇门之后还需转入目标房。",
        "cf_intervention": "lock_gate2_forever",
    },
    {
        "family": "redblue",
        "template_id": "migration_redblue_structural",
        "seed": 203,
        "note": "结构增强：四房间骨架中保持红门先于蓝门的规则。",
        "cf_intervention": "lock_gate2_forever",
    },
    {
        "family": "doorkey",
        "template_id": "migration_doorkey_structural_v2",
        "seed": 211,
        "note": "结构增强v2：开门后转入分支，再二次转向进入终点房。",
        "cf_intervention": "remove_key",
    },
    {
        "family": "multiroom",
        "template_id": "migration_multiroom_structural_v2",
        "seed": 212,
        "note": "结构增强v2：第二扇门放到第二个转角后，之后再到目标。",
        "cf_intervention": "lock_gate2_forever",
    },
    {
        "family": "redblue",
        "template_id": "migration_redblue_structural_v2",
        "seed": 213,
        "note": "结构增强v2：红门在主轴，蓝门在转角后的终点房入口。",
        "cf_intervention": "lock_gate2_forever",
    },
)


from utils.miniworld_records import relpath_payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate structural MiniWorld Task D migration showcase")
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskD-MW-structural-showcase",
    )
    ap.add_argument("--obs_width", type=int, default=320)
    ap.add_argument("--obs_height", type=int, default=240)
    ap.add_argument("--force_lane_z", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = ensure_output_dir(Path(args.out_dir))
    gym, _ = try_import_stack()
    register_local_miniworld_envs(gym)

    rows: list[dict[str, Any]] = []
    for spec in STRUCTURAL_SPECS:
        family = str(spec["family"])
        template_id = str(spec["template_id"])
        seed = int(spec["seed"])
        cf_intervention = str(spec["cf_intervention"])
        family_dir = ensure_output_dir(out_dir / family / template_id)

        full_payload = rollout_variant(
            gym,
            seed=seed,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
            out_dir=family_dir,
            variant="full",
            intervention_type=None,
            env_kwargs={"force_lane_z": args.force_lane_z, "force_template_id": template_id},
        )
        prompt = build_prompt(full_payload["action_sequence"])
        (family_dir / "storyboard_prompt.txt").write_text(prompt, encoding="utf-8")
        full_payload["prompt_relpath"] = "storyboard_prompt.txt"
        dump_json(family_dir / "storyboard_full_manifest.json", full_payload)

        cf_payload = rollout_variant(
            gym,
            seed=seed,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
            out_dir=family_dir,
            variant="cf",
            intervention_type=cf_intervention,
            env_kwargs={"force_lane_z": args.force_lane_z, "force_template_id": template_id},
        )
        cf_payload["prompt_relpath"] = "storyboard_prompt.txt"
        dump_json(family_dir / "storyboard_cf_manifest.json", cf_payload)

        pair_manifest = {
            "family": family,
            "template_id": template_id,
            "seed": seed,
            "cf_intervention": cf_intervention,
            "variants": {
                "full": "storyboard_full_manifest.json",
                "cf": "storyboard_cf_manifest.json",
            },
            "shared_action_sequence": full_payload["action_sequence"],
            "full_event_steps": full_payload.get("event_steps"),
            "cf_event_steps": cf_payload.get("event_steps"),
            "pair_ok": bool(full_payload["success"]) and not bool(cf_payload["success"]),
        }
        dump_json(family_dir / "storyboard_pair_manifest.json", pair_manifest)

        rows.append(
            {
                "family": family,
                "template_id": template_id,
                "seed": seed,
                "note": spec["note"],
                "cf_intervention": cf_intervention,
                "full_success": full_payload["success"],
                "cf_success": cf_payload["success"],
                "pair_ok": bool(full_payload["success"]) and not bool(cf_payload["success"]),
                "full_event_steps": full_payload["event_steps"],
                "cf_event_steps": cf_payload["event_steps"],
                "layout_metadata": full_payload["layout_metadata"],
                "manifest_relpath": f"{family}/{template_id}/storyboard_full_manifest.json",
                "pair_manifest_relpath": f"{family}/{template_id}/storyboard_pair_manifest.json",
                "full_storyboard_relpath": relpath_payload(full_payload, family_dir=family_dir, out_dir=out_dir)["model_input_image_relpath"],
                "cf_storyboard_relpath": relpath_payload(cf_payload, family_dir=family_dir, out_dir=out_dir)["model_input_image_relpath"],
                "event_coverage_relpath": relpath_payload(full_payload, family_dir=family_dir, out_dir=out_dir).get("event_coverage_image_relpath"),
            }
        )

    summary_path = out_dir / "structural_showcase_summary.json"
    dump_json(summary_path, {"rows": rows})
    with (out_dir / "structural_showcase_index.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved structural showcase to {out_dir}")


if __name__ == "__main__":
    main()
