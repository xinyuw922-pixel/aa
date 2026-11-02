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




from utils.miniworld_records import relpath_payload


def cf_intervention_for_family(family: str) -> str:
    if family == "doorkey":
        return "remove_key"
    if family == "redblue":
        return "swap_redblue_order"
    return "lock_gate2_forever"


def main() -> None:
    """Dispatch the unchanged named template-pack workflows."""
    parser = argparse.ArgumentParser(description=main.__doc__, add_help=False)
    parser.add_argument('--template-set', choices=['round1', 'round2', 'round3'], default='round1')
    args, remainder = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remainder]
    {'round1': run_round1, 'round2': run_round2, 'round3': run_round3}[args.template_set]()


ROUND1_SPECS = (
    {
        "family": "doorkey",
        "template_id": "migration_doorkey_highdisc_branch_lower",
        "seed": 1401,
        "note": "高区分度：开门后必须下转进入更深的目标房，不靠 clutter。",
    },
    {
        "family": "doorkey",
        "template_id": "migration_doorkey_highdisc_clear_zigzag",
        "seed": 1402,
        "note": "高区分度：双转向终点房，但去掉低价值 clutter，只保留关键结构差异。",
    },
    {
        "family": "multiroom",
        "template_id": "migration_multiroom_highdisc_clear_gate2_after_turn",
        "seed": 1411,
        "note": "高区分度：第二扇门放到转角之后，不能只看主轴直线过程。",
    },
    {
        "family": "redblue",
        "template_id": "migration_redblue_highdisc_clear_order_zigzag",
        "seed": 1421,
        "note": "高区分度：红门在主轴，蓝门在转角后；错误顺序会更显眼，而且去掉背景噪声。",
    },
)


def run_round1() -> None:
    ap = argparse.ArgumentParser(description="Generate first high-discriminability MiniWorld Task D expansion pack")
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskD-MW-highdisc-expansion-round1",
    )
    ap.add_argument("--obs_width", type=int, default=320)
    ap.add_argument("--obs_height", type=int, default=240)
    ap.add_argument("--force_lane_z", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = ensure_output_dir(Path(args.out_dir))
    gym, _ = try_import_stack()
    register_local_miniworld_envs(gym)

    rows: list[dict[str, Any]] = []
    for spec in ROUND1_SPECS:
        family = str(spec["family"])
        template_id = str(spec["template_id"])
        seed = int(spec["seed"])
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

        cf_intervention = cf_intervention_for_family(family)
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

        summary_row = {
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
        rows.append(summary_row)

    summary_path = out_dir / "highdisc_round1_summary.json"
    dump_json(summary_path, {"rows": rows})

    index_path = out_dir / "highdisc_round1_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved high-discriminability round1 pack to {out_dir}")


ROUND2_SPECS = (
    {
        "family": "doorkey",
        "template_id": "migration_doorkey_highdisc_clear_branch_upper",
        "seed": 1431,
        "note": "高区分度扩题：上侧分支目标房，无 clutter，突出 door 后转向分叉。",
    },
    {
        "family": "multiroom",
        "template_id": "migration_multiroom_highdisc_clear_branch_upper",
        "seed": 1432,
        "note": "高区分度扩题：两门后再进入上侧分支目标房，无 clutter。",
    },
    {
        "family": "redblue",
        "template_id": "migration_redblue_highdisc_clear_four_room",
        "seed": 1433,
        "note": "高区分度扩题：四房间结构里保持 red-before-blue 规则，无背景噪声。",
    },
)


def run_round2() -> None:
    ap = argparse.ArgumentParser(description="Generate second high-discriminability MiniWorld Task D expansion pack")
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskD-MW-highdisc-expansion-round2",
    )
    ap.add_argument("--obs_width", type=int, default=320)
    ap.add_argument("--obs_height", type=int, default=240)
    ap.add_argument("--force_lane_z", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = ensure_output_dir(Path(args.out_dir))
    gym, _ = try_import_stack()
    register_local_miniworld_envs(gym)

    rows: list[dict[str, Any]] = []
    for spec in ROUND2_SPECS:
        family = str(spec["family"])
        template_id = str(spec["template_id"])
        seed = int(spec["seed"])
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

        cf_intervention = cf_intervention_for_family(family)
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

        summary_row = {
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
        rows.append(summary_row)

    summary_path = out_dir / "highdisc_round2_summary.json"
    dump_json(summary_path, {"rows": rows})

    index_path = out_dir / "highdisc_round2_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved high-discriminability round2 pack to {out_dir}")


ROUND3_SPECS = (
    {
        "family": "doorkey",
        "template_id": "migration_doorkey_highdisc_clear_branch_lower",
        "seed": 1441,
        "note": "高区分度 round3：下侧分支目标房，镜像于 branch_upper，突出 door 后下转。",
    },
    {
        "family": "doorkey",
        "template_id": "migration_doorkey_highdisc_clear_branch_lower_altopen",
        "seed": 1442,
        "note": "高区分度 round3：下侧分支目标房，门开启方向改为 lower，强化门态视觉差异。",
    },
    {
        "family": "multiroom",
        "template_id": "migration_multiroom_highdisc_clear_branch_lower",
        "seed": 1451,
        "note": "高区分度 round3：两门后必须下转进入更深目标房。",
    },
    {
        "family": "multiroom",
        "template_id": "migration_multiroom_highdisc_clear_branch_lower_altopen",
        "seed": 1452,
        "note": "高区分度 round3：下侧分支目标房，双门开启方向改为 lower，强化门态视觉差异。",
    },
    {
        "family": "redblue",
        "template_id": "migration_redblue_highdisc_clear_four_room_lower",
        "seed": 1461,
        "note": "高区分度 round3：四房间骨架的下侧变体，保持 red-before-blue 规则。",
    },
    {
        "family": "redblue",
        "template_id": "migration_redblue_highdisc_clear_order_zigzag_lower",
        "seed": 1462,
        "note": "高区分度 round3：下侧 zigzag，blue 门置于转角后，顺序错误更显眼。",
    },
)


def run_round3() -> None:
    ap = argparse.ArgumentParser(description="Generate third high-discriminability MiniWorld Task D expansion pack")
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskD-MW-highdisc-expansion-round3",
    )
    ap.add_argument("--obs_width", type=int, default=320)
    ap.add_argument("--obs_height", type=int, default=240)
    ap.add_argument("--force_lane_z", type=float, default=3.0)
    args = ap.parse_args()

    out_dir = ensure_output_dir(Path(args.out_dir))
    gym, _ = try_import_stack()
    register_local_miniworld_envs(gym)

    rows: list[dict[str, Any]] = []
    for spec in ROUND3_SPECS:
        family = str(spec["family"])
        template_id = str(spec["template_id"])
        seed = int(spec["seed"])
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

        cf_intervention = cf_intervention_for_family(family)
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

        pair_ok = bool(full_payload["success"]) and not bool(cf_payload["success"])
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
            "pair_ok": pair_ok,
        }
        dump_json(family_dir / "storyboard_pair_manifest.json", pair_manifest)

        summary_row = {
            "family": family,
            "template_id": template_id,
            "seed": seed,
            "note": spec["note"],
            "cf_intervention": cf_intervention,
            "full_success": full_payload["success"],
            "cf_success": cf_payload["success"],
            "pair_ok": pair_ok,
            "full_event_steps": full_payload["event_steps"],
            "cf_event_steps": cf_payload["event_steps"],
            "layout_metadata": full_payload["layout_metadata"],
            "manifest_relpath": f"{family}/{template_id}/storyboard_full_manifest.json",
            "pair_manifest_relpath": f"{family}/{template_id}/storyboard_pair_manifest.json",
            "full_storyboard_relpath": relpath_payload(full_payload, family_dir=family_dir, out_dir=out_dir)["model_input_image_relpath"],
            "cf_storyboard_relpath": relpath_payload(cf_payload, family_dir=family_dir, out_dir=out_dir)["model_input_image_relpath"],
            "event_coverage_relpath": relpath_payload(full_payload, family_dir=family_dir, out_dir=out_dir).get("event_coverage_image_relpath"),
        }
        rows.append(summary_row)

    summary_path = out_dir / "highdisc_round3_summary.json"
    dump_json(summary_path, {"rows": rows})

    index_path = out_dir / "highdisc_round3_index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved high-discriminability round3 pack to {out_dir}")


if __name__ == '__main__':
    main()
