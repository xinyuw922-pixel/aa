from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from utils.habitat_types import GeneratorConfig
from utils.habitat_io import relative_path
from tools.habitat.dataset.simulator import configure_simulator, pose_from_agent_state, yaw_quaternion
from tools.habitat.dataset.task_design import action_chunks, critical_frame_selection, evaluate_state_cycle, task_c_state_sequence_prompt
from tools.habitat.objects import COMPONENT_PATTERN, family_for


FAMILY_ZH = {
    "refrigerator": ("目标冰箱", "冰箱门"),
    "oven": ("目标烤箱或微波炉", "门"),
    "wardrobe": ("目标衣柜", "衣柜门或抽屉"),
    "cabinet": ("目标橱柜或储物柜", "柜门或抽屉"),
    "drawer_storage": ("目标抽屉家具", "柜门或抽屉"),
    "appliance": ("目标家电", "门或抽屉"),
}
FAMILY_ORDER = {
    family: index
    for index, family in enumerate(
        ("oven", "appliance", "refrigerator", "wardrobe", "drawer_storage", "cabinet")
    )
}
ACTIONS = ("turn_left", "move_forward", "move_forward", "open", "close", "stop")
INTERACTION_TARGET_PATTERN = re.compile(r"door|drawer|cabinet|lid", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render HSSD articulated OPEN-CLOSE-STOP Task C pairs."
    )
    parser.add_argument(
        "--scene-glob",
        default=(
            "Habitat/data/scene_datasets/hssd-hab/"
            "scenes-articulated/*.scene_instance.json"
        ),
    )
    parser.add_argument(
        "--scene-dataset-config",
        default=(
            "Habitat/data/scene_datasets/hssd-hab/"
            "hssd-hab-articulated.scene_dataset_config.json"
        ),
    )
    parser.add_argument(
        "--config", default="Habitat/config.hssd-preview.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Habitat/outputs/hssd-taskc-articulated"),
    )
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help=(
            "Stop after this many accepted interaction instances globally. "
            "Each instance produces one Full and one CF question; zero is "
            "unlimited."
        ),
    )
    parser.add_argument("--max-targets-per-scene", type=int, default=6)
    parser.add_argument("--max-probes-per-scene", type=int, default=48)
    parser.add_argument("--min-open-pixel-difference", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Explicit scene path; repeat to select a pilot set.",
    )
    return parser.parse_args()


from utils.habitat_images import save_rgb


from utils.habitat_images import image_qc


from utils.habitat_images import visual_change


from tools.habitat.interaction_core import set_agent


from tools.habitat.interaction_core import observe


from utils.habitat_geometry import yaw_to_target


from tools.habitat.interaction_core import set_joint


def eligible_components(obj: Any) -> list[dict[str, Any]]:
    positions = np.asarray(obj.joint_positions, dtype=float)
    lower, upper = (
        np.asarray(values, dtype=float) for values in obj.joint_position_limits
    )
    components = []
    seen_offsets = set()
    for link_id in obj.get_link_ids():
        joint_type = str(obj.get_link_joint_type(link_id))
        if not (
            joint_type.endswith(".Revolute")
            or joint_type.endswith(".Prismatic")
        ):
            continue
        name = obj.get_link_joint_name(link_id)
        if not (
            COMPONENT_PATTERN.search(name)
            and INTERACTION_TARGET_PATTERN.search(name)
        ):
            continue
        offset = int(obj.get_link_joint_pos_offset(link_id))
        if offset in seen_offsets or not 0 <= offset < len(positions):
            continue
        closed = float(positions[offset])
        endpoints = [float(lower[offset]), float(upper[offset])]
        opened = max(endpoints, key=lambda value: abs(value - closed))
        if abs(opened - closed) <= 1e-4:
            continue
        seen_offsets.add(offset)
        components.append(
            {
                "link_id": int(link_id),
                "component_id": name,
                "joint_type": joint_type.rsplit(".", 1)[-1].lower(),
                "joint_offset": offset,
                "closed_position": closed,
                "open_position": opened,
            }
        )
    return components


def object_family(obj: Any) -> tuple[str | None, str]:
    names = [
        obj.get_link_name(link_id)
        for link_id in obj.get_link_ids()
        if str(obj.get_link_joint_type(link_id)).endswith(".Fixed")
    ]
    if not names:
        names = [obj.get_link_name(min(obj.get_link_ids()))]
    semantic_name = names[0]
    return family_for(semantic_name), semantic_name


from tools.habitat.interaction_core import candidate_start_poses


from tools.habitat.interaction_core import probe


def variant_joint_progress(kind: str) -> tuple[float, float, list[bool]]:
    if kind == "open_no_effect":
        return 0.0, 0.0, [False, True]
    if kind == "partial_open":
        return 0.30, 0.0, [True, True]
    if kind == "close_no_effect":
        return 1.0, 1.0, [True, False]
    raise ValueError(kind)


def render_target_pair(
    habitat_sim: Any,
    sim: Any,
    obj: Any,
    family: str,
    semantic_name: str,
    component: dict[str, Any],
    probe_result: dict[str, Any],
    scene_slug: str,
    target_number: int,
    output_root: Path,
    cf_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_id = f"{obj.handle}|{component['component_id']}"
    episode_id = f"{scene_slug}_{target_number:03d}"
    episode_dir = output_root / "episodes" / scene_slug / episode_id
    original_positions = np.asarray(obj.joint_positions, dtype=float).copy()
    open_position = component["open_position"]
    closed_position = component["closed_position"]
    frame_paths: dict[str, list[str]] = {"full": [], "cf": []}

    for index, image in enumerate(probe_result["navigation_frames"]):
        path = episode_dir / "shared" / f"frame_{index:03d}.png"
        save_rgb(path, image)
        frame_paths["full"].append(relative_path(path, output_root))
        frame_paths["cf"].append(relative_path(path, output_root))

    set_agent(
        sim,
        habitat_sim,
        probe_result["positions"][-1],
        probe_result["yaw_deg"] + 30.0,
    )
    set_joint(obj, component["joint_offset"], open_position)
    full_open = observe(sim)
    set_joint(obj, component["joint_offset"], closed_position)
    full_closed = observe(sim)
    for index, image in enumerate((full_open, full_closed, full_closed), start=4):
        path = episode_dir / "full" / f"frame_{index:03d}.png"
        save_rgb(path, image)
        frame_paths["full"].append(relative_path(path, output_root))

    cf_open_progress, cf_close_progress, cf_interactions = variant_joint_progress(cf_kind)
    cf_open_position = closed_position + (
        open_position - closed_position
    ) * cf_open_progress
    cf_close_position = closed_position + (
        open_position - closed_position
    ) * cf_close_progress
    set_joint(obj, component["joint_offset"], cf_open_position)
    cf_open = observe(sim)
    set_joint(obj, component["joint_offset"], cf_close_position)
    cf_closed = observe(sim)
    for index, image in enumerate((cf_open, cf_closed, cf_closed), start=4):
        path = episode_dir / "cf" / f"frame_{index:03d}.png"
        save_rgb(path, image)
        frame_paths["cf"].append(relative_path(path, output_root))

    full_progress = [0.0] * 4 + [1.0, 0.0, 0.0]
    cf_progress = [0.0] * 4 + [
        cf_open_progress,
        cf_close_progress,
        cf_close_progress,
    ]
    common_success = [True, True, True] + [True, True, True]
    full_eval = evaluate_state_cycle(
        ACTIONS,
        full_progress,
        interaction_succeeded=common_success,
        interaction_targets=[None, None, None, target_id, target_id, None],
        target_id=target_id,
    )
    cf_eval = evaluate_state_cycle(
        ACTIONS,
        cf_progress,
        interaction_succeeded=[True, True, True, *cf_interactions, True],
        interaction_targets=[None, None, None, target_id, target_id, None],
        target_id=target_id,
    )
    collision_flags = [False, *probe_result["movement_collisions"][1:], False, False, False]
    indices, reasons = critical_frame_selection(
        ACTIONS,
        collision_flags,
        positions=[position.tolist() for position in probe_result["positions"]]
        + [probe_result["positions"][-1].tolist()] * 3,
        min_frames=6,
        event_frames=full_eval["event_frames"],
    )
    target_description, component_description = FAMILY_ZH[family]
    prompt = task_c_state_sequence_prompt(
        target_description, component_description
    )
    rows = []
    for condition, evaluation, progress in (
        ("full", full_eval, full_progress),
        ("cf", cf_eval, cf_progress),
    ):
        rows.append(
            {
                "uid": f"C.HAB.ART.{episode_id}.{condition.upper()}",
                "task": "C",
                "task_family": "articulated_state_sequence",
                "episode_id": episode_id,
                "scene": scene_slug,
                "condition": condition,
                "goal_image": frame_paths["full"][3],
                "prompt": prompt,
                "prompt_zh": prompt,
                "target": {
                    "category": family,
                    "instance_id": obj.handle,
                    "semantic_name": semantic_name,
                    **component,
                },
                "required_events": ["open", "close", "stop"],
                "thresholds": {"open": 0.60, "closed": 0.05},
                "keyframe_indices": indices,
                "keyframe_images": [frame_paths[condition][i] for i in indices],
                "action_sequence": list(ACTIONS),
                "action_chunks": action_chunks(ACTIONS, indices),
                "trajectory_metadata": relative_path(
                    episode_dir / "episode.json", output_root
                ),
                "selection_rule": "articulated_event_complete_v1",
                "keyframe_reasons_oracle": {
                    str(index): values for index, values in reasons.items()
                },
                "oracle": {
                    "target_progress_by_frame": progress,
                    **evaluation,
                    "cf_intervention": None if condition == "full" else cf_kind,
                },
                "gold": evaluation["outcome"].lower(),
            }
        )
    metadata = {
        "episode_id": episode_id,
        "scene": scene_slug,
        "family": family,
        "target_id": target_id,
        "semantic_name": semantic_name,
        "component": component,
        "cf_kind": cf_kind,
        "visual_change": probe_result["visual_change"],
        "image_qc": probe_result["image_qc"],
        "full_outcome": full_eval["outcome"],
        "cf_outcome": cf_eval["outcome"],
    }
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "episode.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    obj.joint_positions = original_positions
    obj.joint_velocities = np.zeros_like(original_positions)
    return rows, metadata


from utils.habitat_records import write_jsonl


def main() -> int:
    args = parse_args()
    config = GeneratorConfig.from_json(args.config)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.scene:
        scenes = [str(Path(path).resolve()) for path in args.scene]
    else:
        scenes = sorted(
            str(Path(path).resolve())
            for path in glob.glob(args.scene_glob)
            if Path(path).is_file()
        )
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    dataset_config = str(Path(args.scene_dataset_config).resolve())
    rng = np.random.default_rng(args.seed)
    all_rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []
    rejection_path = output_root / "rejections.jsonl"
    rejections: list[dict[str, Any]] = []
    if rejection_path.is_file():
        rejections = [
            json.loads(line)
            for line in rejection_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    family_counts: Counter[str] = Counter()

    for scene_index, scene in enumerate(scenes, start=1):
        if args.max_episodes > 0 and len(episode_records) >= args.max_episodes:
            break
        slug = Path(scene).name.removesuffix(".scene_instance.json")
        scene_done = output_root / "episodes" / slug / "_scene_complete.json"
        if scene_done.is_file():
            payload = json.loads(scene_done.read_text(encoding="utf-8"))
            for row in payload["rows"]:
                row.setdefault(
                    "trajectory_metadata",
                    (
                        f"episodes/{slug}/{row['episode_id']}/"
                        "episode.json"
                    ),
                )
                all_rows.append(row)
            episode_records.extend(payload["episodes"])
            family_counts.update(payload["family_counts"])
            print(f"[{scene_index}/{len(scenes)}] resume {slug}", flush=True)
            continue
        print(f"[{scene_index}/{len(scenes)}] load {slug}", flush=True)
        habitat_sim, sim = configure_simulator(scene, config, dataset_config)
        scene_rows = []
        scene_episodes = []
        scene_family_counts: Counter[str] = Counter()
        probes = 0
        try:
            objects = []
            manager = sim.get_articulated_object_manager()
            for _, obj in manager.get_objects_by_handle_substring("").items():
                family, semantic_name = object_family(obj)
                if family not in FAMILY_ZH:
                    continue
                components = eligible_components(obj)
                if not components:
                    continue
                objects.append((family, semantic_name, obj, components))
            rng.shuffle(objects)
            objects.sort(
                key=lambda item: (
                    scene_family_counts[item[0]],
                    FAMILY_ORDER[item[0]],
                    item[1],
                    item[2].handle,
                )
            )
            accepted_objects: set[str] = set()
            while (
                objects
                and len(scene_episodes) < args.max_targets_per_scene
                and probes < args.max_probes_per_scene
                and (
                    args.max_episodes <= 0
                    or len(episode_records) + len(scene_episodes)
                    < args.max_episodes
                )
            ):
                objects.sort(
                    key=lambda item: (
                        scene_family_counts[item[0]],
                        FAMILY_ORDER[item[0]],
                        item[1],
                        item[2].handle,
                    )
                )
                family, semantic_name, obj, components = objects.pop(0)
                if obj.handle in accepted_objects:
                    continue
                target = np.asarray(obj.translation, dtype=float)
                starts = candidate_start_poses(sim, target, rng)
                best_component = None
                best_probe = None
                for component in components[:3]:
                    if probes >= args.max_probes_per_scene:
                        break
                    probes += 1
                    result = probe(
                        habitat_sim,
                        sim,
                        obj,
                        component,
                        target,
                        starts[:24],
                        args.min_open_pixel_difference,
                    )
                    if result is not None and (
                        best_probe is None
                        or result["visual_change"]["changed_pixel_fraction"]
                        > best_probe["visual_change"]["changed_pixel_fraction"]
                    ):
                        best_component = component
                        best_probe = result
                if best_probe is None or best_component is None:
                    rejections.append(
                        {
                            "scene": slug,
                            "instance_id": obj.handle,
                            "semantic_name": semantic_name,
                            "family": family,
                            "reason": "no_reachable_visually_distinct_probe",
                        }
                    )
                    continue
                cf_kinds = ("open_no_effect", "partial_open", "close_no_effect")
                cf_kind = cf_kinds[len(scene_episodes) % len(cf_kinds)]
                rows, metadata = render_target_pair(
                    habitat_sim,
                    sim,
                    obj,
                    family,
                    semantic_name,
                    best_component,
                    best_probe,
                    slug,
                    len(scene_episodes) + 1,
                    output_root,
                    cf_kind,
                )
                scene_rows.extend(rows)
                scene_episodes.append(metadata)
                scene_family_counts[family] += 1
                accepted_objects.add(obj.handle)
                print(
                    f"  accept {family} {semantic_name} "
                    f"diff={best_probe['visual_change']['changed_pixel_fraction']:.3f}",
                    flush=True,
                )
        finally:
            sim.close()
        scene_payload = {
            "scene": slug,
            "rows": scene_rows,
            "episodes": scene_episodes,
            "family_counts": dict(scene_family_counts),
            "probes": probes,
        }
        scene_done.parent.mkdir(parents=True, exist_ok=True)
        scene_done.write_text(
            json.dumps(scene_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        all_rows.extend(scene_rows)
        episode_records.extend(scene_episodes)
        family_counts.update(scene_family_counts)
        write_jsonl(output_root / "taskc" / "task_c.jsonl", all_rows)
        write_jsonl(rejection_path, rejections)

    write_jsonl(output_root / "taskc" / "task_c.jsonl", all_rows)
    write_jsonl(rejection_path, rejections)
    # Keep the current candidate-pool web viewer usable without changing its
    # three-index contract. These files intentionally contain zero rows.
    write_jsonl(output_root / "taska" / "task_a.jsonl", [])
    write_jsonl(output_root / "taskd" / "task_d.jsonl", [])
    manifest = {
        "schema_version": "hssd-taskc-articulated-render-v1",
        "resolution": [config.width, config.height],
        "scene_count": len(scenes),
        "completed_scene_count": sum(
            (
                output_root / "episodes"
                / Path(scene).name.removesuffix(".scene_instance.json")
                / "_scene_complete.json"
            ).is_file()
            for scene in scenes
        ),
        "episode_count": len(episode_records),
        "max_episode_limit": args.max_episodes or None,
        "question_count": len(all_rows),
        "balance": dict(Counter(row["gold"] for row in all_rows)),
        "family_counts": dict(family_counts),
        "cf_counts": dict(
            Counter(
                row["oracle"]["cf_intervention"]
                for row in all_rows
                if row["condition"] == "cf"
            )
        ),
        "episodes": episode_records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: manifest[k] for k in (
        "completed_scene_count", "episode_count", "question_count",
        "balance", "family_counts", "cf_counts"
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    raise SystemExit(main())
