from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.habitat_types import GeneratorConfig
from utils.habitat_io import relative_path
from tools.habitat.dataset.simulator import configure_simulator
from tools.habitat.dataset.task_design import TASK_D_ARTICULATED_PROMPT_STYLES, action_chunks, evaluate_state_cycle, task_d_articulated_prompts
from tools.habitat.interaction_core import candidate_start_poses, observe, probe, set_agent, set_joint
from utils.habitat_images import save_rgb, visual_change


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render independent successful references for Habitat D-R."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Habitat/outputs/hssd-taskc-articulated-full"),
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=Path(
            "Habitat/data/scene_datasets/hssd-hab/scenes-articulated"
        ),
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=Path,
        default=Path(
            "Habitat/data/scene_datasets/hssd-hab/"
            "hssd-hab-articulated.scene_dataset_config.json"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("Habitat/config.hssd-preview.json"),
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--min-reference-query-difference", type=float, default=0.20)
    parser.add_argument("--min-open-pixel-difference", type=float, default=0.03)
    parser.add_argument("--max-candidates-per-target", type=int, default=36)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument('--reference-phase-attempts', type=int, default=4,
                        help='Random start-angle phase attempts per reference action; does not change acceptance gates')
    parser.add_argument('--taskc-index', type=Path, help='Optional immutable index of completed fresh Task C scenes')
    parser.add_argument('--index-output-root', type=Path, help='Optional separate destination for incremental Task D indexes; images stay under --root')
    args = parser.parse_args()
    if args.reference_phase_attempts <= 0:
        parser.error('--reference-phase-attempts must be positive')
    return args


from utils.habitat_records import read_jsonl


from utils.habitat_records import write_jsonl


def query_images(root: Path, row: dict[str, Any]) -> list[str]:
    episode = root / "episodes" / row["scene"] / row["episode_id"]
    paths = [
        episode / "shared" / f"frame_{index:03d}.png"
        for index in range(4)
    ]
    paths.extend(
        episode / row["condition"] / f"frame_{index:03d}.png"
        for index in range(4, 7)
    )
    return [relative_path(path, root) for path in paths]


def find_object(sim: Any, handle: str) -> Any:
    objects = sim.get_articulated_object_manager().get_objects_by_handle_substring(
        handle
    )
    if handle in objects:
        return objects[handle]
    exact = [obj for key, obj in objects.items() if key == handle]
    if len(exact) != 1:
        raise RuntimeError(f"Could not resolve articulated target {handle}")
    return exact[0]


def render_reference(
    habitat_sim: Any,
    sim: Any,
    obj: Any,
    target: dict[str, Any],
    selected: dict[str, Any],
    episode_dir: Path,
    root: Path,
) -> tuple[list[str], dict[str, Any]]:
    turn_action = selected["turn_action"]
    reference_actions = (
        turn_action,
        "move_forward",
        "move_forward",
        "open",
        "close",
        "stop",
    )
    offset = int(target["joint_offset"])
    closed = float(target["closed_position"])
    opened = float(target["open_position"])
    original = np.asarray(obj.joint_positions, dtype=float).copy()
    set_joint(obj, offset, closed)
    set_agent(
        sim,
        habitat_sim,
        selected["start"],
        selected["yaw_deg"],
    )
    frames = [observe(sim)]
    positions = [
        np.asarray(sim.get_agent(0).get_state().position, dtype=float)
    ]
    collisions = [False]
    sim.step(turn_action)
    frames.append(observe(sim))
    positions.append(
        np.asarray(sim.get_agent(0).get_state().position, dtype=float)
    )
    for _ in range(2):
        before = np.asarray(
            sim.get_agent(0).get_state().position, dtype=float
        )
        sim.step("move_forward")
        after = np.asarray(
            sim.get_agent(0).get_state().position, dtype=float
        )
        collisions.append(float(np.linalg.norm(after - before)) < 0.15)
        positions.append(after)
        frames.append(observe(sim))
    set_joint(obj, offset, opened)
    frames.append(observe(sim))
    set_joint(obj, offset, closed)
    frames.append(observe(sim))
    frames.append(frames[-1].copy())

    image_paths = []
    for index, image in enumerate(frames):
        path = episode_dir / "reference_d" / f"frame_{index:03d}.png"
        save_rgb(path, image)
        image_paths.append(relative_path(path, root))
    progress = [0.0] * 4 + [1.0, 0.0, 0.0]
    target_id = f"{target['instance_id']}|{target['component_id']}"
    evaluation = evaluate_state_cycle(
        reference_actions,
        progress,
        interaction_succeeded=[True] * len(reference_actions),
        interaction_targets=[None, None, None, target_id, target_id, None],
        target_id=target_id,
    )
    obj.joint_positions = original
    obj.joint_velocities = np.zeros_like(original)
    return image_paths, {
        "actions": list(reference_actions),
        "action_chunks": action_chunks(
            reference_actions, list(range(len(frames)))
        ),
        "positions": [position.tolist() for position in positions]
        + [positions[-1].tolist()] * 3,
        "collisions": collisions + [False, False, False],
        "target_progress_by_frame": progress,
        "oracle": evaluation,
        "reference_query_view_difference": selected[
            "avoid_image_change"
        ],
        "open_visual_change": selected["visual_change"],
    }


def make_rows(
    root: Path,
    full: dict[str, Any],
    cf: dict[str, Any],
    reference_images: list[str],
    reference_metadata: dict[str, Any],
    prompt_style: str,
) -> list[dict[str, Any]]:
    prompt_en, prompt_zh = task_d_articulated_prompts(prompt_style)
    rows = []
    for query in (full, cf):
        condition = query["condition"]
        rows.append(
            {
                "uid": (
                    f"D-R.HAB.ART.{query['episode_id']}."
                    f"{condition.upper()}"
                ),
                "task": "D-R",
                "task_family": "articulated_reference_outcome",
                "episode_id": query["episode_id"],
                "scene": query["scene"],
                "condition": condition,
                "prompt": prompt_en,
                "prompt_zh": prompt_zh,
                "prompt_style": prompt_style,
                "prompt_template_id": f"taskd_articulated_{prompt_style}_v1",
                "frame_policy": "all_frames_in_chronological_order",
                "reference": {
                    "images": reference_images,
                    "action_sequence": reference_metadata["actions"],
                    "action_chunks": reference_metadata["action_chunks"],
                    "gold": "success",
                    "known_success": True,
                },
                "query": {
                    "condition": condition,
                    "images": query_images(root, query),
                    "action_sequence": query["action_sequence"],
                    "action_chunks": action_chunks(
                        query["action_sequence"], list(range(7))
                    ),
                    "gold": query["gold"],
                    "intervention": query["oracle"]["cf_intervention"],
                },
                "target_oracle": query["target"],
                "reference_oracle": reference_metadata,
                "gold": query["gold"],
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    index_root = args.index_output_root.resolve() if args.index_output_root else root
    config = GeneratorConfig.from_json(args.config)
    rows_c = read_jsonl(args.taskc_index.resolve() if args.taskc_index else root / "taskc" / "task_c.jsonl")
    by_episode: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows_c:
        by_episode[row["episode_id"]][row["condition"]] = row
    episodes = sorted(by_episode)
    if args.max_targets > 0:
        episodes = episodes[: args.max_targets]
    prompt_style_by_episode = {
        episode_id: TASK_D_ARTICULATED_PROMPT_STYLES[
            index % len(TASK_D_ARTICULATED_PROMPT_STYLES)
        ]
        for index, episode_id in enumerate(episodes)
    }
    by_scene: dict[str, list[str]] = defaultdict(list)
    for episode_id in episodes:
        by_scene[by_episode[episode_id]["full"]["scene"]].append(episode_id)

    all_rows: list[dict[str, Any]] = []
    records = []
    rejections = []
    rng = np.random.default_rng(args.seed)
    total = len(episodes)
    done = 0
    for scene_index, (scene_slug, scene_episodes) in enumerate(
        sorted(by_scene.items()), start=1
    ):
        scene_path = args.scene_root / f"{scene_slug}.scene_instance.json"
        print(
            f"[scene {scene_index}/{len(by_scene)}] {scene_slug}",
            flush=True,
        )
        habitat_sim, sim = configure_simulator(
            str(scene_path.resolve()),
            config,
            str(args.scene_dataset_config.resolve()),
        )
        try:
            for episode_id in scene_episodes:
                done += 1
                pair = by_episode[episode_id]
                full, cf = pair["full"], pair["cf"]
                episode_dir = root / "episodes" / scene_slug / episode_id
                cached = episode_dir / "taskd_reference.json"
                cached_rows = episode_dir / "taskd_rows.json"
                if cached.is_file() and cached_rows.is_file():
                    metadata = json.loads(cached.read_text(encoding="utf-8"))
                    rows = json.loads(cached_rows.read_text(encoding="utf-8"))
                    prompt_style = prompt_style_by_episode[episode_id]
                    prompt_en, prompt_zh = task_d_articulated_prompts(
                        prompt_style
                    )
                    for row in rows:
                        row["prompt"] = prompt_en
                        row["prompt_zh"] = prompt_zh
                        row["prompt_style"] = prompt_style
                        row["prompt_template_id"] = (
                            f"taskd_articulated_{prompt_style}_v1"
                        )
                    cached_rows.write_text(
                        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    all_rows.extend(rows)
                    records.append(metadata)
                    print(f"  [{done}/{total}] resume {episode_id}", flush=True)
                    continue
                target = full["target"]
                obj = find_object(sim, target["instance_id"])
                component = {
                    key: target[key]
                    for key in (
                        "link_id",
                        "component_id",
                        "joint_type",
                        "joint_offset",
                        "closed_position",
                        "open_position",
                    )
                }
                target_point = np.asarray(obj.translation, dtype=float)
                query_view = np.asarray(
                    Image.open(root / full["goal_image"]).convert("RGB")
                )
                selected = None
                # Existing references prefer a right turn because Task C
                # queries use a left turn. For uncached hard targets, try an
                # independent start with a different camera pitch first.
                for turn_action in (
                    "look_down",
                    "look_up",
                    "turn_right",
                    "turn_left",
                ):
                    for _ in range(args.reference_phase_attempts):
                        starts = candidate_start_poses(
                            sim, target_point, rng
                        )
                        if turn_action == "turn_right":
                            starts = [
                                (position, yaw + 60.0)
                                for position, yaw in starts
                            ]
                        elif turn_action in {"look_down", "look_up"}:
                            # candidate_start_poses anticipates a 30-degree
                            # left turn. Point directly at the target before
                            # changing camera pitch instead.
                            starts = [
                                (position, yaw + 30.0)
                                for position, yaw in starts
                            ]
                        selected = probe(
                            habitat_sim,
                            sim,
                            obj,
                            component,
                            target_point,
                            starts[: args.max_candidates_per_target],
                            args.min_open_pixel_difference,
                            turn_action=turn_action,
                            avoid_image=query_view,
                            min_avoid_difference=(
                                args.min_reference_query_difference
                            ),
                        )
                        if selected is not None:
                            break
                    if selected is not None:
                        break
                if selected is None:
                    rejections.append(
                        {
                            "episode_id": episode_id,
                            "scene": scene_slug,
                            "reason": (
                                "no_distinct_successful_reference_view"
                            ),
                        }
                    )
                    print(
                        f"  [{done}/{total}] reject {episode_id}",
                        flush=True,
                    )
                    continue
                reference_images, reference_metadata = render_reference(
                    habitat_sim,
                    sim,
                    obj,
                    target,
                    selected,
                    episode_dir,
                    root,
                )
                rows = make_rows(
                    root,
                    full,
                    cf,
                    reference_images,
                    reference_metadata,
                    prompt_style_by_episode[episode_id],
                )
                metadata = {
                    "episode_id": episode_id,
                    "scene": scene_slug,
                    "target": target,
                    **reference_metadata,
                }
                cached.write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                cached_rows.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                all_rows.extend(rows)
                records.append(metadata)
                print(
                    f"  [{done}/{total}] accept {episode_id} "
                    f"ref-query-diff="
                    f"{selected['avoid_image_change']['changed_pixel_fraction']:.3f}",
                    flush=True,
                )
                write_jsonl(index_root / "taskd" / "task_d.jsonl", all_rows)
        finally:
            sim.close()

    write_jsonl(index_root / "taskd" / "task_d.jsonl", all_rows)
    write_jsonl(index_root / "taskd" / "rejections.jsonl", rejections)
    manifest = {
        "schema_version": "hssd-taskd-articulated-reference-v1",
        "source_task_c": "taskc/task_c.jsonl",
        "target_count": len(records),
        "question_count": len(all_rows),
        "balance": dict(Counter(row["gold"] for row in all_rows)),
        "condition_counts": dict(
            Counter(row["condition"] for row in all_rows)
        ),
        "prompt_style_counts": dict(
            Counter(row["prompt_style"] for row in all_rows)
        ),
        "rejection_count": len(rejections),
        "reference_contract": {
            "same_scene": True,
            "same_target_instance_and_component": True,
            "independent_start_and_view": True,
            "reference_known_success": True,
            "query_full_cf_same_actions": True,
            "minimum_reference_query_pixel_difference": (
                args.min_reference_query_difference
            ),
        },
        "records": records,
    }
    (index_root / "taskd" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "targets": len(records),
                "questions": len(all_rows),
                "balance": manifest["balance"],
                "conditions": manifest["condition_counts"],
                "rejections": len(rejections),
            },
            indent=2,
        )
    )
    return 0 if len(records) == len(episodes) else 2


if __name__ == "__main__":
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    raise SystemExit(main())
