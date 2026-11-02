from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from utils.habitat_types import GeneratorConfig, episode_seed
from utils.habitat_io import relative_path, save_rgb_png, scene_slug, write_json, write_jsonl
from tools.habitat.dataset.simulator import EpisodeSamplingError, configure_simulator, find_cf_yaw, render_goal_image, run_rollout, sample_expert_episode
from tools.habitat.dataset.task_design import TASK_C_PROMPT, TASK_D_REFERENCE_PROMPT, action_chunks, shared_critical_frame_selection


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _write_trajectory(path: Path, rollout: Any) -> None:
    write_jsonl(path, (transition.to_dict() for transition in rollout.transitions))


def sample_episode_pair(habitat_sim: Any, simulator: Any,
                        config: GeneratorConfig, seed: int) -> tuple[Any, Any, int, int]:
    """Preflight a distinct, same-goal successful reference before rendering."""
    last_error: EpisodeSamplingError | None = None
    for query_attempt in range(config.reference_sampling_retries):
        query_seed = seed + query_attempt * 100_000_000
        try:
            query = sample_expert_episode(habitat_sim, simulator, config, query_seed)
        except EpisodeSamplingError as exc:
            last_error = exc
            continue
        for reference_attempt in range(config.reference_sampling_retries):
            reference_seed = query_seed + 50_000_000 + reference_attempt * 1_000_000
            try:
                reference = sample_expert_episode(
                    habitat_sim, simulator, config, reference_seed,
                    fixed_goal=query.goal, excluded_start=query.start,
                )
                return query, reference, query_seed, reference_seed
            except EpisodeSamplingError as exc:
                last_error = exc
        print(f"No independent reference for query seed {query_seed}; "
              "resampling the pair with unchanged constraints", flush=True)
    raise EpisodeSamplingError(
        f"Could not sample a valid query/reference pair for seed={seed} "
        f"after {config.reference_sampling_retries} query seeds"
    ) from last_error


def _copy_current_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _distance_from_last_pose(
    transitions: list[dict[str, Any]],
    goal: list[float],
) -> float:
    final_position = np.asarray(
        transitions[-1]["pose"]["position"], dtype=float
    )
    return float(np.linalg.norm(final_position - np.asarray(goal, dtype=float)))


def _load_completed_episode(
    output_root: Path,
    episode_dir: Path,
    condition_index: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Rebuild global index rows from a completed on-disk episode."""
    episode = json.loads(
        (episode_dir / "episode.json").read_text(encoding="utf-8")
    )
    task_a = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((episode_dir / "taska").glob("*/question.json"))
    ]
    task_c = json.loads(
        (episode_dir / "taskc" / "question.json").read_text(encoding="utf-8")
    )
    full = _read_jsonl(episode_dir / "full" / "trajectory.jsonl")
    cf = _read_jsonl(episode_dir / "cf" / "trajectory.jsonl")
    reference = _read_jsonl(
        episode_dir / "reference" / "trajectory.jsonl"
    )
    d_variant = "cf" if condition_index % 2 == 0 else "full"
    query = cf if d_variant == "cf" else full
    query_gold = "fail" if d_variant == "cf" else "success"
    reference_distance = _distance_from_last_pose(
        reference, episode["goal"]
    )
    query_distance = _distance_from_last_pose(query, episode["goal"])
    full_distance = _distance_from_last_pose(full, episode["goal"])
    cf_distance = _distance_from_last_pose(cf, episode["goal"])
    task_d = {
        "uid": (
            f"D-R.HAB.{episode['episode_id']}.{d_variant.upper()}"
        ),
        "task": "D-R",
        "episode_id": episode["episode_id"],
        "seed": episode["seed"],
        "goal_image": episode["goal_image"],
        "reference": {
            "images": [row["image_path"] for row in reference],
            "action_sequence": episode["reference_action_sequence"],
            "trajectory_metadata": relative_path(
                episode_dir / "reference" / "trajectory.jsonl",
                output_root,
            ),
            "gold": "success",
            "final_distance_to_goal_m": reference_distance,
            "seed": episode["reference_seed"],
            "start": episode["reference_start"],
        },
        "query": {
            "condition": d_variant,
            "images": [row["image_path"] for row in query],
            "action_sequence": episode["action_sequence"],
            "trajectory_metadata": relative_path(
                episode_dir / d_variant / "trajectory.jsonl",
                output_root,
            ),
            "gold": query_gold,
            "final_distance_to_goal_m": query_distance,
        },
    }
    if d_variant == "cf":
        task_d["query"]["intervention"] = {
            "type": "initial_yaw_offset_deg",
            "value": episode["cf_initial_yaw_offset_deg"],
            "same_start_position": True,
            "same_action_sequence": True,
        }
    action_seed = {
        "uid": f"SEED.HAB.{episode['episode_id']}",
        "schema_version": "habitat-action-seed-v1",
        "episode_id": episode["episode_id"],
        "scene": episode["scene"],
        "scene_slug": episode["scene_slug"],
        "seed": episode["seed"],
        "start": {
            "position": episode["start"],
            "yaw_deg": episode["initial_yaw_deg"],
        },
        "goal": {
            "position": episode["goal"],
            "render_pose": episode["goal_render_pose"],
            "image": episode["goal_image"],
        },
        "geodesic_distance_m": episode["geodesic_distance_m"],
        "action_sequence": episode["action_sequence"],
        "reference": {
            "seed": episode["reference_seed"],
            "start": {
                "position": episode["reference_start"],
                "yaw_deg": episode["reference_initial_yaw_deg"],
            },
            "action_sequence": episode["reference_action_sequence"],
            "gold": "success",
            "final_distance_to_goal_m": reference_distance,
        },
        "cf_intervention": {
            "type": "initial_yaw_offset_deg",
            "value": episode["cf_initial_yaw_offset_deg"],
            "same_start_position": True,
            "same_goal": True,
            "same_action_sequence": True,
        },
        "oracle_outcomes": {
            "full": {
                "gold": "success",
                "final_distance_to_goal_m": full_distance,
            },
            "cf": {
                "gold": "fail",
                "final_distance_to_goal_m": cf_distance,
            },
        },
    }
    return task_a, task_c, task_d, action_seed


def generate_dataset(
    scenes: list[str],
    config: GeneratorConfig,
    scene_dataset_config: str | None = None,
) -> dict[str, Any]:
    if not scenes:
        raise ValueError("No scenes were supplied")
    if config.generate_task_a:
        raise ValueError(
            "In-episode Task A generation is disabled because alternative "
            "actions at one state violate the canonical same-action "
            "distractor contract. Render trajectories with generate_task_a="
            "false, then run Habitat/build_template_candidate_pool.py."
        )
    scenes = scenes[: config.max_scenes]
    output_root = Path(config.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    task_a_rows: list[dict[str, Any]] = []
    task_c_rows: list[dict[str, Any]] = []
    task_d_rows: list[dict[str, Any]] = []
    action_seed_rows: list[dict[str, Any]] = []
    episodes_written = 0
    habitat_version = "unknown"

    for scene_index, scene in enumerate(scenes):
        habitat_sim, simulator = configure_simulator(
            scene, config, scene_dataset_config
        )
        habitat_version = _version(habitat_sim)
        slug = scene_slug(scene)
        try:
            for episode_index in range(config.episodes_per_scene):
                seed = episode_seed(
                    config.base_seed, scene_index, episode_index
                )
                episode_id = f"{slug}_s{seed:010d}"
                episode_dir = output_root / "episodes" / slug / episode_id
                condition_index = (
                    scene_index * config.episodes_per_scene + episode_index
                )
                if (episode_dir / "episode.json").is_file():
                    (
                        existing_a,
                        existing_c,
                        existing_d,
                        existing_seed,
                    ) = _load_completed_episode(
                        output_root, episode_dir, condition_index
                    )
                    task_a_rows.extend(existing_a)
                    task_c_rows.append(existing_c)
                    task_d_rows.append(existing_d)
                    action_seed_rows.append(existing_seed)
                    episodes_written += 1
                    continue
                sampled, reference_sampled, sampling_seed, reference_seed = sample_episode_pair(
                    habitat_sim, simulator, config, seed
                )

                full = run_rollout(
                    habitat_sim=habitat_sim,
                    simulator=simulator,
                    start=sampled.start,
                    yaw_deg=sampled.initial_yaw_deg,
                    goal=sampled.goal,
                    actions=sampled.actions,
                    output_dir=episode_dir / "full",
                    output_root=output_root,
                    goal_radius_m=config.goal_radius_m,
                    forward_step_m=config.forward_step_m,
                )
                if not full.success:
                    raise RuntimeError(
                        f"Expert rollout failed unexpectedly: {episode_id}"
                    )
                _write_trajectory(
                    episode_dir / "full" / "trajectory.jsonl", full
                )
                goal_image, goal_pose = render_goal_image(
                    habitat_sim=habitat_sim,
                    simulator=simulator,
                    goal_position=sampled.goal,
                    rotation_xyzw=full.transitions[-1].pose.rotation_xyzw,
                    output_path=episode_dir / "goal_image.png",
                    output_root=output_root,
                )

                cf_offset, cf_probe_distance, cf_probe_success = find_cf_yaw(
                    habitat_sim,
                    simulator,
                    sampled,
                    config.cf_yaw_offsets_deg,
                    config.goal_radius_m,
                )
                if cf_probe_success:
                    raise RuntimeError(
                        "No configured initial-yaw intervention changed the "
                        f"outcome for {episode_id}; farthest replay ended "
                        f"{cf_probe_distance:.3f} m from the goal. Expand "
                        "cf_yaw_offsets_deg or resample this episode."
                    )
                cf = run_rollout(
                    habitat_sim=habitat_sim,
                    simulator=simulator,
                    start=sampled.start,
                    yaw_deg=sampled.initial_yaw_deg + cf_offset,
                    goal=sampled.goal,
                    actions=sampled.actions,
                    output_dir=episode_dir / "cf",
                    output_root=output_root,
                    goal_radius_m=config.goal_radius_m,
                    forward_step_m=config.forward_step_m,
                )
                _write_trajectory(
                    episode_dir / "cf" / "trajectory.jsonl", cf
                )
                if cf.success:
                    raise RuntimeError(
                        f"CF replay did not flip the outcome: {episode_id}"
                    )

                reference = run_rollout(
                    habitat_sim=habitat_sim,
                    simulator=simulator,
                    start=reference_sampled.start,
                    yaw_deg=reference_sampled.initial_yaw_deg,
                    goal=sampled.goal,
                    actions=reference_sampled.actions,
                    output_dir=episode_dir / "reference",
                    output_root=output_root,
                    goal_radius_m=config.goal_radius_m,
                    forward_step_m=config.forward_step_m,
                )
                if not reference.success:
                    raise RuntimeError(
                        f"Reference rollout failed unexpectedly: {episode_id}"
                    )
                _write_trajectory(
                    episode_dir / "reference" / "trajectory.jsonl",
                    reference,
                )

                motion_actions = tuple(
                    action for action in sampled.actions if action != "stop"
                )
                c_variant = "full" if condition_index % 2 == 0 else "cf"
                c_rollout = full if c_variant == "full" else cf
                full_collisions = [
                    transition.collided
                    for transition in full.transitions[1:]
                ]
                cf_collisions = [
                    transition.collided
                    for transition in cf.transitions[1:]
                ]
                c_indices, c_reasons = shared_critical_frame_selection(
                    motion_actions,
                    full_collisions,
                    cf_collisions,
                    full_positions=[
                        transition.pose.position
                        for transition in full.transitions
                    ],
                    cf_positions=[
                        transition.pose.position
                        for transition in cf.transitions
                    ],
                    goal=sampled.goal,
                    goal_radius_m=config.goal_radius_m,
                )
                c_images = [
                    c_rollout.transitions[index].image_path
                    for index in c_indices
                ]
                task_c_row = {
                    "uid": f"C.HAB.{episode_id}.{c_variant.upper()}",
                    "task": "C",
                    "episode_id": episode_id,
                    "seed": seed,
                    "condition": c_variant,
                    "goal_image": goal_image,
                    "prompt": TASK_C_PROMPT,
                    "keyframe_indices": c_indices,
                    "keyframe_images": c_images,
                    "action_sequence": list(sampled.actions),
                    "action_chunks": action_chunks(
                        motion_actions, c_indices
                    ),
                    "trajectory_metadata": relative_path(
                        episode_dir / c_variant / "trajectory.jsonl",
                        output_root,
                    ),
                    "selection_rule": (
                        "shared_full_cf:event_complete+collision_run_onset+"
                        "goal_entry+distance_progress+terminal_pair"
                    ),
                    "keyframe_reasons_oracle": {
                        str(index): values
                        for index, values in c_reasons.items()
                    },
                    "gold": "success" if c_rollout.success else "fail",
                }
                write_json(
                    episode_dir / "taskc" / "question.json", task_c_row
                )
                task_c_rows.append(task_c_row)

                d_variant = "cf" if condition_index % 2 == 0 else "full"
                d_query = cf if d_variant == "cf" else full
                task_d_row = {
                    "uid": f"D-R.HAB.{episode_id}.{d_variant.upper()}",
                    "task": "D-R",
                    "episode_id": episode_id,
                    "seed": seed,
                    "goal_image": goal_image,
                    "prompt": TASK_D_REFERENCE_PROMPT,
                    "frame_policy": "all_frames_in_chronological_order",
                    "reference": {
                        "images": [
                            transition.image_path
                            for transition in reference.transitions
                        ],
                        "action_sequence": list(reference_sampled.actions),
                        "trajectory_metadata": relative_path(
                            episode_dir
                            / "reference"
                            / "trajectory.jsonl",
                            output_root,
                        ),
                        "gold": "success",
                        "final_distance_to_goal_m": (
                            reference.final_distance_to_goal_m
                        ),
                        "seed": reference_seed,
                        "start": list(reference_sampled.start),
                    },
                    "query": {
                        "condition": d_variant,
                        "images": [
                            transition.image_path
                            for transition in d_query.transitions
                        ],
                        "action_sequence": list(sampled.actions),
                        "trajectory_metadata": relative_path(
                            episode_dir / d_variant / "trajectory.jsonl",
                            output_root,
                        ),
                        "gold": "success" if d_query.success else "fail",
                        "final_distance_to_goal_m": (
                            d_query.final_distance_to_goal_m
                        ),
                    },
                }
                if d_variant == "cf":
                    task_d_row["query"]["intervention"] = {
                        "type": "initial_yaw_offset_deg",
                        "value": cf_offset,
                        "same_start_position": True,
                        "same_action_sequence": True,
                    }
                task_d_rows.append(task_d_row)

                action_seed_row = {
                    "uid": f"SEED.HAB.{episode_id}",
                    "schema_version": "habitat-action-seed-v1",
                    "episode_id": episode_id,
                    "scene": str(Path(scene).resolve()),
                    "scene_slug": slug,
                    "seed": seed,
                    "start": {
                        "position": list(sampled.start),
                        "yaw_deg": sampled.initial_yaw_deg,
                    },
                    "goal": {
                        "position": list(sampled.goal),
                        "render_pose": goal_pose.to_dict(),
                        "image": goal_image,
                    },
                    "geodesic_distance_m": sampled.geodesic_distance_m,
                    "action_sequence": list(sampled.actions),
                    "reference": {
                        "seed": reference_seed,
                        "start": {
                            "position": list(reference_sampled.start),
                            "yaw_deg": (
                                reference_sampled.initial_yaw_deg
                            ),
                        },
                        "action_sequence": list(reference_sampled.actions),
                        "gold": "success",
                        "final_distance_to_goal_m": (
                            reference.final_distance_to_goal_m
                        ),
                    },
                    "cf_intervention": {
                        "type": "initial_yaw_offset_deg",
                        "value": cf_offset,
                        "same_start_position": True,
                        "same_goal": True,
                        "same_action_sequence": True,
                    },
                    "oracle_outcomes": {
                        "full": {
                            "gold": "success",
                            "final_distance_to_goal_m": (
                                full.final_distance_to_goal_m
                            ),
                        },
                        "cf": {
                            "gold": "fail",
                            "final_distance_to_goal_m": (
                                cf.final_distance_to_goal_m
                            ),
                        },
                    },
                }
                action_seed_rows.append(action_seed_row)

                episode_payload = {
                    "schema_version": "habitat-pilot-v1",
                    "episode_id": episode_id,
                    "scene": str(Path(scene).resolve()),
                    "scene_slug": slug,
                    "seed": seed,
                    "sampling_seed": sampling_seed,
                    "start": list(sampled.start),
                    "goal": list(sampled.goal),
                    "goal_image": goal_image,
                    "goal_render_pose": goal_pose.to_dict(),
                    "initial_yaw_deg": sampled.initial_yaw_deg,
                    "geodesic_distance_m": sampled.geodesic_distance_m,
                    "action_sequence": list(sampled.actions),
                    "full_success": full.success,
                    "cf_success": cf.success,
                    "cf_initial_yaw_offset_deg": cf_offset,
                    "reference_seed": reference_seed,
                    "reference_start": list(reference_sampled.start),
                    "reference_initial_yaw_deg": (
                        reference_sampled.initial_yaw_deg
                    ),
                    "reference_action_sequence": list(
                        reference_sampled.actions
                    ),
                    "reference_success": reference.success,
                }
                write_json(episode_dir / "episode.json", episode_payload)
                episodes_written += 1
        finally:
            simulator.close()

    write_jsonl(output_root / "taska" / "task_a.jsonl", task_a_rows)
    write_jsonl(output_root / "taskc" / "task_c.jsonl", task_c_rows)
    write_jsonl(output_root / "taskd" / "task_d.jsonl", task_d_rows)
    write_jsonl(
        output_root / "action_seed_bank.jsonl", action_seed_rows
    )
    manifest = {
        "schema_version": "habitat-pilot-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "Habitat/habitat_dataset",
        "python_version": platform.python_version(),
        "habitat_sim_version": habitat_version,
        "config": config.to_dict(),
        "scene_dataset_config": scene_dataset_config,
        "scenes": [str(Path(scene).resolve()) for scene in scenes],
        "counts": {
            "scenes": len(scenes),
            "episodes": episodes_written,
            "task_a_rows": len(task_a_rows),
            "task_c_rows": len(task_c_rows),
            "task_d_rows": len(task_d_rows),
            "action_seed_rows": len(action_seed_rows),
        },
        "action_vocabulary": [
            "move_forward",
            "turn_left",
            "turn_right",
            "look_up",
            "look_down",
            "stop",
        ],
        "reserved_interaction_actions": ["open", "pickup"],
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest
