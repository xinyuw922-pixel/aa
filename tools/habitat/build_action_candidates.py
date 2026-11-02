#!/usr/bin/env python3
"""Render Task A-Action branches from the exact same Habitat agent pose."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tools.habitat.build_candidates import _quality
from utils.habitat_images import _ssim, _visual
from utils.habitat_types import GeneratorConfig
from utils.habitat_io import write_json, write_jsonl
from tools.habitat.dataset.simulator import (
    configure_simulator,
    render_action_sequence_candidates,
)
from tools.habitat.dataset.task_design import task_a_prompt
from tools.habitat.dataset.validation import read_jsonl


SEQUENCES: dict[str, tuple[str, ...]] = {
    "move_forward_x2": ("move_forward", "move_forward"),
    "move_forward_x3": ("move_forward",) * 3,
    "move_forward_x4": ("move_forward",) * 4,
    "turn_left": ("turn_left",),
    "turn_right": ("turn_right",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True,
                        help='Navigation candidate pool containing episode states and source images')
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh output for same-pose action branches')
    parser.add_argument("--anchors-per-episode", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--candidate-pair-ssim-max", type=float, default=0.995)
    parser.add_argument(
        "--min-pixel-difference-ratio", type=float, default=0.20
    )
    parser.add_argument(
        "--pixel-change-threshold", type=int, default=24,
        help=(
            "A pixel is changed when any RGB channel differs by at least "
            "this 8-bit intensity value."
        ),
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_anchor_indices(frame_count: int, count: int) -> list[int]:
    if frame_count < 2 or count <= 0:
        return []
    available = frame_count - 1
    actual = min(count, available)
    return sorted(
        {
            max(
                0,
                min(
                    available - 1,
                    round((slot + 1) * available / (actual + 1)),
                ),
            )
            for slot in range(actual)
        }
    )


def distinct_enough(
    root: Path,
    paths: list[str],
    maximum_similarity: float,
) -> bool:
    arrays = [_visual(str(root / path))[0] for path in paths]
    return all(
        _ssim(left, right) < maximum_similarity
        for index, left in enumerate(arrays)
        for right in arrays[index + 1 :]
    )


def pairwise_pixel_difference_ratios(
    root: Path,
    paths: list[str],
    pixel_change_threshold: int,
) -> list[dict[str, Any]]:
    images: list[np.ndarray] = []
    for path in paths:
        with Image.open(root / path) as image:
            images.append(np.asarray(image.convert("RGB"), dtype=np.int16))
    ratios: list[dict[str, Any]] = []
    for left_index, left in enumerate(images):
        for right_index in range(left_index + 1, len(images)):
            right = images[right_index]
            if left.shape != right.shape:
                raise ValueError(
                    f"Candidate image shape mismatch: "
                    f"{paths[left_index]}={left.shape}, "
                    f"{paths[right_index]}={right.shape}"
                )
            changed = np.max(np.abs(left - right), axis=2)
            ratios.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "ratio": float(
                        np.mean(changed >= pixel_change_threshold)
                    ),
                }
            )
    return ratios


def candidate_labels(query: str, anchor_number: int) -> list[str]:
    if query.startswith("move_forward"):
        turn = "turn_left" if anchor_number % 2 == 0 else "turn_right"
        return [
            "move_forward_x2",
            "move_forward_x3",
            "move_forward_x4",
            turn,
        ]
    return ["turn_left", "turn_right", "move_forward_x2", "move_forward_x4"]


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.min_pixel_difference_ratio < 1.0:
        raise ValueError("--min-pixel-difference-ratio must be in [0, 1)")
    if not 1 <= args.pixel_change_threshold <= 255:
        raise ValueError("--pixel-change-threshold must be in [1, 255]")
    if args.anchors_per_episode < 1:
        raise ValueError('--anchors-per-episode must be positive')
    source = args.source.resolve(strict=True)
    root = args.output.resolve()
    if root == source or root in source.parents or source in root.parents or root.exists():
        raise ValueError('Use a fresh output directory separate from the source')
    manifest = load(source / 'manifest.json')
    manifest.setdefault('counts', {})
    manifest.setdefault('selection', {})
    manifest.setdefault('distributions', {})
    root.mkdir(parents=True)
    (root / 'episodes').symlink_to(source / 'episodes', target_is_directory=True)
    manifest_path = root / 'manifest.json'
    config_payload = dict(manifest["config"])
    config_payload["cf_yaw_offsets_deg"] = tuple(
        config_payload["cf_yaw_offsets_deg"]
    )
    config = GeneratorConfig(**config_payload)
    config.validate()
    scene_dataset_config = manifest.get("scene_dataset_config")
    if scene_dataset_config:
        scene_dataset_config = str(Path(scene_dataset_config).resolve())

    episode_paths = sorted(
        (root / "episodes").glob("*/*/episode.json")
    )
    by_scene: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in episode_paths:
        episode = load(path)
        by_scene[episode["scene"]].append((path.parent, episode))

    min_brightness = float(
        manifest.get("selection", {}).get("min_mean_brightness", 25.0)
    )
    max_dark_fraction = float(
        manifest.get("selection", {}).get("max_dark_fraction", 0.65)
    )
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    anchors_rendered = 0
    pixel_ratio_cache: dict[
        tuple[str, ...], list[dict[str, Any]]
    ] = {}

    for scene, scene_episodes in sorted(by_scene.items()):
        habitat_sim, simulator = configure_simulator(
            scene, config, scene_dataset_config
        )
        try:
            for episode_dir, episode in scene_episodes:
                transitions = read_jsonl(
                    episode_dir / "full" / "trajectory.jsonl"
                )
                anchor_indices = select_anchor_indices(
                    len(transitions), args.anchors_per_episode
                )
                for local_anchor, frame_index in enumerate(anchor_indices):
                    transition = transitions[frame_index]
                    current_image = transition["image_path"]
                    audit = {
                        "episode_id": episode["episode_id"],
                        "frame_index": frame_index,
                    }
                    if not _quality(
                        root,
                        current_image,
                        min_brightness,
                        max_dark_fraction,
                    ):
                        rejected.append(
                            {**audit, "reason": "dark_current_observation"}
                        )
                        continue
                    branch_dir = (
                        root
                        / "taska_action_assets"
                        / episode["scene_slug"]
                        / episode["episode_id"]
                        / f"frame_{frame_index:04d}"
                    )
                    branch_json = branch_dir / "branches.json"
                    if branch_json.is_file():
                        branches = load(branch_json)
                    else:
                        pose = transition["pose"]
                        branches = render_action_sequence_candidates(
                            habitat_sim=habitat_sim,
                            simulator=simulator,
                            position=tuple(pose["position"]),
                            rotation_xyzw=tuple(pose["rotation_xyzw"]),
                            action_sequences=SEQUENCES,
                            output_dir=branch_dir,
                            output_root=root,
                            forward_step_m=config.forward_step_m,
                        )
                        write_json(branch_json, branches)
                    anchors_rendered += 1
                    anchor_number = (
                        int(episode["seed"]) + frame_index + local_anchor
                    )
                    for query_label, query_sequence in SEQUENCES.items():
                        labels = candidate_labels(query_label, anchor_number)
                        paths = [
                            branches[label]["image_path"] for label in labels
                        ]
                        if not all(
                            _quality(
                                root,
                                path,
                                min_brightness,
                                max_dark_fraction,
                            )
                            for path in paths
                        ):
                            rejected.append(
                                {
                                    **audit,
                                    "query": query_label,
                                    "reason": "dark_candidate",
                                }
                            )
                            continue
                        if not distinct_enough(
                            root, paths, args.candidate_pair_ssim_max
                        ):
                            rejected.append(
                                {
                                    **audit,
                                    "query": query_label,
                                    "reason": "candidate_images_too_similar",
                                }
                            )
                            continue
                        cache_key = tuple(paths)
                        if cache_key not in pixel_ratio_cache:
                            pixel_ratio_cache[cache_key] = (
                                pairwise_pixel_difference_ratios(
                                    root,
                                    paths,
                                    args.pixel_change_threshold,
                                )
                            )
                        pixel_ratios = pixel_ratio_cache[cache_key]
                        minimum_pixel_ratio = min(
                            pair["ratio"] for pair in pixel_ratios
                        )
                        if (
                            minimum_pixel_ratio
                            <= args.min_pixel_difference_ratio
                        ):
                            rejected.append(
                                {
                                    **audit,
                                    "query": query_label,
                                    "reason": (
                                        "candidate_pair_pixel_difference_"
                                        "not_above_threshold"
                                    ),
                                    "minimum_pair_pixel_difference_ratio": (
                                        minimum_pixel_ratio
                                    ),
                                    "required_strictly_above": (
                                        args.min_pixel_difference_ratio
                                    ),
                                    "pixel_change_threshold": (
                                        args.pixel_change_threshold
                                    ),
                                }
                            )
                            continue
                        order = list(range(4))
                        rng = np.random.default_rng(
                            args.seed
                            + int(episode["seed"])
                            + frame_index * 100
                            + len(query_sequence)
                            + sum(ord(char) for char in query_label)
                        )
                        rng.shuffle(order)
                        ordered_labels = [labels[index] for index in order]
                        ordered_paths = [paths[index] for index in order]
                        answer_index = ordered_labels.index(query_label)
                        rows.append(
                            {
                                "uid": (
                                    f"A-ACTION.HAB.{episode['episode_id']}."
                                    f"F{frame_index:04d}.{query_label.upper()}"
                                ),
                                "task": "A",
                                "subtask": "A-Action",
                                "episode_id": episode["episode_id"],
                                "seed": episode["seed"],
                                "step_index": frame_index,
                                "current_image": current_image,
                                "correct_next_image": branches[
                                    query_label
                                ]["image_path"],
                                "action": query_sequence[0],
                                "action_sequence": list(query_sequence),
                                "action_sequence_label": query_label,
                                "prompt": task_a_prompt(query_sequence),
                                "candidate_images": ordered_paths,
                                "candidate_action_contract": {
                                    "same_current_state": True,
                                    "options_use_distinct_action_sequences": True,
                                },
                                "option_action_sequences_oracle": [
                                    branches[label]["action_sequence"]
                                    for label in ordered_labels
                                ],
                                "option_final_poses_oracle": [
                                    branches[label]["final_pose"]
                                    for label in ordered_labels
                                ],
                                "candidate_pair_pixel_difference_oracle": {
                                    "definition": (
                                        "fraction_of_pixels_where_any_rgb_"
                                        "channel_abs_difference_gte_threshold"
                                    ),
                                    "pixel_change_threshold": (
                                        args.pixel_change_threshold
                                    ),
                                    "required_strictly_above": (
                                        args.min_pixel_difference_ratio
                                    ),
                                    "minimum_ratio": (
                                        minimum_pixel_ratio
                                    ),
                                    "pairs_before_option_shuffle": (
                                        pixel_ratios
                                    ),
                                },
                                "answer_index": answer_index,
                                "answer_label": chr(
                                    ord("A") + answer_index
                                ),
                                "pose_before": transition["pose"],
                            }
                        )
        finally:
            simulator.close()

    task_a_path = root / "taska" / "task_a.jsonl"
    write_jsonl(task_a_path, rows)
    write_jsonl(root / "taska" / "task_a_action_rejections.jsonl", rejected)

    manifest["schema_version"] = "habitat-template-candidate-pool-v3"
    manifest["config"]["output_dir"] = str(root)
    manifest["counts"]["task_a_rows"] = len(rows)
    manifest["counts"]["task_a_action_rejections"] = len(rejected)
    manifest["selection"]["task_a_subtask"] = "A-Action"
    manifest["selection"]["task_a_same_current_state"] = True
    manifest["selection"]["task_a_candidate_action_sequences"] = {
        label: list(sequence) for label, sequence in SEQUENCES.items()
    }
    manifest["selection"]["task_a_anchors_per_episode"] = (
        args.anchors_per_episode
    )
    manifest["selection"]["task_a_pixel_difference_filter"] = {
        "scope": "all_six_unordered_pairs_among_four_options",
        "definition": (
            "fraction_of_pixels_where_any_rgb_channel_abs_difference_"
            "gte_threshold"
        ),
        "pixel_change_threshold_8bit": args.pixel_change_threshold,
        "required_strictly_above": args.min_pixel_difference_ratio,
    }
    manifest["distributions"]["task_a_action_sequences"] = dict(
        sorted(Counter(row["action_sequence_label"] for row in rows).items())
    )
    manifest["task_a_action_rendering"] = {
        "anchors_rendered_or_reused": anchors_rendered,
        "same_pose_for_all_options": True,
        "source_manifest": str(source / 'manifest.json'),
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "anchors": anchors_rendered,
                "accepted_questions": len(rows),
                "rejected_questions": len(rejected),
                "distribution": manifest["distributions"][
                    "task_a_action_sequences"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
