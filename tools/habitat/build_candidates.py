#!/usr/bin/env python3
"""Rebuild all qualified Habitat A/C/D-R questions without a fixed quota.

The output reuses the source episode image tree through a relative symlink, so
building a larger question index does not duplicate rendered PNG assets.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from utils.habitat_io import write_json, write_jsonl
from tools.habitat.dataset.task_design import TASK_C_PROMPT, TASK_D_REFERENCE_PROMPT, action_chunks, shared_critical_frame_selection, task_a_prompt
from tools.habitat.dataset.validation import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an uncapped MiniGrid/MiniWorld-aligned Habitat candidate "
            "pool from existing rendered episodes."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Habitat/outputs/hssd-pilot"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Habitat/outputs/hssd-template-candidate-pool"),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--task-c-min-frames", type=int, default=6)
    parser.add_argument("--task-a-ssim-min", type=float, default=0.70)
    parser.add_argument("--task-a-ssim-max", type=float, default=0.95)
    parser.add_argument(
        "--task-a-distractor-pair-max", type=float, default=0.98
    )
    parser.add_argument("--min-mean-brightness", type=float, default=25.0)
    parser.add_argument("--max-dark-fraction", type=float, default=0.65)
    parser.add_argument("--shortlist-size", type=int, default=160)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_episode_link(source: Path, output: Path) -> None:
    link = output / "episodes"
    target = source / "episodes"
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"{link} points to an unexpected target")
        return
    if link.exists():
        raise RuntimeError(f"Refusing to replace existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target.resolve(), link.parent.resolve()))


from utils.habitat_images import _visual


from utils.habitat_images import _ssim


def _quality(
    source: Path,
    relative: str,
    min_brightness: float,
    max_dark_fraction: float,
) -> bool:
    _, _, brightness, dark_fraction = _visual(str(source / relative))
    return brightness >= min_brightness and dark_fraction <= max_dark_fraction


def _episode_records(
    source: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    episodes: dict[str, dict[str, Any]] = {}
    for episode_path in sorted((source / "episodes").glob("*/*/episode.json")):
        episode = _load(episode_path)
        episode_dir = episode_path.parent
        episode_id = episode["episode_id"]
        episodes[episode_id] = episode
        full = read_jsonl(episode_dir / "full" / "trajectory.jsonl")
        cf = read_jsonl(episode_dir / "cf" / "trajectory.jsonl")
        motion_actions = [
            action for action in episode["action_sequence"] if action != "stop"
        ]
        if len(full) != len(motion_actions) + 1 or len(cf) != len(full):
            continue
        for step, action in enumerate(motion_actions):
            horizons: list[int] = []
            if action == "move_forward":
                # A single 0.25 m step is often visually negligible. Retain
                # 0.50, 0.75, and 1.00 m homogeneous forward macros.
                for horizon in (2, 3, 4):
                    if motion_actions[step : step + horizon] == [
                        "move_forward"
                    ] * horizon:
                        horizons.append(horizon)
            elif action in {"turn_left", "turn_right"}:
                horizons.append(1)
                if motion_actions[step : step + 2] == [action, action]:
                    horizons.append(2)
            elif action in {"look_up", "look_down"}:
                horizons.append(1)
            for horizon in horizons:
                action_sequence = motion_actions[step : step + horizon]
                records.append(
                    {
                        "episode_id": episode_id,
                        "scene": episode["scene_slug"],
                        "seed": episode["seed"],
                        "step_index": step,
                        "horizon": horizon,
                        "action": action,
                        "action_sequence": action_sequence,
                        "action_sequence_label": (
                            f"{action}_x{horizon}"
                            if horizon > 1
                            else action
                        ),
                        "current_image": full[step]["image_path"],
                        "next_image": full[step + horizon]["image_path"],
                        "pose_before": full[step]["pose"],
                    }
                )
    return records, episodes


def _build_task_a(
    source: Path,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pools: dict[tuple[str, tuple[str, ...]], list[int]] = defaultdict(list)
    hashes: list[int] = []
    eligible: list[bool] = []
    for index, record in enumerate(records):
        pools[
            (record["scene"], tuple(record["action_sequence"]))
        ].append(index)
        next_path = source / record["next_image"]
        current_ok = _quality(
            source,
            record["current_image"],
            args.min_mean_brightness,
            args.max_dark_fraction,
        )
        next_ok = _quality(
            source,
            record["next_image"],
            args.min_mean_brightness,
            args.max_dark_fraction,
        )
        hashes.append(_visual(str(next_path))[1])
        eligible.append(current_ok and next_ok)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        reject = {
            "episode_id": record["episode_id"],
            "step_index": record["step_index"],
            "action_sequence": record["action_sequence"],
        }
        if not eligible[index]:
            rejected.append({**reject, "reason": "dark_or_low_information"})
            continue
        pool = [
            candidate
            for candidate in pools[
                (record["scene"], tuple(record["action_sequence"]))
            ]
            if candidate != index
            and eligible[candidate]
            and (
                records[candidate]["episode_id"] != record["episode_id"]
                or records[candidate]["step_index"] != record["step_index"]
            )
        ]
        pool.sort(
            key=lambda candidate: (
                bin(hashes[index] ^ hashes[candidate]).count("1"),
                records[candidate]["episode_id"],
                records[candidate]["step_index"],
            )
        )
        correct_gray = _visual(str(source / record["next_image"]))[0]
        scored: list[tuple[float, int]] = []
        for candidate in pool[: args.shortlist_size]:
            similarity = _ssim(
                correct_gray,
                _visual(str(source / records[candidate]["next_image"]))[0],
            )
            if args.task_a_ssim_min <= similarity <= args.task_a_ssim_max:
                scored.append((similarity, candidate))
        scored.sort(reverse=True)

        chosen: list[tuple[float, int]] = []
        for similarity, candidate in scored:
            candidate_gray = _visual(
                str(source / records[candidate]["next_image"])
            )[0]
            if all(
                _ssim(
                    candidate_gray,
                    _visual(str(source / records[prior]["next_image"]))[0],
                )
                < args.task_a_distractor_pair_max
                for _, prior in chosen
            ):
                chosen.append((similarity, candidate))
            if len(chosen) == 3:
                break
        if len(chosen) < 3:
            rejected.append(
                {
                    **reject,
                    "reason": "fewer_than_three_same_action_hard_negatives",
                    "hard_negative_count": len(chosen),
                }
            )
            continue

        rng = np.random.default_rng(
            args.seed
            + int(record["seed"])
            + int(record["step_index"])
            + int(record["horizon"]) * 10_000
        )
        option_sources: list[int | None] = [None, *(item[1] for item in chosen)]
        rng.shuffle(option_sources)
        answer_index = option_sources.index(None)
        candidates = [
            record["next_image"]
            if source_index is None
            else records[source_index]["next_image"]
            for source_index in option_sources
        ]
        distractor_contexts = [
            {
                "episode_id": records[source_index]["episode_id"],
                "step_index": records[source_index]["step_index"],
                "action_sequence": record["action_sequence"],
                "ssim_to_answer": similarity,
            }
            for similarity, source_index in chosen
        ]
        accepted.append(
            {
                "uid": (
                    f"A.HAB.{record['episode_id']}.S"
                    f"{record['step_index']:04d}.H{record['horizon']}"
                ),
                "task": "A",
                "episode_id": record["episode_id"],
                "seed": record["seed"],
                "step_index": record["step_index"],
                "horizon": record["horizon"],
                "current_image": record["current_image"],
                "correct_next_image": record["next_image"],
                "action": record["action"],
                "action_sequence": record["action_sequence"],
                "action_sequence_label": record["action_sequence_label"],
                "prompt": task_a_prompt(record["action_sequence"]),
                "candidate_images": candidates,
                "candidate_action_contract": {
                    "all_options_use_action_sequence": (
                        record["action_sequence"]
                    ),
                    "distractors_from_other_state_contexts": True,
                },
                "candidate_state_contexts_oracle": distractor_contexts,
                "answer_index": answer_index,
                "answer_label": chr(ord("A") + answer_index),
                "pose_before": record["pose_before"],
            }
        )
    return accepted, rejected


def _trajectory_data(
    source: Path, episode: dict[str, Any], variant: str
) -> list[dict[str, Any]]:
    episode_dir = (
        source
        / "episodes"
        / episode["scene_slug"]
        / episode["episode_id"]
    )
    return read_jsonl(episode_dir / variant / "trajectory.jsonl")


def _build_task_c(
    source: Path,
    source_rows: list[dict[str, Any]],
    episodes: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for old_row in source_rows:
        episode = episodes[old_row["episode_id"]]
        full = _trajectory_data(source, episode, "full")
        cf = _trajectory_data(source, episode, "cf")
        actions = [
            action for action in episode["action_sequence"] if action != "stop"
        ]
        indices, reasons = shared_critical_frame_selection(
            actions,
            [bool(row["collided"]) for row in full[1:]],
            [bool(row["collided"]) for row in cf[1:]],
            full_positions=[row["pose"]["position"] for row in full],
            cf_positions=[row["pose"]["position"] for row in cf],
            goal=episode["goal"],
            goal_radius_m=float(args.source_manifest["config"]["goal_radius_m"]),
            min_frames=args.task_c_min_frames,
        )
        variant = old_row["condition"]
        trajectory = full if variant == "full" else cf
        images = [trajectory[index]["image_path"] for index in indices]
        visible = [episode["goal_image"], *images]
        if not all(
            _quality(
                source,
                image,
                args.min_mean_brightness,
                args.max_dark_fraction,
            )
            for image in visible
        ):
            rejected.append(
                {
                    "uid": old_row["uid"],
                    "reason": "dark_or_low_information_keyframe",
                }
            )
            continue
        accepted.append(
            {
                **old_row,
                "prompt": TASK_C_PROMPT,
                "keyframe_indices": indices,
                "keyframe_images": images,
                "action_chunks": action_chunks(actions, indices),
                "selection_rule": (
                    "shared_full_cf:event_complete+collision_run_onset+"
                    "goal_entry+distance_progress+terminal_pair"
                ),
                "keyframe_reasons_oracle": {
                    str(index): values for index, values in reasons.items()
                },
            }
        )
    return accepted, rejected


def _build_task_d(
    source: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        representative = [
            row["goal_image"],
            row["reference"]["images"][0],
            row["reference"]["images"][-1],
            row["query"]["images"][0],
            row["query"]["images"][-1],
        ]
        if not all(
            _quality(
                source,
                image,
                args.min_mean_brightness,
                args.max_dark_fraction,
            )
            for image in representative
        ):
            rejected.append(
                {"uid": row["uid"], "reason": "dark_or_low_information"}
            )
            continue
        accepted.append(
            {
                **row,
                "prompt": TASK_D_REFERENCE_PROMPT,
                "frame_policy": "all_frames_in_chronological_order",
            }
        )
    return accepted, rejected


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("--output must differ from --source")
    args.source_manifest = _load(source / "manifest.json")
    output.mkdir(parents=True, exist_ok=True)
    _ensure_episode_link(source, output)

    records, episodes = _episode_records(source)
    task_a, rejected_a = _build_task_a(source, records, args)
    source_c = read_jsonl(source / "taskc" / "task_c.jsonl")
    source_d = read_jsonl(source / "taskd" / "task_d.jsonl")
    task_c, rejected_c = _build_task_c(
        source, source_c, episodes, args
    )
    task_d, rejected_d = _build_task_d(source, source_d, args)

    write_jsonl(output / "taska" / "task_a.jsonl", task_a)
    write_jsonl(output / "taskc" / "task_c.jsonl", task_c)
    write_jsonl(output / "taskd" / "task_d.jsonl", task_d)
    seeds = read_jsonl(source / "action_seed_bank.jsonl")
    write_jsonl(output / "action_seed_bank.jsonl", seeds)
    write_jsonl(
        output / "rejections.jsonl",
        [
            *({"task": "A", **row} for row in rejected_a),
            *({"task": "C", **row} for row in rejected_c),
            *({"task": "D-R", **row} for row in rejected_d),
        ],
    )

    manifest = {
        **args.source_manifest,
        "schema_version": "habitat-template-candidate-pool-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source),
        "selection": {
            "fixed_question_quota": None,
            "policy": "retain_all_qualified_candidates",
            "task_a_same_action_different_state": True,
            "task_a_global_grayscale_ssim_range": [
                args.task_a_ssim_min,
                args.task_a_ssim_max,
            ],
            "task_a_distractor_pair_ssim_max": (
                args.task_a_distractor_pair_max
            ),
            "task_c_shared_full_cf_indices": True,
            "task_c_min_frames": args.task_c_min_frames,
            "task_d_frame_policy": "all_frames",
            "min_mean_brightness": args.min_mean_brightness,
            "max_dark_fraction": args.max_dark_fraction,
        },
        "counts": {
            "scenes": args.source_manifest["counts"]["scenes"],
            "episodes": len(episodes),
            "task_a_rows": len(task_a),
            "task_c_rows": len(task_c),
            "task_d_rows": len(task_d),
            "action_seed_rows": len(seeds),
            "rejected_task_a": len(rejected_a),
            "rejected_task_c": len(rejected_c),
            "rejected_task_d": len(rejected_d),
        },
        "distributions": {
            "task_a_action_sequences": dict(
                sorted(
                    Counter(
                        row["action_sequence_label"] for row in task_a
                    ).items()
                )
            ),
            "task_c_outcomes": dict(
                sorted(Counter(row["gold"] for row in task_c).items())
            ),
            "task_d_outcomes": dict(
                sorted(
                    Counter(row["query"]["gold"] for row in task_d).items()
                )
            ),
        },
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(manifest["distributions"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
