from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def validate_output(
    output_root: str | Path,
    check_images: bool = False,
) -> list[str]:
    root = Path(output_root).resolve()
    errors: list[str] = []

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return [f"Missing manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {})
    expected_size = (int(config.get("width", 0)), int(config.get("height", 0)))

    task_paths = {
        "A": root / "taska" / "task_a.jsonl",
        "C": root / "taskc" / "task_c.jsonl",
        "D": root / "taskd" / "task_d.jsonl",
        "SEED": root / "action_seed_bank.jsonl",
    }
    rows: dict[str, list[dict[str, Any]]] = {}
    for task, path in task_paths.items():
        if not path.is_file():
            errors.append(f"Missing Task {task} index: {path}")
            rows[task] = []
            continue
        try:
            rows[task] = read_jsonl(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            rows[task] = []

    expected_counts = manifest.get("counts", {})
    count_keys = {
        "A": "task_a_rows",
        "C": "task_c_rows",
        "D": "task_d_rows",
        "SEED": "action_seed_rows",
    }
    for task, key in count_keys.items():
        expected = expected_counts.get(key)
        if expected is not None and len(rows[task]) != expected:
            errors.append(
                f"Task {task} count mismatch: {len(rows[task])} != {expected}"
            )

    image_paths: set[str] = set()
    pixel_filter = manifest.get("selection", {}).get(
        "task_a_pixel_difference_filter"
    )
    for row in rows["A"]:
        uid = row.get("uid", "<missing uid>")
        candidates = row.get("candidate_images", [])
        if len(candidates) != 4 or len(set(candidates)) != 4:
            errors.append(f"{uid}: Task A requires four distinct candidates")
        answer_index = row.get("answer_index")
        if not isinstance(answer_index, int) or not 0 <= answer_index < 4:
            errors.append(f"{uid}: invalid answer_index={answer_index!r}")
        current = row.get("current_image")
        if isinstance(current, str):
            image_paths.add(current)
        else:
            errors.append(f"{uid}: missing current_image")
        correct_next = row.get("correct_next_image")
        if isinstance(correct_next, str):
            image_paths.add(correct_next)
        else:
            errors.append(f"{uid}: missing correct_next_image")
        image_paths.update(path for path in candidates if isinstance(path, str))
        if (
            isinstance(answer_index, int)
            and 0 <= answer_index < len(candidates)
            and isinstance(correct_next, str)
        ):
            candidate_path = root / candidates[answer_index]
            correct_path = root / correct_next
            if candidate_path.is_file() and correct_path.is_file():
                if candidate_path.read_bytes() != correct_path.read_bytes():
                    errors.append(
                        f"{uid}: answer candidate does not match the "
                        "recorded Full next frame"
                    )
        contract = row.get("candidate_action_contract")
        if contract is not None:
            if contract.get("same_current_state") and contract.get(
                "options_use_distinct_action_sequences"
            ):
                option_sequences = row.get(
                    "option_action_sequences_oracle", []
                )
                if len(option_sequences) != 4 or len(
                    {tuple(sequence) for sequence in option_sequences}
                ) != 4:
                    errors.append(
                        f"{uid}: A-Action requires four distinct option "
                        "action sequences"
                    )
            elif (
                asserted_sequence := contract.get(
                    "all_options_use_action_sequence"
                )
            ) is not None:
                if asserted_sequence != row.get("action_sequence"):
                    errors.append(
                        f"{uid}: Task A options do not assert the question "
                        "action sequence"
                    )
            elif contract.get("all_options_use_action") != row.get("action"):
                errors.append(
                    f"{uid}: Task A options do not assert the question action"
                )
            if (
                not contract.get("same_current_state")
                and not contract.get("distractors_from_other_state_contexts")
            ):
                errors.append(
                    f"{uid}: Task A distractors are not from other states"
                )
        if pixel_filter:
            required = float(pixel_filter["required_strictly_above"])
            threshold = int(pixel_filter["pixel_change_threshold_8bit"])
            oracle = row.get(
                "candidate_pair_pixel_difference_oracle", {}
            )
            if float(oracle.get("minimum_ratio", -1.0)) <= required:
                errors.append(
                    f"{uid}: recorded pixel difference does not exceed "
                    f"{required:.3f}"
                )
            if check_images and all(
                (root / path).is_file()
                for path in candidates
                if isinstance(path, str)
            ):
                arrays = []
                for path in candidates:
                    with Image.open(root / path) as image:
                        arrays.append(
                            np.asarray(
                                image.convert("RGB"), dtype=np.int16
                            )
                        )
                ratios = [
                    float(
                        np.mean(
                            np.max(np.abs(left - right), axis=2)
                            >= threshold
                        )
                    )
                    for index, left in enumerate(arrays)
                    for right in arrays[index + 1 :]
                ]
                if not ratios or min(ratios) <= required:
                    errors.append(
                        f"{uid}: recomputed option pixel difference does "
                        f"not strictly exceed {required:.3f}"
                    )

    for row in rows["C"]:
        uid = row.get("uid", "<missing uid>")
        condition = row.get("condition")
        gold = row.get("gold")
        if condition not in {"full", "cf"}:
            errors.append(f"{uid}: Task C condition must be full or cf")
        expected_gold = "success" if condition == "full" else "fail"
        if gold != expected_gold:
            errors.append(
                f"{uid}: Task C {condition} must have gold={expected_gold}"
            )
        goal_image = row.get("goal_image")
        if isinstance(goal_image, str):
            image_paths.add(goal_image)
        else:
            errors.append(f"{uid}: missing goal_image")
        indices = row.get("keyframe_indices", [])
        images = row.get("keyframe_images", [])
        if len(indices) != len(images):
            errors.append(f"{uid}: Task C index/image count mismatch")
        if indices != sorted(set(indices)):
            errors.append(f"{uid}: Task C indices are not sorted and unique")
        image_paths.update(path for path in images if isinstance(path, str))
        actions = row.get("action_sequence", [])
        if not actions or actions[-1] != "stop":
            errors.append(f"{uid}: Task C actions must terminate with stop")
        trajectory = row.get("trajectory_metadata")
        if not isinstance(trajectory, str):
            errors.append(f"{uid}: Task C missing trajectory metadata")
        elif not (root / trajectory).is_file():
            errors.append(f"{uid}: Task C missing {trajectory}")

    seed_by_episode = {
        row.get("episode_id"): row
        for row in rows["SEED"]
        if isinstance(row.get("episode_id"), str)
    }
    for row in rows["D"]:
        uid = row.get("uid", "<missing uid>")
        episode_id = row.get("episode_id")
        if row.get("task") != "D-R":
            errors.append(f"{uid}: Task D must use the D-R schema")
        goal_image = row.get("goal_image")
        if isinstance(goal_image, str):
            image_paths.add(goal_image)
        else:
            errors.append(f"{uid}: missing goal_image")
        for role in ("reference", "query"):
            payload = row.get(role, {})
            actions = payload.get("action_sequence", [])
            if not actions or actions[-1] != "stop":
                errors.append(
                    f"{uid}/{role}: actions must terminate with stop"
                )
            expected_frames = (
                sum(action != "stop" for action in actions) + 1
            )
            images = payload.get("images", [])
            if len(images) != expected_frames:
                errors.append(
                    f"{uid}/{role}: {len(images)} frames, "
                    f"expected {expected_frames}"
                )
            image_paths.update(path for path in images if isinstance(path, str))
            trajectory = payload.get("trajectory_metadata")
            if not isinstance(trajectory, str):
                errors.append(f"{uid}/{role}: missing trajectory metadata")
            else:
                trajectory_path = root / trajectory
                if not trajectory_path.is_file():
                    errors.append(f"{uid}/{role}: missing {trajectory}")
                else:
                    try:
                        transitions = read_jsonl(trajectory_path)
                        if len(transitions) != expected_frames:
                            errors.append(
                                f"{uid}/{role}: trajectory has "
                                f"{len(transitions)} rows, expected "
                                f"{expected_frames}"
                            )
                    except (OSError, ValueError) as exc:
                        errors.append(str(exc))
        reference = row.get("reference", {})
        query = row.get("query", {})
        if reference.get("gold") != "success":
            errors.append(f"{uid}: reference must be success")
        condition = query.get("condition")
        if condition not in {"full", "cf"}:
            errors.append(f"{uid}: query condition must be full or cf")
        expected_gold = "success" if condition == "full" else "fail"
        if query.get("gold") != expected_gold:
            errors.append(
                f"{uid}: {condition} query must have gold={expected_gold}"
            )
        intervention = query.get("intervention", {})
        if condition == "cf" and not intervention.get(
            "same_action_sequence"
        ):
            errors.append(
                f"{uid}: CF query does not assert same action sequence"
            )
        seed_row = seed_by_episode.get(episode_id)
        if seed_row is None:
            errors.append(f"{uid}: missing action seed row")
        else:
            if seed_row.get("action_sequence") != query.get(
                "action_sequence"
            ):
                errors.append(f"{uid}: action seed sequence mismatch")
            seed_reference = seed_row.get("reference", {})
            if seed_reference.get("action_sequence") != reference.get(
                "action_sequence"
            ):
                errors.append(f"{uid}: reference seed sequence mismatch")
            seed_goal = seed_row.get("goal", {}).get("image")
            if seed_goal != goal_image:
                errors.append(f"{uid}: action seed goal image mismatch")
            seed_cf = seed_row.get("cf_intervention", {})
            if (
                condition == "cf"
                and seed_cf.get("value") != intervention.get("value")
            ):
                errors.append(f"{uid}: action seed CF intervention mismatch")

    # A candidate pool keeps every qualified question and is not expected to
    # be balanced. Outcome balancing belongs to a later exam-set export.
    if manifest.get("selection", {}).get("fixed_question_quota", "legacy"):
        for task in ("C", "D"):
            task_golds = [
                row.get("gold")
                if task == "C"
                else row.get("query", {}).get("gold")
                for row in rows[task]
            ]
            successes = task_golds.count("success")
            failures = task_golds.count("fail")
            if abs(successes - failures) > 1:
                errors.append(
                    f"Task {task} outcomes are not balanced: "
                    f"{successes} success / {failures} fail"
                )

    for relative in sorted(image_paths):
        path = root / relative
        if not path.is_file():
            errors.append(f"Missing image: {relative}")
            continue
        if check_images:
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    if expected_size != (0, 0) and image.size != expected_size:
                        errors.append(
                            f"Wrong image size {image.size}: {relative}; "
                            f"expected {expected_size}"
                        )
            except OSError as exc:
                errors.append(f"Unreadable image {relative}: {exc}")
    return errors
