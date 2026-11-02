from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from tools.habitat.dataset.task_design import evaluate_state_cycle


from utils.habitat_records import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    rows = read_jsonl(root / "taskc" / "task_c.jsonl")
    errors: list[str] = []
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        uid = row.get("uid", "<missing>")
        condition = row.get("condition")
        pairs[row.get("episode_id", "<missing>")][condition] = row
        if row.get("task_family") != "articulated_state_sequence":
            errors.append(f"{uid}: wrong task_family")
        if condition not in {"full", "cf"}:
            errors.append(f"{uid}: invalid condition")
        if row.get("action_sequence", [])[-1:] != ["stop"]:
            errors.append(f"{uid}: missing terminal stop")
        oracle = row.get("oracle", {})
        target = row.get("target", {})
        target_id = (
            f"{target.get('instance_id')}|{target.get('component_id')}"
        )
        progress = oracle.get("target_progress_by_frame", [])
        actions = row.get("action_sequence", [])
        interaction_success = [True] * len(actions)
        intervention = oracle.get("cf_intervention")
        if intervention == "open_no_effect":
            interaction_success[3] = False
        elif intervention == "close_no_effect":
            interaction_success[4] = False
        recomputed = evaluate_state_cycle(
            actions,
            progress,
            interaction_succeeded=interaction_success,
            interaction_targets=[
                None, None, None, target_id, target_id, None
            ],
            target_id=target_id,
        )
        if recomputed["outcome"].lower() != row.get("gold"):
            errors.append(f"{uid}: oracle/gold mismatch")
        indices = row.get("keyframe_indices", [])
        images = row.get("keyframe_images", [])
        if len(indices) != len(images) or indices != sorted(set(indices)):
            errors.append(f"{uid}: invalid keyframe alignment")
        for image_path in [row.get("goal_image"), *images]:
            path = root / str(image_path)
            if not path.is_file():
                errors.append(f"{uid}: missing image {image_path}")
                continue
            if args.check_images:
                with Image.open(path) as image:
                    if image.size != (512, 512) or image.mode != "RGB":
                        errors.append(
                            f"{uid}: invalid image {image_path}: "
                            f"{image.mode} {image.size}"
                        )
        metadata = root / str(row.get("trajectory_metadata"))
        if not metadata.is_file():
            errors.append(f"{uid}: missing episode metadata")

    for episode_id, pair in pairs.items():
        if set(pair) != {"full", "cf"}:
            errors.append(f"{episode_id}: missing Full/CF pair")
            continue
        full, cf = pair["full"], pair["cf"]
        if full["gold"] != "success" or cf["gold"] != "fail":
            errors.append(f"{episode_id}: pair labels are not Success/Fail")
        if full["action_sequence"] != cf["action_sequence"]:
            errors.append(f"{episode_id}: paired actions differ")
        if full["keyframe_indices"] != cf["keyframe_indices"]:
            errors.append(f"{episode_id}: paired keyframe indices differ")

    if errors:
        print("\n".join(errors))
        return 1
    print(
        json.dumps(
            {
                "root": str(root),
                "questions": len(rows),
                "pairs": len(pairs),
                "images_checked": bool(args.check_images),
                "status": "ok",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
