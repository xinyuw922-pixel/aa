from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from tools.habitat.dataset.task_design import TASK_D_ARTICULATED_PROMPT_STYLES, task_d_articulated_prompts


from utils.habitat_records import read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--expected-targets", type=int, default=100)
    parser.add_argument(
        "--min-reference-query-difference",
        type=float,
        default=0.15,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = read_jsonl(root / "taskd" / "task_d.jsonl")
    errors: list[str] = []
    pairs: dict[str, dict[str, dict]] = defaultdict(dict)
    checked_paths = set()
    for row in rows:
        uid = row.get("uid", "<missing>")
        episode_id = row.get("episode_id", "<missing>")
        condition = row.get("condition")
        pairs[episode_id][condition] = row
        if row.get("task") != "D-R":
            errors.append(f"{uid}: task must be D-R")
        if row.get("task_family") != "articulated_reference_outcome":
            errors.append(f"{uid}: wrong task family")
        if condition not in {"full", "cf"}:
            errors.append(f"{uid}: invalid query condition")
        expected = "success" if condition == "full" else "fail"
        if row.get("gold") != expected:
            errors.append(f"{uid}: expected gold={expected}")
        reference = row.get("reference", {})
        query = row.get("query", {})
        if reference.get("gold") != "success" or not reference.get(
            "known_success"
        ):
            errors.append(f"{uid}: reference is not known-success")
        if query.get("gold") != expected:
            errors.append(f"{uid}: query gold mismatch")
        prompt_style = row.get("prompt_style")
        if prompt_style not in TASK_D_ARTICULATED_PROMPT_STYLES:
            errors.append(f"{uid}: invalid prompt style {prompt_style!r}")
        else:
            expected_en, expected_zh = task_d_articulated_prompts(prompt_style)
            if row.get("prompt") != expected_en:
                errors.append(f"{uid}: English prompt/style mismatch")
            if row.get("prompt_zh") != expected_zh:
                errors.append(f"{uid}: Chinese prompt/style mismatch")
            expected_template = (
                f"taskd_articulated_{prompt_style}_v1"
            )
            if row.get("prompt_template_id") != expected_template:
                errors.append(f"{uid}: prompt template ID mismatch")
        for role, payload in (("reference", reference), ("query", query)):
            images = payload.get("images", [])
            actions = payload.get("action_sequence", [])
            if len(images) != len(actions) + 1:
                errors.append(
                    f"{uid}/{role}: expected one frame per action plus initial"
                )
            if actions[-1:] != ["stop"]:
                errors.append(f"{uid}/{role}: missing STOP")
            for image_path in images:
                path = root / str(image_path)
                if not path.is_file():
                    errors.append(f"{uid}/{role}: missing {image_path}")
                    continue
                if args.check_images and path not in checked_paths:
                    checked_paths.add(path)
                    with Image.open(path) as image:
                        if image.mode != "RGB" or image.size != (512, 512):
                            errors.append(
                                f"{uid}/{role}: invalid {image_path}: "
                                f"{image.mode} {image.size}"
                            )
        reference_oracle = row.get("reference_oracle", {})
        if reference_oracle.get("oracle", {}).get("outcome") != "Success":
            errors.append(f"{uid}: reference oracle is not Success")
        difference = reference_oracle.get(
            "reference_query_view_difference", {}
        ).get("changed_pixel_fraction", 0.0)
        if difference < args.min_reference_query_difference:
            errors.append(
                f"{uid}: reference/query view difference {difference:.3f}"
            )

    for episode_id, pair in pairs.items():
        if set(pair) != {"full", "cf"}:
            errors.append(f"{episode_id}: missing Full/CF query pair")
            continue
        full, cf = pair["full"], pair["cf"]
        if full["reference"]["images"] != cf["reference"]["images"]:
            errors.append(f"{episode_id}: paired references differ")
        if (
            full["query"]["action_sequence"]
            != cf["query"]["action_sequence"]
        ):
            errors.append(f"{episode_id}: Full/CF query actions differ")
        if (
            full["reference"]["images"]
            == full["query"]["images"]
        ):
            errors.append(f"{episode_id}: reference copied from query")
        if full.get("prompt_style") != cf.get("prompt_style"):
            errors.append(f"{episode_id}: paired prompt styles differ")

    style_targets = Counter(
        pair["full"].get("prompt_style")
        for pair in pairs.values()
        if "full" in pair
    )
    expected_styles = set(TASK_D_ARTICULATED_PROMPT_STYLES)
    if set(style_targets) != expected_styles:
        errors.append(
            f"prompt styles incomplete: {dict(style_targets)}"
        )
    elif max(style_targets.values()) - min(style_targets.values()) > 1:
        errors.append(
            f"prompt styles are not target-balanced: {dict(style_targets)}"
        )

    if len(pairs) != args.expected_targets:
        errors.append(
            f"expected {args.expected_targets} target pairs, found {len(pairs)}"
        )
    if len(rows) != 2 * args.expected_targets:
        errors.append(
            f"expected {2 * args.expected_targets} questions, found {len(rows)}"
        )
    if errors:
        print("\n".join(errors))
        return 1
    print(
        json.dumps(
            {
                "root": str(root),
                "targets": len(pairs),
                "questions": len(rows),
                "prompt_style_targets": dict(style_targets),
                "unique_images_checked": len(checked_paths),
                "status": "ok",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
