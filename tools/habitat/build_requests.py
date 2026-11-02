#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


SYSTEM_PROMPT = """You are judging an embodied agent from time-ordered
first-person observations. Use only the supplied images and action text.
Return exactly the requested label and no explanation."""


ACTION_LABELS = {
    "move_forward": "MOVE_FORWARD",
    "turn_left": "TURN_LEFT",
    "turn_right": "TURN_RIGHT",
    "look_up": "LOOK_UP",
    "look_down": "LOOK_DOWN",
    "open": "OPEN",
    "close": "CLOSE",
    "stop": "STOP",
}


def compact_actions(actions: list[str]) -> str:
    runs: list[dict[str, Any]] = []
    for action in actions:
        if runs and runs[-1]["action"] == action:
            runs[-1]["count"] += 1
        else:
            runs.append({"action": action, "count": 1})
    return " → ".join(
        f"{ACTION_LABELS.get(run['action'], run['action'].upper())}"
        + (f" × {run['count']}" if run["count"] > 1 else "")
        for run in runs
    )


def task_a_request(row: dict[str, Any], root: Path) -> tuple[list[str], str]:
    images = [
        str((root / row["current_image"]).resolve()),
        *[
            str((root / path).resolve())
            for path in row["candidate_images"]
        ],
    ]
    prompt = f"""{SYSTEM_PROMPT}

{row["prompt"]}

The five supplied images are ordered as follows:
1. Current observation
2. Option A
3. Option B
4. Option C
5. Option D

Action sequence: {compact_actions(row["action_sequence"])}

Answer with exactly one uppercase letter: A, B, C, or D."""
    return images, prompt


def task_c_request(row: dict[str, Any], root: Path) -> tuple[list[str], str]:
    images = [
        str((root / row["goal_image"]).resolve()),
        *[
            str((root / path).resolve())
            for path in row["keyframe_images"]
        ],
    ]
    chunk_by_end = {
        chunk["to_frame_index"]: chunk
        for chunk in row.get("action_chunks", [])
    }
    transitions = []
    for index, source_index in enumerate(row["keyframe_indices"]):
        if index == 0:
            transitions.append("Storyboard frame 1: initial observation.")
            continue
        chunk = chunk_by_end.get(source_index)
        if chunk:
            chunk_actions = []
            for run in chunk["runs"]:
                chunk_actions.extend([run["action"]] * int(run["count"]))
            transitions.append(
                f"Before storyboard frame {index + 1}: "
                f"{compact_actions(chunk_actions)}."
            )
    image_order = "\n".join(
        ["1. Goal View"]
        + [
            f"{index + 2}. Storyboard frame {index + 1}"
            for index in range(len(row["keyframe_images"]))
        ]
    )
    prompt = f"""{SYSTEM_PROMPT}

{row["prompt"]}

The supplied images are ordered as follows:
{image_order}

Complete action sequence:
{compact_actions(row["action_sequence"])}

Sparse-storyboard transition text:
{chr(10).join(transitions)}

Answer with exactly one token: Success or Fail."""
    return images, prompt


def trajectory_montage(
    paths: list[Path],
    destination: Path,
    role: str,
    columns: int = 4,
) -> str:
    if destination.is_file():
        try:
            with Image.open(destination) as existing:
                existing.verify()
            return str(destination.resolve())
        except Exception:
            pass
    label_height = 34
    tile_size = 512
    rows = math.ceil(len(paths) / columns)
    canvas = Image.new(
        "RGB",
        (columns * tile_size, rows * (tile_size + label_height)),
        (24, 31, 29),
    )
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(paths):
        row, column = divmod(index, columns)
        x = column * tile_size
        y = row * (tile_size + label_height)
        draw.text(
            (x + 10, y + 9),
            f"{role.upper()} FRAME {index}",
            fill=(238, 248, 245),
        )
        with Image.open(path) as image:
            canvas.paste(
                image.convert("RGB").resize((tile_size, tile_size)),
                (x, y + label_height),
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", compress_level=1)
    return str(destination.resolve())


def task_d_request(
    row: dict[str, Any],
    root: Path,
    montage_dir: Path | None = None,
) -> tuple[list[str], str]:
    reference_images = [
        (root / path).resolve()
        for path in row["reference"]["images"]
    ]
    query_images = [
        (root / path).resolve()
        for path in row["query"]["images"]
    ]
    if montage_dir is not None:
        reference_payload = trajectory_montage(
            reference_images,
            montage_dir / f"{row['question_id']}_reference.png",
            "reference",
        )
        query_payload = trajectory_montage(
            query_images,
            montage_dir / f"{row['question_id']}_query.png",
            "query",
        )
        images = [reference_payload, query_payload]
        image_mapping = """- Image 1 is a labeled montage of the known-success
  reference trajectory. Panels REFERENCE FRAME 0–6 are chronological.
- Image 2 is a labeled montage of the query trajectory. Panels QUERY FRAME
  0–6 are chronological."""
    else:
        images = [
            *[str(path) for path in reference_images],
            *[str(path) for path in query_images],
        ]
        ref_end = len(reference_images)
        query_start = ref_end + 1
        query_end = ref_end + len(query_images)
        image_mapping = f"""- Images 1–{ref_end}: known-success reference
  trajectory, in chronological order.
- Images {query_start}–{query_end}: query trajectory, in chronological order."""
    prompt = f"""{SYSTEM_PROMPT}

{row["prompt"]}

Image-role mapping:
{image_mapping}

Reference action sequence:
{compact_actions(row["reference"]["action_sequence"])}

Query action sequence:
{compact_actions(row["query"]["action_sequence"])}

Answer with exactly one token: Success or Fail."""
    return images, prompt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exam-dir",
        type=Path,
        default=Path("Habitat/outputs/hssd-exam-en-550"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "Habitat/outputs/hssd-exam-en-550/inference_requests"
        ),
    )
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument(
        "--task-d-montage",
        action="store_true",
        help="Send Task D as two labeled 4x2 trajectory montages.",
    )
    args = parser.parse_args()
    root = args.exam_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = json.loads(
        (root / "questions.json").read_text(encoding="utf-8")
    )
    counts: dict[str, int] = {}
    for task in ("A", "C", "D"):
        selected = [row for row in questions if row["task"] == task]
        if args.limit_per_task > 0:
            selected = selected[: args.limit_per_task]
        requests = []
        for row in selected:
            if task == "A":
                images, prompt = task_a_request(row, root)
            elif task == "C":
                images, prompt = task_c_request(row, root)
            else:
                images, prompt = task_d_request(
                    row,
                    root,
                    (
                        output_dir / "montages_d"
                        if args.task_d_montage
                        else None
                    ),
                )
            uid = f"{task}.HABITAT.EXAM.{row['question_id']}"
            requests.append(
                {
                    "uid": uid,
                    "exam_id": uid,
                    "images": images,
                    "prompt": prompt,
                    "exam": {
                        "uid": uid,
                        "exam_id": uid,
                        "task": task,
                        "question_id": row["question_id"],
                        "source_uid": row["source_uid"],
                        "prompt_style": row.get("prompt_style"),
                    },
                }
            )
        destination = output_dir / f"requests_{task}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(
                    json.dumps(request, ensure_ascii=False) + "\n"
                )
        counts[task] = len(requests)
    manifest = {
        "schema_version": "habitat-inference-requests-v1",
        "exam_dir": str(root),
        "limit_per_task": args.limit_per_task,
        "counts": counts,
        "image_protocol": (
            "task_d_two_labeled_montages"
            if args.task_d_montage
            else "native_multi_image_with_explicit_role_mapping"
        ),
        "english_prompts_only": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
