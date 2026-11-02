#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from tools.miniworld.common import register_local_miniworld_envs, try_import_stack
    from tools.miniworld.trajectory_core import rollout_variant
else:
    from tools.miniworld.common import register_local_miniworld_envs, try_import_stack
    from tools.miniworld.trajectory_core import rollout_variant


from utils.miniworld_images import FONT_PATHS

REFERENCE_TEMPLATE = {
    "doorkey": "migration_doorkey_canonical",
    "multiroom": "migration_multiroom_canonical",
    "redblue": "migration_redblue_canonical",
}

QUERY_TEMPLATES = {
    "doorkey": (
        ("migration_doorkey_medium", "paired"),
        ("migration_doorkey_hard", "paired"),
        ("migration_doorkey_structural", "structural"),
        ("migration_doorkey_structural_v2", "structural"),
    ),
    "multiroom": (
        ("migration_multiroom_medium", "paired"),
        ("migration_multiroom_hard", "paired"),
        ("migration_multiroom_structural", "structural"),
        ("migration_multiroom_structural_v2", "structural"),
    ),
    "redblue": (
        ("migration_redblue_medium", "paired"),
        ("migration_redblue_hard", "paired"),
        ("migration_redblue_structural", "structural"),
        ("migration_redblue_structural_v2", "structural"),
    ),
}

PROMPT = (
    "Below are two time-ordered MiniWorld trajectory storyboards from the same task family.\n\n"
    "Reference trajectory:\n"
    "This is a known completed example of the task family.\n\n"
    "Query trajectory:\n"
    "Determine whether the query trajectory also completes the same task goal as the reference example.\n"
    "The room layout and path complexity may differ, but the task rule is the same.\n\n"
    "Answer with ONLY: Yes or No."
)


from utils.miniworld_images import load_font


from utils.miniworld_images import fit_contain


def load_summary_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def build_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["family"], row["template_id"]): row for row in rows}


def load_manifest(base_dir: Path, variant: str) -> dict[str, Any]:
    path = base_dir / f"storyboard_{variant}_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def expected_frame_paths(manifest: dict[str, Any], base_dir: Path) -> list[tuple[int, str, Path]]:
    frames_dir = base_dir / manifest["frames_dir_relpath"]
    out: list[tuple[int, str, Path]] = []
    for rec in manifest["step_records"]:
        t = int(rec["t"])
        action = rec["action"] or "start"
        exact = frames_dir / f"t{t:02d}_{action}.png"
        if exact.exists():
            out.append((t, action, exact))
            continue
        alts = sorted(frames_dir.glob(f"t{t:02d}_*.png"))
        if alts:
            out.append((t, action, alts[0]))
    return out


def build_square_montage(manifest: dict[str, Any], base_dir: Path, *, target_side: int = 980) -> Image.Image:
    items = expected_frame_paths(manifest, base_dir)
    cols = max(4, math.ceil(math.sqrt(len(items))))
    rows = math.ceil(len(items) / cols)
    gap = 12
    cell_side = 180
    canvas = Image.new(
        "RGB",
        (cols * cell_side + (cols + 1) * gap, rows * cell_side + (rows + 1) * gap),
        (248, 248, 248),
    )
    draw = ImageDraw.Draw(canvas)
    for idx, (t, action, path) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = gap + col * (cell_side + gap)
        y = gap + row * (cell_side + gap)
        draw.rounded_rectangle((x, y, x + cell_side, y + cell_side), radius=12, fill=(255, 255, 255), outline=(210, 210, 210), width=2)
        frame = Image.open(path).convert("RGB")
        fitted = fit_contain(frame, cell_side - 10, cell_side - 10, bg=(255, 255, 255))
        canvas.paste(fitted, (x + 5, y + 5))
    return fit_contain(canvas, target_side, target_side, bg=(255, 255, 255))


def compose_reference_query_image(reference_montage: Image.Image, query_montage: Image.Image) -> Image.Image:
    margin = 28
    gap = 36
    width = max(reference_montage.width, query_montage.width) + 2 * margin
    height = reference_montage.height + query_montage.height + gap + 2 * margin
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(reference_montage, ((width - reference_montage.width) // 2, margin))
    canvas.paste(query_montage, ((width - query_montage.width) // 2, margin + reference_montage.height + gap))
    return canvas


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_redblue_swapped_query(
    *,
    family: str,
    query_template: str,
    query_row: dict[str, Any],
    out_dir: Path,
) -> tuple[dict[str, Any], Path]:
    gym, _ = try_import_stack()
    register_local_miniworld_envs(gym)
    base_dir = out_dir / "_generated_queries" / family / query_template / "swap_redblue_order"
    base_dir.mkdir(parents=True, exist_ok=True)
    payload = rollout_variant(
        gym,
        seed=int(query_row["seed"]),
        obs_width=320,
        obs_height=240,
        out_dir=base_dir,
        variant="cf",
        intervention_type="swap_redblue_order",
        env_kwargs={"force_lane_z": 3.0, "force_template_id": query_template},
    )
    write_json(base_dir / "storyboard_cf_manifest.json", payload)
    return payload, base_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MiniWorld reference-family Task D showcase")
    ap.add_argument(
        "--paired_summary",
        default="experiments/generated/miniworld/taskD-MW-migration-showcase-families-paired/migration_showcase_summary.json",
    )
    ap.add_argument(
        "--structural_summary",
        default="experiments/generated/miniworld/taskD-MW-structural-showcase-v2/structural_showcase_summary.json",
    )
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskD-MW-reference-family-showcase",
    )
    args = ap.parse_args()

    paired_summary = Path(args.paired_summary)
    structural_summary = Path(args.structural_summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    paired_rows = load_summary_rows(paired_summary)
    structural_rows = load_summary_rows(structural_summary)
    paired_lookup = build_lookup(paired_rows)
    structural_lookup = build_lookup(structural_rows)

    records: list[dict[str, Any]] = []
    for family in ("doorkey", "multiroom", "redblue"):
        ref_template = REFERENCE_TEMPLATE[family]
        ref_row = paired_lookup[(family, ref_template)]
        ref_base = paired_summary.parent / family / ref_template
        ref_manifest = load_manifest(ref_base, "full")
        ref_montage = build_square_montage(ref_manifest, ref_base)

        for query_template, source_name in QUERY_TEMPLATES[family]:
            query_row = (paired_lookup if source_name == "paired" else structural_lookup)[(family, query_template)]
            query_root = paired_summary.parent if source_name == "paired" else structural_summary.parent
            query_base = query_root / family / query_template
            for variant in ("full", "cf"):
                answer = "Success" if variant == "full" else "Fail"
                query_manifest: dict[str, Any]
                query_manifest_base: Path
                if family == "redblue" and variant == "cf":
                    query_manifest, query_manifest_base = build_redblue_swapped_query(
                        family=family,
                        query_template=query_template,
                        query_row=query_row,
                        out_dir=out_dir,
                    )
                else:
                    query_manifest = load_manifest(query_base, variant)
                    query_manifest_base = query_base
                query_montage = build_square_montage(query_manifest, query_manifest_base)
                combined = compose_reference_query_image(ref_montage, query_montage)

                image_rel = Path("images") / family / f"{query_template}_{variant}.png"
                image_path = out_dir / image_rel
                image_path.parent.mkdir(parents=True, exist_ok=True)
                combined.save(image_path, "PNG")

                record = {
                    "task": "D-MW-RF",
                    "exam_id": f"mw_rf__{family}__{query_template}__{variant}",
                    "family": family,
                    "reference_template_id": ref_template,
                    "reference_difficulty": ref_row["layout_metadata"].get("difficulty"),
                    "query_template_id": query_template,
                    "query_difficulty": query_row["layout_metadata"].get("difficulty"),
                    "query_variant": variant,
                    "answer": answer,
                    "image": image_rel.as_posix(),
                    "prompt": PROMPT,
                    "note": query_row.get("note", ""),
                    "reference_manifest_relpath": str(ref_base / "storyboard_full_manifest.json"),
                    "query_manifest_relpath": str(
                        (query_manifest_base / f"storyboard_{variant}_manifest.json")
                        if query_manifest_base == query_base
                        else (query_manifest_base / "storyboard_cf_manifest.json")
                    ),
                    "source_bucket": source_name,
                    "full_success": query_row["full_success"],
                    "cf_success": query_row["cf_success"],
                    "redblue_failure_rule": "swap_redblue_order" if family == "redblue" and variant == "cf" else None,
                }
                records.append(record)

    exam_jsonl = out_dir / "task_d_mw_reference_family_exam.jsonl"
    with exam_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "num_records": len(records),
        "families": {
            family: {
                "reference_template": REFERENCE_TEMPLATE[family],
                "num_queries": sum(1 for r in records if r["family"] == family),
            }
            for family in ("doorkey", "multiroom", "redblue")
        },
        "records": records,
    }
    write_json(out_dir / "reference_family_summary.json", summary)
    (out_dir / "task_d_mw_reference_family_prompt.txt").write_text(PROMPT, encoding="utf-8")
    print(f"Saved reference-family showcase to {out_dir}")


if __name__ == "__main__":
    main()
