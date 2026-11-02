#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from tools.miniworld.reference_scene_logic import PROMPT, REFERENCE_TEMPLATE, build_lookup, build_redblue_swapped_query, build_square_montage, compose_reference_query_image, load_manifest, load_summary_rows, write_json
else:
    from tools.miniworld.reference_scene_logic import PROMPT, REFERENCE_TEMPLATE, build_lookup, build_redblue_swapped_query, build_square_montage, compose_reference_query_image, load_manifest, load_summary_rows, write_json


SOURCE_SPECS = (
    {
        "name": "paired",
        "summary": "experiments/generated/miniworld/taskD-MW-migration-showcase-families-paired/migration_showcase_summary.json",
        "templates": {
            "doorkey": ("migration_doorkey_medium", "migration_doorkey_hard"),
            "multiroom": ("migration_multiroom_medium", "migration_multiroom_hard"),
            "redblue": ("migration_redblue_medium", "migration_redblue_hard"),
        },
    },
    {
        "name": "structural",
        "summary": "experiments/generated/miniworld/taskD-MW-structural-showcase-v2/structural_showcase_summary.json",
        "templates": {
            "doorkey": ("migration_doorkey_structural", "migration_doorkey_structural_v2"),
            "multiroom": ("migration_multiroom_structural", "migration_multiroom_structural_v2"),
            "redblue": ("migration_redblue_structural", "migration_redblue_structural_v2"),
        },
    },
    {
        "name": "highdisc_round1",
        "summary": "experiments/generated/miniworld/taskD-MW-highdisc-expansion-round1-clean/highdisc_round1_summary.json",
        "templates": {
            "doorkey": (
                "migration_doorkey_highdisc_branch_lower",
                "migration_doorkey_highdisc_clear_zigzag",
            ),
            "multiroom": ("migration_multiroom_highdisc_clear_gate2_after_turn",),
            "redblue": ("migration_redblue_highdisc_clear_order_zigzag",),
        },
    },
    {
        "name": "highdisc_round2",
        "summary": "experiments/generated/miniworld/taskD-MW-highdisc-expansion-round2/highdisc_round2_summary.json",
        "templates": {
            "doorkey": ("migration_doorkey_highdisc_clear_branch_upper",),
            "multiroom": ("migration_multiroom_highdisc_clear_branch_upper",),
            "redblue": ("migration_redblue_highdisc_clear_four_room",),
        },
    },
    {
        "name": "highdisc_round3",
        "summary": "experiments/generated/miniworld/taskD-MW-highdisc-expansion-round3/highdisc_round3_summary.json",
        "templates": {
            "doorkey": (
                "migration_doorkey_highdisc_clear_branch_lower",
                "migration_doorkey_highdisc_clear_branch_lower_altopen",
            ),
            "multiroom": (
                "migration_multiroom_highdisc_clear_branch_lower",
                "migration_multiroom_highdisc_clear_branch_lower_altopen",
            ),
            "redblue": (
                "migration_redblue_highdisc_clear_four_room_lower",
                "migration_redblue_highdisc_clear_order_zigzag_lower",
            ),
        },
    },
)


from utils.json_io import write_jsonl_utf8 as write_jsonl


def ensure_image_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 50-question MiniWorld reference-family exam pack")
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskD-MW-reference-family-exam50",
    )
    ap.add_argument('--source-bucket', action='append', choices=[s['name'] for s in SOURCE_SPECS],
                    help='Explicit fresh source buckets; default retains the complete historical cohort')
    ap.add_argument('--max-pairs', type=int, default=0)
    args = ap.parse_args()
    if args.max_pairs < 0:
        ap.error('--max-pairs must be nonnegative')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_payloads = []
    for spec in SOURCE_SPECS:
        if args.source_bucket and spec['name'] not in args.source_bucket:
            continue
        summary_path = Path(str(spec["summary"]))
        rows = load_summary_rows(summary_path)
        source_payloads.append(
            {
                "name": str(spec["name"]),
                "summary_path": summary_path,
                "root": summary_path.parent,
                "lookup": build_lookup(rows),
                "templates": spec["templates"],
            }
        )

    paired_payload = next(payload for payload in source_payloads if payload["name"] == "paired")
    records: list[dict[str, Any]] = []

    for family in ("doorkey", "multiroom", "redblue"):
        ref_template = REFERENCE_TEMPLATE[family]
        ref_row = paired_payload["lookup"][(family, ref_template)]
        ref_base = paired_payload["root"] / family / ref_template
        ref_manifest = load_manifest(ref_base, "full")
        ref_montage = build_square_montage(ref_manifest, ref_base)

        for payload in source_payloads:
            for query_template in payload["templates"].get(family, ()):
                if args.max_pairs and len(records) // 2 >= args.max_pairs:
                    continue
                query_row = payload["lookup"][(family, query_template)]
                query_base = payload["root"] / family / query_template
                for variant in ("full", "cf"):
                    answer = "Success" if variant == "full" else "Fail"

                    if family == "redblue" and variant == "cf" and query_row.get("cf_intervention") != "swap_redblue_order":
                        query_manifest, query_manifest_base = build_redblue_swapped_query(
                            family=family,
                            query_template=query_template,
                            query_row=query_row,
                            out_dir=out_dir,
                        )
                        redblue_failure_rule = "swap_redblue_order"
                    else:
                        query_manifest = load_manifest(query_base, variant)
                        query_manifest_base = query_base
                        redblue_failure_rule = (
                            query_row.get("cf_intervention")
                            if family == "redblue" and variant == "cf"
                            else None
                        )

                    query_montage = build_square_montage(query_manifest, query_manifest_base)
                    combined = compose_reference_query_image(ref_montage, query_montage)

                    image_rel = Path("images") / family / payload["name"] / f"{query_template}_{variant}.png"
                    image_path = out_dir / image_rel
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    combined.save(image_path, "PNG")

                    record = {
                        "task": "D-MW-RF",
                        "exam_id": f"mw_rf50__{family}__{query_template}__{variant}",
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
                        "query_manifest_relpath": str(query_manifest_base / f"storyboard_{variant}_manifest.json"),
                        "source_bucket": payload["name"],
                        "full_success": query_row["full_success"],
                        "cf_success": query_row["cf_success"],
                        "redblue_failure_rule": redblue_failure_rule,
                    }
                    records.append(record)

    records.sort(
        key=lambda r: (
            ("doorkey", "multiroom", "redblue").index(str(r["family"])),
            (
                "paired",
                "structural",
                "highdisc_round1",
                "highdisc_round2",
                "highdisc_round3",
            ).index(str(r["source_bucket"])),
            str(r["query_template_id"]),
            str(r["query_variant"]),
        )
    )

    exam_jsonl = out_dir / "task_d_mw_reference_family_exam50.jsonl"
    write_jsonl(exam_jsonl, records)
    write_jsonl(out_dir / "task_d_exam.jsonl", records)

    summary = {
        "num_records": len(records),
        "num_query_templates": len(records) // 2,
        "selection": {
            "families": ["doorkey", "multiroom", "redblue"],
            "variants": ["full", "cf"],
            "source_buckets": [payload["name"] for payload in source_payloads],
        },
        "families": {
            family: {
                "reference_template": REFERENCE_TEMPLATE[family],
                "num_records": sum(1 for record in records if record["family"] == family),
                "num_query_templates": sum(1 for record in records if record["family"] == family and record["query_variant"] == "full"),
            }
            for family in ("doorkey", "multiroom", "redblue")
        },
        "source_buckets": {
            payload["name"]: sum(1 for record in records if record["source_bucket"] == payload["name"])
            for payload in source_payloads
        },
        "records": records,
    }
    write_json(out_dir / "reference_family_exam50_summary.json", summary)
    (out_dir / "task_d_mw_reference_family_prompt.txt").write_text(PROMPT, encoding="utf-8")
    print(f"Saved {len(records)}-question reference-family exam pack to {out_dir}")


if __name__ == "__main__":
    main()
