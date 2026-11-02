#!/usr/bin/env python3
"""Merge multiple MiniGrid triplet banks into one output root."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("GRIDWM_REPORT_ROOT", str(ROOT / "rebuttal_packet/10_minigrid_10x_expansion"))).resolve()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


from utils.minigrid_io import load_jsonl


from utils.minigrid_groups import group_rows


def source_roots(root: Path) -> list[Path]:
    if (root / "triplets.jsonl").exists():
        return [root]
    return sorted(p for p in root.iterdir() if (p / "triplets.jsonl").exists())


def render_md(report: dict[str, Any]) -> str:
    rows = [
        f"| `{src['name']}` | `{src['root']}` | {src['groups']} | {src['rows']} |"
        for src in report["sources"]
    ]
    env_rows = [
        f"| `{env}` | {count} |"
        for env, count in sorted(report["groups_by_task"].items())
    ]
    return "\n".join([
        "# MiniGrid Merged Triplet Bank Report",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        f"- Output root: `{report['output_root']}`",
        f"- Total groups: `{report['total_groups']}`",
        f"- Total rows: `{report['total_rows']}`",
        "",
        "## Sources",
        "",
        "| Source | Root | Groups | Rows |",
        "| --- | --- | ---: | ---: |",
        *rows,
        "",
        "## Groups By Task",
        "",
        "| Task/env key | Groups |",
        "| --- | ---: |",
        *env_rows,
    ]) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="NAME=ROOT source bank. May repeat.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--report-stem", default="merged_triplet_bank_report")
    args = ap.parse_args()

    out_root = resolve(args.out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    merged_rows: list[dict[str, Any]] = []
    seen_groups = set()
    source_reports = []
    groups_by_task: dict[str, int] = defaultdict(int)
    for spec in args.source:
        if "=" not in spec:
            raise SystemExit(f"Invalid source spec: {spec}")
        name, raw_root = spec.split("=", 1)
        src_root = resolve(raw_root)
        roots = source_roots(src_root)
        rows_total = 0
        groups_total = 0
        for leaf_root in roots:
            rows = load_jsonl(leaf_root / "triplets.jsonl")
            grouped = group_rows(rows)
            rows_total += len(rows)
            groups_total += len(grouped)
            for group_id, group in grouped.items():
                if group_id in seen_groups:
                    raise SystemExit(f"Duplicate group_id across sources: {group_id}")
                seen_groups.add(group_id)
                src_group_dir = leaf_root / group_id
                dst_group_dir = out_root / group_id
                if src_group_dir.exists():
                    shutil.copytree(src_group_dir, dst_group_dir)
                merged_rows.extend(group)
                task = str((group[0].get("task") or group[0].get("env_task") or "unknown"))
                groups_by_task[task] += 1
        source_reports.append({
            "name": name,
            "root": str(src_root),
            "leaf_roots": [str(p) for p in roots],
            "groups": groups_total,
            "rows": rows_total,
        })

    with (out_root / "triplets.jsonl").open("w", encoding="utf-8") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(out_root),
        "total_groups": len(seen_groups),
        "total_rows": len(merged_rows),
        "sources": source_reports,
        "groups_by_task": dict(groups_by_task),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{args.report_stem}.json"
    md_path = OUT_DIR / f"{args.report_stem}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    print(json.dumps({
        "groups": report["total_groups"],
        "rows": report["total_rows"],
        "json": str(json_path),
        "markdown": str(md_path),
        "out_root": str(out_root),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
