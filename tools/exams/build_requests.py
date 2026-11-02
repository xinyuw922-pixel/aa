#!/usr/bin/env python3
"""Build MiniGrid A/C/D request bundles for smoke and expanded reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.minigrid.task_a import export_split_panels, representation_prompt


TASK_FILES = {
    "A": "taskA_expanded_manifest.jsonl",
    "C": "taskC_expanded_manifest.jsonl",
    "D": "taskD_expanded_manifest.jsonl",
}

EXAM_FILES = {
    "A": "task_a_exam.jsonl",
    "C": "task_c_exam.jsonl",
    "D": "task_d_exam.jsonl",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


from utils.json_io import write_jsonl_utf8 as write_jsonl


def resolve_image(row: dict[str, Any]) -> Path:
    root = row.get("source_exam_root")
    image = row.get("image")
    if not root or not image:
        raise ValueError(f"{row.get('uid')}: missing source_exam_root/image")
    path = Path(root) / image
    if not path.exists():
        raise FileNotFoundError(f"{row.get('uid')}: image not found: {path}")
    return path.resolve()


def select_task_a(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_envs: set[str] = set()
    for row in rows:
        env = str(row.get("env_task", ""))
        if env in seen_envs:
            continue
        selected.append(row)
        seen_envs.add(env)
        if len(selected) >= n:
            return selected
    return rows[:n]


def select_complete_cd_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("variant") not in {"full", "nocue", "cf"}:
            continue
        if row.get("temporal", "orig") != "orig":
            continue
        if row.get("visual", "clean") != "clean":
            continue
        framing = str(row.get("framing", ""))
        if framing and framing != "neu":
            continue
        key = (
            str(row.get("env_task") or row.get("env_id") or ""),
            str(row.get("group_id") or ""),
            str(row.get("visual", "")),
            framing,
        )
        groups[key].append(row)

    for key in sorted(groups):
        group_rows = groups[key]
        by_variant = {row.get("variant"): row for row in group_rows}
        if {"full", "nocue", "cf"}.issubset(by_variant):
            return [by_variant["full"], by_variant["nocue"], by_variant["cf"]]
    raise RuntimeError("No complete full/nocue/cf group found")


def select_complete_cd_groups_by_env(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("variant") not in {"full", "nocue", "cf"}:
            continue
        if row.get("temporal", "orig") != "orig":
            continue
        if row.get("visual", "clean") != "clean":
            continue
        framing = str(row.get("framing", ""))
        if framing and framing != "neu":
            continue
        key = (
            str(row.get("env_task") or row.get("env_id") or ""),
            str(row.get("group_id") or ""),
            str(row.get("visual", "")),
            framing,
        )
        groups[key].append(row)

    selected: list[dict[str, Any]] = []
    seen_envs: set[str] = set()
    for key in sorted(groups):
        env = key[0]
        if env in seen_envs:
            continue
        by_variant = {row.get("variant"): row for row in groups[key]}
        if {"full", "nocue", "cf"}.issubset(by_variant):
            selected.extend([by_variant["full"], by_variant["nocue"], by_variant["cf"]])
            seen_envs.add(env)
    if not selected:
        raise RuntimeError("No complete full/nocue/cf groups found")
    return selected


def request_from_exam_row(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row["prompt"])
    if row.get("task") == "A" and row.get('prompt_contract') != 'base_native':
        prompt = representation_prompt(row)

    return {
        "uid": row["uid"],
        "exam_id": row.get("exam_id", row["uid"]),
        "schema_version": "gridwm.exam.v1",
        "images": [str(resolve_image(row))],
        "prompt": prompt,
        "exam": {**{k: v for k, v in row.items() if not k.startswith("_")}, "prompt": prompt},
        "api_content_order": "text_first",
    }


def select_rows_by_mode(manifest_dir: Path, task_a_n: int, mode: str) -> dict[str, list[dict[str, Any]]]:
    all_rows = {task: load_jsonl(manifest_dir / file_name) for task, file_name in TASK_FILES.items()}

    if mode == "full":
        return all_rows

    if mode == "balanced":
        return {
            "A": select_task_a(all_rows["A"], task_a_n),
            "C": select_complete_cd_groups_by_env(all_rows["C"]),
            "D": select_complete_cd_groups_by_env(all_rows["D"]),
        }

    selected: dict[str, list[dict[str, Any]]] = {
        "A": select_task_a(all_rows["A"], task_a_n),
        "C": select_complete_cd_group(all_rows["C"]),
        "D": select_complete_cd_group(all_rows["D"]),
    }
    if mode == "triad":
        selected["A"] = selected["A"][:1]
        selected["C"] = [row for row in selected["C"] if row.get("variant") == "full"][:1]
        selected["D"] = [row for row in selected["D"] if row.get("variant") == "full"][:1]
    return selected


def build_bundle(manifest_dir: Path, out_dir: Path, task_a_n: int, mode: str) -> dict[str, Any]:
    exam_dir = out_dir / "exam"
    request_dir = out_dir / "_requests_from_manifest"

    selected = select_rows_by_mode(manifest_dir, task_a_n, mode)

    combined_requests: list[dict[str, Any]] = []
    manifest_tasks: dict[str, Any] = {}
    for task, rows in selected.items():
        if task == 'A':
            rows = [{**row, 'prompt': row['prompt'] if row.get('prompt_contract') == 'base_native'
                     else representation_prompt(row)} for row in rows]
        write_jsonl(exam_dir / EXAM_FILES[task], rows)
        requests = [request_from_exam_row(row) for row in rows]
        if task == 'A':
            split_requests = []
            for row, request in zip(rows, requests):
                source = resolve_image(row)
                # Keep the exact same UID in separate representation response streams.
                request.update(representation='composite', source_uid=row['uid'])
                request['exam'].update(representation='composite', source_uid=row['uid'])
                name = hashlib.sha256(row['uid'].encode()).hexdigest()
                panels = export_split_panels(row, source, out_dir / 'split/images/taskA' / name)
                split_requests.append({**request, 'representation': 'split',
                                       'images': [str(path) for path in panels],
                                       'exam': {**request['exam'], 'representation': 'split'}})
            write_jsonl(out_dir / 'split/exam/task_a_exam.jsonl',
                        [request['exam'] for request in split_requests])
            write_jsonl(out_dir / 'split/requests_A.jsonl', split_requests)
        write_jsonl(request_dir / f"requests_{task}.jsonl", requests)
        combined_requests.extend(requests)
        manifest_tasks[task] = {
            "exam": str((exam_dir / EXAM_FILES[task]).resolve()),
            "requests": str((request_dir / f"requests_{task}.jsonl").resolve()),
            "rows": len(rows),
            "uids": [row["uid"] for row in rows],
        }
        if task == 'A':
            manifest_tasks[task]['representations'] = {
                'composite': str((request_dir / 'requests_A.jsonl').resolve()),
                'split': str((out_dir / 'split/requests_A.jsonl').resolve()),
            }

    combined_name = "requests_ACD_full.jsonl" if mode == "full" else "requests_ACD_smoke.jsonl"
    combined_path = request_dir / combined_name
    write_jsonl(combined_path, combined_requests)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_kind": f"minigrid_acd_{mode}_request_bundle",
        "manifest_dir": str(manifest_dir.resolve()),
        "exam_dir": str(exam_dir.resolve()),
        "combined_requests": str(combined_path.resolve()),
        "total_requests": len(combined_requests),
        "mode": mode,
        "tasks": manifest_tasks,
        "notes": [
            "Images are referenced by absolute path in requests.",
            "Exam JSONL files are a scoring-only subset and retain source_exam_root.",
            "Task A/C rows come from expanded manifests; Task D rows use generated DRX images.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "request_bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("rebuttal_packet/03_minigrid_expanded/expanded"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("rebuttal_packet/03_minigrid_expanded/request_bundle/one_model_smoke"),
    )
    parser.add_argument("--task-a-n", type=int, default=3)
    parser.add_argument("--mode", choices=["paired", "triad", "balanced", "full"], default="paired")
    args = parser.parse_args()

    manifest = build_bundle(args.manifest_dir, args.out_dir, args.task_a_n, args.mode)
    print(json.dumps({
        "exam_dir": manifest["exam_dir"],
        "combined_requests": manifest["combined_requests"],
        "total_requests": manifest["total_requests"],
        "tasks": {k: v["rows"] for k, v in manifest["tasks"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
