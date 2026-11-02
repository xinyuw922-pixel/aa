#!/usr/bin/env python3
"""Build a prompt-corrected Scale-2 Task C manifest and request bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.minigrid_prompts import TASK_C_ENVIRONMENT_CONTEXT, build_task_c_prompt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / "rebuttal_packet/10_minigrid_10x_expansion/manifests_scale2_v1"
    / "taskC_expanded_manifest.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rebuttal_packet/10_minigrid_10x_expansion"
    / "taskc_prompt_contract_v2"
)
VARIANTS = {"full", "nocue", "cf"}


from utils.minigrid_io import read_jsonl


from utils.minigrid_io import write_jsonl


from utils.file_hash import sha256


def image_path(source_manifest: Path, row: dict[str, Any]) -> Path:
    path = source_manifest.parent / str(row["image"])
    if not path.is_file():
        raise FileNotFoundError(f"{row['uid']}: missing image {path}")
    return path.resolve()


def validate_source(rows: list[dict[str, Any]]) -> None:
    envs = {str(row.get("env_task")) for row in rows}
    expected = set(TASK_C_ENVIRONMENT_CONTEXT)
    if envs != expected:
        raise RuntimeError(
            f"environment contract mismatch: missing={sorted(expected - envs)}, "
            f"unexpected={sorted(envs - expected)}"
        )
    if any(not str(row.get("mission") or "").strip() for row in rows):
        raise RuntimeError("every Scale-2 Task C row must have a simulator mission")
    if len({row["uid"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate Task C UIDs")

    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(str(row["env_task"]), str(row["group_id"]))].add(
            str(row["variant"])
        )
    incomplete = [key for key, values in grouped.items() if values != VARIANTS]
    if incomplete:
        raise RuntimeError(f"incomplete Full/NoCue/CF groups: {incomplete[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--profile', choices=('default', 'matched-probes'), default='default',
                        help='Optional Full/CF clean/noise/style/reversal requests; default behavior is unchanged')
    parser.add_argument('--raw-root', type=Path, help='Optional triplet-bank override for matched-probes')
    args = parser.parse_args()

    if args.profile == 'matched-probes':
        build_matched_probes(args.source, args.output_dir, args.raw_root)
        return

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    rows = read_jsonl(source)
    validate_source(rows)

    corrected = []
    requests = []
    prompt_counts: Counter[str] = Counter()
    env_missions: dict[str, set[str]] = defaultdict(set)
    for source_row in rows:
        row = dict(source_row)
        env = str(row["env_task"])
        mission = str(row["mission"]).strip()
        row["prompt"] = build_task_c_prompt(env, mission)
        row["source_exam_root"] = str(source.parent)
        row["prompt_contract_version"] = "taskc_explicit_mission_v2"
        row["prompt_contract_source"] = str(source)
        corrected.append(row)
        prompt_counts[row["prompt"]] += 1
        env_missions[env].add(mission)
        requests.append({
            "uid": row["uid"],
            "exam_id": row.get("exam_id", row["uid"]),
            "schema_version": "gridwm.exam.v1",
            "images": [str(image_path(source, row))],
            "prompt": row["prompt"],
            "exam": row,
            "api_content_order": "text_first",
        })

    manifest_path = output_dir / "taskC_expanded_manifest.jsonl"
    exam_path = output_dir / "exam/task_c_exam.jsonl"
    requests_path = output_dir / "_requests_from_manifest/requests_C.jsonl"
    write_jsonl(manifest_path, corrected)
    write_jsonl(exam_path, corrected)
    write_jsonl(requests_path, requests)

    audit_rows = []
    for env in sorted(TASK_C_ENVIRONMENT_CONTEXT):
        selected = [row for row in corrected if row["env_task"] == env]
        audit_rows.append({
            "environment": env,
            "rows": len(selected),
            "groups": len({row["group_id"] for row in selected}),
            "unique_missions": len(env_missions[env]),
            "environment_task_description": TASK_C_ENVIRONMENT_CONTEXT[env],
            "example_mission": sorted(env_missions[env])[0],
            "example_prompt": selected[0]["prompt"].replace("\n", "\\n"),
        })
    audit_path = output_dir / "taskc_prompt_coverage.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    invariant_fields = (
        "uid",
        "exam_id",
        "task",
        "env_task",
        "env_id",
        "group_id",
        "variant",
        "temporal",
        "visual",
        "framing",
        "image",
        "answer",
        "label",
        "mission",
    )
    mismatches = []
    for before, after in zip(rows, corrected):
        for field in invariant_fields:
            if before.get(field) != after.get(field):
                mismatches.append({
                    "uid": before.get("uid"),
                    "field": field,
                    "before": before.get(field),
                    "after": after.get(field),
                })

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "Task C explicit environment task and simulator mission",
        "source_manifest": str(source),
        "source_manifest_sha256": sha256(source),
        "corrected_manifest": str(manifest_path),
        "corrected_manifest_sha256": sha256(manifest_path),
        "requests": str(requests_path),
        "requests_sha256": sha256(requests_path),
        "rows": len(corrected),
        "requests_rows": len(requests),
        "unique_uids": len({row["uid"] for row in corrected}),
        "environments": dict(Counter(row["env_task"] for row in corrected)),
        "variants": dict(Counter(row["variant"] for row in corrected)),
        "unique_prompts": len(prompt_counts),
        "missing_missions": sum(not row["mission"] for row in corrected),
        "missing_images": sum(
            not Path(request["images"][0]).is_file() for request in requests
        ),
        "invariant_field_mismatches": mismatches,
        "passed": (
            len(corrected) == len(rows)
            and len(requests) == len(rows)
            and len({row["uid"] for row in corrected}) == len(rows)
            and not mismatches
        ),
    }
    report_path = output_dir / "taskc_prompt_contract_validation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not report["passed"]:
        raise RuntimeError(f"prompt-contract validation failed: {report_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))



SCHEMA_VERSION = "gridwm.exam.v1"
from utils.minigrid_panels import compose_storyboard
import argparse

import json




import math


from pathlib import Path

from typing import Any, List, Tuple

from collections import defaultdict


from PIL import Image


from utils.exam_schema import EXAM_SCHEMA_VERSION

def build_base_task_c(raw_root: Path, rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    out_rows = []
    for row in sorted(rows, key=lambda r: (str(r.get("task")), str(r.get("group_id")), str(r.get("variant")))):
        env = str(row.get("task") or "unknown")
        group_id = str(row.get("group_id"))
        variant = str(row.get("variant"))
        rel_img = Path("images/taskC") / env / f"{group_id}_{variant}.png"
        compose_storyboard(raw_root, row, out_dir / rel_img)
        answer = "Success" if row.get("success") else "Fail"
        uid = f"C10X.{env}.{group_id}.{variant}.orig.clean.neu"
        out_rows.append({
            "exam_id": uid,
            "uid": uid,
            "schema_version": SCHEMA_VERSION,
            "task": "C",
            "env_task": env,
            "env_id": row.get("env_id"),
            "group_id": group_id,
            "variant": variant,
            "temporal": "orig",
            "visual": "clean",
            "framing": "neu",
            "image": rel_img.as_posix(),
            "source_exam_root": str(out_dir),
            "prompt": (
                "Below is a MiniGrid trajectory storyboard. Determine whether "
                "the agent completed the task. Answer with ONLY: Success or Fail."
            ),
            "answer": answer,
            "label": answer,
            "mission": row.get("mission"),
            "source_raw_root": str(raw_root),
            "source_group_id": group_id,
            "source_variant": variant,
            "generator": "rebuttal_packet/scripts/build_minigrid_10x_acd_manifests.py",
        })
    return out_rows


def build_task_c(raw_root: Path, rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    out_rows = []
    for row in sorted(rows, key=lambda r: (str(r.get("task")), str(r.get("group_id")), str(r.get("variant")))):
        env = str(row.get("task") or "unknown")
        group_id = str(row.get("group_id"))
        variant = str(row.get("variant"))
        rel_img = Path("images/taskC") / env / f"{group_id}_{variant}.png"
        compose_storyboard(raw_root, row, out_dir / rel_img)
        answer = "Success" if row.get("success") else "Fail"
        uid = f"C10X.{env}.{group_id}.{variant}.orig.clean.neu"
        out_rows.append({
            "exam_id": uid,
            "uid": uid,
            "schema_version": SCHEMA_VERSION,
            "task": "C",
            "env_task": env,
            "env_id": row.get("env_id"),
            "group_id": group_id,
            "variant": variant,
            "temporal": "orig",
            "visual": "clean",
            "framing": "neu",
            "image": rel_img.as_posix(),
            "source_exam_root": str(out_dir),
            "prompt": build_task_c_prompt(env, str(row.get("mission") or "")),
            "answer": answer,
            "label": answer,
            "mission": row.get("mission"),
            "source_raw_root": str(raw_root),
            "source_group_id": group_id,
            "source_variant": variant,
            "generator": 'tools/minigrid/exam_logic.py',
        })
    return out_rows


from tools.minigrid.exam_source import _apply_visual_variant, _stable_seed, load_frame, parse_traj, read_jsonl, sb_dispatch


def build_matched_probes(source: Path, output: Path, raw_override: Path | None = None) -> None:
    """Export four presentation conditions per Full/CF pair, retaining source prompts."""
    from utils.minigrid_panels import compose_storyboard_frames, open_rgb, selected_indices
    import shutil

    source, output = source.resolve(strict=True), output.resolve()
    if output == ROOT or ROOT in output.parents or output in ROOT.parents or output.exists():
        raise ValueError('Use a fresh matched-probes output outside the source repository')
    rows = read_jsonl(source)
    grouped = defaultdict(dict)
    seen_uids = set()
    for row in rows:
        if row.get('task') != 'C' or row.get('variant') not in ('full', 'cf'):
            continue
        if row.get('temporal', 'orig') != 'orig' or row.get('visual', 'clean') != 'clean':
            continue
        if row.get('framing', 'neu') not in ('', 'neu'):
            continue
        key = (str(row['env_task']), str(row['group_id']))
        if row['uid'] in seen_uids or row['variant'] in grouped[key]:
            raise ValueError('Duplicate matched-probe source UID/group variant')
        seen_uids.add(row['uid'])
        grouped[key][row['variant']] = row
    if not grouped or any(set(variants) != {'full', 'cf'} for variants in grouped.values()):
        raise ValueError('matched-probes requires complete Full/CF source groups')
    cache, sources = {}, []
    for key in sorted(grouped):
        for variant in ('full', 'cf'):
            row = grouped[key][variant]
            raw = raw_override or row.get('source_raw_root')
            if not raw:
                raise ValueError('Source row lacks source_raw_root; supply --raw-root')
            raw = Path(raw).resolve(strict=True)
            if raw not in cache:
                cache[raw] = {}
                for record in read_jsonl(raw / 'triplets.jsonl'):
                    raw_key = (str(record.get('task') or record.get('env_task')),
                               str(record['group_id']), str(record['variant']))
                    if raw_key in cache[raw]:
                        raise ValueError('Duplicate raw triplet key')
                    cache[raw][raw_key] = record
            raw_key = (key[0], str(row.get('source_group_id') or row['group_id']), variant)
            record = cache[raw].get(raw_key)
            if record is None:
                raise ValueError(f'Missing raw triplet: {raw_key}')
            expected = 'Success' if record['success'] else 'Fail'
            if row['answer'] != expected:
                raise ValueError('Source/raw answer mismatch: ' + row['uid'])
            if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                raise ValueError('Missing source prompt')
            indices = selected_indices(len(record['frames']))
            if not indices:
                raise ValueError('Empty trajectory')
            frames = [open_rgb(raw / record['frames'][idx], max_size=180) for idx in indices]
            clean = Path(row.get('source_exam_root') or source.parent) / row['image']
            clean = clean.resolve(strict=True)
            sources.append((row, frames, indices, clean))
    output.mkdir(parents=True)
    exams, requests = [], []
    reverse_note = ('\nNote: The frames are shown in reverse chronological order '
                    '(final state first, initial state last). '
                    'Use this ordering to judge whether the trajectory succeeded.')
    for row, frames, indices, clean in sources:
        for arm, temporal, visual in [('clean', 'orig', 'clean'), ('noise', 'orig', 'noisy'),
                                      ('style', 'orig', 'style'), ('temporal_reversal', 'rev', 'clean')]:
            # UID-derived paths never interpret environment/group IDs as directories.
            folder = hashlib.sha256(row['uid'].encode()).hexdigest()
            relative = Path('images') / folder / (arm + '.png')
            path = output / relative
            if arm == 'clean':
                compose_storyboard_frames(frames, [f't={idx}' for idx in indices], path)
                with Image.open(clean) as original, Image.open(path) as rendered:
                    if original.size != rendered.size or original.convert('RGB').tobytes() != rendered.convert('RGB').tobytes():
                        raise ValueError('Clean storyboard differs from selected raw frames: ' + row['uid'])
                shutil.copy2(clean, path)
            else:
                transformed = [_apply_visual_variant(frame, visual, 0, profile='matched-probes',
                                seed_material=f"{row['group_id']}:{row['variant']}:{idx}:noise-v1")
                               for frame, idx in zip(frames, indices)]
                ordered = list(indices)
                if temporal == 'rev':
                    transformed.reverse()
                    ordered.reverse()
                compose_storyboard_frames(transformed, [f't={idx}' for idx in ordered], path)
            uid = row['uid'] + '.matched_probes.' + arm
            prompt = row['prompt'] + (reverse_note if temporal == 'rev' else '')
            exam = {**row, 'uid': uid, 'exam_id': uid, 'source_uid': row['uid'],
                    'image': relative.as_posix(), 'source_exam_root': str(output), 'prompt': prompt,
                    'temporal': temporal, 'visual': visual, 'presentation_profile': 'matched-probes',
                    'presentation_arm': arm, 'selected_frame_indices': list(reversed(indices)) if temporal == 'rev' else indices}
            exams.append(exam)
            requests.append(dict(uid=uid, exam_id=uid, schema_version=exam['schema_version'],
                                 images=[str(path)], prompt=prompt, exam=exam, api_content_order='text_first'))
    write_jsonl(output / 'taskC_expanded_manifest.jsonl', exams)
    write_jsonl(output / 'exam/task_c_exam.jsonl', exams)
    write_jsonl(output / '_requests_from_manifest/requests_C.jsonl', requests)
    print(json.dumps(dict(profile='matched-probes', groups=len(grouped), rows=len(exams), output=str(output)), indent=2))

def build_taskC(task_dirs: List[Path], out_dir: Path, args: Any) -> Tuple[List[dict], dict]:
    rows, stats = [], defaultdict(lambda: defaultdict(int))
    img_root = out_dir / "images" / "taskC"

    for td in task_dirs:
        groups = defaultdict(dict)
        for rec in read_jsonl(td / "triplets.jsonl"):
            tr = parse_traj(rec); groups[tr.group_id][tr.variant] = tr

        for gid, g in groups.items():
            if not g: continue
            any_tr = next(iter(g.values()))

            if not all(v in g for v in ["full", "nocue", "cf"]):
                stats[any_tr.task]["skip_incomplete"] += 1
                continue

            full = g["full"]
            base_idxs = sb_dispatch(full, args.c_k)

            for variant in ["full", "nocue", "cf"]:
                tr = g[variant]
                if tr.n_frames == 0: continue

                if variant == "nocue" and max(base_idxs) < tr.n_frames:
                    curr_idxs = base_idxs
                    stats[tr.task]["aligned_nocue"] += 1
                else:
                    curr_idxs = sb_dispatch(tr, args.c_k)

                for temporal in ["orig", "rev"]:
                    t_idx = list(reversed(curr_idxs)) if temporal == "rev" else curr_idxs

                    try:
                        frames = [load_frame(td, tr.frames[i]) for i in t_idx]
                    except FileNotFoundError:
                        stats[tr.task]["skip_missing_frames"] += 1 # Guard: Log missing frames
                        continue

                    if len(frames) == 0: continue
                    w, h = frames[0].size
                    cols = 4; rows_cnt = math.ceil(len(frames)/cols)
                    grid = Image.new("RGB", (w*cols, h*rows_cnt))
                    for i, f in enumerate(frames):
                        grid.paste(f, ((i%cols)*w, (i//cols)*h))

                    for visual in ["clean", "noisy", "style"]:
                        seed = _stable_seed(tr.task, gid, variant, temporal, visual)
                        vgrid = _apply_visual_variant(grid, visual, seed)
                        rel_path = f"{tr.task}/{gid}/{variant}_{temporal}_{visual}.png"
                        (img_root / rel_path).parent.mkdir(parents=True, exist_ok=True)
                        vgrid.save(img_root / rel_path)

                        rows.append({
                            "exam_id": f"C.{tr.task}.{gid}.{variant}.{temporal}.{visual}",
                            "uid": f"C.{tr.task}.{gid}.{variant}.{temporal}.{visual}",
                            "schema_version": EXAM_SCHEMA_VERSION,
                            "task": "C", "variant": variant, "visual": visual,
                            "image": f"images/taskC/{rel_path}",
                            "prompt": "Determine whether the agent accomplished its goal based on the storyboard.",
                            "answer": "Success" if tr.success else "Fail",
                            "label": int(tr.success)
                        })
                        stats[tr.task]["made"] += 1

    return rows, dict(stats)


if __name__ == "__main__":
    main()
