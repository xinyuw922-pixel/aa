#!/usr/bin/env python3
"""Build a full mission-matched Scale-2 Task D-R request bundle."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.minigrid_panels import compose_reference_query, compose_storyboard
from utils.minigrid_prompts import TASK_D_REFERENCE_PROMPT


ROOT = Path(__file__).resolve().parents[2]
SCALE2_ROOT = (
    ROOT / "rebuttal_packet/10_minigrid_10x_expansion/manifests_scale2_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "rebuttal_packet/10_minigrid_10x_expansion"
    / "taskd_mission_matched_v2"
)
LEGACY_ROOT = ROOT / "data/scale2/raw_bank_runs"
DEFAULT_RAW_SOURCES = (
    (
        "scale2_putnear_pool",
        LEGACY_ROOT
        / "minigrid_10x_raw_bank_runs/rawbank_scale2_v1/putnear_pool",
    ),
    (
        "scale1_putnear_pool",
        LEGACY_ROOT
        / "minigrid_10x_raw_bank_runs/rawbank_scale1_v1/putnear_pool",
    ),
    (
        "pilot_putnear_pool",
        LEGACY_ROOT
        / "minigrid_10x_raw_bank_runs/rawbank_pilot_v1/putnear_pool",
    ),
)
VARIANTS = {"full", "nocue", "cf"}


from utils.minigrid_io import read_jsonl


from utils.minigrid_io import write_jsonl


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


from utils.file_hash import sha256


def full_raw_rows(path: Path, source_tag: str) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path / "triplets.jsonl"):
        if row.get("variant") != "full" or row.get("success") is not True:
            continue
        enriched = dict(row)
        enriched["_source_tag"] = source_tag
        enriched["_source_root"] = str(path.resolve())
        enriched["_candidate_id"] = f"{source_tag}:{row['group_id']}"
        enriched["_episode_identity"] = episode_identity(row, path)
        rows.append(enriched)
    return rows


def episode_identity(row: dict[str, Any], root: Path) -> str:
    if row.get("seed") is not None:
        return f"{row.get('env_id') or row.get('task') or row.get('env_task')}:{row['seed']}"
    return f"{root.resolve()}:{row['group_id']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale2-root", type=Path, default=SCALE2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-query-groups", type=int, default=432)
    parser.add_argument("--generated-source-tag", default="generated_putnear")
    parser.add_argument(
        "--raw-source", action="append", metavar="TAG=PATH",
        help="Successful raw reference pool; repeat to override default pools.",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=DEFAULT_OUTPUT / "_reference_sources/generated_putnear",
    )
    args = parser.parse_args()

    scale2_root = args.scale2_root.resolve()
    output_dir = args.output_dir.resolve()
    generated_root = args.generated_root.resolve()
    task_c_path = scale2_root / "taskC_expanded_manifest.jsonl"
    task_d_path = scale2_root / "taskD_expanded_manifest.jsonl"
    task_c = read_jsonl(task_c_path)
    task_d = read_jsonl(task_d_path)
    c_by_uid = {row["uid"]: row for row in task_c}

    # The corrected package preserves the original 432 query groups.
    query_groups: dict[tuple[str, str], dict[str, Any]] = {}
    d_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_d:
        key = (str(row["env_task"]), str(row["group_id"]))
        d_by_group[key].append(row)
        query = c_by_uid[str(row["query_uid"])]
        if query["variant"] == "full":
            query_groups[key] = query
    if len(query_groups) != args.expected_query_groups:
        raise RuntimeError(f"expected {args.expected_query_groups} query groups, got {len(query_groups)}")

    raw_identities = {}
    for raw_root in {Path(row['source_raw_root']) for row in task_c}:
        for row in read_jsonl(raw_root / 'triplets.jsonl'):
            if row.get('variant') == 'full':
                raw_identities[(str(row.get('task') or row.get('env_task')), str(row['group_id']))] = episode_identity(row, raw_root)

    candidates = []
    for row in task_c:
        if row["variant"] != "full" or row["answer"] != "Success":
            continue
        enriched = dict(row)
        enriched["_source_tag"] = "scale2_selected"
        enriched["_source_root"] = str(scale2_root)
        enriched["_candidate_id"] = f"scale2_selected:{row['group_id']}"
        enriched["_storyboard"] = str((scale2_root / row["image"]).resolve())
        enriched["_episode_identity"] = raw_identities[(str(row['env_task']), str(row['group_id']))]
        candidates.append(enriched)
    raw_sources = DEFAULT_RAW_SOURCES
    if args.raw_source:
        raw_sources = []
        for value in args.raw_source:
            source_tag, separator, path = value.partition("=")
            if not separator or not source_tag or not path:
                parser.error("--raw-source requires TAG=PATH")
            raw_sources.append((source_tag, Path(path).resolve()))
    for source_tag, source_root in raw_sources:
        if not (source_root / "triplets.jsonl").is_file():
            raise FileNotFoundError(source_root / "triplets.jsonl")
        candidates.extend(full_raw_rows(source_root, source_tag))
    if not (generated_root / "triplets.jsonl").is_file():
        raise FileNotFoundError(
            f"generated references missing: run "
            f"generate_scale2_taskd_missing_references.py first: {generated_root}"
        )
    candidates.extend(full_raw_rows(generated_root, args.generated_source_tag))

    by_signature: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for candidate in candidates:
        signature = (
            str(candidate.get("env_task") or candidate.get("task")),
            str(candidate["mission"]),
        )
        by_signature[signature].setdefault(candidate["_episode_identity"], candidate)

    usage: Counter[str] = Counter()
    selected_refs = {}
    audit = []
    for key, query in sorted(query_groups.items()):
        env, group_id = key
        mission = str(query["mission"])
        eligible = [
            row
            for row in by_signature.get((env, mission), {}).values()
            if not (
                row["_source_tag"] == "scale2_selected"
                and str(row["group_id"]) == group_id
            )
            and row['_episode_identity'] != raw_identities[(env, group_id)]
        ]
        if not eligible:
            raise RuntimeError(f"no exact mission reference for {env}/{group_id}: {mission}")
        old_reference_uid = str(d_by_group[key][0]["reference_uid"])
        old_reference = c_by_uid[old_reference_uid]
        reusable = [
            row
            for row in eligible
            if row["_source_tag"] == "scale2_selected"
            and str(row.get("uid")) == old_reference_uid
            and str(old_reference["mission"]) == mission
        ]
        reference = reusable[0] if reusable else min(
            eligible,
            key=lambda row: (
                usage[row["_candidate_id"]],
                row["_source_tag"] != "scale2_selected",
                row["_candidate_id"],
            ),
        )
        usage[reference["_candidate_id"]] += 1
        selected_refs[key] = reference
        audit.append({
            "env_task": env,
            "query_group_id": group_id,
            "query_mission": mission,
            "reference_id": reference["_candidate_id"],
            "reference_group_id": reference["group_id"],
            "reference_mission": reference["mission"],
            "reference_seed": reference.get("seed", ""),
            "reference_source": reference["_source_tag"],
            "exact_mission_match": reference["mission"] == mission,
            "distinct_selected_group": not (
                reference["_source_tag"] == "scale2_selected"
                and str(reference["group_id"]) == group_id
            ),
            "eligible_references": len(eligible),
            "distinct_simulator_episode": reference['_episode_identity'] != raw_identities[(env, group_id)],
            "reference_episode_identity": reference['_episode_identity'],
            "query_episode_identity": raw_identities[(env, group_id)],
        })

    reference_storyboards: dict[str, Path] = {}
    for reference in {
        row["_candidate_id"]: row for row in selected_refs.values()
    }.values():
        candidate_id = str(reference["_candidate_id"])
        if reference.get("_storyboard"):
            reference_storyboards[candidate_id] = Path(reference["_storyboard"])
            continue
        safe_id = candidate_id.replace(":", "__")
        relative = (
            Path("reference_storyboards")
            / str(reference["task"])
            / f"{safe_id}_full.png"
        )
        compose_storyboard(
            Path(reference["_source_root"]),
            reference,
            output_dir / relative,
        )
        reference_storyboards[candidate_id] = output_dir / relative

    corrected = []
    requests = []
    rebuilt_images = 0
    for key in sorted(d_by_group):
        env, query_group = key
        reference = selected_refs[key]
        query_mission = str(query_groups[key]["mission"])
        for old_row in sorted(d_by_group[key], key=lambda row: str(row["variant"])):
            query = c_by_uid[str(old_row["query_uid"])]
            candidate_id = str(reference["_candidate_id"])
            old_reference = c_by_uid[str(old_row["reference_uid"])]
            can_reuse = (
                old_reference["mission"] == query_mission
                and reference["_source_tag"] == "scale2_selected"
                and str(reference["uid"]) == str(old_row["reference_uid"])
            )
            if can_reuse:
                image = str(old_row["image"])
                source_exam_root = str(scale2_root)
            else:
                safe_ref = candidate_id.replace(":", "__")
                image = str(
                    Path("images")
                    / env
                    / safe_ref
                    / f"{query_group}__{old_row['variant']}_ref_query.png"
                )
                compose_reference_query(
                    reference_storyboards[candidate_id],
                    scale2_root / str(query["image"]),
                    output_dir / image,
                )
                source_exam_root = str(output_dir)
                rebuilt_images += 1

            uid = (
                f"D10XV2.{env}.{candidate_id.replace(':', '__')}."
                f"{query_group}.{old_row['variant']}.orig.clean.neu"
            )
            row = {
                **old_row,
                "uid": uid,
                "exam_id": uid,
                "audit_task": "D-R-10x-scale-mission-matched-v2",
                "image": image,
                "source_exam_root": source_exam_root,
                "prompt": TASK_D_REFERENCE_PROMPT,
                "reference_uid": reference.get("uid", candidate_id),
                "reference_group_id": reference["group_id"],
                "reference_source": reference["_source_tag"],
                "reference_seed": reference.get("seed"),
                "reference_mission": str(reference["mission"]),
                "query_mission": query_mission,
                "goal_signature": query_mission,
                "goal_signature_kind": "exact_simulator_mission",
                "model_facing_mission": False,
                "source_scale2_v1_uid": old_row["uid"],
                "generator": (
                    'tools/minigrid/task_d.py'
                ),
            }
            corrected.append(row)
            image_path = Path(source_exam_root) / image
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            requests.append({
                "uid": uid,
                "exam_id": uid,
                "schema_version": "gridwm.exam.v1",
                "images": [str(image_path.resolve())],
                "prompt": row["prompt"],
                "exam": row,
                "api_content_order": "text_first",
            })

    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in corrected:
        grouped[(row["env_task"], row["group_id"])].add(row["variant"])
    failures = []
    if len(corrected) != args.expected_query_groups * 3:
        failures.append(f"rows={len(corrected)}")
    if len(grouped) != args.expected_query_groups:
        failures.append(f"groups={len(grouped)}")
    if any(values != VARIANTS for values in grouped.values()):
        failures.append("incomplete_variants")
    if any(not row["exact_mission_match"] for row in audit):
        failures.append("mission_mismatch")
    if any(not row["distinct_selected_group"] for row in audit):
        failures.append("same_selected_group")
    if any(not row['distinct_simulator_episode'] for row in audit):
        failures.append('same_simulator_episode')
    if len({row["uid"] for row in corrected}) != len(corrected):
        failures.append("duplicate_uid")

    manifest_path = output_dir / "taskD_expanded_manifest.jsonl"
    exam_path = output_dir / "exam/task_d_exam.jsonl"
    request_path = output_dir / "_requests_from_manifest/requests_D.jsonl"
    write_jsonl(manifest_path, corrected)
    write_jsonl(exam_path, corrected)
    write_jsonl(request_path, requests)
    write_csv(output_dir / "matched_reference_audit.csv", audit)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "source_task_c": str(task_c_path),
        "source_task_c_sha256": sha256(task_c_path),
        "source_task_d": str(task_d_path),
        "source_task_d_sha256": sha256(task_d_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "requests": str(request_path),
        "requests_sha256": sha256(request_path),
        "rows": len(corrected),
        "groups": len(grouped),
        "environments": dict(Counter(row["env_task"] for row in audit)),
        "reference_sources": dict(
            Counter(row["reference_source"] for row in audit)
        ),
        "unique_references": len({row["reference_id"] for row in audit}),
        "max_reference_reuse": max(
            Counter(row["reference_id"] for row in audit).values()
        ),
        "exact_mission_matches": sum(row["exact_mission_match"] for row in audit),
        "rebuilt_images": rebuilt_images,
        "model_facing_mission": False,
        "prompt_contract": "original Task D-R reference/query semantics",
    }
    (output_dir / "taskd_mission_match_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)



SCHEMA_VERSION = "gridwm.exam.v1"
def build_base_task_d(task_c_rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for row in task_c_rows:
        env = str(row["env_task"])
        if row["variant"] == "full" and row["answer"] == "Success" and env not in refs:
            refs[env] = row
    out_rows = []
    for row in task_c_rows:
        env = str(row["env_task"])
        ref = refs.get(env)
        if ref is None or row["group_id"] == ref["group_id"]:
            continue
        rel_img = (
            Path("images/taskD")
            / env
            / str(ref["group_id"])
            / f"{row['group_id']}__{row['variant']}_ref_query.png"
        )
        compose_reference_query(out_dir / ref["image"], out_dir / row["image"], out_dir / rel_img)
        uid = f"D10X.{env}.{ref['group_id']}.{row['group_id']}.{row['variant']}.orig.clean.neu"
        out_rows.append({
            "exam_id": uid,
            "uid": uid,
            "schema_version": SCHEMA_VERSION,
            "task": "D",
            "audit_task": "D-R-10x-scale",
            "env_task": env,
            "env_id": row.get("env_id"),
            "group_id": row["group_id"],
            "variant": row["variant"],
            "query_variant": row["variant"],
            "temporal": "orig",
            "visual": "clean",
            "framing": "neu",
            "image": rel_img.as_posix(),
            "source_exam_root": str(out_dir),
            "prompt": (
                "Below are two MiniGrid storyboard trajectories from the same environment. "
                "The reference trajectory is a known successful example. Determine whether "
                "the query trajectory also completes the task. Answer with ONLY: Success or Fail."
            ),
            "answer": row["answer"],
            "label": row["answer"],
            "reference_uid": ref["uid"],
            "reference_group_id": ref["group_id"],
            "query_uid": row["uid"],
            "source_group_id": row["group_id"],
            "generator": "rebuttal_packet/scripts/build_minigrid_10x_acd_manifests.py",
        })
    return out_rows


def build_task_d(task_c_rows: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    refs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_c_rows:
        env = str(row["env_task"])
        mission = str(row.get("mission") or "")
        if row["variant"] == "full" and row["answer"] == "Success":
            refs[(env, mission)].append(row)
    out_rows = []
    for row in task_c_rows:
        env = str(row["env_task"])
        mission = str(row.get("mission") or "")
        candidates = [
            candidate
            for candidate in refs.get((env, mission), [])
            if candidate["group_id"] != row["group_id"]
        ]
        if not candidates:
            raise RuntimeError(
                f"no distinct exact-mission Task D reference for "
                f"{env}/{row['group_id']}: {mission}"
            )
        ref = min(candidates, key=lambda candidate: str(candidate["group_id"]))
        rel_img = (
            Path("images/taskD")
            / env
            / str(ref["group_id"])
            / f"{row['group_id']}__{row['variant']}_ref_query.png"
        )
        compose_reference_query(out_dir / ref["image"], out_dir / row["image"], out_dir / rel_img)
        uid = f"D10X.{env}.{ref['group_id']}.{row['group_id']}.{row['variant']}.orig.clean.neu"
        out_rows.append({
            "exam_id": uid,
            "uid": uid,
            "schema_version": SCHEMA_VERSION,
            "task": "D",
            "audit_task": "D-R-10x-scale",
            "env_task": env,
            "env_id": row.get("env_id"),
            "group_id": row["group_id"],
            "variant": row["variant"],
            "query_variant": row["variant"],
            "temporal": "orig",
            "visual": "clean",
            "framing": "neu",
            "image": rel_img.as_posix(),
            "source_exam_root": str(out_dir),
            "prompt": TASK_D_REFERENCE_PROMPT,
            "answer": row["answer"],
            "label": row["answer"],
            "reference_uid": ref["uid"],
            "reference_group_id": ref["group_id"],
            "reference_mission": ref["mission"],
            "query_mission": mission,
            "goal_signature": mission,
            "goal_signature_kind": "exact_simulator_mission",
            "model_facing_mission": False,
            "query_uid": row["uid"],
            "source_group_id": row["group_id"],
            "generator": 'tools/minigrid/exam_logic.py',
        })
    return out_rows


if __name__ == "__main__":
    main()
