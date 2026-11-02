#!/usr/bin/env python3
"""Assemble fresh base MiniGrid cohorts with explicit task-size contracts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from tools.minigrid.base_task_a import build_base_task_a
from tools.minigrid.task_b import build_full_task_b
from tools.minigrid.base_task_c import build_final_task_c
from tools.minigrid.base_task_d import build_final_task_d
from utils.json_io import load_jsonl, write_json, write_jsonl
from utils.minigrid_prompts import TASK_B_PROMPT
from utils.file_hash import pixel_sha256

ENVIRONMENTS = ('doorkey', 'keycorridor', 'lavagap', 'memory', 'multiroom', 'redblue')
C_ENVIRONMENTS = ('keycorridor', 'multiroom', 'redblue')


def select_groups(rows: list[dict], environments: tuple[str, ...], quota: int | dict[str, int]) -> list[dict]:
    """Select whole simulator-valid triplets in fixed environment/group order."""
    grouped = defaultdict(dict)
    for row in rows:
        key = (str(row.get('task')), str(row['group_id']))
        variant = row['variant']
        if variant in grouped[key]:
            raise ValueError(f'Duplicate source variant: {key}/{variant}')
        grouped[key][variant] = row
    result = []
    for env in environments:
        required = quota[env] if isinstance(quota, dict) else quota
        eligible = []
        for (task, gid), variants in sorted(grouped.items()):
            if task != env or set(variants) != {'full', 'nocue', 'cf'}:
                continue
            if not (variants['full'].get('success') is True and
                    variants['nocue'].get('success') is True and
                    variants['cf'].get('success') is False):
                continue
            if not all(row.get('frames') and row.get('state_seq') for row in variants.values()):
                continue
            eligible.append(variants)
        if len(eligible) < required:
            raise ValueError(f'{env}: need {required} complete valid groups, found {len(eligible)}')
        for variants in eligible[:required]:
            result.extend(variants[v] for v in ('full', 'nocue', 'cf'))
    return result


def build_views(raw: Path, rows: list[dict], output: Path) -> list[Path]:
    """Expose selected native trajectories without copying their images."""
    grouped = defaultdict(list)
    for row in rows:
        copied = dict(row)
        copied['frames'] = [str((raw/frame).resolve(strict=True)) for frame in row['frames']]
        grouped[row['task']].append(copied)
    for env, records in sorted(grouped.items()):
        write_jsonl(output/env/'triplets.jsonl', records)
    return [output/env for env in sorted(grouped)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--lavagap-pool', type=Path, required=True)
    parser.add_argument('--task-b-raw-root', type=Path,
                        help='Optional Full trajectory pool; defaults to the fresh raw bank')
    parser.add_argument('--task-a-cohort', type=Path)
    parser.add_argument('--task-a-raw-root', type=Path, required=True,
                        help='Native base Task A source cohort, separate from the expanded bank')
    parser.add_argument('--task-a-per-traj', type=int, default=6)
    parser.add_argument('--exam-seed', type=int, default=42)
    args = parser.parse_args()
    raw, output = args.raw_root.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error('Output exists; choose a fresh output directory')
    rows = load_jsonl(raw/'triplets.jsonl')
    # LavaGap S7 has few distinct observation sequences. Use exact-distinct,
    # simulator-validated candidates rather than duplicating a selected trajectory.
    combined=[]
    for row in rows:
        if row['task']=='lavagap':
            continue
        combined.append({**row,'frames':[str((raw/f).resolve(strict=True)) for f in row['frames']]})
    pool=args.lavagap_pool.resolve()
    pool_groups=defaultdict(list)
    for row in load_jsonl(pool/'triplets.jsonl'):
        pool_groups[row['group_id']].append(row)
    seen=set()
    for gid, records in sorted(pool_groups.items()):
        full=next((r for r in records if r['variant']=='full'),None)
        if full is None or full.get('success') is not True:
            continue
        fingerprint=tuple(pixel_sha256(pool/f) for f in full['frames'])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        combined.extend({**r,'frames':[str((pool/f).resolve(strict=True)) for f in r['frames']]} for r in records)
    c_source = select_groups(combined, C_ENVIRONMENTS, 6)
    output.mkdir(parents=True)
    raw=output/'raw'
    write_jsonl(raw/'triplets.jsonl',combined)
    a_root = args.task_a_raw_root.resolve()
    a_order = ('memory', 'redblue', 'doorkey', 'multiroom', 'keycorridor', 'lavagap')
    views_a = [a_root/env for env in a_order]
    for directory in views_a:
        if not (directory/'triplets.jsonl').is_file():
            raise ValueError(f'Missing Task A source: {directory}')
    cohort = None
    if args.task_a_cohort:
        cohort = [row['uid'] for row in load_jsonl(args.task_a_cohort)]
    a_rows = build_base_task_a(views_a, output, seed=args.exam_seed,
                               per_trajectory=args.task_a_per_traj, cohort_uids=cohort)
    if len(a_rows) != 103:
        raise ValueError(f'Native Task A generated {len(a_rows)} rows, not 103. '
                         'Use the intended source cohort and original exam parameters; '
                         'automatic trimming or replacement sampling is not allowed.')
    write_jsonl(output/'taskA_expanded_manifest.jsonl', a_rows)
    b_root = args.task_b_raw_root.resolve() if args.task_b_raw_root else args.raw_root.resolve()
    b_sources = ([b_root] if (b_root/'triplets.jsonl').is_file() else
                 sorted(p.parent for p in b_root.rglob('triplets.jsonl')))
    b_rows, _ = build_full_task_b(b_sources, output, seed=args.exam_seed)
    if len(b_rows) != 144:
        raise ValueError(f'Task B requires 144 rows, got {len(b_rows)}')
    for row in b_rows:
        row.update(prompt=TASK_B_PROMPT, source_exam_root=str(output))
    write_jsonl(output/'task_b_exam.jsonl', b_rows)
    c_rows = build_final_task_c(c_source, output)
    if len(c_rows) != 486 or len({r['uid'] for r in c_rows}) != 486:
        raise ValueError(f'Task C requires 486 unique rows, got {len(c_rows)}')
    write_jsonl(output/'taskC_expanded_manifest.jsonl', c_rows)
    d_reference_source = select_groups(combined, ENVIRONMENTS, 3)
    base_d = build_final_task_d(d_reference_source, output)
    if len(base_d) != 144:
        raise ValueError(f'Task D requires 36 base queries x 4 conditions, got {len(base_d)} rows')
    write_jsonl(output/'taskD_expanded_manifest.jsonl', base_d)
    write_json(output/'cohort_contract.json', dict(kind='fresh_base_minigrid_formal',
        historical_uid_replay=False, seed=args.seed, task_a=103, task_b=144,
        task_b_protocol='full_only_144', task_b_per_environment=24,
        task_c=486, task_d=144, environments=list(ENVIRONMENTS),
        a_source_groups_per_environment={env:3 for env in ENVIRONMENTS},
        a_exam_seed=args.exam_seed, a_per_trajectory=args.task_a_per_traj,
        d_query_groups_per_environment=2, d_probe_conditions=4,
        c_environments=list(C_ENVIRONMENTS), c_source_groups_per_environment=6,
        c_factor_counts=dict(variants=3, baseline_temporal_visual=6, framing=3),
        c_variant_counts=dict(Counter(r['variant'] for r in c_rows))))


if __name__ == '__main__':
    main()
