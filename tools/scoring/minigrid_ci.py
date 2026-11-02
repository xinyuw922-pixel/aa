"""Cluster confidence intervals over native MiniGrid and Scale-2 question scores."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from utils.cluster_bootstrap import ratio_interval
from utils.exam_schema import build_exam_meta


def confidence_report(rows: list[dict], gold: dict, iterations: int, seed: int,
                      exam_metadata: dict | None = None) -> dict:
    """Compute A/C/D accuracy and complete-triplet diagnostic intervals.

    A group is a simulator trajectory family within an environment. All of its
    questions, styles and interventions stay together in each resampling draw.
    """
    selected = [r for r in rows if r['task'] in ('A', 'C', 'D')]
    if not selected:
        raise ValueError('CI requires at least one Task A/C/D question')
    for row in selected:
        if not row.get('env_task') or not row.get('group_id'):
            raise ValueError('CI requires env_task and group_id: '+row['uid'])
    results, coverage = [], []

    def emit(task: str, scope: dict, metric: str, observations: list) -> None:
        for unit in ('group', 'environment'):
            sums = defaultdict(lambda: np.zeros(2))
            for env, group, numerator, denominator in observations:
                key = (env, group) if unit == 'group' else env
                sums[key] += (numerator, denominator)
            identity = json.dumps([task, scope, metric, unit], sort_keys=True)
            local_seed = (seed + int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little')) % (2**63-1)
            interval = ratio_interval(np.asarray(list(sums.values())), iterations, local_seed)
            results.append(dict(task=task, scope=scope, metric=metric, resampling_unit=unit,
                                observations=len(observations), numerator=sum(o[2] for o in observations),
                                denominator=sum(o[3] for o in observations), **interval))

    for task in ('A', 'C', 'D'):
        task_rows = [r for r in selected if r['task'] == task]
        if not task_rows:
            continue
        slices = [(dict(), task_rows)]
        if task in ('C', 'D'):
            for variant in sorted({r.get('variant') or '' for r in task_rows}):
                slices.append((dict(variant=variant), [r for r in task_rows if (r.get('variant') or '') == variant]))
        for scope, subset in slices:
            emit(task, scope, 'Accuracy', [(r['env_task'], r['group_id'], int(r['correct']), 1) for r in subset])
        if task == 'A':
            continue
        pairs = defaultdict(dict)
        for row in task_rows:
            meta = build_exam_meta(row['uid'], row=(exam_metadata or {}).get(row['uid'], row))
            source = (exam_metadata or {}).get(row['uid'], {})
            key = (row['env_task'], row['group_id'], meta.get('temporal') or '',
                   meta.get('visual') or '', meta.get('framing') or '',
                   source.get('reference_group_id') or '')
            variant = row.get('variant')
            if variant not in ('full', 'nocue', 'cf'):
                raise ValueError('Unsupported diagnostic variant: '+str(variant))
            if variant in pairs[key]:
                raise ValueError('Duplicate variant within diagnostic pairing: '+str(key))
            pairs[key][variant] = row
        observations = defaultdict(list)
        complete = 0
        for key, variants in pairs.items():
            if set(variants) != {'full', 'nocue', 'cf'}:
                continue
            if any(str(gold[r['uid']].c_answer).strip().lower() != ('fail' if v == 'cf' else 'success')
                   for v, r in variants.items()):
                raise ValueError('Diagnostic metrics require Success/Success/Fail gold labels: '+str(key))
            complete += 1
            env, group = key[:2]
            active = int(variants['full'].get('pred_norm') == 'Success')
            joint = active * int(variants['cf'].get('pred_norm') == 'Fail')
            for metric, num, den in [('p_active', active, 1), ('IDR_raw', joint, 1), ('IDR_cond', joint, active)]:
                observations[metric].append((env, group, num, den))
        coverage.append(dict(task=task, pair_cells=len(pairs), complete_triplets=complete,
                             incomplete_triplets=len(pairs)-complete))
        for metric, data in observations.items():
            emit(task, dict(pairing='complete_triplets'), metric, data)
    return dict(method='cluster_percentile', confidence_level=.95, iterations=iterations, seed=seed,
                estimand='pooled question accuracy; pooled complete-triplet diagnostic ratios',
                failure_policy='Native missing/API/parse failures retained in accuracy denominator',
                seed_derivation='SHA256(task, scope, metric, resampling_unit) plus base seed modulo 2**63-1',
                pairing_fields=['env_task', 'group_id', 'temporal', 'visual', 'framing', 'reference_group_id'],
                diagnostic_coverage=coverage, results=results)


def write_report(output: Path, rows: list[dict], gold: dict, iterations: int, seed: int,
                 exam_metadata: dict | None = None) -> None:
    """Write machine-readable intervals without altering native score reports."""
    report = confidence_report(rows, gold, iterations, seed, exam_metadata)
    (output/'confidence_intervals.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    flattened = [dict(r, scope=json.dumps(r['scope'], sort_keys=True)) for r in report['results']]
    with (output/'confidence_intervals.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)
