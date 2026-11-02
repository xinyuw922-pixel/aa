#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native Task B visible-state exams from simulator trajectories.

Answers use visible-image coordinates and simulator observation visibility.
The standalone CLI writes exam and image references with the Task B prompt.
The default CLI selects 24 pixel-distinct Full frames per environment (144 total).
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, List, Tuple
from collections import defaultdict

import numpy as np

from utils.exam_schema import EXAM_SCHEMA_VERSION

from tools.minigrid.exam_source import parse_traj, read_jsonl, stable_int
from utils.minigrid_visible_state import visible_state_answer
from utils.file_hash import pixel_sha256

ENVIRONMENTS = ('doorkey', 'keycorridor', 'lavagap', 'memory', 'multiroom', 'redblue')


def build_full_task_b(task_dirs: List[Path], out_dir: Path, *, per_environment: int = 24,
                      seed: int = 42) -> Tuple[List[dict], dict]:
    """Select balanced Full-only questions without duplicate images per environment."""
    if per_environment < 1:
        raise ValueError('Task B per-environment count must be positive')
    from utils.json_io import load_jsonl
    from utils.minigrid_prompts import TASK_B_PROMPT
    groups = defaultdict(dict)
    for source in sorted(task_dirs):
        for record in load_jsonl(source/'triplets.jsonl'):
            if record['variant'] != 'full':
                continue
            env = record['task']
            gid = f"{record['env_id']}__{record['group_id']}"
            if env not in ENVIRONMENTS:
                continue
            if gid in groups[env]:
                raise ValueError(f'Duplicate Full source: {env}/{gid}')
            frames, states = record['frames'], record['state_seq']
            if not frames or len(frames) != len(states):
                raise ValueError(f'Unaligned frames/states: {env}/{gid}')
            rng = np.random.default_rng(stable_int(f'{seed}|B-full|{env}|{gid}'))
            candidates = []
            for step in rng.permutation(len(frames)):
                step = int(step)
                image = (source/frames[step]).resolve(strict=True)
                candidates.append((gid, step, image, pixel_sha256(image), states[step]))
            groups[env][gid] = candidates
    selected = []
    stats = {}
    # Round-robin across groups prevents long trajectories from owning the exam.
    for env in ENVIRONMENTS:
        queues = [list(candidates) for _, candidates in sorted(groups[env].items())]
        seen = set()
        chosen = []
        while len(chosen) < per_environment:
            progress = False
            for queue in queues:
                while queue:
                    gid, step, image, fingerprint, state = queue.pop(0)
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    chosen.append((gid, step, image, state))
                    progress = True
                    break
                if len(chosen) == per_environment:
                    break
            if not progress:
                raise ValueError(f'{env}: need {per_environment} pixel-distinct Full frames, '
                                 f'found {len(chosen)}; generate more Full trajectories')
        stats[env] = len(chosen)
        selected.extend((env, *item) for item in chosen)
    rows = []
    for env, gid, step, source, state in selected:
        uid = f'B.{env}.{gid}.full.t{step}'
        image_rel = f'images/taskB/{env}/{uid}.png'
        destination = out_dir/image_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        answer = visible_state_answer(state)
        rows.append(dict(exam_id=uid, uid=uid, schema_version=EXAM_SCHEMA_VERSION,
                         task='B', image=image_rel, answer_json=answer,
                         answer=json.dumps(answer), prompt=TASK_B_PROMPT,
                         variant='full', _variant='full', group_id=gid, source_frame=str(source),
                         source_step=step, source_exam_root=str(out_dir.resolve())))
    return rows, stats

def build_taskB(task_dirs: List[Path], out_dir: Path, args: Any) -> Tuple[List[dict], dict]:
    rows, stats = [], defaultdict(int)
    img_root = out_dir / "images" / "taskB"

    for td in task_dirs:
        for rec in read_jsonl(td / "triplets.jsonl"):
            tr = parse_traj(rec)
            if tr.variant != "full": continue

            rng = np.random.default_rng(stable_int(f"{args.seed}|B|{tr.group_id}"))
            valid_frames = min(tr.n_frames, len(tr.state_seq))
            if valid_frames == 0: continue

            ts = rng.choice(range(valid_frames), size=min(valid_frames, args.b_per_traj), replace=False)

            for t in ts:
                rel_path = f"{tr.task}/{tr.group_id}_t{t}.png"
                dest = img_root / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)

                try:
                    shutil.copy(td / tr.frames[t], dest)
                except FileNotFoundError:
                    stats[f"{tr.task}_skip_missing_frame"] += 1
                    continue

                ans = visible_state_answer(tr.state_seq[t])
                rows.append({
                    "exam_id": f"B.{tr.task}.{tr.group_id}.t{t}",
                    "uid": f"B.{tr.task}.{tr.group_id}.t{t}",
                    "schema_version": EXAM_SCHEMA_VERSION,
                    "task": "B", "image": f"images/taskB/{rel_path}",
                    "answer_json": ans, "answer": json.dumps(ans)
                })
                stats[tr.task] += 1
    return rows, dict(stats)


def main() -> None:
    """Build visible-state Task B exams directly from trajectory records."""
    from utils.minigrid_prompts import TASK_B_PROMPT
    from utils.json_io import write_jsonl
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--per-environment', type=int, default=24)
    parser.add_argument('--b-per-traj', type=int, help='Explicit variable-size per-trajectory sampling')
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error('Output exists; use a fresh directory')
    sources = ([args.root] if (args.root/'triplets.jsonl').is_file() else
               sorted(p.parent for p in args.root.rglob('triplets.jsonl')))
    if not sources:
        parser.error('No environment triplets.jsonl files found under --root')
    if args.per_environment < 1 or (args.b_per_traj is not None and args.b_per_traj < 1):
        parser.error('Question counts must be positive')
    rows, stats = (build_taskB(sources, args.out_dir, args) if args.b_per_traj is not None else
                   build_full_task_b(sources, args.out_dir, per_environment=args.per_environment,
                                     seed=args.seed))
    for row in rows:
        row.update(prompt=TASK_B_PROMPT, source_exam_root=str(args.out_dir.resolve()))
    write_jsonl(args.out_dir/'task_b_exam.jsonl', rows)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
