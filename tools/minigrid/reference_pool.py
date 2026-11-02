#!/usr/bin/env python3
"""Collect fresh successful references and simulate missing PutNear missions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/minigrid'))
from tools.minigrid.task_d import episode_identity
from tools.minigrid.generate_interactions import INTERACTION_ENVS, generate_one
import gymnasium as gym
import minigrid  # noqa: F401


from utils.json_io import load_jsonl as load


from utils.json_io import write_jsonl as dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed-start', type=int, default=7000000)
    parser.add_argument('--max-seeds', type=int, default=100000)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error('Reference output already exists and is preserved; choose another directory')
    base, run, out = args.base_root.resolve(), args.run_root.resolve(), args.out_dir.resolve()
    rows_c = load(base / 'taskC_expanded_manifest.jsonl')
    by_uid = {row['uid']: row for row in rows_c}
    queries = [by_uid[row['query_uid']] for row in load(base / 'taskD_expanded_manifest.jsonl') if row['variant'] == 'full']
    identities, candidates = {}, {}
    for path in sorted(run.rglob('triplets.jsonl')):
        for row in load(path):
            if row.get('variant') != 'full' or row.get('success') is not True:
                continue
            env = str(row.get('task') or row.get('env_task'))
            identity = episode_identity(row, path.parent)
            identities[(env, str(row['group_id']))] = identity
            normalized = dict(row)
            normalized['frames'] = [str((path.parent / frame).resolve()) for frame in row['frames']]
            candidates.setdefault(identity, normalized)
    signatures = defaultdict(set)
    for identity, row in candidates.items():
        signatures[(str(row.get('task') or row.get('env_task')), str(row['mission']))].add(identity)
    missing = set()
    for query in queries:
        key = (str(query['env_task']), str(query['mission']))
        identity = identities[(key[0], str(query['group_id']))]
        if not signatures[key] - {identity}:
            missing.add(key)
    unsupported = [key for key in missing if key[0] not in {'putnear', 'keycorridor'}]
    if unsupported:
        raise RuntimeError(f'Need additional simulator reference generators for: {unsupported}')
    out.mkdir(parents=True)
    dump(out / 'existing/triplets.jsonl', list(candidates.values()))
    generated_root = out / 'generated'
    generated_root.mkdir()
    generated, audit = [], []
    for key in sorted(key for key in missing if key[0] == 'keycorridor'):
        env_id = next(row['env_id'] for row in rows_c if row['env_task'] == key[0])
        search_env = gym.make(env_id)
        try:
            for seed in range(args.seed_start, args.seed_start + args.max_seeds):
                search_env.reset(seed=seed)
                if str(search_env.unwrapped.mission) != key[1]:
                    continue
                attempt = generated_root / f'keycorridor_{seed}'
                cmd = [sys.executable, str(ROOT / 'tools/minigrid/environments/keycorridor.py'), 'generate',
                       '--out-dir', str(attempt), '--env-id', env_id, '--num', '1',
                       '--seed-start', str(seed), '--max-attempts-total', '1']
                log_path = generated_root / f'keycorridor_{seed}.log'
                with log_path.open('w') as log:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, timeout=120)
                path = attempt / 'triplets.jsonl'
                if result.returncode or not path.is_file():
                    continue
                rows = load(path)
                full = next((row for row in rows if row['variant'] == 'full'), None)
                if not full or full.get('success') is not True or full['mission'] != key[1]:
                    continue
                for row in rows:
                    row['frames'] = [str((attempt / frame).resolve()) for frame in row['frames']]
                    row['group_id'] = f'keycorridor_reference_s{seed}'
                generated.extend(rows)
                audit.append(dict(env='keycorridor', mission=key[1], seed=seed, group_id=rows[0]['group_id']))
                missing.remove(key)
                break
        finally:
            search_env.close()
    env = gym.make(INTERACTION_ENVS['putnear']['env_id'])
    try:
        for seed in range(args.seed_start, args.seed_start + args.max_seeds):
            if not any(key[0] == 'putnear' for key in missing):
                break
            env.reset(seed=seed)
            mission = str(env.unwrapped.mission)
            key = ('putnear', mission)
            if key not in missing:
                continue
            rows = generate_one('putnear', INTERACTION_ENVS['putnear'], seed, 950000 + len(audit), generated_root)
            if rows is None:
                continue
            full = next(row for row in rows if row['variant'] == 'full')
            if full.get('success') is not True or str(full['mission']) != mission:
                raise RuntimeError(f'Invalid simulator reference at seed {seed}')
            generated.extend(rows)
            audit.append(dict(env='putnear', mission=mission, seed=seed, group_id=full['group_id']))
            missing.remove(key)
            print(f'Generated reference seed={seed}, remaining mission gaps={len(missing)}', flush=True)
    finally:
        env.close()
    dump(generated_root / 'triplets.jsonl', generated)
    report = dict(status='PASS' if not missing else 'FAIL', existing_successful_episodes=len(candidates),
                  generated_groups=len(audit), generated=audit, unresolved=sorted(missing),
                  rule='Exact simulator mission; independent environment/seed; no reassignment of mission labels')
    (out / 'reference_pool_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(bool(missing))


if __name__ == '__main__':
    main()
