#!/usr/bin/env python3
"""Meet the formal group budget using unchanged selection gates on fresh pools."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


from utils.json_io import load_jsonl as load


def count(root):
    roots = [root] if (root / 'triplets.jsonl').is_file() else [p for p in root.iterdir() if (p / 'triplets.jsonl').is_file()]
    return sum(sum(row['variant'] == 'full' for row in load(p / 'triplets.jsonl')) for p in roots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--target-groups', type=int, default=442)
    args = parser.parse_args()
    run = args.run_root.resolve()
    output, selection = run / 'raw_bank_formal_size', run / 'putnear_formal_size'
    if output.exists() or selection.exists():
        parser.error('Formal-size output already exists and is preserved')
    tier1, nav = run / 'tier1_selected', run / 'tier2_navigation_selected'
    n1, nn = count(tier1), count(nav)
    putnear_target = args.target_groups - n1 - nn
    if putnear_target < 2:
        raise SystemExit('Formal budget leaves no independent PutNear query/reference pair')
    commands = [
        ['select_bank.py', '--source-root', run / 'putnear_pool', '--out-root', selection,
         '--target-per-env', str(putnear_target), '--report-stem', run.name + '_formal_putnear'],
        ['merge_banks.py', '--source', f'tier1={tier1}', '--source', f'tier2nav={nav}',
         '--source', f'putnear={selection}', '--out-root', output, '--report-stem', run.name + '_formal_size'],
        ['validate_navigation.py', '--root', output],
        ['similarity.py', '--roots', output, '--report-stem', run.name + '_formal_similarity'],
    ]
    for index, command in enumerate(commands):
        with (run / f'formal_size_stage_{index + 1}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(ROOT / 'tools/minigrid' / command[0]), *map(str, command[1:])],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        if result.returncode:
            raise SystemExit(f'Formal-size stage {index + 1} failed; all intermediate files retained')
        if index == 0 and count(selection) != putnear_target:
            raise SystemExit(f'Need additional fresh candidates: requested {putnear_target} PutNear groups, got {count(selection)}')
    observed = count(output)
    if observed != args.target_groups:
        raise SystemExit(f'Expected {args.target_groups} formal groups, found {observed}')
    report = dict(status='PASS', target_groups=args.target_groups, observed_groups=observed,
                  tier1_groups=n1, navigation_groups=nn, putnear_groups=putnear_target,
                  selection_rule='Unchanged exact/near-duplicate filtering; adjust only the requested PutNear quota',
                  output=str(output))
    (run / 'formal_size_selection.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
