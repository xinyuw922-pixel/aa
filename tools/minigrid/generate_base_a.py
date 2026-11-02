"""Generate the six-environment native source cohort for base Task A."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils.json_io import load_jsonl, write_jsonl

SOURCE_CONFIG = (
    ('memory', 'MiniGrid-MemoryS13-v0', 0),
    ('redblue', 'MiniGrid-RedBlueDoors-8x8-v0', 0),
    ('doorkey', 'MiniGrid-DoorKey-8x8-v0', 10),
    ('multiroom', 'MiniGrid-MultiRoom-N6-v0', 0),
    ('keycorridor', 'MiniGrid-KeyCorridorS6R3-v0', 0),
    ('lavagap', 'MiniGrid-LavaGapS7-v0', 0),
)


def main() -> None:
    """Run internal simulators, preserving seed-derived Task A group identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('Output exists; use a fresh directory')
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(repo)}
    for name, env_id, seed_start in SOURCE_CONFIG:
        raw = output/'generated'/name
        command = [sys.executable, '-B', '-m', f'tools.minigrid.environments.{name}',
                   'generate', '--out-dir', str(raw), '--num', '3',
                   '--seed-start', str(seed_start), '--env-id', env_id]
        with (output/f'{name}.log').open('w') as log:
            subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        full = [r for r in load_jsonl(raw/'triplets.jsonl') if r['variant']=='full']
        if len(full)!=3:
            raise ValueError(f'{name}: expected three Full trajectories, found {len(full)}')
        rows = [{**r, 'group_id': f"{name}_s{r['seed']:06d}",
                 'frames': [str((raw/frame).resolve(strict=True)) for frame in r['frames']]}
                for r in full]
        write_jsonl(output/'sources'/name/'triplets.jsonl', rows)
        print(f'{name}: '+', '.join(str(r['seed']) for r in full), flush=True)


if __name__ == '__main__':
    main()
