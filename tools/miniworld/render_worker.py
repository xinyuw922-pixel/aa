#!/usr/bin/env python3
"""Render one native FourRooms candidate, then release all process resources."""
import argparse
import json
from pathlib import Path
import random
import resource

from tools.miniworld.generate_fourrooms import make_question
from utils.random_state import tuples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    if args.result.exists():
        parser.error('Worker result exists; nothing overwritten')
    request = json.loads(args.request.read_text())
    random.setstate(tuples(request['random_state']))
    kwargs = request['kwargs']
    kwargs['img_dir'] = Path(kwargs['img_dir'])
    try:
        result = dict(status='ok', record=make_question(**kwargs))
    except ValueError as exc:
        if 'Degenerate' not in str(exc) and 'Subtle' not in str(exc):
            raise
        result = dict(status='rejected', reason=str(exc))
    result.update(random_state=random.getstate(), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.result.open('x') as handle:
        json.dump(result, handle)


if __name__ == '__main__':
    main()
