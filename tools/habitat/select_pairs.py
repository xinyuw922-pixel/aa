#!/usr/bin/env python3
"""Select complete, simulator-valid C/D pairs before any model evaluation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
from tools.habitat.dataset.task_design import TASK_D_ARTICULATED_PROMPT_STYLES, task_d_articulated_prompts


from utils.json_io import load_jsonl as load


from utils.file_hash import sha256 as sha


from utils.json_io import write_jsonl as dump


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', required=True, type=Path,
                        help='Fresh rendered source root; earlier sources have priority')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--targets', type=int, default=100)
    parser.add_argument('--min-reference-query-difference', type=float, default=0.20)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Qualified output already exists and is preserved')
    eligible, excluded, source_records, seen = [], [], [], set()
    for source in args.source:
        source = source.resolve()
        path = source / 'taskc/task_c.jsonl'
        pairs = defaultdict(dict)
        for row in load(path):
            pairs[row['episode_id']][row['condition']] = row
        source_records.append(dict(root=str(source), taskc_sha256=sha(path), target_count=len(pairs)))
        for episode_id in sorted(pairs):
            if episode_id in seen:
                raise RuntimeError(f'Duplicate episode ID across fresh sources: {episode_id}')
            seen.add(episode_id)
            pair = pairs[episode_id]
            if set(pair) != {'full', 'cf'} or pair['full']['gold'] != 'success' or pair['cf']['gold'] != 'fail':
                raise RuntimeError(f'Invalid source Full/CF pair: {episode_id}')
            folder = source / 'episodes' / pair['full']['scene'] / episode_id
            cached_rows, cached_metadata = folder / 'taskd_rows.json', folder / 'taskd_reference.json'
            if not cached_rows.is_file() or not cached_metadata.is_file():
                excluded.append(dict(episode_id=episode_id, source=str(source), reason='no_qualified_independent_reference_cache'))
                continue
            rows_d = json.loads(cached_rows.read_text())
            metadata = json.loads(cached_metadata.read_text())
            difference = metadata['reference_query_view_difference']['changed_pixel_fraction']
            if len(rows_d) != 2 or {r['condition'] for r in rows_d} != {'full', 'cf'}:
                raise RuntimeError(f'Invalid cached D pair: {episode_id}')
            if difference < args.min_reference_query_difference:
                excluded.append(dict(episode_id=episode_id, source=str(source), reason='reference_view_difference_below_gate', difference=difference))
                continue
            eligible.append(dict(episode_id=episode_id, folder=folder, c=pair, d=rows_d, source=source,
                                 difference=difference, reference_metadata_sha256=sha(cached_metadata),
                                 reference_rows_sha256=sha(cached_rows)))
    if len(eligible) < args.targets:
        raise SystemExit(f'Insufficient qualified joint targets: {len(eligible)}/{args.targets}; generate more sources, do not relax gates')
    selected = eligible[:args.targets]
    for entry in eligible[args.targets:]:
        excluded.append(dict(episode_id=entry['episode_id'], source=str(entry['source']), reason='qualified_surplus_outside_fixed_pre_inference_quota'))
    output.mkdir(parents=True)
    rows_c, rows_d, records = [], [], []
    for index, entry in enumerate(selected):
        full = entry['c']['full']
        destination = output / 'episodes' / full['scene'] / entry['episode_id']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(os.path.relpath(entry['folder'], destination.parent), target_is_directory=True)
        rows_c.extend([entry['c']['full'], entry['c']['cf']])
        style = TASK_D_ARTICULATED_PROMPT_STYLES[index % len(TASK_D_ARTICULATED_PROMPT_STYLES)]
        en, zh = task_d_articulated_prompts(style)
        for row in sorted(entry['d'], key=lambda r:r['condition']):
            row = dict(row)
            row.update(prompt_style=style, prompt=en, prompt_zh=zh,
                       prompt_template_id=f'taskd_articulated_{style}_v1')
            rows_d.append(row)
        records.append(dict(episode_id=entry['episode_id'], source=str(entry['source']), scene=full['scene'],
                            reference_view_difference=entry['difference'], prompt_style=style,
                            reference_metadata_sha256=entry['reference_metadata_sha256'],
                            reference_rows_sha256=entry['reference_rows_sha256']))
    dump(output / 'taskc/task_c.jsonl', rows_c)
    dump(output / 'taskd/task_d.jsonl', rows_d)
    report = dict(status='PASS', selected_targets=len(selected), eligible_targets=len(eligible),
                  task_c_rows=len(rows_c), task_d_rows=len(rows_d), sources=source_records,
                  selected=records, excluded=excluded,
                  minimum_reference_query_difference=args.min_reference_query_difference,
                  prompt_style_targets=dict(Counter(r['prompt_style'] for r in records)),
                  policy='Fixed source-priority order; simulator-valid complete Full/CF pairs with independent successful references; no model response or score used',
                  style_policy='Counterbalance all five unchanged prompt templates across selected targets before inference; one style shared by each Full/CF pair')
    (output / 'qualification_report.json').write_text(json.dumps(report, indent=2) + '\n')
    manifest = dict(schema_version='hssd-qualified-joint-pool-v1', status='PASS', episode_count=len(selected),
                    question_count=len(rows_c), source_records=source_records,
                    qualification_report_sha256=sha(output / 'qualification_report.json'))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['selected', 'excluded']}, indent=2))


if __name__ == '__main__':
    main()
