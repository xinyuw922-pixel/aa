#!/usr/bin/env python3
"""Assemble and audit formal exams from newly rendered simulator outputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from functools import lru_cache

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


from utils.json_io import load_jsonl as load


from utils.file_hash import sha256 as sha


from utils.file_hash import pixel_sha256 as pixel_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['scale2', 'habitat'])
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--assets', type=Path, help='Independent HSSD asset copy; required for Habitat reference rendering')
    parser.add_argument('--candidate-only', action='store_true', help='Habitat only: finish the navigation-based candidate pool while interaction rendering continues')
    parser.add_argument('--interaction-root', type=Path, help='Optional Habitat pool qualified by complete C/D pair and independent-reference gates')
    args = parser.parse_args()
    if args.candidate_only and args.dataset != 'habitat':
        parser.error('--candidate-only is only supported for Habitat')
    run = args.run_root.resolve()
    interaction_input = args.interaction_root.resolve() if args.interaction_root else run / 'interaction'
    logs = run / 'formal_assembly_logs'
    logs.mkdir(parents=True, exist_ok=True)
    checkpoint = logs / 'status.json'
    records = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    source_paths = ([run / 'raw_bank/triplets.jsonl'] if args.dataset == 'scale2' else
                    [run / 'navigation/manifest.json'] + ([] if args.candidate_only else [interaction_input / 'manifest.json']))
    if any(not path.is_file() for path in source_paths):
        raise SystemExit('Simulator sources are not complete yet; no assembly has been started')
    source_contract = {str(path): sha(path) for path in source_paths}
    contract_path = logs / 'source_contract.json'
    if contract_path.exists():
        previous_contract = json.loads(contract_path.read_text())
        if any(path in previous_contract and previous_contract[path] != value for path, value in source_contract.items()):
            raise SystemExit('Sources changed after assembly started; preserve this run and use a separate assembly output')
        source_contract = {**previous_contract, **source_contract}
    contract_path.write_text(json.dumps(source_contract, indent=2) + '\n')

    def step(name, target, *extra, timeout=14400):
        command = [sys.executable, str(ROOT / target), *map(str, extra)]
        previous = next((r for r in records if r['name'] == name and r['status'] == 'PASS'), None)
        if previous:
            if previous['command'] != command:
                raise RuntimeError(f'Stage command changed after completion: {name}')
            print(f'Resume completed stage {name}', flush=True)
            return
        attempt = sum(r['name'] == name for r in records) + 1
        log_path = logs / f'{name}_{attempt}.log'
        record = dict(name=name, command=command, log=str(log_path), status='RUNNING')
        records.append(record)
        checkpoint.write_text(json.dumps(records, indent=2) + '\n')
        print(f'Running {name}; log={log_path}', flush=True)
        try:
            with log_path.open('w') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            record.update(returncode=result.returncode, status='PASS' if result.returncode == 0 else 'FAIL')
        except subprocess.TimeoutExpired:
            record.update(status='TIMEOUT')
        checkpoint.write_text(json.dumps(records, indent=2) + '\n')
        if record['status'] != 'PASS':
            raise SystemExit(f'Stage {name} failed; inspect {log_path}. All outputs are retained.')

    if args.dataset == 'scale2':
        raw = run / 'raw_bank'
        validation = json.loads((raw / 'validation_audit.json').read_text())
        if validation.get('invalid_groups') or not validation.get('groups'):
            raise SystemExit('Raw-bank validation did not pass')
        if validation['groups'] != 442:
            step('formal_size_selection', 'tools/minigrid/calibrate_bank.py', '--run-root', run)
            raw = run / 'raw_bank_formal_size'
        audit_suffix = '_formal_similarity' if raw.name == 'raw_bank_formal_size' else '_similarity_audit'
        similarity_root = Path(os.environ.get('GRIDWM_REPORT_ROOT', str(ROOT / 'rebuttal_packet/10_minigrid_10x_expansion')))
        similarity_path = similarity_root / (run.name + audit_suffix + '.json')
        similarity = json.loads(similarity_path.read_text())
        duplicate_fields = ['exact_action_duplicates', 'exact_state_duplicates', 'exact_image_duplicates',
                            'near_action_pairs_lt_0_15', 'near_image_mean_pairs_le_4', 'missing_selected_frames']
        if len(similarity['env_rows']) != 10 or any(row[key] for row in similarity['env_rows'] for key in duplicate_fields):
            raise RuntimeError(f'Formal similarity/environment coverage audit failed: {similarity_path}')
        base = run / 'base_exams'
        step('base_exams', 'tools/minigrid/build_exams.py', '--raw-root', raw,
             '--out-dir', base, '--task-a-target', 1030)
        expected = {'A': 1030, 'C': 1326, 'D': 1296}
        counts = {task: len(load(base / f'task{task}_expanded_manifest.jsonl')) for task in expected}
        if counts != expected:
            raise SystemExit(f'Formal size not met: expected={expected}, observed={counts}; expand the selected bank before acceptance')
        references = run / 'reference_pool'
        step('references', 'tools/minigrid/reference_pool.py', '--base-root', base,
             '--run-root', run, '--out-dir', references)
        c_root, d_root = run / 'corrected_c', run / 'matched_d'
        step('correct_c', 'tools/minigrid/task_c.py',
             '--source', base / 'taskC_expanded_manifest.jsonl', '--output-dir', c_root)
        step('matched_d', 'tools/minigrid/task_d.py', '--scale2-root', base,
             '--output-dir', d_root, '--raw-source', f'fresh={references / "existing"}',
             '--generated-root', references / 'generated', '--generated-source-tag', 'new_simulator_references')
        a_root = run / 'validated_a'
        step('a_candidate_qc', 'tools/minigrid/task_a.py', '--source',
             base / 'taskA_expanded_manifest.jsonl', '--output-dir', a_root)
        final = run / 'final_manifests_validated'
        final.mkdir(exist_ok=True)
        source_files = {'A': a_root, 'C': c_root, 'D': d_root}
        for task, source in source_files.items():
            destination = final / f'task{task}_expanded_manifest.jsonl'
            if destination.exists():
                if sha(destination) != sha(source / destination.name):
                    raise RuntimeError(f'Refusing changed final manifest: {destination}')
            else:
                shutil.copy2(source / destination.name, destination)
        step('requests_validated', 'tools/exams/build_requests.py', '--manifest-dir', final,
             '--out-dir', run / 'requests_validated', '--mode', 'full')
        report = dict(dataset='scale2', counts=counts, status='PASS', manifests={}, tasks={}, images_checked=0)
        run_config = json.loads((run / 'run_config.json').read_text())
        report['seed_offset'] = run_config['seed_offset']
        report['historical_seed_offset'] = 830000
        report['historical_seed_replay'] = run_config['seed_offset'] == 830000
        report['historical_byte_equality_claimed'] = False
        report['similarity_audit'] = dict(path=str(similarity_path), sha256=sha(similarity_path),
                                          groups=similarity['total_groups'], environments=similarity['env_rows'])
        checked = set()
        for task, size in expected.items():
            path = final / f'task{task}_expanded_manifest.jsonl'
            rows = load(path)
            if len(rows) != size or len({r['uid'] for r in rows}) != size:
                raise RuntimeError(f'Count or unique UID mismatch for {task}')
            if task in {'C', 'D'}:
                groups = defaultdict(list)
                for row in rows:
                    groups[row['group_id']].append(row)
                    if row['answer'] != ('Fail' if row['variant'] == 'cf' else 'Success'):
                        raise RuntimeError(f'Incorrect variant label: {row["uid"]}')
                if any(len(g) != 3 or {r['variant'] for r in g} != {'full', 'nocue', 'cf'} for g in groups.values()):
                    raise RuntimeError(f'Incomplete variants in {task}')
            elif set(r['answer'] for r in rows) != {'A', 'B', 'C', 'D'}:
                raise RuntimeError('Invalid Task A answers')
            if task == 'A':
                for row in rows:
                    source = Path(row['source_raw_root'])
                    candidates = [pixel_sha(source / p) for p in row['candidate_frames']]
                    target = pixel_sha(source / row['correct_next_frame'])
                    if len(candidates) != 4 or len(set(candidates)) != 4 or candidates['ABCD'.index(row['answer'])] != target:
                        raise RuntimeError(f'A candidate/answer ambiguity: {row["uid"]}')
            for row in rows:
                image_path = Path(row['source_exam_root']) / row['image']
                if not image_path.resolve().is_relative_to(run):
                    raise RuntimeError(f'Nonlocal image input: {image_path}')
                if image_path not in checked:
                    with Image.open(image_path) as image:
                        image.load()
                        if all(low == high for low, high in image.convert('RGB').getextrema()):
                            raise RuntimeError(f'Blank image: {image_path}')
                    checked.add(image_path)
            report['manifests'][task] = dict(path=str(path.relative_to(run)), sha256=sha(path))
            report['tasks'][task] = dict(rows=len(rows), environments=dict(Counter(r['env_task'] for r in rows)),
                                        answers=dict(Counter(r['answer'] for r in rows)))
        report['images_checked'] = len(checked)
        report['task_a_candidate_qc'] = json.loads((a_root / 'candidate_qc_report.json').read_text())
        audit = load_csv(d_root / 'matched_reference_audit.csv')
        if not all(r['exact_mission_match'] == 'True' and r['distinct_simulator_episode'] == 'True' for r in audit):
            raise RuntimeError('D reference mission/independence audit failed')
        request_path = run / 'requests_validated/_requests_from_manifest/requests_ACD_full.jsonl'
        request_rows = load(request_path)
        if len(request_rows) != sum(expected.values()):
            raise RuntimeError('Request count mismatch')
        report['request_rows'] = len(request_rows)
    else:
        if not args.assets:
            parser.error('--assets is required for Habitat')
        assets = args.assets.resolve()
        navigation, interaction, pool = run / 'navigation', interaction_input, run / 'candidate_pool'
        for path in [navigation / 'manifest.json'] + ([] if args.candidate_only else [interaction / 'manifest.json']):
            if not path.is_file():
                raise SystemExit(f'Wait for simulator rendering to finish: {path}')
        step('candidate_pool', 'tools/habitat/build_candidates.py',
             '--source', navigation, '--output', pool)
        if args.candidate_only:
            manifest = json.loads((pool / 'manifest.json').read_text())
            print(json.dumps(dict(status='CANDIDATE_POOL_ONLY', counts=manifest['counts'],
                                  formal_550_complete=False), indent=2))
            return
        qualification_path = interaction / 'qualification_report.json'
        if qualification_path.is_file():
            qualification = json.loads(qualification_path.read_text())
            if qualification['status'] != 'PASS' or qualification['selected_targets'] != 100:
                raise RuntimeError('Qualified Habitat joint pool did not meet the 100-target contract')
        else:
            step('references', 'tools/habitat/generate_references.py', '--root', interaction,
                 '--scene-root', assets / 'scenes-articulated', '--scene-dataset-config',
                 assets / 'hssd-hab-articulated.scene_dataset_config.json', '--config',
                 ROOT / 'tools/habitat/config.hssd-pilot.json', '--max-targets', 100,
                 '--reference-phase-attempts', 8)
        step('validate_c', 'tools/habitat/validate_interactions.py', interaction, '--check-images')
        step('validate_d', 'tools/habitat/validate_references.py', interaction,
             '--check-images', '--expected-targets', 100, '--min-reference-query-difference', 0.20)
        exam = run / 'exam_550'
        step('assemble_550', 'tools/habitat/build_exams.py', '--task-a-root', pool,
             '--interaction-root', interaction, '--output-dir', exam,
             '--task-a-count', 150, '--task-c-count', 200, '--task-d-count', 200)
        step('requests', 'tools/habitat/build_requests.py',
             '--exam-dir', exam, '--output-dir', run / 'requests', '--task-d-montage')
        manifest = json.loads((exam / 'manifest.json').read_text())
        if manifest['task_counts'] != {'A': 150, 'C': 200, 'D': 200} or manifest['total_questions'] != 550:
            raise RuntimeError('Habitat formal count mismatch')
        questions = json.loads((exam / 'questions.json').read_text())
        ids = [row['question_id'] for row in questions]
        if len(set(ids)) != 550:
            raise RuntimeError('Duplicate Habitat question identifiers')
        report = dict(dataset='habitat', status='PASS', counts=manifest['task_counts'], total=550,
                      image_references_checked=manifest['image_references_checked'],
                      files={p: sha(exam / p) for p in ['questions.json', 'answer_key.json', 'answer_sheet.csv', 'manifest.json', 'questions.html']})
        if qualification_path.is_file():
            report['joint_pool_qualification'] = dict(path=str(qualification_path), sha256=sha(qualification_path),
                                                      selected_targets=qualification['selected_targets'], eligible_targets=qualification['eligible_targets'])
        request_audit = {}
        all_ids = []
        for task, expected in manifest['task_counts'].items():
            path = run / 'requests' / f'requests_{task}.jsonl'
            rows = load(path)
            if len(rows) != expected:
                raise RuntimeError(f'Habitat {task} offline request count mismatch')
            for row in rows:
                all_ids.append(row['uid'])
                if any(not Path(image).is_file() for image in row['images']):
                    raise RuntimeError(f'Unavailable model input image: {row["uid"]}')
            request_audit[task] = dict(rows=len(rows), sha256=sha(path))
        if len(set(all_ids)) != 550:
            raise RuntimeError('Habitat offline request UIDs are not unique')
        c_ids = {row['episode_id'] for row in load(interaction / 'taskc/task_c.jsonl')}
        d_ids = {row['episode_id'] for row in load(interaction / 'taskd/task_d.jsonl')}
        if c_ids != d_ids or len(c_ids) != 100:
            raise RuntimeError('Habitat C/D target cohorts differ')
        report['requests'] = request_audit
        report['unique_request_ids'] = len(set(all_ids))
        report['same_c_d_target_cohort'] = True
    report['model_api_calls'] = 0
    report['source_scope'] = 'New simulator runs in this independent reproduction copy, not archived exam images'
    (run / 'formal_acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def load_csv(path):
    import csv
    with path.open() as handle:
        return list(csv.DictReader(handle))


if __name__ == '__main__':
    main()
