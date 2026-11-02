"""Score recorded responses offline with an explicit native experiment protocol."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from tools.scoring.protocols import PROTOCOLS, canonical_protocol, describe
from utils.exam_schema import resolve_uid
from utils.file_hash import sha256
from utils.json_io import load_jsonl

ROOT = Path(__file__).resolve().parents[2]


def unique(rows: list, key, label: str) -> dict:
    """Require unambiguous IDs before invoking a native scorer."""
    result = {}
    for row in rows:
        uid = key(row)
        if not isinstance(uid, str) or not uid:
            raise ValueError(label + ' has a missing/non-string ID')
        if uid in result:
            raise ValueError(label + ' has a duplicate ID: ' + uid)
        result[uid] = row
    return result


def write_receipt(path: Path, value: dict) -> None:
    """Atomically record invocation status without modifying native reports."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def score(args) -> dict:
    """Validate inputs and delegate metric/parse computation to internal scorers."""
    contract = describe(args.protocol)
    family = contract['family']
    if args.ci and (family != 'ssot' or args.per_row_only):
        raise ValueError('--ci requires the SSOT protocol without --per-row-only')
    exam_task = 'D' if args.exam_task in ('DR', 'D-R') else args.exam_task
    if args.exam_task and family != 'ssot':
        raise ValueError('--exam-task is only valid with the SSOT protocol')
    output = args.output.resolve()
    if output.exists() or output == ROOT or ROOT in output.parents or output in ROOT.parents:
        raise ValueError('Use a fresh scoring output directory outside the source repository')
    responses = args.responses.resolve(strict=True)
    rows = load_jsonl(responses)
    if any(not isinstance(r, dict) or (r.get('meta') is not None and not isinstance(r.get('meta'), dict)) for r in rows):
        raise ValueError('Response records must be objects with object/null metadata')
    if any(r.get('synthetic') or (r.get('meta') or {}).get('synthetic') for r in rows):
        raise ValueError('This entry accepts recorded model responses, not synthetic validation responses')
    inputs = [responses]
    command = [sys.executable, str(ROOT / contract['script'])]
    native = output / 'native'
    gold_aliases = {}
    iterations = contract['iterations'] if args.iterations is None else args.iterations
    seed = contract['seed'] if args.seed is None else args.seed
    if family in ('structured', 'oracle_state'):
        if args.requests is None or args.exam_dir or args.exam_file:
            raise ValueError('This protocol requires --requests only')
        requests = args.requests.resolve(strict=True)
        expected = unique(load_jsonl(requests), lambda r: r.get('uid'), 'Requests')
        inputs.append(requests)
        if iterations is None or iterations <= 0:
            raise ValueError('Bootstrap iterations must be positive')
        if not args.model:
            raise ValueError('--model is required; native statistical seeds depend on model ID')
        if family == 'structured':
            command += ['--requests', str(requests), '--response', f'{args.model}={responses}',
                        '--output-dir', str(native), '--iterations', str(iterations), '--seed', str(seed)]
        else:
            command += ['--requests', str(requests), '--model-response', f'{args.model}={responses}',
                        '--output-dir', str(native), '--bootstrap-iterations', str(iterations), '--seed', str(seed)]
        if args.per_row_only:
            if args.iterations is not None or args.seed is not None:
                raise ValueError('Per-row scoring does not use bootstrap overrides')
            command += ['--per-row-only']
            iterations, seed = None, None
    else:
        if args.requests or (not args.ci and (args.iterations is not None or args.seed is not None)):
            raise ValueError('This protocol does not accept request bundles or bootstrap overrides')
        if args.ci:
            iterations = 20000 if args.iterations is None else args.iterations
            seed = 20261103 if args.seed is None else args.seed
            if iterations <= 0 or seed < 0:
                raise ValueError('CI requires positive iterations and a nonnegative seed')
        if family == 'miniworld':
            if args.exam_file is None or args.exam_dir:
                raise ValueError('MiniWorld requires --exam-file')
            exam = args.exam_file.resolve(strict=True)
            inputs.append(exam)
            expected = unique(load_jsonl(exam), resolve_uid, 'Exam')
            command += ['--exam_jsonl', str(exam), '--responses', str(responses),
                        '--out', str(native / 'score_report.json'), '--per_row_out', str(native / 'score_per_row.jsonl')]
        else:
            if family == 'ssot':
                from tools.scoring.minigrid import load_gold_from_exam_dir
                if args.exam_file and not args.exam_dir and args.exam_task:
                    original = args.exam_file.resolve(strict=True)
                    filename = f'task_{exam_task.lower()}_exam'+('_v13' if exam_task == 'E' else '')+'.jsonl'
                    unique(load_jsonl(original), resolve_uid, 'Exam')
                    # The native loader requires canonical filenames; alias bytes, not rows.
                    with tempfile.TemporaryDirectory(prefix='gridwm-score-gold-') as temporary:
                        (Path(temporary) / filename).symlink_to(original)
                        expected = load_gold_from_exam_dir(Path(temporary))
                    gold_aliases = {filename: original}
                    paths = [original]
                    exam = output / 'inputs' / 'gold'
                elif args.exam_dir and not args.exam_file and not args.exam_task:
                    exam = args.exam_dir.resolve(strict=True)
                    paths = sorted(exam.glob('task_*_exam*.jsonl'))
                    historical_d = exam / 'task_d_r_exam.jsonl'
                    if historical_d.exists():
                        if (exam / 'task_d_exam.jsonl').exists():
                            raise ValueError('Both Task D filenames exist; select one with --exam-file and --exam-task D')
                        gold_aliases = {p.name: p.resolve(strict=True) for p in paths if p != historical_d}
                        gold_aliases['task_d_exam.jsonl'] = historical_d.resolve(strict=True)
                        with tempfile.TemporaryDirectory(prefix='gridwm-score-gold-') as temporary:
                            for filename, original in gold_aliases.items():
                                (Path(temporary) / filename).symlink_to(original)
                            expected = load_gold_from_exam_dir(Path(temporary))
                        exam = output / 'inputs' / 'gold'
                    else:
                        expected = load_gold_from_exam_dir(exam)
                else:
                    raise ValueError('Use --exam-dir, or --exam-file with explicit --exam-task for a named slice')
                unique([r for p in paths for r in load_jsonl(p)], resolve_uid, 'Exams')
                inputs.extend(paths)
                command += ['--exam_dir', str(exam), '--responses', str(responses),
                            '--out', str(native / 'score_report.json'), '--dump_rows']
            else:
                if args.exam_dir is None or args.exam_file or args.exam_task:
                    raise ValueError('Habitat requires --exam-dir only')
                exam = args.exam_dir.resolve(strict=True)
                key = exam / 'answer_key.json'
                expected = unique(json.loads(key.read_text())['answers'], lambda r: r.get('question_id'), 'Answer key')
                inputs.append(key)
                if not args.model:
                    raise ValueError('Habitat requires --model')
                command += ['--exam-dir', str(exam), '--runs-root', str(output / 'inputs'), '--output-dir', str(native)]
    if args.model and not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', args.model):
        raise ValueError('--model must be a safe model ID')
    if args.model and '..' in args.model:
        raise ValueError('Unsafe model ID')
    if not expected:
        raise ValueError('No gold records found')
    if args.ci:
        relevant = [g for g in expected.values() if g.task in ('A', 'C', 'D')]
        if not relevant or any(not g.env_task or not g.group_id for g in relevant):
            raise ValueError('CI requires A/C/D gold with env_task and group_id')
    def response_id(row):
        if family == 'habitat':
            return (row.get('meta') or {}).get('exam', {}).get('question_id') or str(row.get('uid', '')).rsplit('.', 1)[-1]
        return resolve_uid(row)
    indexed = unique(rows, response_id, 'Responses')
    extra = set(indexed) - set(expected)
    if extra:
        raise ValueError('Unexpected response IDs: ' + str(sorted(extra)[:5]))
    input_hashes = {str(p): sha256(p) for p in inputs}
    if args.per_row_only and family not in ('structured', 'oracle_state'):
        command += ['--per-row-only']
        if family == 'miniworld':
            command[command.index('--per_row_out') + 1] = str(native / 'scored_rows.jsonl')
    sources = sorted((ROOT / 'tools/scoring').glob('*.py')) + sorted((ROOT / 'utils').glob('*.py'))
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    output.mkdir(parents=True)
    native.mkdir()
    if gold_aliases:
        folder = output / 'inputs' / 'gold'
        folder.mkdir(parents=True)
        for filename, original in gold_aliases.items():
            (folder / filename).symlink_to(original)
    if family == 'habitat':
        folder = output / 'inputs' / args.model
        folder.mkdir(parents=True)
        (folder / 'responses.jsonl').symlink_to(responses)
    receipt = dict(status='RUNNING', protocol=contract, model=args.model, api_calls=0,
                   confidence_intervals=args.ci,
                   per_row_only=args.per_row_only,
                   effective_parameters=dict(iterations=iterations, seed=seed), command=command,
                   exam_task=exam_task,
                   exam_filename_aliases={name: str(path) for name, path in gold_aliases.items()},
                   coverage=dict(expected=len(expected), received=len(indexed), missing=len(set(expected)-set(indexed))),
                   input_sha256=input_hashes, source_sha256=source_hashes,
                   started_utc=datetime.now(timezone.utc).isoformat())
    write_receipt(output / 'scoring_receipt.json', receipt)
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=os.environ.get('GRIDWM_IMPORT_ROOT', str(ROOT)))
        with (output / 'scoring.log').open('x') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Native scorer failed; inspect ' + str(output / 'scoring.log'))
        if args.ci:
            from tools.scoring.minigrid_ci import write_report
            scored = json.loads((native / 'score_report.json').read_text())['rows']
            metadata = {resolve_uid(r): r for path in paths for r in load_jsonl(path)}
            write_report(native, scored, expected, iterations, seed, metadata)
        if any(sha256(Path(p)) != value for p, value in input_hashes.items()):
            raise RuntimeError('Scoring input changed during execution')
        if any(sha256(ROOT / p) != value for p, value in source_hashes.items()):
            raise RuntimeError('Scoring source changed during execution')
        receipt.update(status='PASS', output_sha256={str(p.relative_to(native)): sha256(p) for p in native.rglob('*') if p.is_file()})
    except Exception as exc:
        receipt.update(status='FAIL', error=str(exc))
        raise
    finally:
        write_receipt(output / 'scoring_receipt.json', receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=canonical_protocol, choices=sorted(PROTOCOLS))
    parser.add_argument('--responses', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--exam-dir', type=Path)
    parser.add_argument('--exam-file', type=Path)
    parser.add_argument('--exam-task', choices=list('ABCDE') + ['DR', 'D-R'],
                        help='SSOT task for an explicit exam file; DR/D-R are aliases of D; preserves raw IDs and rows')
    parser.add_argument('--requests', type=Path)
    parser.add_argument('--model')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ci', action='store_true',
                        help='Add A/C/D group and environment bootstrap 95%% intervals to SSOT scores')
    parser.add_argument('--per-row-only', action='store_true', help='Question scores only for the selected native protocol; no aggregate tables')
    args = parser.parse_args()
    try:
        receipt = score(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(1, 'ERROR: ' + str(exc) + '\n')
    print(json.dumps(dict(status=receipt['status'], protocol=args.protocol, output=str(args.output)), indent=2))


if __name__ == '__main__':
    main()
