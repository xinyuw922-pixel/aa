"""Run preassembled multimodal requests and save native-scorer-compatible responses."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

from tqdm import tqdm

from tools.inference.client import APIClient
from utils.exam_schema import RESPONSE_SCHEMA_VERSION, build_exam_meta, resolve_uid
from utils.file_hash import sha256
from utils.inference_payload import build_payload, encode_image
from utils.json_io import load_jsonl

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict) -> None:
    """Replace a small runtime record atomically."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare(args) -> tuple[list[dict], dict]:
    """Validate the whole bundle before any API call or response-file creation."""
    url = urlsplit(args.base_url)
    if (url.scheme not in ('https', 'http') or not url.hostname or url.username or url.password
            or url.query or url.fragment):
        raise ValueError('Use an explicit API base URL without credentials, query or fragment')
    if url.scheme == 'http' and url.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Plain HTTP is permitted only for a local test server')
    if min(args.workers, args.max_tokens, args.num_shards) <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError('Workers, tokens and shard count must be positive; shard ID must be in range')
    if (args.retries < 0 or args.max_side < 0 or not math.isfinite(args.timeout) or args.timeout <= 0
            or not math.isfinite(args.backoff) or args.backoff < 0
            or not math.isfinite(args.rps) or args.rps < 0
            or (args.temperature is not None and not math.isfinite(args.temperature))):
        raise ValueError('Invalid timeout, retry, rate, resize or temperature value')
    args.output = args.output.resolve()
    if args.output == ROOT or ROOT in args.output.parents or args.output in ROOT.parents:
        raise ValueError('Runtime output must be outside the source repository')
    if args.resume and args.dry_run:
        raise ValueError('--resume cannot be combined with --dry-run')
    args.requests = args.requests.resolve(strict=True)
    extra = json.loads(args.extra_body.read_text()) if args.extra_body else {}
    if not isinstance(extra, dict):
        raise ValueError('Extra body must be a JSON object')
    forbidden = {'model', 'messages', 'input', 'contents', 'stream', 'stream_options',
                 'max_tokens', 'max_completion_tokens', 'max_output_tokens', 'temperature', 'seed'}
    if forbidden & extra.keys():
        raise ValueError('Extra body cannot override model, evidence, streaming or explicit decoding parameters')
    if args.protocol == 'gemini':
        config = extra.get('generationConfig', {})
        if not isinstance(config, dict) or {'temperature', 'maxOutputTokens', 'seed'} & config.keys():
            raise ValueError('generationConfig cannot override explicit decoding parameters')
    if any(word in str(k).lower() for k in extra for word in ('key', 'authorization', 'token_secret')):
        raise ValueError('Credentials must not be placed in extra body')
    rows, seen, selected, image_hashes = load_jsonl(args.requests), set(), [], {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Every request must be an object')
        uid = resolve_uid(row)
        if not uid or uid in seen:
            raise ValueError('Missing or duplicate request UID: ' + str(uid))
        seen.add(uid)
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError('Missing prompt for ' + uid)
        if not isinstance(row.get('images'), list) or any(not isinstance(p, str) for p in row['images']):
            raise ValueError('Images must be an ordered list of local paths, including [] for text-only: ' + uid)
        if 'exam' in row and not isinstance(row['exam'], dict):
            raise ValueError('Exam metadata must be an object: ' + uid)
        order = row.get('api_content_order', args.content_order)
        if order not in ('images_first', 'text_first'):
            raise ValueError('Unsupported content order: ' + str(order))
        detail = row.get('api_image_detail', args.image_detail)
        if detail not in (None, 'auto', 'low', 'high'):
            raise ValueError('Invalid image detail')
        images = []
        for name in row['images']:
            path = Path(name)
            path = (args.requests.parent / path if not path.is_absolute() else path).resolve(strict=True)
            if str(path) not in image_hashes:
                image_hashes[str(path)] = sha256(path)
                # Decode before dispatch so a corrupt later image cannot cause a partial paid run.
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
            images.append(str(path))
        shard = int.from_bytes(hashlib.blake2b(uid.encode(), digest_size=8).digest(), 'little') % args.num_shards
        if shard == args.shard_id:
            selected.append({'row': row, 'uid': uid, 'images': images, 'order': order, 'detail': detail})
    if not selected:
        raise ValueError('No requests selected for this shard')
    config = dict(protocol=args.protocol, base_url=args.base_url.rstrip('/'), model=args.model,
                  max_tokens=args.max_tokens, temperature=args.temperature, seed=args.seed,
                  content_order=args.content_order, image_detail=args.image_detail, max_side=args.max_side,
                  timeout=args.timeout, retries=args.retries, backoff=args.backoff, workers=args.workers,
                  rps=args.rps, trust_env=args.trust_env, num_shards=args.num_shards, shard_id=args.shard_id,
                  extra_body=extra, requests=str(args.requests), requests_sha256=sha256(args.requests),
                  image_sha256=image_hashes, selected=len(selected))
    return selected, config


def infer_one(item: dict, args, client: APIClient, config: dict) -> dict:
    """Send only prompt/images and preserve unparsed text plus scoring metadata."""
    row, uid = item['row'], item['uid']
    for name in item['images']:
        if sha256(Path(name)) != config['image_sha256'][name]:
            raise ValueError('Input image changed before request dispatch: ' + uid)
    payload = build_payload(args.protocol, args.model, row['prompt'],
                            [encode_image(Path(p), args.max_side) for p in item['images']],
                            args.max_tokens, args.temperature, args.seed, item['order'], item['detail'],
                            config['extra_body'])
    result = client.generate(payload)
    exam = dict(row['exam']) if 'exam' in row else build_exam_meta(uid, row=row)
    exam.setdefault('uid', uid)
    exam.setdefault('exam_id', row.get('exam_id', uid))
    return dict(uid=uid, exam_id=row.get('exam_id', uid), schema_version=RESPONSE_SCHEMA_VERSION,
                exam_schema_version=row.get('schema_version', 'gridwm.exam.v1'),
                pred=result['text'] if result['ok'] else '',
                raw=result['text'] if result['text'] else '__GW_ERR__: ' + str(result['error']),
                provider_response=result['provider_response'],
                meta=dict(ok=result['ok'], error=result['error'], model=args.model, provider='custom',
                          backend='gemini' if args.protocol == 'gemini' else 'openai_compatible',
                          protocol=args.protocol, shard=args.shard_id, fused=False, exam=exam,
                          finish_reason=result['finish_reason'], usage=result['usage'],
                          latency_ms=result['latency_ms'], attempts=result['attempts']))


def run(args) -> dict:
    selected, config = prepare(args)
    if args.dry_run:
        return dict(status='VALIDATED', requests=len(selected), protocol=args.protocol, api_calls=0)
    if args.env_file:
        from dotenv import dotenv_values
        key = dotenv_values(args.env_file).get(args.api_key_env) or os.environ.get(args.api_key_env)
    else:
        key = os.environ.get(args.api_key_env)
    if not key or not key.strip():
        raise ValueError('Missing API key in the named environment variable')
    key = key.strip()
    if key in json.dumps(config):
        raise ValueError('API key must not appear in persisted configuration')
    if args.output.exists() and not args.resume:
        raise ValueError('Output exists; choose a fresh output directory or --resume')
    if args.resume and not args.output.is_dir():
        raise ValueError('Resume directory does not exist')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / '.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process owns this output; use separate shard directories') from None
        record = args.output / 'run_config.json'
        output = args.output / 'responses.jsonl'
        done = {}
        if args.resume:
            if json.loads(record.read_text()) != config:
                raise ValueError('Resume configuration or input hashes differ from the original run')
            for row in load_jsonl(output) if output.exists() else []:
                uid = resolve_uid(row)
                if not uid or uid in done:
                    raise ValueError('Resume responses contain missing/duplicate IDs; file not modified')
                done[uid] = row
            if set(done) - {r['uid'] for r in selected}:
                raise ValueError('Resume responses contain unexpected IDs')
            if output.exists() and output.stat().st_size:
                with output.open('rb') as handle:
                    handle.seek(-1, 2)
                    if handle.read(1) != b'\n':
                        raise ValueError('Incomplete response-file tail; preserve and inspect it before resuming')
        else:
            write_json(record, config)
        pending = [item for item in selected if item['uid'] not in done]
        summary = dict(status='RUNNING', expected=len(selected), completed=len(done),
                       api_errors=sum(not row['meta']['ok'] for row in done.values()),
                       resumed=len(done), model=args.model, protocol=args.protocol)
        client = APIClient(args.base_url, args.protocol, args.model, key, args.timeout,
                           args.retries, args.backoff, args.rps, args.trust_env)
        write_json(args.output / 'run_summary.json', summary)
        try:
            with output.open('a', encoding='utf-8') as handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
                remaining = iter(pending)
                futures = set()
                for item in remaining:
                    futures.add(executor.submit(infer_one, item, args, client, config))
                    if len(futures) >= args.workers:
                        break
                with tqdm(total=len(selected), initial=len(done), unit='request', desc=args.model) as progress:
                    while futures:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for future in completed:
                            row = future.result()
                            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                            handle.flush()
                            os.fsync(handle.fileno())
                            summary['completed'] += 1
                            summary['api_errors'] += not row['meta']['ok']
                            write_json(args.output / 'run_summary.json', summary)
                            progress.update()
                            progress.set_postfix(errors=summary['api_errors'])
                        for _ in completed:
                            item = next(remaining, None)
                            if item is not None:
                                futures.add(executor.submit(infer_one, item, args, client, config))
            if sha256(args.requests) != config['requests_sha256']:
                raise ValueError('Request bundle changed during execution')
            summary['status'] = 'COMPLETE_WITH_ERRORS' if summary['api_errors'] else 'PASS'
        except BaseException as exc:
            summary.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL',
                           error=type(exc).__name__)
            raise
        finally:
            client.close()
            write_json(args.output / 'run_summary.json', summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', choices=('chat', 'responses', 'gemini'), required=True)
    parser.add_argument('--base-url', required=True, help='API base including /v1 for compatible endpoints')
    parser.add_argument('--model', required=True, help='Exact provider model ID; no automatic alias substitution')
    parser.add_argument('--api-key-env', default='VLM_API_KEY')
    parser.add_argument('--env-file', type=Path, help='Optional external dotenv file; only the named key is read')
    parser.add_argument('--max-tokens', type=int, required=True)
    parser.add_argument('--temperature', type=float, help='Omitted unless explicitly set; use the experiment value')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--rps', type=float, default=0, help='Global request rate, including retries; 0 means unlimited')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--backoff', type=float, default=2)
    parser.add_argument('--max-side', type=int, default=0, help='0 preserves image dimensions')
    parser.add_argument('--content-order', choices=('images_first', 'text_first'), default='images_first')
    parser.add_argument('--image-detail', choices=('auto', 'low', 'high'))
    parser.add_argument('--extra-body', type=Path, help='External JSON with provider-specific non-evidence options')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--trust-env', action='store_true', help='Opt in to environment HTTP proxy settings')
    parser.add_argument('--resume', action='store_true', help='Skip all recorded IDs, including API errors')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs without credentials, output or API calls')
    args = parser.parse_args()
    try:
        summary = run(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, 'ERROR: ' + str(exc) + '\n')
    print(json.dumps(summary, indent=2))
    if summary['status'] == 'COMPLETE_WITH_ERRORS':
        parser.exit(2)


if __name__ == '__main__':
    main()
