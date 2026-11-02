"""HTTP transport, bounded retry policy and shared request-rate limiting."""
from __future__ import annotations

import json
import threading
import time
from urllib.parse import quote

import httpx

from utils.inference_payload import extract_output


class APIClient:
    """Share one thread-safe HTTP pool; never retry based on answer correctness."""

    def __init__(self, base_url: str, protocol: str, model: str, key: str,
                 timeout: float, retries: int, backoff: float, rps: float,
                 trust_env: bool):
        self.protocol, self.key = protocol, key
        base = base_url.rstrip('/')
        if protocol == 'gemini':
            if not base.endswith('/v1beta'):
                base += '/v1beta'
            self.url = base + '/models/' + quote(model.removeprefix('models/'), safe='') + ':generateContent'
            headers = {'x-goog-api-key': key}
        else:
            self.url = base + ('/chat/completions' if protocol == 'chat' else '/responses')
            headers = {'Authorization': 'Bearer ' + key}
        self.http = httpx.Client(headers=headers, timeout=timeout, trust_env=trust_env,
                                 follow_redirects=False)
        self.retries, self.backoff, self.rps = retries, backoff, rps
        self.lock = threading.Lock()
        self.next_request = 0.0

    def _wait(self) -> None:
        if self.rps:
            with self.lock:
                now = time.monotonic()
                time.sleep(max(0.0, self.next_request - now))
                self.next_request = time.monotonic() + 1.0 / self.rps

    def generate(self, payload: dict) -> dict:
        """Keep provider JSON and attempt history separate from answer text."""
        attempts = []
        started = time.monotonic()
        for index in range(self.retries + 1):
            self._wait()
            data, status, finish, usage, text, retry_after = None, None, None, None, '', None
            try:
                response = self.http.post(self.url, json=payload)
                status = response.status_code
                try:
                    data = response.json()
                except ValueError:
                    data = {'body': response.text}
                # Providers occasionally echo credentials in errors: never persist them.
                data = json.loads(json.dumps(data).replace(self.key, '[REDACTED]'))
                retryable = status in (408, 429, 500, 502, 503, 504)
                if status == 200 and isinstance(data, dict) and not data.get('error'):
                    text, finish, usage = extract_output(self.protocol, data)
                    ok = isinstance(text, str) and bool(text.strip())
                    error = None if ok else 'empty_model_output'
                    if not isinstance(text, str):
                        text = ''
                    retryable = False
                else:
                    ok, error = False, 'http_' + str(status) if status != 200 else 'provider_error'
                header = response.headers.get('retry-after', '')
                retry_after = float(header) if header.replace('.', '', 1).isdigit() else None
            except (httpx.TransportError, ValueError, TypeError, AttributeError, KeyError) as exc:
                ok, error = False, type(exc).__name__
                retryable = isinstance(exc, httpx.TransportError)
            attempts.append({'attempt': index + 1, 'http_status': status, 'error': error,
                             'provider_response': data, 'finish_reason': finish})
            if ok or not retryable or index == self.retries:
                return {'text': text, 'ok': ok, 'error': error, 'finish_reason': finish,
                        'usage': usage, 'provider_response': data, 'attempts': attempts,
                        'latency_ms': (time.monotonic() - started) * 1000}
            time.sleep(min(60.0, retry_after if retry_after is not None else self.backoff * 2 ** index))
        raise AssertionError('Unreachable retry loop')

    def close(self) -> None:
        self.http.close()
