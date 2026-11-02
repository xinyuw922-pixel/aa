"""Leaf conversions for multimodal API payloads and unparsed provider outputs."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


def encode_image(path: Path, max_side: int = 0) -> str:
    """Encode RGB PNG without resizing unless an explicit maximum is supplied."""
    with Image.open(path) as original:
        image = original.convert('RGB')
        if max_side > 0:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format='PNG', optimize=True)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


def build_payload(protocol: str, model: str, prompt: str, images: list[str],
                  max_tokens: int, temperature: float | None, seed: int | None,
                  order: str, detail: str | None, extra: dict[str, Any]) -> dict:
    """Construct only model-facing text/images, never exam metadata or labels."""
    if protocol == 'gemini':
        parts = [{'inline_data': {'mime_type': 'image/png', 'data': image}} for image in images]
        text = {'text': prompt}
    else:
        parts = []
        for image in images:
            url = 'data:image/png;base64,' + image
            if protocol == 'chat':
                item = {'url': url}
                if detail:
                    item['detail'] = detail
                parts.append({'type': 'image_url', 'image_url': item})
            else:
                item = {'type': 'input_image', 'image_url': url}
                if detail:
                    item['detail'] = detail
                parts.append(item)
        text = {'type': 'text' if protocol == 'chat' else 'input_text', 'text': prompt}
    parts = [text, *parts] if order == 'text_first' else [*parts, text]
    if protocol == 'gemini':
        config = {'maxOutputTokens': max_tokens}
        if temperature is not None:
            config['temperature'] = temperature
        if seed is not None:
            config['seed'] = seed
        config.update(extra.get('generationConfig', {}))
        return {**extra, 'contents': [{'parts': parts}], 'generationConfig': config}
    payload = {'model': model, 'stream': False}
    if protocol == 'chat':
        payload['messages'] = [{'role': 'user', 'content': parts}]
        short_name = model.lower().rsplit('/', 1)[-1]
        reasoning = short_name.startswith(('gpt-5', 'chatgpt-5', 'o1', 'o3', 'o4', 'o5'))
        payload['max_completion_tokens' if reasoning else 'max_tokens'] = max_tokens
    else:
        payload['input'] = [{'role': 'user', 'content': parts}]
        payload['max_output_tokens'] = max_tokens
    if temperature is not None:
        payload['temperature'] = temperature
    if seed is not None:
        payload['seed'] = seed
    payload.update(extra)
    return payload


def extract_output(protocol: str, data: dict) -> tuple[str, str | None, dict | None]:
    """Extract answer text without grading, stripping fences or recovering verdicts."""
    if protocol == 'chat':
        choices = data.get('choices') or []
        first = choices[0] if choices else {}
        text = (first.get('message') or {}).get('content') or ''
        if isinstance(text, list):
            text = ''.join(p.get('text', '') for p in text if isinstance(p, dict) and p.get('type') == 'text')
        return text, first.get('finish_reason'), data.get('usage')
    if protocol == 'responses':
        text = data.get('output_text') or ''.join(
            part.get('text', '') for item in data.get('output', [])
            for part in item.get('content', []) if part.get('type') == 'output_text')
        reason = (data.get('incomplete_details') or {}).get('reason') or data.get('status')
        return text, reason, data.get('usage')
    candidates = data.get('candidates') or []
    first = candidates[0] if candidates else {}
    # Thought parts are metadata, not the final model answer.
    text = ''.join(part.get('text', '') for part in (first.get('content') or {}).get('parts', [])
                   if not part.get('thought'))
    reason = first.get('finishReason') or (data.get('promptFeedback') or {}).get('blockReason')
    return text, reason, data.get('usageMetadata')
