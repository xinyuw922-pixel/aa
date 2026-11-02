"""Deterministic MiniGrid duplicate-detection primitives."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from PIL import Image

def stable_hash(obj: Any) -> str:
    data = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def image_ahash(path: Path, size: int = 8) -> int | None:
    if not path.exists():
        return None
    with Image.open(path) as img:
        gray = img.convert("L").resize((size, size))
        values = list(gray.getdata())
    avg = sum(values) / len(values)
    bits = 0
    for value in values:
        bits = (bits << 1) | int(value >= avg)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def levenshtein(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def selected_frame_indices(frames: list[str]) -> list[int]:
    if not frames:
        return []
    idxs = {0, len(frames) // 2, len(frames) - 1}
    return sorted(idxs)



def normalized_action_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    denom = max(len(left["actions"]), len(right["actions"]), 1)
    return levenshtein(left["actions"], right["actions"]) / denom


def mean_image_hamming(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    pairs = list(zip(left["frame_hashes"], right["frame_hashes"]))
    if not pairs:
        return None
    return sum(hamming(a, b) for a, b in pairs) / len(pairs)

