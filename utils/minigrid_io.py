"""Shared MiniGrid image and durable JSONL I/O."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from PIL import Image


def save_png(arr: np.ndarray, path: Path) -> None:
    """Save an array image, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def append_jsonl_fsync(path: Path, objs: List[Dict]) -> None:
    """Append UTF-8 JSONL and fsync the file before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_rgb_png(path: Path) -> np.ndarray:
    """Read an image as an RGB array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))



def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")



def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

