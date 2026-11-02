from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def save_rgb_png(path: Path, observation: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(observation)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 RGB observation, got {rgb.shape}")
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def scene_slug(scene_path: str) -> str:
    path = Path(scene_path)
    name = path.name
    if name.endswith(".scene_instance.json"):
        name = name[: -len(".scene_instance.json")]
    else:
        name = path.stem
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return safe or "scene"
