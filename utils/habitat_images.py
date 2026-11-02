"""RGB export and visual-change metrics used by Habitat rendering."""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Any
import numpy as np
from PIL import Image


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb[:, :, :3], dtype=np.uint8), "RGB").save(path)


def image_qc(rgb: np.ndarray) -> dict[str, float]:
    image = np.asarray(rgb[:, :, :3], dtype=np.uint8)
    luminance = (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    )
    return {
        "mean_luminance": float(luminance.mean()),
        "black_pixel_fraction": float((luminance < 8.0).mean()),
        "rgb_std": float(image.std()),
    }


def visual_change(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    changed = np.any(
        np.abs(
            np.asarray(a[:, :, :3], dtype=np.int16)
            - np.asarray(b[:, :, :3], dtype=np.int16)
        )
        >= 24,
        axis=2,
    )
    ys, xs = np.where(changed)
    if not len(xs):
        bbox = None
        bbox_width = bbox_height = 0
    else:
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        bbox_width = bbox[2] - bbox[0] + 1
        bbox_height = bbox[3] - bbox[1] + 1
    return {
        "changed_pixel_fraction": float(changed.mean()),
        "changed_bbox_xyxy": bbox,
        "changed_bbox_width": bbox_width,
        "changed_bbox_height": bbox_height,
    }



@lru_cache(maxsize=8192)
def _visual(path_text: str) -> tuple[np.ndarray, int, float, float]:
    with Image.open(path_text) as image:
        gray = np.asarray(
            image.convert("L").resize((96, 96), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        tiny = np.asarray(
            image.convert("L").resize((9, 8), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    bits = tiny[:, 1:] > tiny[:, :-1]
    dhash = 0
    for bit in bits.reshape(-1):
        dhash = (dhash << 1) | int(bit)
    return (
        gray,
        dhash,
        float(gray.mean()),
        float((gray < 20).mean()),
    )


def _ssim(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64)
    y = right.astype(np.float64)
    mean_x = float(x.mean())
    mean_y = float(y.mean())
    var_x = float(x.var())
    var_y = float(y.var())
    covariance = float(((x - mean_x) * (y - mean_y)).mean())
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x**2 + mean_y**2 + c1) * (
        var_x + var_y + c2
    )
    return float(numerator / denominator)

