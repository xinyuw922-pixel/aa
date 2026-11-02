"""MiniWorld image composition primitives with native font fallback."""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
import math
import os
import numpy as np


def windows_font_paths(names: tuple[str, ...]) -> list[Path]:
    """Locate optional fonts using a configured directory or OS font mounts."""
    directories = []
    if os.environ.get('GRIDWM_FONT_DIR'):
        directory = Path(os.environ['GRIDWM_FONT_DIR']).expanduser().resolve(strict=True)
        if not directory.is_dir():
            raise ValueError('GRIDWM_FONT_DIR must name a font directory')
        directories.append(directory)
    if os.environ.get('WINDIR'):
        directories.append(Path(os.environ['WINDIR']) / 'Fonts')
    directories.extend(sorted(Path('/mnt').glob('*/Windows/Fonts')))
    return [directory / name for directory in dict.fromkeys(directories) for name in names]


FONT_PATHS = [
    *windows_font_paths(('msyh.ttc', 'msyhbd.ttc')),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def fit_contain(img: Image.Image, target_w: int, target_h: int, *, bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    scale = min(target_w / img.width, target_h / img.height, 1.0)
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), bg)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas



def concat_vertical_rgb(
    top: np.ndarray,
    bottom: np.ndarray,
    *,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    top_img = Image.fromarray(np.asarray(top, dtype=np.uint8))
    bottom_img = Image.fromarray(np.asarray(bottom, dtype=np.uint8))
    width = max(top_img.width, bottom_img.width)
    height = top_img.height + bottom_img.height
    canvas = Image.new("RGB", (width, height), bg_color)
    canvas.paste(top_img, ((width - top_img.width) // 2, 0))
    canvas.paste(bottom_img, ((width - bottom_img.width) // 2, top_img.height))
    return np.asarray(canvas, dtype=np.uint8)



def annotate_frame(frame: np.ndarray, caption: str) -> np.ndarray:
    img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    canvas = Image.new("RGB", (img.width, img.height + 18), (255, 255, 255))
    canvas.paste(img, (0, 18))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), (img.width, 18)], fill=(245, 245, 245))
    draw.text((6, 3), caption, fill=(0, 0, 0), font=ImageFont.load_default())
    return np.asarray(canvas, dtype=np.uint8)


def build_montage(frames: list[np.ndarray], cols: int = 4) -> np.ndarray:
    pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB") for frame in frames]
    frame_w = max(img.width for img in pil_frames)
    frame_h = max(img.height for img in pil_frames)
    rows = (len(pil_frames) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * frame_w, rows * frame_h), (255, 255, 255))
    for idx, img in enumerate(pil_frames):
        row = idx // cols
        col = idx % cols
        x = col * frame_w + (frame_w - img.width) // 2
        y = row * frame_h + (frame_h - img.height) // 2
        canvas.paste(img, (x, y))
    return np.asarray(canvas, dtype=np.uint8)


def concat_vertical(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    img_a = Image.fromarray(np.asarray(a, dtype=np.uint8)).convert("RGB")
    img_b = Image.fromarray(np.asarray(b, dtype=np.uint8)).convert("RGB")
    width = max(img_a.width, img_b.width)
    height = img_a.height + img_b.height
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(img_a, ((width - img_a.width) // 2, 0))
    canvas.paste(img_b, ((width - img_b.width) // 2, img_a.height))
    return np.asarray(canvas, dtype=np.uint8)



def annotate_frame_with_step(img: Image.Image, step_idx: int) -> Image.Image:
    canvas = Image.new("RGB", (img.width, img.height + 22), (255, 255, 255))
    canvas.paste(img, (0, 22))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, img.width, 22), fill=(245, 245, 245))
    draw.text((6, 4), f"t={step_idx}", fill=(0, 0, 0), font=load_font(14))
    return canvas


def build_grid(frames: list[Image.Image], cols: int = 4) -> Image.Image:
    frame_w = max(frame.width for frame in frames)
    frame_h = max(frame.height for frame in frames)
    rows = math.ceil(len(frames) / cols)
    canvas = Image.new("RGB", (cols * frame_w, rows * frame_h), (255, 255, 255))
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x = col * frame_w + (frame_w - frame.width) // 2
        y = row * frame_h + (frame_h - frame.height) // 2
        canvas.paste(frame, (x, y))
    return canvas

