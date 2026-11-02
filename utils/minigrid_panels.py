"""Leaf image composition shared by MiniGrid tasks; no project imports."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from collections import defaultdict
import os
from PIL import Image, ImageDraw, ImageFont


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
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    *windows_font_paths(('arial.ttf',)),
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_PATHS:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def open_rgb(path: Path, max_size: int | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_size and max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def add_top_bar(img: Image.Image, label: str, fill: tuple[int, int, int]) -> Image.Image:
    bar_h = 34
    out = Image.new("RGB", (img.width, img.height + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, img.width, bar_h], fill=fill)
    font = load_font(18)
    tw, th = text_size(draw, label, font)
    draw.text(((img.width - tw) / 2, (bar_h - th) / 2 - 1), label, fill=(255, 255, 255), font=font)
    return out


def selected_indices(n: int, max_frames: int = 6) -> list[int]:
    if n <= max_frames:
        return list(range(n))
    return sorted({round(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)})


def compose_storyboard(raw_root: Path, row: dict[str, Any], out_path: Path) -> None:
    frame_paths = [raw_root / rel for rel in row.get("frames", [])]
    frames = [open_rgb(frame_paths[idx], max_size=180) for idx in selected_indices(len(frame_paths))]
    labels = [f"t={idx}" for idx in selected_indices(len(frame_paths))]
    compose_storyboard_frames(frames, labels, out_path)


def compose_storyboard_frames(frames: list[Image.Image], labels: list[str], out_path: Path) -> None:
    """Compose already selected/transformed frames without modifying their order."""
    if not frames or len(frames) != len(labels):
        raise ValueError('Storyboard requires nonempty aligned frames and labels')
    gap = 8
    label_h = 24
    width = sum(img.width for img in frames) + gap * (len(frames) - 1)
    height = max(img.height for img in frames) + label_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = load_font(14)
    x = 0
    for img, label in zip(frames, labels):
        canvas.paste(img, (x, label_h))
        tw, _ = text_size(draw, label, font)
        draw.text((x + (img.width - tw) / 2, 3), label, fill=(40, 40, 40), font=font)
        x += img.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG")


def compose_task_a_image(
    raw_root: Path,
    current: str,
    candidates: list[str],
    labels: list[str],
    out_path: Path,
) -> None:
    tile = 150
    gap = 12
    header_h = 36
    current_img = add_top_bar(open_rgb(raw_root / current, max_size=tile), "Current", (49, 91, 143))
    cand_imgs = [
        add_top_bar(open_rgb(raw_root / rel, max_size=tile), label, (86, 86, 86))
        for rel, label in zip(candidates, labels)
    ]
    width = current_img.width + gap + sum(img.width for img in cand_imgs) + gap * (len(cand_imgs) - 1)
    height = max([current_img.height, *[img.height for img in cand_imgs]]) + header_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = load_font(16)
    draw.text((0, 6), "Choose the observation after taking the specified action.", fill=(20, 20, 20), font=font)
    x = 0
    y = header_h
    canvas.paste(current_img, (x, y))
    x += current_img.width + gap
    for img in cand_imgs:
        canvas.paste(img, (x, y))
        x += img.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG")


def compose_reference_query(reference: Path, query: Path, out_path: Path) -> None:
    ref = add_top_bar(open_rgb(reference, max_size=980), "Reference (known success)", (49, 91, 143))
    qry = add_top_bar(open_rgb(query, max_size=980), "Query", (86, 86, 86))
    gap = 18
    width = max(ref.width, qry.width)
    height = ref.height + gap + qry.height
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(ref, ((width - ref.width) // 2, 0))
    canvas.paste(qry, ((width - qry.width) // 2, ref.height + gap))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG")


def split_task_a_panels(source: Path, boxes: list[tuple[int, int, int, int]],
                        output_dir: Path) -> list[Path]:
    """Export five exact pixel crops; do not resize, reorder or relabel evidence."""
    if len(boxes) != 5:
        raise ValueError('Task A requires current state and four ordered candidates')
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = []
    with Image.open(source) as original:
        image = original.convert('RGB')
        for name, (left, top, right, bottom) in zip(('current', 'A', 'B', 'C', 'D'), boxes):
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError('Task A panel lies outside its composite image')
            path = output_dir / f'{name}.png'
            image.crop((left, top, right, bottom)).save(path, 'PNG')
            paths.append(path.resolve())
    return paths


def by_group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("group_id"))].append(row)
    return dict(grouped)
