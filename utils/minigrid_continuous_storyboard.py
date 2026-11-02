"""Continuous annotated storyboard rendering for MiniGrid Task D."""
import math
from typing import Dict, List, Literal, Tuple
from PIL import Image, ImageDraw

DEFAULT_LAYOUT_MODE: Literal["fixed-5", "adaptive"] = "adaptive"

DEFAULT_MIN_COLS = 4

DEFAULT_MAX_COLS = 8

TASK_D_PROMPT = (
    "Below is the full time-ordered trajectory storyboard, arranged from left to right "
    "and top to bottom. Judge whether the agent ultimately completed the task. "
    "Answer with ONLY: Success or Fail."
)

OVERLAY_PAD = 6

OVERLAY_BOX_PADDING = 2

def annotate_frame(img: Image.Image, step_idx: int) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    label = f"t={step_idx}"

    try:
        from PIL import ImageFont

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                max(10, min(w, h) // 12),
            )
        except OSError:
            font = ImageFont.load_default()
    except ImportError:
        font = None

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    bx0 = OVERLAY_PAD
    by0 = OVERLAY_PAD
    bx1 = bx0 + text_w + 2 * OVERLAY_BOX_PADDING
    by1 = by0 + text_h + 2 * OVERLAY_BOX_PADDING
    draw.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0, 160))
    draw.text(
        (bx0 + OVERLAY_BOX_PADDING, by0 + OVERLAY_BOX_PADDING),
        label,
        fill=(255, 255, 255, 255),
        font=font,
    )
    return img

def choose_cols(
    n_frames: int,
    mode: str = DEFAULT_LAYOUT_MODE,
    min_cols: int = DEFAULT_MIN_COLS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> int:
    if mode == "fixed-5":
        return 5
    raw = round(math.sqrt(n_frames))
    return max(min_cols, min(max_cols, raw))

def make_montage(
    frames: List[Image.Image],
    layout_mode: str = DEFAULT_LAYOUT_MODE,
    min_cols: int = DEFAULT_MIN_COLS,
    max_cols: int = DEFAULT_MAX_COLS,
) -> Tuple[Image.Image, Dict[str, int | str]]:
    if not frames:
        raise ValueError("make_montage received empty frames")

    w, h = frames[0].size
    n_frames = len(frames)
    cols = choose_cols(n_frames, layout_mode, min_cols, max_cols)
    rows = math.ceil(n_frames / cols)
    montage = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))

    for i, frame in enumerate(frames):
        col = i % cols
        row = i // cols
        montage.paste(frame, (col * w, row * h))

    return montage, {
        "cols": cols,
        "rows": rows,
        "n_frames": n_frames,
        "layout_mode": layout_mode,
        "min_cols": min_cols,
        "max_cols": max_cols,
    }
