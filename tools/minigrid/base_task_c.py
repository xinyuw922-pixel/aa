"""Event-anchored MiniGrid Task C temporal, visual and framing conditions."""
from pathlib import Path
from typing import Any, Dict, List
import cv2
import numpy as np
from PIL import Image
from utils.exam_schema import EXAM_SCHEMA_VERSION
from utils.minigrid_base_prompts import TASK_C_GOALS, TASK_C_PROMPT_TEMPLATE
from utils.minigrid_event_keyframes import compute_layout, select_keyframes_for_group

FRAMING_PREAMBLES = {
    'pos': 'The agent seems to handle the task competently in the following trajectory. ',
    'neu': 'The following trajectory is described in a neutral way. ',
    'neg': 'The agent seems to struggle with the task in the following trajectory. ',
}


def add_framing_rows(rows: list[dict]) -> list[dict]:
    """Add the three framing conditions only to original clean evidence."""
    added = []
    for row in rows:
        if row['temporal'] != 'orig' or row['visual'] != 'clean':
            continue
        for framing, preamble in FRAMING_PREAMBLES.items():
            uid = row['uid'] + '.' + framing
            dimensions = {**row['probe_dimensions'], 'framing': framing}
            added.append({**row, 'uid': uid, 'exam_id': uid, 'framing': framing,
                          'prompt': preamble + row['prompt'], 'probe_dimensions': dimensions,
                          'prompt_version': 'probe-rich-v1-plus-framing-v2-mid-2026-04-03',
                          'prompt_source': 'framing_overlay'})
    return rows + added


def build_final_task_c(records: list[dict], output: Path) -> list[dict]:
    """Assemble all nine conditions from shared event-anchored simulator frames."""
    grouped = {}
    for record in records:
        variants = grouped.setdefault((record['task'], record['group_id']), {})
        if record['variant'] in variants:
            raise ValueError('Duplicate Task C source variant')
        variants[record['variant']] = record
    rows = []
    for (env, gid), variants in sorted(grouped.items()):
        if set(variants) != {'full', 'nocue', 'cf'}:
            raise ValueError('Task C requires complete Full/NoCue/CF groups')
        full = variants['full']
        steps = select_keyframes_for_group(env, full['state_seq'], variants)
        if not steps:
            raise ValueError('Task C cannot use an empty storyboard')
        layout = compute_layout(len(steps))
        for variant, record in sorted(variants.items()):
            if max(steps) >= len(record['frames']):
                raise ValueError(f'Unaligned Task C frames: {env}/{gid}/{variant}')
            frames = []
            for step in steps:
                with Image.open(record['frames'][step]) as image:
                    frames.append(image.convert('RGB'))
            for temporal in ('orig', 'rev'):
                for visual in ('clean', 'noisy', 'style'):
                    transform = {'clean': lambda image: image, 'noisy': apply_noisy,
                                 'style': apply_style_cartoon_medium}[visual]
                    row = build_row(env, gid, variant, temporal, visual,
                                    bool(record['success']), steps, len(full['frames']),
                                    TASK_C_GOALS[env], layout)
                    render_montage([transform(frame) for frame in frames], layout,
                                   output / row['image'], reverse=temporal == 'rev')
                    row.update(source_exam_root=str(output), source_raw_root=str(output / 'raw'),
                               source_group_id=gid, source_variant=variant,
                               mission=record.get('mission'))
                    rows.append(row)
    return add_framing_rows(rows)

def apply_noisy(img: Image.Image, sigma: float = 20.0 / 255.0) -> Image.Image:
    """Add Gaussian noise to image."""
    arr = np.array(img, dtype=np.float64)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_style_cartoon_medium(img: Image.Image) -> Image.Image:
    """Cartoon edges (medium): edge-preserving smoothing that simplifies flat regions
    while retaining crisp object boundaries and color discrimination.

    Uses cv2.edgePreservingFilter (RECURS_FILTER, sigma_s=60, sigma_r=0.4) to produce
    a cartoon-like appearance: walls/objects become solid-color regions, but red/blue
    color distinction is preserved. This is semantically richer than grayscale, which
    destroys color cues entirely.

    IMPORTANT: The transformation is deterministic (no randomness) because edgePreservingFilter
    is a deterministic filter. However, frame order (orig/rev) and the actual trajectory
    content (full/nocue/cf) are the independent probe dimensions.
    """
    arr = np.array(img)
    out = cv2.edgePreservingFilter(arr, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
    return Image.fromarray(out)


def render_montage(
    frames: List[Image.Image],
    layout: Dict[str, Any],
    out_path: Path,
    reverse: bool = False,
) -> None:
    """Render a sparse storyboard montage. If reverse=True, flip frame order."""
    target_frames = list(reversed(frames)) if reverse else frames
    if not target_frames:
        raise ValueError(f"No frames for montage: {out_path}")

    width, height = target_frames[0].size
    rows = int(layout.get("rows", 1))
    cols = int(layout.get("cols", len(target_frames)))
    canvas = Image.new("RGB", (width * cols, height * rows), (0, 0, 0))

    for idx, frame in enumerate(target_frames):
        x = (idx % cols) * width
        y = (idx // cols) * height
        canvas.paste(frame, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_image_rel_path(
    short_env: str, group_id: str,
    variant: str, temporal: str, visual: str
) -> str:
    return f"images/taskC/{short_env}/{group_id}/{variant}_{temporal}_{visual}.png"


def build_row(
    short_env: str,
    group_id: str,
    variant: str,
    temporal: str,   # "orig" or "rev"
    visual: str,     # "clean", "noisy", or "style"
    success: bool,
    selected_steps: List[int],
    n_frames_available: int,
    task_goal: str,
    layout: Dict[str, Any],
) -> dict:
    """Build a single probe-rich exam row."""
    uid = f"C.{short_env}.{group_id}.{variant}.{temporal}.{visual}"
    answer = "Success" if success else "Fail"
    label = 1 if success else 0
    img_rel = make_image_rel_path(short_env, group_id, variant, temporal, visual)

    # For rev: add ordering note to prompt
    if temporal == "rev":
        prompt_suffix = (
            "\nNote: The frames are shown in reverse chronological order "
            "(final state first, initial state last). "
            "Use this ordering to judge whether the trajectory succeeded."
        )
        full_prompt = TASK_C_PROMPT_TEMPLATE.format(goal=task_goal) + prompt_suffix
    else:
        full_prompt = TASK_C_PROMPT_TEMPLATE.format(goal=task_goal)

    return {
        "exam_id": uid,
        "uid": uid,
        "schema_version": EXAM_SCHEMA_VERSION,
        "task": "C",
        "env_task": short_env,
        "env_id": short_env,
        "group_id": group_id,
        "variant": variant,
        "temporal": temporal,
        "visual": visual,
        "image": img_rel,
        "prompt": full_prompt,
        "answer": answer,
        "label": label,
        "n_frames_available": n_frames_available,
        "n_frames_selected": len(selected_steps),
        "selected_steps": selected_steps,
        "task_goal": f"Task goal: {task_goal}",
        "success_criterion_text": task_goal,
        "prompt_version": "probe-rich-v1-2026-03-28",
        "layout": layout,
        "n_frames": len(selected_steps),
        "evidence_scope": _evidence_scope(short_env),
        "keyframe_selector": "event-anchored-variable-k-v1",
        "probe_dimensions": {
            "variant": variant,       # full/nocue/cf
            "temporal": temporal,   # orig/rev
            "visual": visual,       # clean/noisy/style
        },
    }


def _evidence_scope(short_env: str) -> str:
    if short_env in ("memory", "keycorridor", "redblue"):
        return "cross_frame"
    return "final_frame"
