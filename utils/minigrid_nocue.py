from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


State = Dict[str, Any]


def select_nocue_indices(
    state_seq: Sequence[State],
    first_interaction_step: int,
    max_visible: int,
    is_target_visible: Callable[[State], bool],
    is_d5_excluded: Callable[[State], bool],
) -> Tuple[List[int], List[int], List[int]]:
    leak_indices: List[int] = []
    d5_excluded_indices: List[int] = []
    for t in range(max(0, int(first_interaction_step))):
        st = state_seq[t]
        if not is_target_visible(st):
            continue
        if is_d5_excluded(st):
            d5_excluded_indices.append(int(t))
            continue
        leak_indices.append(int(t))

    masked_indices = leak_indices[: max(0, int(max_visible))]
    if not masked_indices:
        raise ValueError("No cues to mask")
    if any(t >= int(first_interaction_step) for t in masked_indices):
        raise ValueError("NoCue window violation")
    if any(is_d5_excluded(state_seq[t]) for t in masked_indices):
        raise ValueError("NoCue D5 violation")

    return masked_indices, leak_indices, d5_excluded_indices


def compute_nocue_metrics(
    *,
    per_frame_mask_ratios: Sequence[float],
    alignment_ratios: Sequence[float],
    masked_tiles_total: int,
    gh: int,
    gw: int,
    masked_count: int,
    mask_ratio_max: float,
    alignment_min: float,
) -> Dict[str, float | int]:
    max_mask_ratio = float(np.max(per_frame_mask_ratios)) if per_frame_mask_ratios else 0.0
    avg_mask_ratio = float(np.mean(per_frame_mask_ratios)) if per_frame_mask_ratios else 0.0
    alignment_score = float(np.mean(alignment_ratios)) if alignment_ratios else 0.0
    mask_budget_tiles = int(np.floor(float(mask_ratio_max) * float(gh * gw) * float(masked_count)))

    if alignment_score < float(alignment_min):
        raise ValueError("Alignment fail")
    if max_mask_ratio > float(mask_ratio_max):
        raise ValueError("Mask ratio exceeded")
    if int(masked_tiles_total) > int(mask_budget_tiles):
        raise ValueError("Mask budget exceeded")

    return {
        "alignment_score": float(alignment_score),
        "max_mask_ratio": float(max_mask_ratio),
        "avg_mask_ratio": float(avg_mask_ratio),
        "mask_budget_tiles": int(mask_budget_tiles),
        "masked_tiles_total": int(masked_tiles_total),
    }


def build_nocue_meta(
    *,
    targets: Iterable[str],
    first_interaction_step: int,
    masked_indices: Sequence[int],
    leak_indices: Sequence[int],
    d5_excluded_indices: Sequence[int],
    mask_strength_target: float,
    mask_strength_actual: float,
    mask_strength_threshold: float,
    alignment_threshold: float,
    metrics: Dict[str, float | int],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "targets": sorted([str(x) for x in targets]),
        "window_policy": "EARLY",
        "first_interaction_step": int(first_interaction_step),
        "window_steps": [int(x) for x in masked_indices],
        "masked_indices": [int(x) for x in masked_indices],
        "leak_indices": [int(x) for x in leak_indices],
        "d5_excluded_indices": [int(x) for x in d5_excluded_indices],
        "mask_type": "tile_suppression",
        "mask_strength_target": float(mask_strength_target),
        "mask_strength_actual": float(mask_strength_actual),
        "mask_strength_threshold": float(mask_strength_threshold),
        "alignment_score": float(metrics.get("alignment_score", 0.0)),
        "alignment_threshold": float(alignment_threshold),
        "masked_frames": int(len(masked_indices)),
        "physics_check_passed": True,
        "max_mask_ratio": float(metrics.get("max_mask_ratio", 0.0)),
        "avg_mask_ratio": float(metrics.get("avg_mask_ratio", 0.0)),
        "masked_tiles_total": int(metrics.get("masked_tiles_total", 0)),
        "mask_budget_tiles": int(metrics.get("mask_budget_tiles", 0)),
    }
    if extra:
        out.update(extra)
    return out
