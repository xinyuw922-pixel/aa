"""Shared MiniGrid triplet alignment, mask and door-event checks."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX


def check_triplet_alignment(records: List[Dict]) -> None:
    """Require complete variants with aligned actions, frames and metadata."""
    by_variant = {str(r.get("variant", "")): r for r in records}
    required = ("full", "nocue", "cf")
    if set(by_variant.keys()) != set(required):
        raise ValueError(f"triplet_variants_invalid:{sorted(by_variant.keys())}")

    full = by_variant["full"]
    nocue = by_variant["nocue"]
    cf = by_variant["cf"]

    for name in required:
        rec = by_variant[name]
        actions = list(rec.get("actions_id", []))
        frames = list(rec.get("frames", []))
        state_seq = list(rec.get("state_seq", []))
        if len(frames) != len(actions) + 1:
            raise ValueError(f"{name}_frame_action_misaligned:{len(frames)}!={len(actions)}+1")
        if len(state_seq) != len(frames):
            raise ValueError(f"{name}_state_frame_misaligned:{len(state_seq)}!={len(frames)}")

    if list(nocue.get("actions_id", [])) != list(full.get("actions_id", [])):
        raise ValueError("nocue_actions_mismatch")
    if list(cf.get("actions_id", [])) != list(full.get("actions_id", [])):
        raise ValueError("cf_actions_mismatch")

    for key in ("success", "terminated", "truncated"):
        if nocue.get(key) != full.get(key):
            raise ValueError(f"nocue_{key}_mismatch")
    if float(nocue.get("reward", 0.0)) != float(full.get("reward", 0.0)):
        raise ValueError("nocue_reward_mismatch")


def _die(msg: str) -> None:
    raise SystemExit(msg)


def pixel_mask_from_tiles(tile_mask: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    """Transpose semantic X/Y tiles into a pixel row/column mask."""
    image_tile_mask = tile_mask.T
    if frame_h % 7 != 0 or frame_w % 7 != 0:
        _die(f"Frame shape not divisible by 7: {(frame_h, frame_w)}")
    th, tw = frame_h // 7, frame_w // 7
    return np.kron(image_tile_mask.astype(np.uint8), np.ones((th, tw), dtype=np.uint8)).astype(bool)


def cue_tile_mask(state_encoding: List[List[List[int]]], cue_type: str, cue_color: str) -> np.ndarray:
    """Select object/color cells in local semantic-grid coordinates."""
    enc = np.asarray(state_encoding, dtype=np.uint8)  # (7,7,3)
    obj = OBJECT_TO_IDX.get(cue_type, None)
    col = COLOR_TO_IDX.get(cue_color, None)
    if obj is None or col is None:
        return np.zeros((7, 7), dtype=bool)
    return (enc[:, :, 0] == int(obj)) & (enc[:, :, 1] == int(col))


def door_state_at(st: Dict, door_pos: Tuple[int, int]) -> Optional[int]:
    """Read a door state at a world-grid position, if present."""
    for o in st.get("objects", []):
        if o.get("type") == "door" and tuple(o.get("pos", [])) == door_pos:
            return int(o.get("state", -1))
    return None


def door_opens_after_action(states: List[Dict], t: int, door_pos: Tuple[int, int], horizon: int = 4) -> bool:
    """Check the original bounded post-action window for an open door."""
    for j in range(t + 1, min(t + 1 + horizon, len(states))):
        ds = door_state_at(states[j], door_pos)
        if ds == 0:
            return True
    return False

