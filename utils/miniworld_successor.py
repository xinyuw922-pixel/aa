"""Shared MiniWorld successor-choice primitives."""
from __future__ import annotations
import math
from pathlib import Path
from PIL import Image


MACRO_ACTIONS = {
    "turn_left_30": ["turn_left"] * 2,
    "turn_right_30": ["turn_right"] * 2,
    "turn_left_60": ["turn_left"] * 4,
    "turn_right_60": ["turn_right"] * 4,
    "forward_5": ["move_forward"] * 5,
    "forward_10": ["move_forward"] * 10,
}


MIN_SAME_DIR_POS_DELTA = 0.12


def expand_macro_sequence(sequence: list[str]) -> list[str]:
    primitive: list[str] = []
    for macro in sequence:
        primitive.extend(MACRO_ACTIONS[macro])
    return primitive


def get_agent_state(env) -> dict:
    target = getattr(env, "unwrapped", env)
    try:
        agent = getattr(target, "agent", None)
        if agent is not None:
            pos = getattr(agent, "pos", None)
            if pos is not None and hasattr(pos, "__iter__"):
                px = float(pos[0])
                pz = float(pos[2])
            else:
                px = pz = 0.0
            dir_angle = float(getattr(agent, "dir", 0.0))
            angle = dir_angle % (2 * math.pi)
            sector = round(angle / (math.pi / 2)) % 4
            cardinal = ["S", "W", "N", "E"][sector]
        else:
            px = pz = 0.0
            cardinal = "?"
    except Exception:
        px = pz = 0.0
        cardinal = "?"
    return {"pos": [round(px, 3), round(pz, 3)], "cardinal": cardinal}


def get_motion_contract(env) -> dict:
    target = getattr(env, "unwrapped", env)
    params = getattr(target, "params", None)
    if params is None:
        return {"forward_step": 0.15, "turn_step_deg": 15.0}
    return {
        "forward_step": float(params.sample(None, "forward_step")),
        "turn_step_deg": float(params.sample(None, "turn_step")),
    }


EN_ACTION_LABELS = {
    "turn_left_30": "turn left by one unit (30°)",
    "turn_right_30": "turn right by one unit (30°)",
    "turn_left_60": "turn left by two units (60°)",
    "turn_right_60": "turn right by two units (60°)",
    "forward_5": "move forward by one unit",
    "forward_10": "move forward by two units",
}


def en_seq(sequence: list[str]) -> str:
    return " -> ".join(EN_ACTION_LABELS.get(a, a) for a in sequence)


def _infer_strategy(seq: list[str], dist_seq: list[str]) -> str:
    if len(seq) != len(dist_seq):
        return "different_length"
    diffs = [(i, seq[i], dist_seq[i]) for i in range(len(seq)) if seq[i] != dist_seq[i]]
    if not diffs:
        return "identical_sequences"
    return (
        "alternative_prefix"
        if diffs[0][0] == 0
        else f"alternative_suffix_pos{diffs[0][0]}"
    )


def _make_distractor_source(seq: list[str], dist_seq: list[str]) -> str:
    return f"{_infer_strategy(seq, dist_seq)}_rollout"


def _state_distance(a: dict, b: dict) -> float:
    return math.dist(a["pos"], b["pos"])


def _is_too_subtle(a: dict, b: dict) -> bool:
    same_cardinal = a["cardinal"] == b["cardinal"]
    return same_cardinal and _state_distance(a, b) < MIN_SAME_DIR_POS_DELTA


def _ego_panels_identical(path_a: Path, path_b: Path) -> bool:
    with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
        a = im_a.convert("RGB")
        b = im_b.convert("RGB")
        ego_h = min(a.height, b.height) // 2
        return list(a.crop((0, 0, a.width, ego_h)).getdata()) == list(
            b.crop((0, 0, b.width, ego_h)).getdata()
        )

