"""memory: simulator trajectory generation and environment-specific validation."""
from __future__ import annotations

# Trajectory generation
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from utils.minigrid_checks import check_triplet_alignment as enforce_triplet_hard_gates
from utils.minigrid_io import load_rgb_png as _load_png
from utils.minigrid_state import render_pov_with_fallback as render_agent_pov


import argparse
import json
import os
import shutil
import time
import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image

from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT, COLOR_TO_IDX
from minigrid.core.world_object import Ball, Box, Key
try:
    from utils.minigrid_nocue import build_nocue_meta, compute_nocue_metrics, select_nocue_indices
except ModuleNotFoundError:
    import sys

    _tasks_root = Path(__file__).resolve().parents[1]
    if str(_tasks_root) not in sys.path:
        sys.path.append(str(_tasks_root))
    from utils.minigrid_nocue import build_nocue_meta, compute_nocue_metrics, select_nocue_indices


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (_repo_root() / p).resolve()

# ---------------------------
# MiniGrid action names
# ---------------------------
ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

DIR2VEC = {
    0: (1, 0),  # right
    1: (0, 1),  # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}

# ---------------------------
# Contract fields
# ---------------------------
MODEL_INPUT_FIELDS = ["frames", "mission"]

# ---------------------------
# NoCue defaults
# ---------------------------
DEFAULT_MASK_STRENGTH_TARGET = 0.02
DEFAULT_MASK_RATIO_MAX = 0.35
DEFAULT_ALIGNMENT_MIN = 0.70
DEFAULT_MIN_VISIBLE_FRAMES = 1
DEFAULT_MAX_VISIBLE_FRAMES = 10

MASK_RGB = (0, 0, 0)
MEMORY_CUE_TYPES = ("key", "ball", "box")

# ---------------------------
# IO utils
# ---------------------------
def append_jsonl_batch(path: Path, objs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_skip_log(path: Path, seed: int, reason: str, detail: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] seed={seed} reason={reason}"
    if detail:
        line += f" detail={detail}"
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())



def _mkdir_clean(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _atomic_replace_dir(tmp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.replace(final_dir)


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)



# ---------------------------
# Rendering & state extraction
# ---------------------------


def extract_state(env_unwrapped) -> Dict[str, Any]:
    """
    Oracle state. Semantic encoding must be env_unwrapped.gen_obs()['image'].
    """
    enc = np.asarray(env_unwrapped.gen_obs()["image"], dtype=np.uint8)  # (7,7,3)

    visible_types_gt: Set[str] = set()
    for oid in np.unique(enc[:, :, 0]).tolist():
        name = IDX_TO_OBJECT.get(int(oid), "unknown")
        if name in ("unseen", "empty", "floor", "wall"):
            continue
        visible_types_gt.add(name)

    # front cell
    fx, fy = env_unwrapped.front_pos
    f_cell = env_unwrapped.grid.get(fx, fy)
    f_type = getattr(f_cell, "type", "empty") if f_cell else "empty"
    if f_type == "floor":
        f_type = "empty"
    front_info: Dict[str, Any] = {"pos": [int(fx), int(fy)], "type": str(f_type)}
    if f_cell and hasattr(f_cell, "color"):
        front_info["color"] = str(getattr(f_cell, "color"))

    # oracle objects
    objects: List[Dict[str, Any]] = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            cell = env_unwrapped.grid.get(x, y)
            if cell is None:
                continue
            t = getattr(cell, "type", None)
            if t in ("wall", "floor"):
                continue
            o: Dict[str, Any] = {"type": str(t), "pos": [int(x), int(y)]}
            if hasattr(cell, "color"):
                o["color"] = str(getattr(cell, "color"))
            objects.append(o)

    return {
        "agent": {
            "pos": [int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1])],
            "dir": int(env_unwrapped.agent_dir),
            "carrying": {
                "type": str(env_unwrapped.carrying.type),
                "color": str(env_unwrapped.carrying.color),
            } if env_unwrapped.carrying else None,
        },
        "front_cell": front_info,
        "visible_types_gt": sorted(list(visible_types_gt)),
        "visible_types": sorted(list(visible_types_gt)),
        "state_encoding": enc.astype(int).tolist(),
        "objects": objects,
        "mask_applied": False,
        "mask_type": None,
    }

# ---------------------------
# Planning (BFS)
# ---------------------------
@dataclass(frozen=True)
class MemState:
    x: int
    y: int
    d: int

def _is_passable(env_unwrapped, x: int, y: int) -> bool:
    if not (0 <= x < env_unwrapped.width and 0 <= y < env_unwrapped.height):
        return False
    cell = env_unwrapped.grid.get(x, y)
    if cell is None:
        return True
    t = getattr(cell, "type", None)
    if t in ("wall", "lava"):
        return False
    if getattr(cell, "can_overlap", False):
        return True
    if t in ("key", "ball", "box", "goal"):
        return True
    return False

def bfs_to_pos(
    env_unwrapped,
    goal_pos: Tuple[int, int],
    max_nodes: int = 300000,
    timeout_s: float = 0.0,
) -> Optional[List[int]]:
    """
    Shortest path to stand on goal_pos (any dir) using {left,right,forward}.
    """
    start = MemState(int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1]), int(env_unwrapped.agent_dir))
    q = deque([start])
    parent: Dict[MemState, Tuple[Optional[MemState], Optional[int]]] = {start: (None, None)}
    visited = {start}

    goal_set = {(goal_pos[0], goal_pos[1], d) for d in range(4)}
    steps = 0

    t0 = time.perf_counter()
    while q:
        if timeout_s > 0 and (time.perf_counter() - t0) > timeout_s:
            return None
        cur = q.popleft()
        steps += 1
        if steps > max_nodes:
            return None

        if (cur.x, cur.y, cur.d) in goal_set:
            acts: List[int] = []
            node = cur
            while parent[node][0] is not None:
                prev, act = parent[node]
                assert act is not None
                acts.append(int(act))
                node = prev  # type: ignore[assignment]
            acts.reverse()
            return acts

        # left/right
        for act in (0, 1):
            nd = (cur.d - 1) % 4 if act == 0 else (cur.d + 1) % 4
            nxt = MemState(cur.x, cur.y, nd)
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = (cur, act)
                q.append(nxt)

        # forward
        dx, dy = DIR2VEC[cur.d]
        nx, ny = cur.x + dx, cur.y + dy
        if _is_passable(env_unwrapped, nx, ny):
            nxt = MemState(nx, ny, cur.d)
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = (cur, 2)
                q.append(nxt)

    return None


def astar_to_pos(
    env_unwrapped,
    goal_pos: Tuple[int, int],
    max_nodes: int = 120000,
    timeout_s: float = 0.0,
) -> Optional[List[int]]:
    start = MemState(int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1]), int(env_unwrapped.agent_dir))
    goal_xy = (int(goal_pos[0]), int(goal_pos[1]))

    def heuristic(x: int, y: int) -> int:
        return abs(x - goal_xy[0]) + abs(y - goal_xy[1])

    frontier: List[Tuple[int, int, int, MemState]] = []
    tie = 0
    heapq.heappush(frontier, (heuristic(start.x, start.y), 0, tie, start))
    parent: Dict[MemState, Tuple[Optional[MemState], Optional[int]]] = {start: (None, None)}
    gscore: Dict[MemState, int] = {start: 0}
    expanded = 0
    t0 = time.perf_counter()

    while frontier:
        if timeout_s > 0 and (time.perf_counter() - t0) > timeout_s:
            return None
        _f, g, _tie, cur = heapq.heappop(frontier)
        if g != gscore.get(cur):
            continue
        expanded += 1
        if expanded > max_nodes:
            return None

        if (cur.x, cur.y) == goal_xy:
            acts: List[int] = []
            node = cur
            while parent[node][0] is not None:
                prev, act = parent[node]
                assert act is not None
                acts.append(int(act))
                node = prev  # type: ignore[assignment]
            acts.reverse()
            return acts

        for act in (0, 1):
            nd = (cur.d - 1) % 4 if act == 0 else (cur.d + 1) % 4
            nxt = MemState(cur.x, cur.y, nd)
            ng = g + 1
            if ng < gscore.get(nxt, 10**9):
                gscore[nxt] = ng
                parent[nxt] = (cur, act)
                tie += 1
                heapq.heappush(frontier, (ng + heuristic(nxt.x, nxt.y), ng, tie, nxt))

        dx, dy = DIR2VEC[cur.d]
        nx, ny = cur.x + dx, cur.y + dy
        if _is_passable(env_unwrapped, nx, ny):
            nxt = MemState(nx, ny, cur.d)
            ng = g + 1
            if ng < gscore.get(nxt, 10**9):
                gscore[nxt] = ng
                parent[nxt] = (cur, 2)
                tie += 1
                heapq.heappush(frontier, (ng + heuristic(nxt.x, nxt.y), ng, tie, nxt))

    return None

# ---------------------------
# Memory object inference
# ---------------------------
def infer_objects(env_unwrapped) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            cell = env_unwrapped.grid.get(x, y)
            if cell is None:
                continue
            t = getattr(cell, "type", None)
            if t not in MEMORY_CUE_TYPES:
                continue
            objs.append({"type": str(t), "color": str(getattr(cell, "color", "grey")), "pos": (int(x), int(y))})
    return objs

def pick_cue_and_ends(objs: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if len(objs) < 3:
        raise ValueError(f"Memory expects >=3 objects, got {len(objs)}")

    xs = [o["pos"][0] for o in objs]
    min_x = min(xs)
    max_x = max(xs)

    cue_candidates = [o for o in objs if o["pos"][0] == min_x]
    cue = sorted(cue_candidates, key=lambda o: (o["pos"][1], o["type"], o["color"]))[0]

    end_candidates = [o for o in objs if o["pos"][0] == max_x]
    if len(end_candidates) < 2:
        end_candidates = sorted(objs, key=lambda o: (o["pos"][0], o["pos"][1]))[-2:]

    end_sorted = sorted(end_candidates, key=lambda o: (o["pos"][1], o["type"], o["color"]))
    return cue, end_sorted[0], end_sorted[1]

def match_good_bad(cue: Dict[str, Any], end_a: Dict[str, Any], end_b: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sig = (cue["type"], cue["color"])
    a_sig = (end_a["type"], end_a["color"])
    b_sig = (end_b["type"], end_b["color"])

    if a_sig == sig and b_sig != sig:
        return end_a, end_b
    if b_sig == sig and a_sig != sig:
        return end_b, end_a

    # relax: type-only
    if end_a["type"] == cue["type"] and end_b["type"] != cue["type"]:
        return end_a, end_b
    if end_b["type"] == cue["type"] and end_a["type"] != cue["type"]:
        return end_b, end_a

    raise ValueError(f"Ambiguous cue/end match: cue={cue}, end_a={end_a}, end_b={end_b}")


MEMORY_OBJ_CLASS = {
    "ball": Ball,
    "box": Box,
    "key": Key,
}


def infer_corridor_start_x(env_unwrapped) -> int:
    counts: List[Tuple[int, int]] = []
    for x in range(env_unwrapped.width):
        passable = 0
        for y in range(env_unwrapped.height):
            if _is_passable(env_unwrapped, x, y):
                passable += 1
        counts.append((x, passable))

    min_pass = min(c for x, c in counts if x > 0)
    cands = [x for x, c in counts if x > 0 and c == min_pass]
    return min(cands) if cands else 0

# ---------------------------
# NoCue: tile mask by (type,color)
# ---------------------------
def cue_tile_mask(enc: np.ndarray, cue_type: str, cue_color: str) -> np.ndarray:
    obj = OBJECT_TO_IDX.get(cue_type, None)
    col = COLOR_TO_IDX.get(cue_color, None)
    if obj is None or col is None:
        return np.zeros(enc.shape[:2], dtype=bool)
    return (enc[:, :, 0] == int(obj)) & (enc[:, :, 1] == int(col))

def encoding_mask_to_image_tile_mask(tile_mask: np.ndarray) -> np.ndarray:
    """
    Convert a state_encoding tile mask into image tile coordinates.

    MiniGrid's `state_encoding` for this Memory pipeline is indexed in the
    environment's local `(vx, vy)` convention, while the rendered image grid is
    addressed in standard image `(row, col) == (y, x)` order. For these Memory
    artifacts, that means we must transpose the 7x7 tile mask before applying it
    to image tiles.
    """
    if tile_mask.ndim != 2:
        raise ValueError(f"tile_mask must be 2D, got shape={tile_mask.shape}")
    return tile_mask.T

def apply_tile_mask(img: np.ndarray, tile_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    out = img.copy()
    image_tile_mask = encoding_mask_to_image_tile_mask(tile_mask)
    gh, gw = image_tile_mask.shape
    H, W = out.shape[:2]
    if H % gh != 0 or W % gw != 0:
        raise ValueError(f"Frame not divisible by tile grid: frame={out.shape}, tile={image_tile_mask.shape}")
    th, tw = H // gh, W // gw

    ys, xs = np.where(image_tile_mask)
    for r, c in zip(ys.tolist(), xs.tolist()):
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = MASK_RGB
    return out, int(image_tile_mask.sum())

def compute_window_steps(
    full_state_seq: Sequence[Dict[str, Any]],
    cue_type: str,
    cue_color: str,
    corridor_start_x: int,
    max_visible_frames: int,
) -> Tuple[List[int], int, List[int], List[int]]:
    """
    EARLY window: frames where cue is visible while agent is still before corridor.
    """
    visible: List[int] = []
    d5_excluded: List[int] = []
    for t, st in enumerate(full_state_seq):
        ax = int(st["agent"]["pos"][0])
        if ax >= corridor_start_x:
            continue
        fc = st.get("front_cell", {})
        fc_type = str(fc.get("type", ""))
        fc_color = fc.get("color", None)
        if fc_type == cue_type and str(fc_color) == str(cue_color):
            d5_excluded.append(t)
            continue
        enc = np.asarray(st["state_encoding"], dtype=np.uint8)
        if cue_tile_mask(enc, cue_type, cue_color).any():
            visible.append(t)

    if len(visible) == 0 or len(visible) > max_visible_frames:
        return [], -1, [], d5_excluded

    window_end = max(visible)
    leak: List[int] = []
    for t, st in enumerate(full_state_seq):
        if t <= window_end:
            continue
        ax = int(st["agent"]["pos"][0])
        # Leak gate is only defined inside the EARLY (pre-corridor) phase.
        # Re-appearance after entering corridor is not part of the masking window.
        if ax >= corridor_start_x:
            continue
        fc = st.get("front_cell", {})
        if str(fc.get("type", "")) == cue_type and str(fc.get("color", None)) == str(cue_color):
            if t not in d5_excluded:
                d5_excluded.append(int(t))
            continue
        enc = np.asarray(st["state_encoding"], dtype=np.uint8)
        if cue_tile_mask(enc, cue_type, cue_color).any():
            leak.append(t)

    return sorted(visible), window_end, leak, d5_excluded

# ---------------------------
# CF: swap end objects (Type-2)
# ---------------------------
def apply_cf_swap_success_failure(env_unwrapped) -> Dict[str, Any]:
    """Swap endpoint objects and recompute success/failure positions.
    
    Rendered endpoints and simulator termination semantics must change together."""
    old_success_floor = tuple(int(x) for x in env_unwrapped.success_pos)
    old_failure_floor = tuple(int(x) for x in env_unwrapped.failure_pos)

    objs = infer_objects(env_unwrapped)
    cue, end_a, end_b = pick_cue_and_ends(objs)
    old_good_end, old_bad_end = match_good_bad(cue, end_a, end_b)

    pos_a = tuple(end_a["pos"])
    pos_b = tuple(end_b["pos"])
    cell_a = env_unwrapped.grid.get(*pos_a)
    cell_b = env_unwrapped.grid.get(*pos_b)
    if cell_a is None or cell_b is None:
        raise ValueError("memory_cf_missing_end_object")

    cls_a = MEMORY_OBJ_CLASS.get(getattr(cell_a, "type", None))
    cls_b = MEMORY_OBJ_CLASS.get(getattr(cell_b, "type", None))
    if cls_a is None or cls_b is None:
        raise ValueError(f"memory_cf_unsupported_object_types:{getattr(cell_a,'type',None)},{getattr(cell_b,'type',None)}")

    new_a = cls_b(getattr(cell_b, "color", "green"))
    new_b = cls_a(getattr(cell_a, "color", "green"))
    env_unwrapped.grid.set(pos_a[0], pos_a[1], new_a)
    env_unwrapped.grid.set(pos_b[0], pos_b[1], new_b)

    objs_cf = infer_objects(env_unwrapped)
    cue_cf, end_a_cf, end_b_cf = pick_cue_and_ends(objs_cf)
    good_end_cf, bad_end_cf = match_good_bad(cue_cf, end_a_cf, end_b_cf)

    env_unwrapped.success_pos = (int(good_end_cf["pos"][0]), int(good_end_cf["pos"][1]) - 1 if int(good_end_cf["pos"][1]) > int(cue_cf["pos"][1]) else int(good_end_cf["pos"][1]) + 1)
    env_unwrapped.failure_pos = (int(bad_end_cf["pos"][0]), int(bad_end_cf["pos"][1]) - 1 if int(bad_end_cf["pos"][1]) > int(cue_cf["pos"][1]) else int(bad_end_cf["pos"][1]) + 1)

    return {
        "type": "swap_success_failure_visual",
        "old_success_floor_pos": [int(old_success_floor[0]), int(old_success_floor[1])],
        "old_failure_floor_pos": [int(old_failure_floor[0]), int(old_failure_floor[1])],
        "old_good_end": [int(old_good_end["pos"][0]), int(old_good_end["pos"][1])],
        "old_bad_end": [int(old_bad_end["pos"][0]), int(old_bad_end["pos"][1])],
        "swap_pos_a": [int(pos_a[0]), int(pos_a[1])],
        "swap_pos_b": [int(pos_b[0]), int(pos_b[1])],
        "cue": {"type": cue_cf["type"], "color": cue_cf["color"], "pos": [int(cue_cf["pos"][0]), int(cue_cf["pos"][1])]},
        "new_success_floor_pos": [int(env_unwrapped.success_pos[0]), int(env_unwrapped.success_pos[1])],
        "new_failure_floor_pos": [int(env_unwrapped.failure_pos[0]), int(env_unwrapped.failure_pos[1])],
        "new_good_end": [int(good_end_cf["pos"][0]), int(good_end_cf["pos"][1])],
        "new_bad_end": [int(bad_end_cf["pos"][0]), int(bad_end_cf["pos"][1])],
    }


def rollout_state_only(env, actions: Sequence[int]) -> Dict[str, Any]:
    state_seq: List[Dict[str, Any]] = [extract_state(env.unwrapped)]
    total_reward = 0.0
    terminated = False
    truncated = False
    executed: List[int] = []
    for a in actions:
        _obs, r, terminated, truncated, _info = env.step(int(a))
        executed.append(int(a))
        total_reward += float(r)
        state_seq.append(extract_state(env.unwrapped))
        if terminated or truncated:
            break
    return {
        "state_seq": state_seq,
        "actions_executed": executed,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward": float(total_reward),
        "success": bool(terminated and total_reward > 0.0 and not truncated),
    }


def prescreen_seed_memory(
    env_id: str,
    seed: int,
    max_plan_steps: int,
    min_visible_frames: int,
    max_visible_frames: int,
    plan_max_nodes: int,
    plan_timeout_s: float,
) -> Tuple[bool, str]:
    env = None
    try:
        env = gym.make(env_id, render_mode="rgb_array")
        env.reset(seed=seed)
        objs = infer_objects(env.unwrapped)
        cue, end_a, end_b = pick_cue_and_ends(objs)
        good_end, _bad_end = match_good_bad(cue, end_a, end_b)

        plan = astar_to_pos(
            env.unwrapped,
            tuple(good_end["pos"]),
            max_nodes=int(plan_max_nodes),
            timeout_s=float(plan_timeout_s),
        )
        if plan is None:
            plan = bfs_to_pos(
                env.unwrapped,
                tuple(good_end["pos"]),
                max_nodes=max(int(plan_max_nodes), 300000),
                timeout_s=max(float(plan_timeout_s), 2.0),
            )
        if plan is None:
            return False, "prescreen_plan_failed"
        if len(plan) > int(max_plan_steps):
            return False, "prescreen_plan_too_long"

        dry = rollout_state_only(env, plan)
        if not dry["success"] or dry["truncated"]:
            return False, "prescreen_full_invalid"
        corridor_x = infer_corridor_start_x(env.unwrapped)
        window_steps, window_end, leak_steps, _d5 = compute_window_steps(
            dry["state_seq"], cue["type"], cue["color"], corridor_x, max_visible_frames
        )
        if len(window_steps) < int(min_visible_frames):
            return False, "prescreen_visible_too_few"
        if window_end < 0:
            return False, "prescreen_window_invalid"
        if leak_steps:
            return False, "prescreen_leak_detected"
        return True, "ok"
    except Exception as exc:
        return False, f"prescreen_exception:{type(exc).__name__}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

# ---------------------------
# Rollout
# ---------------------------
def rollout_and_save(env, actions: Sequence[int], tile_size: int, work_dir: Path, variant: str) -> Dict[str, Any]:
    vdir = work_dir / variant
    vdir.mkdir(parents=True, exist_ok=True)

    frames: List[str] = []
    state_seq: List[Dict[str, Any]] = []
    actions_executed: List[int] = []

    total_reward = 0.0
    terminated = False
    truncated = False

    def snap(t: int) -> None:
        frame = render_agent_pov(env, tile_size=tile_size)
        if frame is None:
            raise RuntimeError("render_agent_pov failed")
        fname = f"step_{t:03d}.png"
        _save_png(frame, vdir / fname)
        frames.append(f"{variant}/{fname}")
        state_seq.append(extract_state(env.unwrapped))

    snap(0)
    for i, a in enumerate(actions):
        _obs, r, terminated, truncated, _info = env.step(int(a))
        actions_executed.append(int(a))
        total_reward += float(r)
        snap(i + 1)
        if terminated or truncated:
            break

    success = bool(terminated and total_reward > 0.0 and not truncated)

    return {
        "frames": frames,
        "state_seq": state_seq,
        "actions_executed": actions_executed,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward": float(total_reward),
        "success": bool(success),
    }

def generate_one(
    env_id: str,
    seed: int,
    group_index: int,
    out_dir: Path,
    tile_size: int,
    max_plan_steps: int,
    max_episode_steps: int,
    # NoCue knobs
    mask_strength_target: float,
    mask_ratio_max: float,
    alignment_min: float,
    min_visible_frames: int,
    max_visible_frames: int,
    plan_max_nodes: int,
    plan_timeout_s: float,
    plan_fallback_max_nodes: int,
    plan_fallback_timeout_s: float,
    skip_cf_nocue: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    gid = f"memory_s{group_index:06d}"
    tmp_root = out_dir / "_tmp" / gid
    final_root = out_dir / gid

    _mkdir_clean(tmp_root)

    env = None
    env_cf = None
    try:
        # -------- FULL --------
        env = gym.make(env_id, render_mode="rgb_array")
        if hasattr(env.unwrapped, "tile_size"):
            env.unwrapped.tile_size = int(tile_size)

        env.reset(seed=seed)

        objs = infer_objects(env.unwrapped)
        cue, end_a, end_b = pick_cue_and_ends(objs)
        good_end, _bad_end = match_good_bad(cue, end_a, end_b)

        plan = astar_to_pos(
            env.unwrapped,
            tuple(good_end["pos"]),
            max_nodes=int(plan_max_nodes),
            timeout_s=float(plan_timeout_s),
        )
        if plan is None:
            plan = bfs_to_pos(
                env.unwrapped,
                tuple(good_end["pos"]),
                max_nodes=int(plan_fallback_max_nodes),
                timeout_s=float(plan_fallback_timeout_s),
            )
        if plan is None or len(plan) > max_plan_steps:
            return None

        full = rollout_and_save(env, plan, tile_size, tmp_root, "full")

        if not full["success"]:
            return None
        if full["truncated"]:
            return None
        if len(full["actions_executed"]) == 0 or len(full["actions_executed"]) > max_episode_steps:
            return None

        T = len(full["actions_executed"])
        if len(full["frames"]) != T + 1 or len(full["state_seq"]) != T + 1:
            return None

        # -------- Early return for testing --------
        if skip_cf_nocue:
            # -------- finalize --------
            mission = getattr(env.unwrapped, "mission", "")
            _atomic_replace_dir(tmp_root, final_root)

            def mk_record(variant: str, traj: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
                frames_rel = [f"{gid}/{variant}/{Path(p).name}" for p in traj["frames"]]
                acts = traj["actions_executed"]
                rec: Dict[str, Any] = {
                    "task": "memory",
                    "env_id": env_id,
                    "seed": int(seed),
                    "group_id": gid,
                    "variant": variant,
                    "mission": str(mission),
                    "model_input_fields": list(MODEL_INPUT_FIELDS),
                    "actions_id": [int(a) for a in acts],
                    "actions_text": [ACTION_NAMES.get(int(a), str(int(a))) for a in acts],
                    "frames": frames_rel,
                    "state_seq": traj["state_seq"],
                    "terminated": bool(traj["terminated"]),
                    "truncated": bool(traj["truncated"]),
                    "reward": float(traj["reward"]),
                    "success": bool(traj["success"]),
                }
                rec.update(extra)
                rec.update({
                    "generator": {"name": 'gen_memory_triplets.py', "created_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                })
                return rec

            rec_full = mk_record("full", full, {})
            return [rec_full]

        # -------- NoCue --------
        corridor_x = infer_corridor_start_x(env.unwrapped)
        first_interaction_step = next(
            (i for i, st in enumerate(full["state_seq"]) if int(st["agent"]["pos"][0]) >= int(corridor_x)),
            len(full["state_seq"]),
        )
        try:
            window_steps, leak_steps, d5_excluded_steps = select_nocue_indices(
                state_seq=full["state_seq"],
                first_interaction_step=int(first_interaction_step),
                max_visible=int(max_visible_frames),
                is_target_visible=lambda st: (
                    int(st["agent"]["pos"][0]) < int(corridor_x)
                    and cue_tile_mask(np.asarray(st["state_encoding"], dtype=np.uint8), cue["type"], cue["color"]).any()
                ),
                is_d5_excluded=lambda st: (
                    int(st["agent"]["pos"][0]) < int(corridor_x)
                    and str(st.get("front_cell", {}).get("type", "")) == str(cue["type"])
                    and str(st.get("front_cell", {}).get("color", None)) == str(cue["color"])
                ),
            )
        except ValueError:
            return None

        if len(window_steps) < min_visible_frames:
            return None

        gh, gw = 7, 7
        masked_frames = len(window_steps)

        nocue_state_seq: List[Dict[str, Any]] = json.loads(json.dumps(full["state_seq"]))

        masked_tiles_total = 0
        target_tiles_total = 0
        per_frame_mask_ratios: List[float] = []

        window_set = set(window_steps)

        for t in range(T + 1):
            src = tmp_root / full["frames"][t]
            dst = tmp_root / "nocue" / Path(full["frames"][t]).name
            dst.parent.mkdir(parents=True, exist_ok=True)

            if t not in window_set:
                shutil.copy2(src, dst)
                nocue_state_seq[t]["visible_types"] = list(nocue_state_seq[t].get("visible_types_gt", []))
                nocue_state_seq[t]["mask_applied"] = False
                nocue_state_seq[t]["mask_type"] = None
                continue

            img = _load_png(src)
            enc = np.asarray(nocue_state_seq[t]["state_encoding"], dtype=np.uint8)
            tm = cue_tile_mask(enc, cue["type"], cue["color"])
            target_tiles = int(tm.sum())
            if target_tiles <= 0:
                return None

            masked_img, masked_tiles = apply_tile_mask(img, tm)
            _save_png(masked_img, dst)

            masked_tiles_total += masked_tiles
            target_tiles_total += target_tiles
            per_frame_mask_ratios.append(masked_tiles / float(gh * gw))

            nocue_state_seq[t]["visible_types"] = []
            nocue_state_seq[t]["mask_applied"] = True
            nocue_state_seq[t]["mask_type"] = "tile_suppression"

        alignment_ratios = [float(masked_tiles_total) / float(max(1, target_tiles_total))]
        try:
            metrics = compute_nocue_metrics(
                per_frame_mask_ratios=per_frame_mask_ratios,
                alignment_ratios=alignment_ratios,
                masked_tiles_total=int(masked_tiles_total),
                gh=int(gh),
                gw=int(gw),
                masked_count=len(window_steps),
                mask_ratio_max=float(mask_ratio_max),
                alignment_min=float(alignment_min),
            )
        except ValueError:
            return None

        mask_strength_actual = float(masked_tiles_total / float(gh * gw * max(1, masked_frames)))

        nocue = {
            "frames": [f"nocue/{Path(p).name}" for p in full["frames"]],
            "state_seq": nocue_state_seq,
            "actions_executed": list(full["actions_executed"]),
            "terminated": bool(full["terminated"]),
            "truncated": bool(full["truncated"]),
            "reward": float(full["reward"]),
            "success": bool(full["success"]),
        }

        nocue_meta = build_nocue_meta(
            targets=[str(cue["type"])],
            first_interaction_step=int(first_interaction_step),
            masked_indices=window_steps,
            leak_indices=leak_steps,
            d5_excluded_indices=d5_excluded_steps,
            mask_strength_target=float(mask_strength_target),
            mask_strength_actual=float(mask_strength_actual),
            mask_strength_threshold=float(mask_ratio_max),
            alignment_threshold=float(alignment_min),
            metrics=metrics,
            extra={
                "corridor_start_x": int(corridor_x),
                "window_end": int(max(window_steps)) if window_steps else -1,
                "mask_ratio_max": float(mask_ratio_max),
                "cue": cue,
            },
        )

        # -------- CF --------
        env_cf = gym.make(env_id, render_mode="rgb_array")
        if hasattr(env_cf.unwrapped, "tile_size"):
            env_cf.unwrapped.tile_size = int(tile_size)
        env_cf.reset(seed=seed)

        objs_cf = infer_objects(env_cf.unwrapped)
        _cue_cf, end_a_cf, end_b_cf = pick_cue_and_ends(objs_cf)
        _good_end_cf, _bad_end_cf = match_good_bad(_cue_cf, end_a_cf, end_b_cf)

        cf_meta = apply_cf_swap_success_failure(env_cf.unwrapped)
        cf = rollout_and_save(env_cf, full["actions_executed"], tile_size, tmp_root, "cf")
        if cf["success"]:
            return None
        if cf["truncated"]:
            return None
        if float(cf["reward"]) != 0.0:
            return None

        if len(cf["state_seq"]) != len(full["state_seq"]):
            return None
        for t in range(T + 1):
            if cf["state_seq"][t]["agent"]["pos"] != full["state_seq"][t]["agent"]["pos"]:
                return None
            if cf["state_seq"][t]["agent"]["dir"] != full["state_seq"][t]["agent"]["dir"]:
                return None

        mission = getattr(env.unwrapped, "mission", "")

        def mk_record(variant: str, traj: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
            frames_rel = [f"{gid}/{variant}/{Path(p).name}" for p in traj["frames"]]
            acts = traj["actions_executed"]
            rec: Dict[str, Any] = {
                "task": "memory",
                "env_id": env_id,
                "seed": int(seed),
                "group_id": gid,
                "variant": variant,
                "mission": str(mission),
                "model_input_fields": list(MODEL_INPUT_FIELDS),
                "actions_id": [int(a) for a in acts],
                "actions_text": [ACTION_NAMES.get(int(a), str(int(a))) for a in acts],
                "frames": frames_rel,
                "state_seq": traj["state_seq"],
                "terminated": bool(traj["terminated"]),
                "truncated": bool(traj["truncated"]),
                "reward": float(traj["reward"]),
                "success": bool(traj["success"]),
                "generator": {"name": 'gen_memory_triplets.py', "created_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            }
            rec.update(extra)
            return rec

        rec_full = mk_record("full", full, {})
        rec_nocue = mk_record("nocue", nocue, {"nocue_meta": nocue_meta})
        rec_cf = mk_record("cf", cf, {"cf_meta": cf_meta})
        enforce_triplet_hard_gates([rec_full, rec_nocue, rec_cf])

        # -------- finalize --------
        _atomic_replace_dir(tmp_root, final_root)

        return [rec_full, rec_nocue, rec_cf]

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            if env_cf is not None:
                env_cf.close()
        except Exception:
            pass
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

def generate() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="MiniGrid-MemoryS13-v0")
    ap.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default=str(_repo_root() / "datasets" / "raw_data" / "memory"),
    )
    ap.add_argument("--num", "--n", dest="num", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--tile-size", type=int, default=32)
    ap.add_argument("--max-plan-steps", type=int, default=512)
    ap.add_argument("--max-episode-steps", type=int, default=2048)
    ap.add_argument("--attempts-per-group", type=int, default=200)
    ap.add_argument("--max-attempts-total", type=int, default=0, help="Global cap for total seeds tried; <=0 disables cap.")
    ap.add_argument("--plan-max-nodes", type=int, default=120000)
    ap.add_argument("--plan-timeout-s", type=float, default=1.0)
    ap.add_argument("--plan-fallback-max-nodes", type=int, default=300000)
    ap.add_argument("--plan-fallback-timeout-s", type=float, default=2.5)
    ap.add_argument("--prescreen-plan-max-nodes", type=int, default=80000)
    ap.add_argument("--prescreen-plan-timeout-s", type=float, default=0.6)
    ap.add_argument("--disable-prescreen", action="store_true")
    ap.add_argument("--log-every", type=int, default=20)

    # NoCue knobs
    ap.add_argument("--mask-strength-target", type=float, default=DEFAULT_MASK_STRENGTH_TARGET)
    ap.add_argument("--mask-ratio-max", type=float, default=DEFAULT_MASK_RATIO_MAX)
    ap.add_argument("--alignment-min", type=float, default=DEFAULT_ALIGNMENT_MIN)
    ap.add_argument("--min-visible-frames", type=int, default=DEFAULT_MIN_VISIBLE_FRAMES)
    ap.add_argument("--max-visible-frames", type=int, default=DEFAULT_MAX_VISIBLE_FRAMES)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-cf-nocue", action="store_true", help="Skip CF and NoCue generation for testing")
    args = ap.parse_args()

    out_dir = _resolve_path(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "triplets.jsonl"
    skip_log_path = out_dir / "skips.log"
    prefix = "memory_s"
    if not args.resume and jsonl_path.exists():
        jsonl_path.unlink()
    if not args.resume and skip_log_path.exists():
        skip_log_path.unlink()
    if not args.resume:
        for child in out_dir.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                shutil.rmtree(child, ignore_errors=True)
        tmp_root = out_dir / "_tmp"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    existing_group_dirs = sorted(
        p.name for p in out_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)
    )
    existing_seeds = []
    if args.resume and jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("variant") == "full":
                existing_seeds.append(int(rec["seed"]))

    made = len(existing_group_dirs) if args.resume else 0
    seed_base = (max(existing_seeds) + 1) if existing_seeds else int(args.seed_start)
    attempts_total = 0
    failure_hist: Dict[str, int] = defaultdict(int)

    while made < int(args.num):
        if int(args.max_attempts_total) > 0 and attempts_total >= int(args.max_attempts_total):
            print(f"[STOP] reached max-attempts-total={args.max_attempts_total}, made={made}/{args.num}")
            break
        recs = None
        used_seed: Optional[int] = None

        for k in range(int(args.attempts_per_group)):
            if int(args.max_attempts_total) > 0 and attempts_total >= int(args.max_attempts_total):
                break
            s_try = seed_base + k
            gid_try = f"memory_s{s_try:06d}"
            if args.resume and (out_dir / gid_try).exists():
                continue
            attempts_total += 1
            t0 = time.perf_counter()

            if not args.disable_prescreen:
                ok_seed, reason = prescreen_seed_memory(
                    env_id=args.env_id,
                    seed=s_try,
                    max_plan_steps=int(args.max_plan_steps),
                    min_visible_frames=int(args.min_visible_frames),
                    max_visible_frames=int(args.max_visible_frames),
                    plan_max_nodes=int(args.prescreen_plan_max_nodes),
                    plan_timeout_s=float(args.prescreen_plan_timeout_s),
                )
                if not ok_seed:
                    failure_hist[reason] += 1
                    append_skip_log(skip_log_path, s_try, reason)
                    if int(args.log_every) > 0 and (attempts_total % int(args.log_every) == 0):
                        print(f"[PROGRESS] attempts={attempts_total} made={made}/{args.num} last_fail={reason}")
                    continue

            recs = generate_one(
                env_id=args.env_id,
                seed=s_try,
                group_index=made,
                out_dir=out_dir,
                tile_size=int(args.tile_size),
                max_plan_steps=int(args.max_plan_steps),
                max_episode_steps=int(args.max_episode_steps),
                mask_strength_target=float(args.mask_strength_target),
                mask_ratio_max=float(args.mask_ratio_max),
                alignment_min=float(args.alignment_min),
                min_visible_frames=int(args.min_visible_frames),
                max_visible_frames=int(args.max_visible_frames),
                plan_max_nodes=int(args.plan_max_nodes),
                plan_timeout_s=float(args.plan_timeout_s),
                plan_fallback_max_nodes=int(args.plan_fallback_max_nodes),
                plan_fallback_timeout_s=float(args.plan_fallback_timeout_s),
                skip_cf_nocue=args.skip_cf_nocue,
            )
            if recs is not None:
                used_seed = s_try
                break
            reason = "generate_failed"
            failure_hist[reason] += 1
            append_skip_log(skip_log_path, s_try, reason, detail=f"elapsed_s={time.perf_counter()-t0:.3f}")
            if int(args.log_every) > 0 and (attempts_total % int(args.log_every) == 0):
                print(f"[PROGRESS] attempts={attempts_total} made={made}/{args.num} last_fail={reason}")

        if recs is None or used_seed is None:
            print(f"[SKIP] seed_base={seed_base} failed after retries; attempts_total={attempts_total}")
            seed_base += 1
            continue

        # Write 3 lines: full -> nocue -> cf
        append_jsonl_batch(jsonl_path, recs)
        print(f"[OK] group=memory_s{made:06d} seed={used_seed} wrote 3 lines")
        made += 1
        seed_base += 1

    if failure_hist:
        top = ", ".join(f"{k}:{v}" for k, v in sorted(failure_hist.items(), key=lambda kv: kv[1], reverse=True)[:8])
        print(f"[SUMMARY] attempts={attempts_total} made={made}/{args.num} failures={top}")



# Private validation helpers must not shadow generator globals.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from utils.minigrid_checks import _die as _check__die
from utils.minigrid_checks import cue_tile_mask as _check__cue_tile_mask
from utils.minigrid_checks import pixel_mask_from_tiles as _check__pixel_mask_from_tile_mask
from utils.minigrid_io import load_rgb_png as _check__load_png


import argparse as _check_argparse
import json as _check_json
from pathlib import Path as _check_Path
from typing import Any as _check_Any, Dict as _check_Dict, List as _check_List, Tuple as _check_Tuple

import numpy as _check_np
from PIL import Image as _check_Image

from minigrid.core.constants import OBJECT_TO_IDX as _check_OBJECT_TO_IDX, COLOR_TO_IDX as _check_COLOR_TO_IDX

# Clean model_input_fields: only what the model sees (no outcome leakage)
# SSOT §4.5 Physical Isolation: no terminated/truncated/reward/success
# Validate both supported model-input field sets.
_check_ALLOWED_INPUTS_CLEAN = ["frames", "mission"]
_check_ALLOWED_INPUTS_LEGACY = ["frames", "mission", "terminated", "truncated"]

# Union for backward compatibility during transition
_check_ALLOWED_INPUTS = _check_ALLOWED_INPUTS_CLEAN + [x for x in _check_ALLOWED_INPUTS_LEGACY if x not in _check_ALLOWED_INPUTS_CLEAN]
_check_MASK_RGB = (0, 0, 0)





def validate() -> None:
    ap = _check_argparse.ArgumentParser()
    repo_root = _check_Path(__file__).resolve().parents[3]
    default_root = repo_root / "datasets" / "raw_data" / "memory"
    ap.add_argument("--root", type=str, default=str(default_root), help="dataset root (where triplets.jsonl exists)")
    ap.add_argument("--max-groups", type=int, default=-1)
    args = ap.parse_args()

    root = _check_Path(args.root)
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    jsonl = root / "triplets.jsonl"
    if not jsonl.exists():
        _check__die(f"Not found: {jsonl}")

    rows: _check_List[_check_Dict[str, _check_Any]] = []
    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_check_json.loads(line))

    if len(rows) % 3 != 0:
        _check__die(f"JSONL lines must be multiple of 3, got {len(rows)}")

    n_groups = len(rows) // 3
    if args.max_groups > 0:
        n_groups = min(n_groups, args.max_groups)

    for gi in range(n_groups):
        full = rows[3 * gi + 0]
        nocue = rows[3 * gi + 1]
        cf = rows[3 * gi + 2]

        gid = full.get("group_id", f"<idx={gi}>")

        # ---- order ----
        if full.get("variant") != "full" or nocue.get("variant") != "nocue" or cf.get("variant") != "cf":
            _check__die(f"[{gid}] variant order must be full/nocue/cf")

        # ---- identity ----
        for k in ("task", "env_id", "seed", "group_id"):
            if not (full.get(k) == nocue.get(k) == cf.get(k)):
                _check__die(f"[{gid}] identity mismatch on {k}")

        if full["task"] != "memory":
            _check__die(f"[{gid}] task != memory: {full['task']}")

        # ---- model_input_fields ----
        for rec, name in ((full, "full"), (nocue, "nocue"), (cf, "cf")):
            mif = rec.get("model_input_fields", None)
            # Accept either supported model-input field set.
            mif_set = set(mif) if mif else set()
            if not mif_set.issubset(set(_check_ALLOWED_INPUTS)):
                _check__die(f"[{gid}:{name}] model_input_fields must be subset of {_check_ALLOWED_INPUTS}, got {mif}")

        # ---- actions parity ----
        if full.get("actions_id") != nocue.get("actions_id") or full.get("actions_id") != cf.get("actions_id"):
            _check__die(f"[{gid}] actions_id mismatch across variants")

        T = len(full["actions_id"])

        # ---- lengths ----
        for rec, name in ((full, "full"), (nocue, "nocue"), (cf, "cf")):
            if len(rec.get("frames", [])) != T + 1:
                _check__die(f"[{gid}:{name}] frames len != T+1")
            if len(rec.get("state_seq", [])) != T + 1:
                _check__die(f"[{gid}:{name}] state_seq len != T+1")

        # ---- physics consistency ----
        for t in range(T + 1):
            for key in ("pos", "dir"):
                a = full["state_seq"][t]["agent"][key]
                b = nocue["state_seq"][t]["agent"][key]
                c = cf["state_seq"][t]["agent"][key]
                if a != b or a != c:
                    _check__die(f"[{gid}] physics mismatch at t={t} agent.{key}")

        # ---- outcomes ----
        if not (full["success"] is True and float(full["reward"]) > 0.0):
            _check__die(f"[{gid}] full must succeed with reward>0")
        if not (cf["success"] is False and float(cf["reward"]) == 0.0):
            _check__die(f"[{gid}] cf must fail with reward==0")
        if not (nocue["success"] == full["success"] and float(nocue["reward"]) == float(full["reward"])):
            _check__die(f"[{gid}] nocue outcome must match full")

        # ---- NoCue meta schema ----
        meta = nocue.get("nocue_meta", None)
        if not isinstance(meta, dict):
            _check__die(f"[{gid}] nocue missing nocue_meta dict")

        required_meta = [
            "targets", "window_policy", "window_steps",
            "mask_strength_target", "mask_strength_actual",
            "alignment_score", "alignment_threshold",
            "masked_frames", "mask_type", "physics_check_passed",
        ]
        for k in required_meta:
            if k not in meta:
                _check__die(f"[{gid}] nocue_meta missing {k}")

        if meta["window_policy"] != "EARLY":
            _check__die(f"[{gid}] window_policy must be EARLY")

        window_steps = [int(x) for x in meta["window_steps"]]
        wset = set(window_steps)
        if len(wset) != len(window_steps):
            _check__die(f"[{gid}] window_steps contains duplicates")
        if any(t < 0 or t > T for t in window_steps):
            _check__die(f"[{gid}] window_steps out of range")

        # We need cue type+color for audit; stored in meta['cue'] by generator
        cue = meta.get("cue", {})
        cue_type = cue.get("type", meta["targets"][0] if meta["targets"] else None)
        cue_color = cue.get("color", None)
        if cue_type is None or cue_color is None:
            _check__die(f"[{gid}] nocue_meta must include cue.type and cue.color for color-safe audit")

        # ---- NoCue image audits ----
        # 1) Unmasked frames must be identical
        for t in range(T + 1):
            if t in wset:
                continue
            f_img = _check__load_png(root / full["frames"][t])
            n_img = _check__load_png(root / nocue["frames"][t])
            if not _check_np.array_equal(f_img, n_img):
                _check__die(f"[{gid}] NoCue illegal diff at unmasked t={t}")

        # 2) After window_end, must be identical (no late leakage)
        if window_steps:
            window_end = max(window_steps)
            for t in range(window_end + 1, T + 1):
                f_img = _check__load_png(root / full["frames"][t])
                n_img = _check__load_png(root / nocue["frames"][t])
                if not _check_np.array_equal(f_img, n_img):
                    _check__die(f"[{gid}] NoCue leakage after window_end at t={t}")

        # 3) Masked frames: diff subset of cue tiles (tile-level -> pixel-level)
        for t in sorted(wset):
            f_img = _check__load_png(root / full["frames"][t])
            n_img = _check__load_png(root / nocue["frames"][t])
            if f_img.shape != n_img.shape:
                _check__die(f"[{gid}] frame shape mismatch at masked t={t}")

            st = full["state_seq"][t]
            tile_mask = _check__cue_tile_mask(st["state_encoding"], cue_type, cue_color)
            if tile_mask.sum() <= 0:
                _check__die(f"[{gid}] masked t={t}: cue tile mask empty (inconsistent)")

            pix_mask = _check__pixel_mask_from_tile_mask(tile_mask, f_img.shape[0], f_img.shape[1])

            # diff must not appear outside pix_mask
            diff = _check_np.any(f_img != n_img, axis=2)
            if _check_np.any(diff & (~pix_mask)):
                _check__die(f"[{gid}] masked t={t}: diff leaks outside cue tiles")

            # diff must actually happen inside pix_mask
            if not _check_np.any(diff & pix_mask):
                _check__die(f"[{gid}] masked t={t}: no diff inside cue tiles")

        print(f"[OK] {gid}")

    print(f"[DONE] validated {n_groups} groups")



def main() -> None:
    """Select generation or validation while preserving each command's arguments."""
    import argparse as _cli_argparse
    import sys as _cli_sys

    parser = _cli_argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['generate', 'validate'])
    if len(_cli_sys.argv) < 2 or _cli_sys.argv[1] not in ('generate', 'validate'):
        parser.parse_args()
        return
    original = _cli_sys.argv
    command = original[1]
    _cli_sys.argv = [original[0] + ' ' + command, *original[2:]]
    try:
        if command == 'generate':
            generate()
        else:
            validate()
    finally:
        _cli_sys.argv = original


if __name__ == '__main__':
    main()
