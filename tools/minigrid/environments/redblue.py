"""redblue: simulator trajectory generation and environment-specific validation."""
from __future__ import annotations

# Trajectory generation
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from utils.minigrid_checks import check_triplet_alignment as enforce_triplet_hard_gates
from utils.minigrid_io import save_png as save_png


import argparse
import copy
import json
import logging
import os
import shutil
from collections import deque
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import gymnasium as gym

# Bug D Fix: Use official constants
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR
from minigrid.core.world_object import Wall
from minigrid.wrappers import RGBImgPartialObsWrapper
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


# -----------------------------
# CLI / Logging / IO helpers
# -----------------------------

ACTION_ID2NAME = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

DIR2VEC = {
    0: (1, 0),   # right
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}

VEC2DIR = {v: k for k, v in DIR2VEC.items()}


def setup_logger(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)




def append_jsonl_batch(path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Append multiple json lines and fsync for crash-safety.
    """
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())



# -----------------------------
# Rendering & State Extraction
# -----------------------------

def render_agent_pov(env: gym.Env, tile_size: int, grid_h: int, grid_w: int) -> np.ndarray:
    """
    Render agent POV RGB frame aligned to the 7x7 observation grid.
    """
    # Assuming env is wrapped with RGBImgPartialObsWrapper
    # We use get_frame from the unwrapped env or the wrapper if accessible
    # Standard MiniGrid: env.get_frame(...) returns the full env render or POV.
    # RGBImgPartialObsWrapper step() returns the POV image directly in obs['image'].
    # To get it manually without stepping:
    frame = env.unwrapped.get_frame(tile_size=tile_size, agent_pov=True)

    if not isinstance(frame, np.ndarray):
        frame = np.array(frame, dtype=np.uint8)
    expected_h = grid_h * tile_size
    expected_w = grid_w * tile_size
    # Basic shape check
    assert frame.shape[0] == expected_h and frame.shape[1] == expected_w, (
        f"render shape mismatch: got {frame.shape[:2]} expected {(expected_h, expected_w)}"
    )
    return frame.astype(np.uint8)


def _door_state(obj: Any) -> int:
    """Use the simulator's door encoding; non-door cells retain sentinel -1."""
    if obj is None or getattr(obj, "type", None) != "door":
        return -1
    state = obj.encode()[2]
    if isinstance(state, bool) or not isinstance(state, Integral) or state not in (0, 1, 2):
        raise ValueError(f"Invalid simulator door state: {state!r}")
    return int(state)


def extract_state(env_unwrapped: Any) -> Dict[str, Any]:
    """
    Bug A fix: always use env_unwrapped.gen_obs()['image'] as semantic encoding.
    """
    obs = env_unwrapped.gen_obs()
    enc = obs["image"]  # (7,7,3) semantic grid

    ax, ay = env_unwrapped.agent_pos
    ad = int(env_unwrapped.agent_dir)

    fx, fy = env_unwrapped.front_pos
    front_obj = None
    if 0 <= fx < env_unwrapped.width and 0 <= fy < env_unwrapped.height:
        front_obj = env_unwrapped.grid.get(fx, fy)

    front_type = getattr(front_obj, "type", "wall" if front_obj is not None else "empty")
    if front_type == "floor":
        front_type = "empty"
    front_cell = {
        "pos": [int(fx), int(fy)],
        "type": front_type,
        "color": getattr(front_obj, "color", None),
        "state": _door_state(front_obj),
    }

    # list non-trivial objects
    objects = []
    grid = env_unwrapped.grid
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            o = grid.get(x, y)
            if o is None:
                continue
            ot = getattr(o, "type", None)
            if ot in ("wall", "floor"):
                continue
            obj_rec = {"type": ot, "pos": [int(x), int(y)]}
            if hasattr(o, "color"):
                obj_rec["color"] = getattr(o, "color", None)
            if ot == "door":
                obj_rec["state"] = _door_state(o)
            objects.append(obj_rec)

    # visible types (GT) from semantic encoding
    visible_types_gt = []
    seen = set()
    for i in range(enc.shape[0]):
        for j in range(enc.shape[1]):
            oid = int(enc[i, j, 0])
            oname = IDX_TO_OBJECT.get(oid, "unknown")
            if oname in ("unseen", "empty", "floor", "wall"):
                continue
            if oname == "door":
                cid = int(enc[i, j, 1])
                cname = IDX_TO_COLOR.get(cid, None)
                if cname == "red":
                    tag = "red_door"
                elif cname == "blue":
                    tag = "blue_door"
                else:
                    tag = "door"
                if tag not in seen:
                    seen.add(tag)
                    visible_types_gt.append(tag)
            else:
                if oname not in seen:
                    seen.add(oname)
                    visible_types_gt.append(oname)

    return {
        "step": int(env_unwrapped.step_count),
        # Retain the flat pose fields consumed by validators.
        "agent_pos": [int(ax), int(ay)],
        "agent_dir": ad,
        # SSOT-aligned schema fields.
        "agent": {
            "pos": [int(ax), int(ay)],
            "dir": ad,
            "carrying": (
                {
                    "type": str(getattr(env_unwrapped.carrying, "type", "unknown")),
                    "color": str(getattr(env_unwrapped.carrying, "color", "unknown")),
                }
                if getattr(env_unwrapped, "carrying", None) is not None
                else None
            ),
        },
        "front_cell": front_cell,
        "objects": objects,
        "state_encoding": enc.tolist(),
        "visible_types_gt": visible_types_gt,
        "visible_types": visible_types_gt,  # may be overwritten in NoCue
        "mask_applied": False,
        "mask_type": None,
    }


# -----------------------------
# Planning (BFS on interaction states)
# -----------------------------

@dataclass(frozen=True)
class DoorPos:
    color: str
    pos: Tuple[int, int]
    open0: bool


def find_red_blue_doors(env_unwrapped: Any) -> Tuple[DoorPos, DoorPos]:
    red = None
    blue = None
    grid = env_unwrapped.grid
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            o = grid.get(x, y)
            if o is None or getattr(o, "type", None) != "door":
                continue
            c = getattr(o, "color", None)
            st = _door_state(o)
            open0 = (st == 0)
            if c == "red":
                red = DoorPos("red", (x, y), open0)
            elif c == "blue":
                blue = DoorPos("blue", (x, y), open0)
    if red is None or blue is None:
        raise RuntimeError("Red/Blue doors not found in grid.")
    return red, blue


@dataclass(frozen=True)
class RBState:
    ax: int
    ay: int
    ad: int
    red_open: bool
    blue_open: bool


def _front_pos(ax: int, ay: int, ad: int) -> Tuple[int, int]:
    dx, dy = DIR2VEC[ad]
    return ax + dx, ay + dy


def _cell_passable(env_unwrapped: Any, x: int, y: int, red_pos: Tuple[int, int], blue_pos: Tuple[int, int]) -> bool:
    if not (0 <= x < env_unwrapped.width and 0 <= y < env_unwrapped.height):
        return False
    # Bug C Fix: Treat doors as obstacles for movement (we toggle them from adjacent)
    if (x, y) == red_pos or (x, y) == blue_pos:
        return False
    o = env_unwrapped.grid.get(x, y)
    if o is None:
        return True
    if getattr(o, "type", None) == "wall":
        return False
    # conservative: block any other object
    return False


def bfs_plan_red_then_blue(env_unwrapped: Any, max_nodes: int = 20000) -> List[int]:
    """
    Plan with BFS over (pos,dir,door_states).
    Strategy: Red Open -> Blue Open.
    """
    red, blue = find_red_blue_doors(env_unwrapped)
    red_pos, blue_pos = red.pos, blue.pos

    sx, sy = map(int, env_unwrapped.agent_pos)
    sd = int(env_unwrapped.agent_dir)

    start = RBState(sx, sy, sd, red.open0, blue.open0)

    q = deque([start])
    parent: Dict[RBState, Tuple[Optional[RBState], Optional[int]]] = {start: (None, None)}
    expanded = 0

    def is_goal(s: RBState) -> bool:
        return s.red_open and s.blue_open

    while q:
        s = q.popleft()
        expanded += 1
        if expanded > max_nodes:
            break
        if is_goal(s):
            actions_rev: List[int] = []
            cur = s
            while True:
                prev, act = parent[cur]
                if prev is None:
                    break
                assert act is not None
                actions_rev.append(act)
                cur = prev
            return list(reversed(actions_rev))

        # Actions: left, right, forward
        # left
        ns = RBState(s.ax, s.ay, (s.ad - 1) % 4, s.red_open, s.blue_open)
        if ns not in parent:
            parent[ns] = (s, 0)
            q.append(ns)

        # right
        ns = RBState(s.ax, s.ay, (s.ad + 1) % 4, s.red_open, s.blue_open)
        if ns not in parent:
            parent[ns] = (s, 1)
            q.append(ns)

        # forward
        nx, ny = _front_pos(s.ax, s.ay, s.ad)
        if _cell_passable(env_unwrapped, nx, ny, red_pos, blue_pos):
            ns = RBState(nx, ny, s.ad, s.red_open, s.blue_open)
            if ns not in parent:
                parent[ns] = (s, 2)
                q.append(ns)

        # toggle (Interaction Logic)
        fx, fy = _front_pos(s.ax, s.ay, s.ad)
        # 1. Open Red
        if (fx, fy) == red_pos and (not s.red_open):
            ns = RBState(s.ax, s.ay, s.ad, True, s.blue_open)
            if ns not in parent:
                parent[ns] = (s, 5)
                q.append(ns)
        # 2. Open Blue (only if Red is open)
        elif (fx, fy) == blue_pos and s.red_open and (not s.blue_open):
            ns = RBState(s.ax, s.ay, s.ad, s.red_open, True)
            if ns not in parent:
                parent[ns] = (s, 5)
                q.append(ns)

    raise RuntimeError("BFS failed to find a valid plan.")


# -----------------------------
# Rollout
# -----------------------------

def rollout_from_current(
    env: gym.Env,
    obs0: Dict[str, Any],
    actions: List[int],
    tile_size: int,
    out_dir: Path,
    rel_dir: str,
) -> Dict[str, Any]:
    """
    Execute actions from current env state, saving frames.
    """
    # obs0['image'] from wrapper is (H,W,3) RGB
    grid_h = int(obs0["image"].shape[0]) // tile_size
    grid_w = int(obs0["image"].shape[1]) // tile_size

    frames: List[str] = []
    state_seq: List[Dict[str, Any]] = []

    # step 0
    img0 = render_agent_pov(env, tile_size, grid_h, grid_w)
    fp0 = out_dir / rel_dir / "step_000.png"
    save_png(img0, fp0)
    frames.append(str(Path(rel_dir) / "step_000.png"))
    state_seq.append(extract_state(env.unwrapped))

    total_reward = 0.0
    terminated = False
    truncated = False

    for t, a in enumerate(actions, start=1):
        obs, r, term, trunc, _ = env.step(a)
        total_reward += float(r)
        terminated = bool(term)
        truncated = bool(trunc)

        img = render_agent_pov(env, tile_size, grid_h, grid_w)
        fp = out_dir / rel_dir / f"step_{t:03d}.png"
        save_png(img, fp)
        frames.append(str(Path(rel_dir) / f"step_{t:03d}.png"))
        state_seq.append(extract_state(env.unwrapped))

        if terminated or truncated:
            break

    return {
        "frames": frames,
        "state_seq": state_seq,
        "actions_id": actions[: len(frames) - 1],
        "actions_text": [ACTION_ID2NAME[x] for x in actions[: len(frames) - 1]],
        "terminated": terminated,
        "truncated": truncated,
        "reward": float(total_reward),
        "success": bool(terminated and total_reward > 0.0),
        "mission": obs0.get("mission", ""),
    }


# -----------------------------
# NoCue Logic
# -----------------------------

def is_red_door_visible(state_encoding: np.ndarray) -> bool:
    door_id = OBJECT_TO_IDX["door"]
    red_id = COLOR_TO_IDX["red"]
    return bool(np.any((state_encoding[:, :, 0] == door_id) & (state_encoding[:, :, 1] == red_id)))


def mask_red_door_tiles_inplace(pov_img: np.ndarray, state_encoding: np.ndarray) -> Tuple[int, int]:
    """
    Black-out tiles where semantic encoding indicates red door.
    """
    H, W, _ = pov_img.shape
    gh, gw = state_encoding.shape[0], state_encoding.shape[1]
    tile_h = H // gh
    tile_w = W // gw

    door_id = OBJECT_TO_IDX["door"]
    red_id = COLOR_TO_IDX["red"]

    target_tiles = 0
    masked_tiles = 0

    for i in range(gh):
        for j in range(gw):
            if int(state_encoding[i, j, 0]) == door_id and int(state_encoding[i, j, 1]) == red_id:
                target_tiles += 1
                # IMPORTANT: state_encoding axes are (obs_x, obs_y). First
                # axis maps to image x, second axis maps to image y.
                x1, x2 = i * tile_w, (i + 1) * tile_w
                y1, y2 = j * tile_h, (j + 1) * tile_h
                # Hard blackout
                pov_img[y1:y2, x1:x2] = 0
                masked_tiles += 1

    return masked_tiles, target_tiles


def build_nocue_from_full(
    full_traj: Dict[str, Any],
    work_dir: Path,
    tile_size: int,
    nocue_max_visible: int,
    alignment_min: float,
    mask_ratio_max: float,
    mask_strength_target: float,
) -> Optional[Dict[str, Any]]:
    state_seq = copy.deepcopy(full_traj["state_seq"])
    actions = full_traj["actions_id"]

    # 1. T_interact: First toggle Red
    t_interact = None
    for t, a in enumerate(actions):
        if a != 5: continue
        fc = state_seq[t]["front_cell"]
        if fc.get("type") == "door" and fc.get("color") == "red":
            t_interact = t
            break
    if t_interact is None: return None

    # 2. First Seen
    t_first_seen = None
    for t in range(len(state_seq)):
        enc = np.array(state_seq[t]["state_encoding"], dtype=np.int64)
        if is_red_door_visible(enc):
            t_first_seen = t
            break
    if t_first_seen is None: return None

    # 3. Window: EARLY (t <= first_seen)
    window_end = min(t_first_seen, t_interact - 1)
    if window_end < 0: return None

    # 4. Filter leak (Interaction Evidence)
    try:
        masked_indices, leak_indices, d5_excluded_indices = select_nocue_indices(
            state_seq=state_seq,
            first_interaction_step=int(window_end + 1),
            max_visible=int(nocue_max_visible),
            is_target_visible=lambda st: is_red_door_visible(np.array(st["state_encoding"], dtype=np.int64)),
            is_d5_excluded=lambda st: (
                st["front_cell"].get("type") == "door" and st["front_cell"].get("color") == "red"
            ),
        )
    except ValueError:
        return None

    first_interaction_step = int(t_interact)
    if any(t >= first_interaction_step for t in masked_indices):
        return None

    (work_dir / "nocue").mkdir(parents=True, exist_ok=True)
    per_frame_mask_ratio = []
    per_frame_alignment = []
    masked_tiles_total = 0

    enc0 = np.array(state_seq[masked_indices[0]]["state_encoding"], dtype=np.int64)
    gh0, gw0 = enc0.shape[0], enc0.shape[1]
    mask_budget_tiles = int(np.floor(mask_ratio_max * float(gh0 * gw0) * float(len(masked_indices))))

    for t in range(len(full_traj["frames"])):
        src = work_dir / full_traj["frames"][t]
        dst = work_dir / "nocue" / Path(full_traj["frames"][t]).name

        with Image.open(src) as im:
            img = np.array(im.convert("RGB"), dtype=np.uint8)
        if t in masked_indices:
            enc = np.array(state_seq[t]["state_encoding"], dtype=np.int64)
            masked_tiles, target_tiles = mask_red_door_tiles_inplace(img, enc)

            gh, gw = enc.shape[0], enc.shape[1]
            ratio = float(masked_tiles) / float(max(1, gh * gw))
            align = float(masked_tiles) / float(max(1, target_tiles))
            masked_tiles_total += int(masked_tiles)

            per_frame_mask_ratio.append(ratio)
            per_frame_alignment.append(align)

            # Metadata update
            state_seq[t]["mask_applied"] = True
            state_seq[t]["mask_type"] = "tile_suppression"
            vgt = list(state_seq[t].get("visible_types_gt", []))
            state_seq[t]["visible_types"] = [v for v in vgt if v != "red_door"]
        else:
            state_seq[t]["mask_applied"] = False
            state_seq[t]["mask_type"] = None
            state_seq[t]["visible_types"] = list(state_seq[t].get("visible_types_gt", []))

        save_png(img, dst)

    if not per_frame_alignment: return None

    mask_strength_actual = float(np.mean(per_frame_mask_ratio))
    try:
        metrics = compute_nocue_metrics(
            per_frame_mask_ratios=per_frame_mask_ratio,
            alignment_ratios=per_frame_alignment,
            masked_tiles_total=int(masked_tiles_total),
            gh=int(gh0),
            gw=int(gw0),
            masked_count=len(masked_indices),
            mask_ratio_max=float(mask_ratio_max),
            alignment_min=float(alignment_min),
        )
    except ValueError:
        return None

    nocue_meta = build_nocue_meta(
        targets=["red_door"],
        first_interaction_step=int(first_interaction_step),
        masked_indices=masked_indices,
        leak_indices=leak_indices,
        d5_excluded_indices=d5_excluded_indices,
        mask_strength_target=float(mask_strength_target),
        mask_strength_actual=float(mask_strength_actual),
        mask_strength_threshold=float(mask_ratio_max),
        alignment_threshold=float(alignment_min),
        metrics=metrics,
        extra={"t_first_seen": int(t_first_seen), "t_interact": int(t_interact)},
    )
    return {"nocue_meta": nocue_meta, "nocue_state_seq": state_seq}


# -----------------------------
# CF (Type-2 Intervention)
# -----------------------------

def apply_cf_blue_door_to_wall(env_unwrapped: Any) -> Dict[str, Any]:
    red, blue = find_red_blue_doors(env_unwrapped)
    bx, by = blue.pos
    env_unwrapped.grid.set(bx, by, Wall())
    return {
        "cf_mode": "blue_door_to_wall",
        "intervention_type": "Type-2",
        "blue_door_pos": [int(bx), int(by)],
    }


# -----------------------------
# Record Packaging
# -----------------------------

def mk_rec(
    gid: str,
    task: str,
    env_id: str,
    seed: int,
    variant: str,
    traj: Dict[str, Any],
    cf_meta: Optional[Dict[str, Any]] = None,
    nocue_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    allowed_input_fields = {"frames", "mission"}
    rec: Dict[str, Any] = {
        "group_id": gid,
        "task": task,
        "variant": variant,
        "env_id": env_id,
        "seed": int(seed),
        "frames": [f"{gid}/{variant}/{Path(p).name}" for p in traj["frames"]],
        "actions_id": traj["actions_id"],
        "actions_text": traj["actions_text"],
        "mission": traj.get("mission", ""),
        "terminated": bool(traj["terminated"]),
        "truncated": bool(traj["truncated"]),
        "success": bool(traj["success"]),
        "reward": float(traj["reward"]),
        "state_seq": traj["state_seq"],
        "model_input_fields": sorted(list(allowed_input_fields)),
    }
    if cf_meta: rec["cf_meta"] = cf_meta
    if nocue_meta: rec["nocue_meta"] = nocue_meta
    return rec


# -----------------------------
# Main
# -----------------------------

def generate_one(
    env_id: str,
    seed: int,
    group_index: int,
    out_dir: Path,
    tile_size: int,
    max_nodes: int,
    nocue_max_visible: int,
    alignment_min: float,
    mask_ratio_max: float,
    mask_strength_target: float,
) -> Optional[List[Dict[str, Any]]]:
    # Canonical group_id is indexed by accepted sample order, not raw env seed.
    # The actual seed remains recorded in the JSONL `seed` field.
    gid = f"redblue_s{group_index:06d}"
    final_dir = out_dir / gid
    if final_dir.exists(): return None

    tmp_root = out_dir / "_tmp"
    work_dir = tmp_root / gid
    if work_dir.exists(): shutil.rmtree(work_dir, ignore_errors=True)
    ensure_dir(work_dir)

    env = None
    try:
        # Full
        env = gym.make(env_id)
        env = RGBImgPartialObsWrapper(env, tile_size=tile_size)

        # Bug B Fix: Plan after reset
        obs0, _ = env.reset(seed=seed)
        scan_prefix = [1, 1, 1, 1] # Deterministic scan
        plan_actions = bfs_plan_red_then_blue(env.unwrapped, max_nodes=max_nodes)
        actions = scan_prefix + plan_actions

        full_traj = rollout_from_current(
            env=env,
            obs0=obs0,
            actions=actions,
            tile_size=tile_size,
            out_dir=work_dir,
            rel_dir="full",
        )

        if full_traj["truncated"]: return None
        if not full_traj["success"]: return None
        if not full_traj["terminated"]: return None

    except Exception:
        return None
    finally:
        if env: env.close()

    # NoCue
    try:
        ensure_dir(work_dir / "full")
        nocue_pack = build_nocue_from_full(
            full_traj=full_traj,
            work_dir=work_dir,
            tile_size=tile_size,
            nocue_max_visible=nocue_max_visible,
            alignment_min=alignment_min,
            mask_ratio_max=mask_ratio_max,
            mask_strength_target=mask_strength_target,
        )
        if not nocue_pack: return None
        nocue_meta = nocue_pack["nocue_meta"]
        nocue_state_seq = nocue_pack["nocue_state_seq"]
    except Exception:
        return None

    # CF
    env_cf = None
    try:
        env_cf = gym.make(env_id)
        env_cf = RGBImgPartialObsWrapper(env_cf, tile_size=tile_size)
        obs_cf0, _ = env_cf.reset(seed=seed)

        cf_meta = apply_cf_blue_door_to_wall(env_cf.unwrapped)

        cf_traj = rollout_from_current(
            env=env_cf,
            obs0=obs_cf0,
            actions=full_traj["actions_id"],
            tile_size=tile_size,
            out_dir=work_dir,
            rel_dir="cf",
        )

        # CF Gates: Type-2
        if cf_traj["truncated"]: return None
        if cf_traj["success"]: return None
        if cf_traj["terminated"]: return None # Wall toggle shouldn't terminate
        if cf_traj["reward"] != 0.0: return None

        # Physics Invariance
        if len(cf_traj["state_seq"]) != len(full_traj["state_seq"]): return None
        for t in range(len(full_traj["state_seq"])):
            sf = full_traj["state_seq"][t]
            sc = cf_traj["state_seq"][t]
            sf_pos = sf.get("agent", {}).get("pos", sf.get("agent_pos"))
            sf_dir = sf.get("agent", {}).get("dir", sf.get("agent_dir"))
            sc_pos = sc.get("agent", {}).get("pos", sc.get("agent_pos"))
            sc_dir = sc.get("agent", {}).get("dir", sc.get("agent_dir"))
            if sf_pos != sc_pos or sf_dir != sc_dir:
                return None

    except Exception:
        return None
    finally:
        if env_cf: env_cf.close()

    # Finalize
    ensure_dir(final_dir.parent)
    if final_dir.exists(): shutil.rmtree(final_dir)
    shutil.move(str(work_dir), str(final_dir))

    full_rec = mk_rec(gid, "redblue", env_id, seed, "full", full_traj)
    # Fix paths for jsonl (point to final)
    full_rec["frames"] = [f"{gid}/full/{Path(p).name}" for p in full_traj["frames"]]

    nocue_traj = {
        "frames": full_traj["frames"],
        "state_seq": nocue_state_seq,
        "actions_id": full_traj["actions_id"],
        "actions_text": full_traj["actions_text"],
        "mission": full_traj.get("mission", ""),
        "terminated": full_traj["terminated"],
        "truncated": full_traj["truncated"],
        "success": full_traj["success"],
        "reward": full_traj["reward"],
    }
    nocue_rec = mk_rec(gid, "redblue", env_id, seed, "nocue", nocue_traj, nocue_meta=nocue_meta)
    nocue_rec["frames"] = [f"{gid}/nocue/{Path(p).name}" for p in full_traj["frames"]]

    cf_rec = mk_rec(gid, "redblue", env_id, seed, "cf", cf_traj, cf_meta=cf_meta)
    cf_rec["frames"] = [f"{gid}/cf/{Path(p).name}" for p in cf_traj["frames"]]

    records = [full_rec, nocue_rec, cf_rec]
    enforce_triplet_hard_gates(records)
    return records


def generate() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", "--env_id", dest="env_id", type=str, default="MiniGrid-RedBlueDoors-8x8-v0")
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default=str(_repo_root() / "datasets" / "raw_data" / "redblue"),
    )
    parser.add_argument("--num", "--n", dest="num", type=int, default=100)
    parser.add_argument("--seed-start", "--seed_start", dest="seed_start", type=int, default=0)
    parser.add_argument("--tile-size", "--tile_size", dest="tile_size", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logger(args.verbose)

    out_dir = _resolve_path(Path(args.out_dir))
    ensure_dir(out_dir)
    ensure_dir(out_dir / "_tmp")
    jsonl_path = out_dir / "triplets.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    # Fresh rebuild: remove stale canonical group directories so numbering starts cleanly
    # from redblue_s000000 without being blocked by leftovers from previous runs.
    for child in out_dir.iterdir():
        if child.is_dir() and child.name.startswith("redblue_s"):
            shutil.rmtree(child, ignore_errors=True)
    if (out_dir / "_tmp").exists():
        shutil.rmtree(out_dir / "_tmp", ignore_errors=True)
    ensure_dir(out_dir / "_tmp")

    target = args.num
    n_ok = 0
    seed = args.seed_start

    while n_ok < target:
        recs = generate_one(
            env_id=args.env_id,
            seed=seed,
            group_index=n_ok,
            out_dir=out_dir,
            tile_size=args.tile_size,
            max_nodes=20000,
            nocue_max_visible=5,
            alignment_min=0.70,
            mask_ratio_max=0.35,
            mask_strength_target=0.05
        )
        if recs:
            append_jsonl_batch(jsonl_path, recs)
            n_ok += 1
            if n_ok % 10 == 0:
                logging.info(f"[OK] {n_ok}/{target} (last accepted seed {seed})")
        seed += 1

    logging.info(f"Done. {jsonl_path}")



# Private validation helpers must not shadow generator globals.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse as _check_argparse
import json as _check_json
import logging as _check_logging
from collections import defaultdict as _check_defaultdict
from pathlib import Path as _check_Path
from typing import Any as _check_Any, Dict as _check_Dict, List as _check_List, Tuple as _check_Tuple

import numpy as _check_np
from PIL import Image as _check_Image

# Bug D Fix
from minigrid.core.constants import OBJECT_TO_IDX as _check_OBJECT_TO_IDX, COLOR_TO_IDX as _check_COLOR_TO_IDX, IDX_TO_OBJECT as _check_IDX_TO_OBJECT, IDX_TO_COLOR as _check_IDX_TO_COLOR


def _check_setup_logger(verbose: bool) -> None:
    level = _check_logging.DEBUG if verbose else _check_logging.INFO
    _check_logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def _check_load_jsonl(path: _check_Path) -> _check_List[_check_Dict[str, _check_Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                records.append(_check_json.loads(line))
            except:
                continue
    return records


def _check_check(cond: bool, msg: str) -> None:
    if not cond: raise AssertionError(msg)


def _check_red_door_tile_mask(enc: _check_np.ndarray, tile_size: int, img_h: int, img_w: int) -> _check_np.ndarray:
    door_id = _check_OBJECT_TO_IDX["door"]
    red_id = _check_COLOR_TO_IDX["red"]
    gh, gw = enc.shape[0], enc.shape[1]

    tile_mask = (enc[:, :, 0] == door_id) & (enc[:, :, 1] == red_id)
    image_tile_mask = tile_mask.T
    px_mask = _check_np.kron(image_tile_mask.astype(bool), _check_np.ones((tile_size, tile_size), dtype=bool))
    return px_mask


def _check_validate_group(gid: str, recs: _check_List[_check_Dict[str, _check_Any]], root: _check_Path, fast: bool) -> None:
    variants = {r["variant"]: r for r in recs}
    _check_check(set(variants.keys()) == {"full", "nocue", "cf"}, "Missing variants")

    full, nocue, cf = variants["full"], variants["nocue"], variants["cf"]

    # Action Identity
    _check_check(full["actions_id"] == nocue["actions_id"], "Action mismatch: Full vs NoCue")
    _check_check(full["actions_id"] == cf["actions_id"], "Action mismatch: Full vs CF")

    # Gates
    _check_check(full["success"] and full["terminated"], "Full must succeed")
    _check_check(not cf["success"] and not cf["terminated"], "CF must fail and not terminate")
    _check_check(full["reward"] > 0 and cf["reward"] == 0, "Reward check failed")

    # Physics Invariance
    T = len(full["state_seq"])
    _check_check(len(cf["state_seq"]) == T, "Length mismatch")
    for t in range(T):
        sf = full["state_seq"][t]
        sc = cf["state_seq"][t]
        _check_check(sf["agent_pos"] == sc["agent_pos"], f"Pos diverged at {t}")
        _check_check(sf["agent_dir"] == sc["agent_dir"], f"Dir diverged at {t}")

    # NoCue Logic
    meta = nocue.get("nocue_meta", {})
    masked = meta.get("masked_indices", [])
    _check_check(len(masked) > 0, "NoCue mask empty")
    t_interact = meta.get("t_interact", -1)

    # Interaction leak check
    for m in masked:
        _check_check(int(m) < t_interact, "Mask leaked into interaction phase")
        fc = full["state_seq"][int(m)]["front_cell"]
        _check_check(not (fc.get("type") == "door" and fc.get("color") == "red"), "Mask leaked into interaction-evidence frame")

    # Pixel Audit
    if fast: return

    full_frames = full["frames"]
    nocue_frames = nocue["frames"]
    sample_img = _check_np.array(_check_Image.open(root / full_frames[0]))
    tile_size = sample_img.shape[0] // 7

    masked_set = set(int(x) for x in masked)

    for t in range(len(full_frames)):
        img_f = _check_np.array(_check_Image.open(root / full_frames[t]))
        img_n = _check_np.array(_check_Image.open(root / nocue_frames[t]))
        diff = _check_np.any(img_f != img_n, axis=2)

        if t not in masked_set:
            _check_check(not _check_np.any(diff), f"Unmasked frame {t} changed")
        else:
            enc = _check_np.array(full["state_seq"][t]["state_encoding"])
            px_mask = _check_red_door_tile_mask(enc, tile_size, sample_img.shape[0], sample_img.shape[1])

            # Diff allowed ONLY in red door tiles
            _check_check(not _check_np.any(diff & (~px_mask)), f"Diff outside red door at {t}")
            _check_check(_check_np.any(diff & px_mask), f"No diff in masked area at {t}")


def validate() -> None:
    parser = _check_argparse.ArgumentParser()
    repo_root = _check_Path(__file__).resolve().parents[3]
    default_root = repo_root / "datasets" / "raw_data" / "redblue"
    parser.add_argument("--jsonl", type=str, default=None, help="Path to triplets.jsonl (defaults to <root>/triplets.jsonl)")
    parser.add_argument("--root", type=str, default=str(default_root))
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    _check_setup_logger(True)

    root = _check_Path(args.root)
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    jsonl = _check_Path(args.jsonl) if args.jsonl else (root / "triplets.jsonl")
    if not jsonl.is_absolute():
        jsonl = (repo_root / jsonl).resolve()

    recs = _check_load_jsonl(jsonl)
    groups = _check_defaultdict(list)
    for r in recs: groups[r["group_id"]].append(r)

    ok, fail = 0, 0
    for gid, g_recs in groups.items():
        try:
            _check_validate_group(gid, g_recs, root, args.fast)
            ok += 1
        except Exception as e:
            fail += 1
            _check_logging.error(f"FAIL {gid}: {e}")

    _check_logging.info(f"OK: {ok}, FAIL: {fail}")



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
