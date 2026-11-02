"""multiroom: simulator trajectory generation and environment-specific validation."""
from __future__ import annotations

# Trajectory generation
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from utils.minigrid_checks import check_triplet_alignment as enforce_triplet_hard_gates
from utils.minigrid_io import append_jsonl_fsync as append_jsonl_batch
from utils.minigrid_io import save_png as save_png
from utils.minigrid_state import door_state as _door_state
from utils.minigrid_state import render_pov_with_fallback as render_agent_pov


import argparse
import json
import os
import shutil
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image

from minigrid.core.constants import IDX_TO_OBJECT
from minigrid.core.world_object import Goal
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

# -------------------------
# Constants
# -------------------------
ACTION_ID2NAME = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

DIR_VEC = {
    0: (1, 0),  # right
    1: (0, 1),  # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}

ALLOWED_INPUT_FIELDS = ["frames", "mission"]

# -------------------------
# IO utils
# -------------------------



def append_skip_log(path: Path, seed: int, reason: str) -> None:
    """Production logging: timestamp, atomic-ish write, fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] Seed {seed}\n{reason}\n{'-'*80}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
        f.flush()
        os.fsync(f.fileno())



def extract_state(env_unwrapped, obs: Dict) -> Dict:
    """Extract a single time-step state snapshot."""

    agent_pos = tuple(int(x) for x in env_unwrapped.agent_pos)
    agent_dir = int(env_unwrapped.agent_dir)
    dx, dy = DIR_VEC[agent_dir]
    front_pos = (agent_pos[0] + dx, agent_pos[1] + dy)

    objects = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            o = env_unwrapped.grid.get(x, y)
            if o is None:
                continue
            if getattr(o, "type", None) in ("wall", "floor"):
                continue
            objects.append({
                "type": getattr(o, "type", "unknown"),
                "color": getattr(o, "color", "unknown"),
                "state": _door_state(o) if getattr(o, "type", None) == "door" else 0,
                "pos": [int(x), int(y)],
            })

    # IMPORTANT: use env.unwrapped.gen_obs()["image"] for 7x7x3 semantic encoding
    state_encoding = env_unwrapped.gen_obs()["image"].tolist()

    arr = np.array(state_encoding, dtype=np.int64)
    visible_types = set()
    H, W, _ = arr.shape

    # Debug: check what objects are actually in the 7x7 observation
    unique_indices = set()
    unique_objects = set()
    door_positions = []
    goal_positions = []

    for i in range(H):
        for j in range(W):
            obj_idx = int(arr[i, j, 0])
            unique_indices.add(obj_idx)
            name = IDX_TO_OBJECT.get(obj_idx, f"unknown_{obj_idx}")
            if name and name != f"unknown_{obj_idx}":
                unique_objects.add(name)

            # Track door and goal positions
            if name == "door":
                door_positions.append((i, j, arr[i, j, 1], arr[i, j, 2]))  # color and state
            elif name == "goal":
                goal_positions.append((i, j))

            # Only include meaningful objects
            if name and name not in ("empty", "wall", "floor", "unseen"):
                visible_types.add(name)


    fobj = None
    if 0 <= front_pos[0] < env_unwrapped.width and 0 <= front_pos[1] < env_unwrapped.height:
        fobj = env_unwrapped.grid.get(front_pos[0], front_pos[1])

    if fobj is None:
        front_cell = {"pos": [int(front_pos[0]), int(front_pos[1])], "type": "empty", "color": "none", "state": 0}
    else:
        f_type = getattr(fobj, "type", "unknown")
        if f_type == "floor":
            f_type = "empty"
        front_cell = {
            "pos": [int(front_pos[0]), int(front_pos[1])],
            "type": f_type,
            "color": getattr(fobj, "color", "unknown"),
            "state": _door_state(fobj) if getattr(fobj, "type", None) == "door" else 0,
        }

    return {
        "agent": {
            "pos": [int(agent_pos[0]), int(agent_pos[1])],
            "dir": int(agent_dir),
            "carrying": (
                {
                    "type": str(getattr(env_unwrapped.carrying, "type", "unknown")),
                    "color": str(getattr(env_unwrapped.carrying, "color", "unknown")),
                }
                if getattr(env_unwrapped, "carrying", None) is not None
                else None
            ),
        },
        "agent_room": None,  # will be computed later
        "front_cell": front_cell,
        "objects": objects,
        "state_encoding": state_encoding,
        "visible_types_gt": sorted(list(visible_types)),
        "visible_types": sorted(list(visible_types)),
        "mask_applied": False,
        "mask_type": None,
    }

def rollout_from_current(
    env,
    obs0,
    actions: List[int],
    tile_size: int,
    out_dir: Path,
    rel_dir: str,
    grid_h: int,
    grid_w: int,
) -> Optional[Dict]:
    out_dir = out_dir / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    state_seq = []

    img0 = render_agent_pov(env, tile_size, grid_h, grid_w)
    if img0 is None:
        return None
    save_png(img0, out_dir / "step_000.png")
    frames.append(f"{rel_dir}/step_000.png")
    state_seq.append(extract_state(env.unwrapped, obs0))

    total_reward = 0.0
    terminated = False
    truncated = False

    obs = obs0
    for t, a in enumerate(actions):
        obs, r, term, trunc, _ = env.step(int(a))
        total_reward += float(r)
        terminated = bool(term)
        truncated = bool(trunc)

        img = render_agent_pov(env, tile_size, grid_h, grid_w)
        if img is None:
            return None
        save_png(img, out_dir / f"step_{t+1:03d}.png")
        frames.append(f"{rel_dir}/step_{t+1:03d}.png")
        state_seq.append(extract_state(env.unwrapped, obs))

        if terminated or truncated:
            break

    return {
        "frames": frames,
        "state_seq": state_seq,
        "reward": float(total_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(terminated and total_reward > 0),
    }

# -------------------------
# Planning (BFS with door handling)
# -------------------------
@dataclass(frozen=True)
class MRState:
    x: int
    y: int
    d: int

def _is_passable(env_unwrapped, x: int, y: int) -> bool:
    if not (0 <= x < env_unwrapped.width and 0 <= y < env_unwrapped.height):
        return False
    o = env_unwrapped.grid.get(x, y)
    if o is None:
        return True
    t = getattr(o, "type", None)
    if t in ("floor",):
        return True
    if t == "goal":
        return True
    if t == "door":
        return True  # For planning, assume doors are passable (will be opened when needed)
    return False

def _find_goal(env_unwrapped) -> Optional[Tuple[int, int]]:
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            c = env_unwrapped.grid.get(x, y)
            if c and getattr(c, "type", None) == "goal":
                return (x, y)
    return None

def plan_multiroom(env_unwrapped, max_nodes: int = 200000) -> Optional[List[int]]:
    goal_pos = _find_goal(env_unwrapped)
    if not goal_pos:
        return None
    start = MRState(int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1]), int(env_unwrapped.agent_dir))
    q = deque([start])
    parent: Dict[MRState, Tuple[Optional[MRState], Optional[int]]] = {start: (None, None)}
    nodes = 0; final_state = None

    while q:
        cur = q.popleft()
        nodes += 1
        if nodes > max_nodes:
            return None
        if (cur.x, cur.y) == goal_pos:
            final_state = cur
            break
        for act in (0, 1):
            nd = (cur.d - 1) % 4 if act == 0 else (cur.d + 1) % 4
            nxt = MRState(cur.x, cur.y, nd)
            if nxt not in parent:
                parent[nxt] = (cur, act)
                q.append(nxt)
        dx, dy = DIR_VEC[cur.d]
        nx, ny = cur.x + dx, cur.y + dy
        if _is_passable(env_unwrapped, nx, ny):
            nxt = MRState(nx, ny, cur.d)
            if nxt not in parent:
                parent[nxt] = (cur, 2)
                q.append(nxt)

    if final_state is None:
        return None
    raw = []
    cur = final_state
    while parent[cur][0] is not None:
        prev, act = parent[cur]
        raw.append(act)
        cur = prev
    raw.reverse()

    opened_doors = set()
    vx, vy, vd = start.x, start.y, start.d
    final_actions = []
    for act in raw:
        if act == 2:
            dx, dy = DIR_VEC[vd]
            nx, ny = vx + dx, vy + dy
            cell = env_unwrapped.grid.get(nx, ny)
            if cell and cell.type == "door":
                if (nx, ny) not in opened_doors and not getattr(cell, "is_open", False):
                    final_actions.append(5)  # toggle
                    opened_doors.add((nx, ny))
            final_actions.append(2)  # forward
            vx, vy = nx, ny
        elif act == 0:
            final_actions.append(0)  # left
            vd = (vd - 1) % 4
        elif act == 1:
            final_actions.append(1)  # right
            vd = (vd + 1) % 4
    return final_actions

# -------------------------
# Room computation
# -------------------------
def compute_room_labels(env_unwrapped):
    W, H = env_unwrapped.width, env_unwrapped.height
    goal_pos = _find_goal(env_unwrapped)
    if not goal_pos:
        raise ValueError("Goal not found")
    passable = np.zeros((W, H), dtype=bool)
    for x in range(W):
        for y in range(H):
            c = env_unwrapped.grid.get(x, y)
            passable[x, y] = (c is None) or (c.type not in ("wall", "door"))
    labels = -np.ones((W, H), dtype=int)
    rid = 0
    for sx in range(W):
        for sy in range(H):
            if passable[sx, sy] and labels[sx, sy] == -1:
                dq = deque([(sx, sy)])
                labels[sx, sy] = rid
                while dq:
                    x, y = dq.popleft()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < W and 0 <= ny < H and passable[nx, ny] and labels[nx, ny] == -1:
                            labels[nx, ny] = rid
                            dq.append((nx, ny))
                rid += 1
    gx, gy = goal_pos
    return labels, goal_pos, int(labels[gx, gy])

def find_enter_final_room_step(states, labels, goal_rid: int) -> Optional[int]:
    for t, st in enumerate(states):
        ax, ay = st["agent"]["pos"]
        if 0 <= ax < labels.shape[0] and 0 <= ay < labels.shape[1]:
            if int(labels[ax, ay]) == goal_rid:
                return t
    return None

# -------------------------
# NoCue masking
# -------------------------
def mask_targets_inplace(img: np.ndarray, state_encoding, targets: Set[str]) -> Tuple[int, int]:
    if state_encoding is None:
        return 0, 0
    enc = np.array(state_encoding, dtype=int)
    Ht, Wt, _ = enc.shape
    H, W = img.shape[0], img.shape[1]
    th, tw = H // Ht, W // Wt
    masked_count = target_count = 0
    for i in range(Ht):
        for j in range(Wt):
            name = IDX_TO_OBJECT.get(int(enc[i, j, 0]), None)
            if name in targets:
                target_count += 1
                # IMPORTANT: state_encoding axes are (obs_x, obs_y), not
                # conventional image (row, col). First axis maps to image x.
                x1, x2 = i * tw, (i + 1) * tw
                y1, y2 = j * th, (j + 1) * th
                img[y1:y2, x1:x2] = 0
                masked_count += 1
    return masked_count, target_count

def _has_targets(state_encoding, targets: Set[str]) -> bool:
    """Whether the 7x7x3 semantic encoding contains an object (optionally color-matched)."""
    if state_encoding is None:
        return False
    arr = np.array(state_encoding, dtype=np.int64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False
    Ht, Wt = arr.shape[0], arr.shape[1]
    for i in range(Ht):
        for j in range(Wt):
            if IDX_TO_OBJECT.get(int(arr[i, j, 0]), None) in targets:
                return True
    return False

# -------------------------
# CF: remove goal
# -------------------------
def remove_goal_cf(env_unwrapped) -> Optional[Dict]:
    gp = _find_goal(env_unwrapped)
    if not gp:
        return None
    env_unwrapped.grid.set(gp[0], gp[1], None)
    return {
        "cf_mode": "remove_goal",
        "goal_from": [int(gp[0]), int(gp[1])],
        "goal_to": None,
        "intervention_step": 0
    }

# -------------------------
# Main
# -------------------------
def generate():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default=str(_repo_root() / "datasets" / "raw_data" / "multiroom"),
    )
    ap.add_argument("--num", "--n", dest="num", type=int, default=50)
    ap.add_argument("--env-id", default="MiniGrid-MultiRoom-N6-v0")
    ap.add_argument("--tile-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--nocue-max-visible", type=int, default=12)
    ap.add_argument("--alignment-min", type=float, default=0.7)
    ap.add_argument("--mask-ratio-max", type=float, default=0.35)
    ap.add_argument("--mask-strength-target", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = _resolve_path(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "triplets.jsonl"
    skip_log_path = out_dir / "skip.log"
    prefix = "multiroom_s"

    if not args.resume:
        if jsonl_path.exists():
            jsonl_path.unlink()
        if skip_log_path.exists():
            skip_log_path.unlink()
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
    seed = (max(existing_seeds) + 1) if existing_seeds else args.seed_start
    while made < args.num:
        gid = f"{prefix}{made:06d}"
        final_root = out_dir / gid
        if final_root.exists():
            if args.resume:
                print(f"[RESUME] Skipping {gid}")
                made += 1
                continue

        work_dir = out_dir / "_tmp" / gid
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        env = None

        try:
            # ---- PLAN (separate env) ----
            env_plan = gym.make(args.env_id, render_mode="rgb_array")
            try:
                env_plan.reset(seed=seed)
                actions = plan_multiroom(env_plan.unwrapped)
            finally:
                env_plan.close()

            if not actions or len(actions) > args.max_steps:
                raise ValueError("Plan failed")

            # ---- FULL ----
            env = gym.make(args.env_id, render_mode="rgb_array")
            obs0, _ = env.reset(seed=seed)
            if "image" not in obs0:
                raise ValueError("Obs missing 'image'")
            gh, gw = obs0["image"].shape[:2]
            mission = getattr(env.unwrapped, "mission", "")
            full_traj = rollout_from_current(env, obs0, actions, args.tile_size, work_dir, "full", gh, gw)
            env.close()
            env = None

            if not full_traj or not full_traj["success"]:
                raise ValueError("Full rollout failed")
            if full_traj["truncated"]:
                raise ValueError("Full truncated")

            # Room analysis
            labels, _, goal_rid = compute_room_labels(env_plan.unwrapped)
            t_enter = find_enter_final_room_step(full_traj["state_seq"], labels, goal_rid)
            if t_enter is None:
                raise ValueError("Never entered goal room")

            # Semantic checks
            toggles = [t for t, a in enumerate(actions) if a == 5 and full_traj["state_seq"][t]["front_cell"]["type"] == "door"]
            if len(toggles) < 2:
                raise ValueError("Semantic fail: toggles < 2")

            # ---- CF ----
            cf_traj, cf_meta = None, None
            env_cf = gym.make(args.env_id, render_mode="rgb_array")
            try:
                env_cf.reset(seed=seed)
                meta = remove_goal_cf(env_cf.unwrapped)
                if not meta:
                    raise ValueError("CF: goal not found")
                obs_cf0 = env_cf.unwrapped.gen_obs()  # Re-fetch obs!
                traj = rollout_from_current(env_cf, obs_cf0, actions, args.tile_size, work_dir, "cf", gh, gw)
                if traj is None:
                    raise ValueError("CF render fail")

                goal_visible = any("goal" in st.get("visible_types_gt", []) for st in traj["state_seq"])
                if (not traj["success"]) and (traj["reward"] == 0.0) and (not traj["truncated"]) and (not goal_visible):
                    cf_traj, cf_meta = traj, meta
                else:
                    raise ValueError("CF gate not met (leaked goal or success)")
            finally:
                env_cf.close()

            # Physics check
            for t in range(len(full_traj["state_seq"])):
                if cf_traj["state_seq"][t]["agent"] != full_traj["state_seq"][t]["agent"]:
                    raise ValueError(f"CF physics mismatch t={t}")

            # ---- NOCUE ----
            targets = {"door", "goal"}
            masked, leak, d5_excluded = select_nocue_indices(
                state_seq=full_traj["state_seq"],
                first_interaction_step=int(t_enter),
                max_visible=int(args.nocue_max_visible),
                is_target_visible=lambda st: (
                    (("door" in set(st.get("visible_types_gt", []))) or ("goal" in set(st.get("visible_types_gt", []))))
                    and _has_targets(st.get("state_encoding"), targets)
                ),
                is_d5_excluded=lambda st: st["front_cell"]["type"] in targets,
            )

            (work_dir / "nocue").mkdir(parents=True, exist_ok=True)
            nocue_states = json.loads(json.dumps(full_traj["state_seq"]))
            per_frame_ratios = []
            align_ratios = []
            masked_tiles_total = 0

            for i, fp_rel in enumerate(full_traj["frames"]):
                src = work_dir / fp_rel
                dst = work_dir / "nocue" / Path(fp_rel).name

                with Image.open(src) as im:
                    img = np.array(im.convert("RGB"))
                st = nocue_states[i]
                if i in masked:
                    cnt, tgt = mask_targets_inplace(img, st.get("state_encoding"), targets)
                    masked_tiles_total += int(cnt)
                    per_frame_ratios.append(cnt / float(gh * gw))
                    align_ratios.append((cnt / tgt) if tgt > 0 else 0.0)

                    st["mask_applied"] = True
                    st["mask_type"] = "tile_suppression"
                    st["visible_types"] = sorted(list(set(st.get("visible_types_gt", [])) - targets))
                    save_png(img, dst)
                else:
                    shutil.copy2(src, dst)
                    st["mask_applied"] = False
                    st["mask_type"] = None
                    st["visible_types"] = st.get("visible_types_gt", [])

            metrics = compute_nocue_metrics(
                per_frame_mask_ratios=per_frame_ratios,
                alignment_ratios=align_ratios,
                masked_tiles_total=int(masked_tiles_total),
                gh=int(gh),
                gw=int(gw),
                masked_count=len(masked),
                mask_ratio_max=float(args.mask_ratio_max),
                alignment_min=float(args.alignment_min),
            )

            # NoCue_spec-aligned meta (extra keys allowed)
            mask_strength_target = float(args.mask_strength_target)
            nocue_meta = build_nocue_meta(
                targets=sorted(list(targets)),
                first_interaction_step=int(t_enter),
                masked_indices=masked,
                leak_indices=leak,
                d5_excluded_indices=d5_excluded,
                mask_strength_target=float(mask_strength_target),
                mask_strength_actual=float(np.mean(per_frame_ratios)) if per_frame_ratios else 0.0,
                mask_strength_threshold=float(args.mask_ratio_max),
                alignment_threshold=float(args.alignment_min),
                metrics=metrics,
                extra={"event_end_action_idx": int(t_enter)},
            )

            # ---- Finalize filesystem ----
            if final_root.exists():
                shutil.rmtree(final_root)
            shutil.move(str(work_dir), str(final_root))

            # ---- Write records ----
            base_rec = {
                "group_id": gid,
                "task": "multiroom",
                "env_id": args.env_id,
                "seed": int(seed),
                "actions_id": [int(a) for a in actions],
                "actions_text": [ACTION_ID2NAME[int(a)] for a in actions],
                "mission": str(mission),
                "model_input_fields": ALLOWED_INPUT_FIELDS,
            }

            nocue_traj_struct = {
                "frames": full_traj["frames"],
                "state_seq": nocue_states,
                "success": full_traj["success"],
                "reward": full_traj["reward"],
                "terminated": full_traj["terminated"],
                "truncated": full_traj["truncated"],
            }

            def mk_rec(variant: str, traj: Dict, meta_c=None, meta_n=None) -> Dict:
                return {
                    **base_rec,
                    "variant": variant,
                    "frames": [f"{gid}/{variant}/{Path(p).name}" for p in traj["frames"]],
                    "state_seq": traj["state_seq"],
                    "success": bool(traj["success"]),
                    "reward": float(traj["reward"]),
                    "terminated": bool(traj["terminated"]),
                    "truncated": bool(traj["truncated"]),
                    "cf_meta": meta_c,
                    "nocue_meta": meta_n,
                }

            records = [
                mk_rec("full", full_traj, None, None),
                mk_rec("nocue", nocue_traj_struct, None, nocue_meta) if nocue_traj_struct else None,
                mk_rec("cf", cf_traj, cf_meta, None) if cf_traj else None,
            ]
            records = [r for r in records if r is not None]
            enforce_triplet_hard_gates(records)
            append_jsonl_batch(jsonl_path, records)

            print(f"[{made+1}/{args.num}] Generated {gid} (seed={seed})")
            made += 1

        except ValueError:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        except Exception as e:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            tb = traceback.format_exc()
            reason = f"gid={gid} env_id={args.env_id} seed={seed}\n{tb}"
            append_skip_log(skip_log_path, seed, reason)

        seed += 1



# Private validation helpers must not shadow generator globals.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from utils.minigrid_checks import _die as _check__die
from utils.minigrid_checks import door_opens_after_action as _check_door_opens_after_action
from utils.minigrid_checks import door_state_at as _check_door_state_at
from utils.minigrid_checks import pixel_mask_from_tiles as _check__pixel_mask_from_tile_mask
from utils.minigrid_state import agent_trajectory as _check__agent_traj


import argparse as _check_argparse
import json as _check_json
import sys as _check_sys
from collections import defaultdict as _check_defaultdict
from pathlib import Path as _check_Path
from typing import Dict as _check_Dict, List as _check_List, Set as _check_Set, Tuple as _check_Tuple, Optional as _check_Optional

import numpy as _check_np
from PIL import Image as _check_Image

from minigrid.core.constants import OBJECT_TO_IDX as _check_OBJECT_TO_IDX, IDX_TO_OBJECT as _check_IDX_TO_OBJECT

# Clean model_input_fields: only what the model sees (no outcome leakage)
# SSOT §4.5 Physical Isolation: no terminated/truncated/reward/success
# Validate both supported model-input field sets.
_check_ALLOWED_INPUTS_CLEAN = ["frames", "mission"]
_check_ALLOWED_INPUTS_LEGACY = ["frames", "mission", "terminated", "truncated"]

# Union for backward compatibility during transition
_check_ALLOWED_INPUTS = _check_ALLOWED_INPUTS_CLEAN + [x for x in _check_ALLOWED_INPUTS_LEGACY if x not in _check_ALLOWED_INPUTS_CLEAN]
_check_MASK_RGB = (0, 0, 0)


def _check__load_png(path: _check_Path) -> _check_np.ndarray:
    return _check_np.array(_check_Image.open(path).convert("RGB"))

def _check__cue_tile_mask(state_encoding: _check_List[_check_List[_check_List[int]]], cue_type: str, cue_color: str) -> _check_np.ndarray:
    enc = _check_np.asarray(state_encoding, dtype=_check_np.uint8)  # (7,7,3)
    obj = _check_OBJECT_TO_IDX.get(cue_type, None)
    col = None  # We don't use color for multiroom since targets are door/goal
    if obj is None:
        return _check_np.zeros((7, 7), dtype=bool)
    if col is not None:
        return (enc[:, :, 0] == int(obj)) & (enc[:, :, 1] == int(col))
    return enc[:, :, 0] == int(obj)

def _check_get_tile_mask(enc, target_names: _check_Set[str]) -> _check_np.ndarray:
    arr = _check_np.array(enc, dtype=int)
    Ht, Wt = arr.shape[:2]
    mask = _check_np.zeros((Ht, Wt), dtype=bool)
    for i in range(Ht):
        for j in range(Wt):
            oid = int(arr[i, j, 0])
            obj_name = _check_IDX_TO_OBJECT.get(oid)
            if obj_name in target_names:
                mask[i, j] = True
    return mask


def _check_validate_state_encoding(enc) -> bool:
    if enc is None:
        return False
    arr = _check_np.array(enc)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False
    H, W, _ = arr.shape
    for i in range(H):
        for j in range(W):
            obj_i, col_i, st_i = int(arr[i, j, 0]), int(arr[i, j, 1]), int(arr[i, j, 2])
            if obj_i not in _check_OBJECT_TO_IDX:
                return False
            # color and state validation minimal here
    return True




def _check_check_door_semantics(full_rec: _check_Dict) -> _check_Tuple[bool, str]:
    actions = full_rec["actions_id"]
    states = full_rec["state_seq"]
    toggle_indices = [i for i, a in enumerate(actions) if a == 5]
    if len(toggle_indices) < 2:
        return False, "toggles_lt_2"
    for t in toggle_indices:
        fc = states[t]["front_cell"]
        if fc.get("type") != "door":
            return False, f"toggle_non_door_t{t}"
        door_pos = tuple(fc["pos"])
        if t + 1 >= len(states):
            return False, f"toggle_last_step_t{t}"
        post = states[t + 1]
        door_obj = next((o for o in post.get("objects", []) if tuple(o.get("pos", [])) == door_pos and o.get("type") == "door"), None)
        if door_obj is not None:
            if int(door_obj.get("state", 999)) != 0:
                return False, f"door_not_open_t{t}"
        else:
            fc2 = post["front_cell"]
            if tuple(fc2.get("pos")) == door_pos and fc2.get("type") == "door":
                if int(fc2.get("state", 999)) != 0:
                    return False, f"door_not_open_fc_t{t}"
            else:
                return False, f"door_obj_missing_t{t}"
    return True, "ok"

def _check__check_state_encoding_consistency(v: _check_Dict, v_name: str) -> _check_Tuple[bool, str, _check_Optional[_check_Tuple[int, int, int]]]:
    ref_shape = None
    for t, st in enumerate(v["state_seq"]):
        enc = st.get("state_encoding", [])
        arr = _check_np.array(enc)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return False, f"bad_state_enc_rank_{v_name}_t{t}", None
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            return False, f"state_enc_drift_{v_name}_t{t}", None
    return True, "ok", ref_shape

def _check_validate_group(group: _check_List[_check_Dict], root: _check_Path, audit_stats: _check_Dict) -> _check_Tuple[bool, str]:
    if len(group) != 3:
        return False, "group_len_not_3"
    full, nocue, cf = group[0], group[1], group[2]
    if [full.get("variant"), nocue.get("variant"), cf.get("variant")] != ["full", "nocue", "cf"]:
        return False, "bad_variant_order"
    gid = full.get("group_id")
    if gid is None or nocue.get("group_id") != gid or cf.get("group_id") != gid:
        return False, "group_id_mismatch"

    ref_shapes = []
    for v_name, v in [("full", full), ("nocue", nocue), ("cf", cf)]:
        mif = v.get("model_input_fields", None)
        # Accept either supported model-input field set.
        mif_set = set(mif) if mif else set()
        if not mif_set.issubset(set(_check_ALLOWED_INPUTS)):
            return False, f"bad_inputs_{v_name}"
        for k in mif:
            if k not in v:
                return False, f"missing_input_{k}_{v_name}"
        T = len(v.get("actions_id", []))
        if len(v.get("frames", [])) != T + 1:
            return False, f"frames_len_mismatch_{v_name}"
        if len(v.get("state_seq", [])) != T + 1:
            return False, f"state_seq_len_mismatch_{v_name}"
        ok, msg, ref_shape = _check__check_state_encoding_consistency(v, v_name)
        if not ok:
            return False, msg
        ref_shapes.append(ref_shape)

    if not (ref_shapes[0] == ref_shapes[1] == ref_shapes[2]):
        return False, "state_enc_shape_mismatch_across_variants"
    if not (full["actions_id"] == nocue["actions_id"] == cf["actions_id"]):
        return False, "action_mismatch"

    traj_f = _check__agent_traj(full)
    traj_n = _check__agent_traj(nocue)
    traj_c = _check__agent_traj(cf)
    if traj_f != traj_n:
        return False, "physics_mismatch_nocue"
    if traj_f != traj_c:
        return False, "physics_mismatch_cf"

    if not (full.get("success", False) and full.get("terminated", False)):
        return False, "full_not_success"
    if cf.get("success", False) or cf.get("terminated", False):
        return False, "cf_must_not_terminate"
    if float(cf.get("reward", 0.0)) != 0.0:
        return False, "cf_reward_nonzero"

    sem_ok, sem_msg = _check_check_door_semantics(full)
    if not sem_ok:
        return False, f"semantic_{sem_msg}"

    for st in cf["state_seq"]:
        if "goal" in st.get("visible_types_gt", []):
            return False, "cf_goal_visible"

    meta = nocue.get("nocue_meta", None)
    if not isinstance(meta, dict):
        return False, "missing_nocue_meta"
    if not all(k in meta for k in ["targets", "window_policy", "window_steps", "mask_strength_target", "mask_strength_actual", "alignment_score", "alignment_threshold", "masked_frames", "mask_type", "physics_check_passed", "masked_tiles_total", "mask_budget_tiles"]):
        return False, "missing_nocue_meta_keys"
    if meta.get("window_policy") != "EARLY":
        return False, "bad_window_policy"
    if not isinstance(meta.get("targets"), list) or set(meta.get("targets")) != {"door", "goal"}:
        return False, "nocue_targets_bad"
    if int(meta.get("masked_frames")) != int(len(meta.get("window_steps", []))):
        return False, "nocue_mask_count_mismatch"
    if float(meta.get("mask_strength_actual")) > float(meta.get("mask_strength_threshold")):
        return False, "nocue_mask_strength_exceeded"
    if float(meta.get("alignment_score")) < float(meta.get("alignment_threshold")):
        return False, "nocue_alignment_fail"

    window = set(meta.get("window_steps", []))
    t_enter = int(meta.get("event_end_action_idx", max(window) + 1))
    for t in window:
        if t >= t_enter:
            return False, "mask_outside_window"
        if full["state_seq"][t]["front_cell"].get("type") in ("door", "goal"):
            return False, "mask_interaction"

    for t in range(len(full["frames"])):
        fp_f = root / full["frames"][t]
        fp_n = root / nocue["frames"][t]
        if not fp_f.exists() or not fp_n.exists():
            return False, "missing_png"

        im_f = _check__load_png(fp_f)
        im_n = _check__load_png(fp_n)
        if im_f.shape != im_n.shape:
            return False, "shape_mismatch"

        diff = _check_np.any(im_f != im_n, axis=2)

        if t not in window:
            if _check_np.any(diff):
                return False, f"diff_unmasked_t{t}"
        else:
            enc = full["state_seq"][t]["state_encoding"]
            tile_mask = _check_get_tile_mask(enc, {"door", "goal"})
            pix_mask = _check__pixel_mask_from_tile_mask(tile_mask, im_f.shape[0], im_f.shape[1])
            if _check_np.any(diff & ~pix_mask):
                return False, f"leak_outside_target_t{t}"

    audit_stats["mask_counts"].append(len(window))
    audit_stats["mask_ratios"].append(float(meta.get("mask_strength_actual", 0)))
    audit_stats["toggles"].append(int(sum(1 for a in full["actions_id"] if a == 5)))
    return True, "pass"

def validate():
    ap = _check_argparse.ArgumentParser()
    repo_root = _check_Path(__file__).resolve().parents[3]
    default_root = repo_root / "datasets" / "raw_data" / "multiroom"
    ap.add_argument("--triplets", default=None, help="Path to triplets.jsonl (defaults to <root>/triplets.jsonl)")
    ap.add_argument("--root", default=str(default_root))
    ap.add_argument("--out-audit", default=None, help="Audit output path (defaults to <root>/audit.json)")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    root = _check_Path(args.root)
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    triplets = _check_Path(args.triplets) if args.triplets else (root / "triplets.jsonl")
    if not triplets.is_absolute():
        triplets = (repo_root / triplets).resolve()
    out_audit = _check_Path(args.out_audit) if args.out_audit else (root / "audit.json")
    if not out_audit.is_absolute():
        out_audit = (repo_root / out_audit).resolve()

    data = []
    with open(triplets, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(_check_json.loads(line))

    if len(data) % 3 != 0:
        _check__die("JSONL lines must be multiple of 3")

    groups = _check_defaultdict(list)
    for r in data:
        groups[r["group_id"]].append(r)

    audit = {"passed": 0, "failed": 0, "reasons": _check_defaultdict(int), "stats": {"mask_counts": [], "mask_ratios": [], "toggles": []}}

    for gid, group in groups.items():
        try:
            ok, msg = _check_validate_group(group, root, audit["stats"])
        except Exception as e:
            ok, msg = False, f"exception_{e}"
        if ok:
            audit["passed"] += 1
        else:
            audit["failed"] += 1
            audit["reasons"][msg] += 1

    summary = {
        "total": len(groups),
        "passed": audit["passed"],
        "failed": audit["failed"],
        "pass_rate": audit["passed"] / len(groups) if groups else 0,
        "avg_masks": _check_np.mean(audit["stats"]["mask_counts"]) if audit["stats"]["mask_counts"] else 0,
        "failure_modes": dict(audit["reasons"])
    }
    print("\n=== Audit Report ===")
    print(_check_json.dumps(summary, indent=2))
    with open(out_audit, "w") as f:
        _check_json.dump(summary, f, indent=2)



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
