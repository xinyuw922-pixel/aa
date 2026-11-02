"""doorkey: simulator trajectory generation and environment-specific validation."""
from __future__ import annotations

# Trajectory generation
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.minigrid_checks import check_triplet_alignment as enforce_triplet_hard_gates
from utils.minigrid_io import append_jsonl_fsync as append_jsonl_batch
from utils.minigrid_io import save_png as save_png


import argparse
import copy
import json
import shutil
import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from PIL import Image
import gymnasium as gym

# MiniGrid imports
from minigrid.core.constants import IDX_TO_OBJECT, OBJECT_TO_IDX
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
# Constants & Helpers
# -------------------------

ACTION_ID2NAME = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}
DIR2VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}




def render_agent_pov(env, tile_size: int, grid_h: int, grid_w: int) -> Optional[np.ndarray]:
    try:
        frame = env.unwrapped.get_frame(tile_size=tile_size, agent_pov=True)
        # Dynamic shape check
        if isinstance(frame, np.ndarray) and frame.shape[0] == grid_h * tile_size and frame.shape[1] == grid_w * tile_size:
            return frame
    except Exception:
        pass
    return None

def _door_state(door_obj) -> int:
    if getattr(door_obj, "is_locked", False): return 2
    if getattr(door_obj, "is_open", False): return 0
    return 1

def extract_state(env_unwrapped, obs: Dict) -> Dict:
    img_enc = obs.get("image", None)
    visible_types_gt: Set[str] = set()
    if img_enc is not None:
        H, W = img_enc.shape[0], img_enc.shape[1]
        for i in range(H):
            for j in range(W):
                obj_idx = int(img_enc[i, j, 0])
                name = IDX_TO_OBJECT.get(obj_idx, None)
                if name and name not in ("unseen", "empty", "wall", "floor"):
                    visible_types_gt.add(name)

    objects = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            cell = env_unwrapped.grid.get(x, y)
            if cell is None: continue
            t = getattr(cell, "type", "unknown")
            if t in ("wall", "floor"): continue
            o = {"type": t, "pos": [int(x), int(y)]}
            if hasattr(cell, "color"): o["color"] = str(getattr(cell, "color"))
            if t == "door": o["state"] = _door_state(cell)
            objects.append(o)

    fx, fy = env_unwrapped.front_pos
    f_cell = env_unwrapped.grid.get(fx, fy)
    f_type = getattr(f_cell, "type", "empty") if f_cell else "empty"
    if f_type == "floor":
        f_type = "empty"
    front_info = {"pos": [int(fx), int(fy)], "type": str(f_type)}
    if f_cell and f_type == "door":
        front_info["state"] = _door_state(f_cell)

    return {
        "agent": {
            "pos": [int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1])],
            "dir": int(env_unwrapped.agent_dir),
            "carrying": {"type": str(env_unwrapped.carrying.type), "color": str(env_unwrapped.carrying.color)} if env_unwrapped.carrying else None
        },
        "front_cell": front_info,
        "visible_types_gt": sorted(list(visible_types_gt)),
        "visible_types": sorted(list(visible_types_gt)),
        "state_encoding": img_enc.astype(int).tolist() if img_enc is not None else None,
        "objects": objects,
        "mask_applied": False,
        "mask_type": None
    }

# -------------------------
# Planning & Rollout
# -------------------------

@dataclass(frozen=True)
class DKState:
    ax: int; ay: int; ad: int; has_key: bool; door_state: int; key_present: bool

def _find_key_door_goal(env_unwrapped):
    key_pos = door_pos = goal_pos = None
    key_obj = door_obj = None
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            c = env_unwrapped.grid.get(x, y)
            if c:
                if c.type == "key": key_pos = (x, y); key_obj = c
                elif c.type == "door": door_pos = (x, y); door_obj = c
                elif c.type == "goal": goal_pos = (x, y)
    return key_pos, key_obj, door_pos, door_obj, goal_pos

def plan_doorkey_bfs(env_unwrapped, max_nodes=200000) -> Optional[List[int]]:
    key_pos, _, door_pos, door_obj, goal_pos = _find_key_door_goal(env_unwrapped)
    if not (key_pos and door_pos and goal_pos and door_obj): return None

    start_node = DKState(int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1]), int(env_unwrapped.agent_dir), False, _door_state(door_obj), True)
    from collections import deque
    q = deque([start_node]); visited = {start_node}; parent = {start_node: (None, None)}

    steps = 0
    while q:
        curr = q.popleft()
        steps += 1
        if steps > max_nodes: return None
        if (curr.ax, curr.ay) == goal_pos:
            path = []
            while parent[curr][0]:
                prev, act = parent[curr]; path.append(act); curr = prev
            return list(reversed(path))

        for act in (0, 1):
            nd = (curr.ad - 1) % 4 if act == 0 else (curr.ad + 1) % 4
            nxt = DKState(curr.ax, curr.ay, nd, curr.has_key, curr.door_state, curr.key_present)
            if nxt not in visited: visited.add(nxt); parent[nxt] = (curr, act); q.append(nxt)
        dx, dy = DIR2VEC[curr.ad]; nx, ny = curr.ax + dx, curr.ay + dy
        if 0 <= nx < env_unwrapped.width and 0 <= ny < env_unwrapped.height:
            passable = True
            if (nx, ny) == door_pos and curr.door_state != 0: passable = False
            if (nx, ny) == key_pos and curr.key_present: passable = False
            cell = env_unwrapped.grid.get(nx, ny);
            if cell and cell.type == "wall": passable = False
            if passable:
                nxt = DKState(nx, ny, curr.ad, curr.has_key, curr.door_state, curr.key_present)
                if nxt not in visited: visited.add(nxt); parent[nxt] = (curr, 2); q.append(nxt)
        if (nx, ny) == key_pos and curr.key_present and not curr.has_key:
            nxt = DKState(curr.ax, curr.ay, curr.ad, True, curr.door_state, False)
            if nxt not in visited: visited.add(nxt); parent[nxt] = (curr, 3); q.append(nxt)
        if (nx, ny) == door_pos:
            if (curr.door_state == 2 and curr.has_key) or curr.door_state == 1:
                nxt = DKState(curr.ax, curr.ay, curr.ad, curr.has_key, 0, curr.key_present)
                if nxt not in visited: visited.add(nxt); parent[nxt] = (curr, 5); q.append(nxt)
    return None

def rollout_from_current(env, obs0: Dict, actions: List[int], tile_size: int, out_dir: Path, rel_dir: str, grid_h: int, grid_w: int) -> Optional[Dict]:
    frames, states = [], []
    reward_sum = 0.0
    terminated = truncated = False

    frame0 = render_agent_pov(env, tile_size, grid_h, grid_w)
    if frame0 is None: return None
    save_png(frame0, out_dir / rel_dir / "step_000.png")
    frames.append(str(Path(rel_dir) / "step_000.png"))
    states.append(extract_state(env.unwrapped, obs0))

    obs = obs0
    for t, a in enumerate(actions):
        obs, r, term, trunc, _ = env.step(a)
        reward_sum += float(r)
        terminated |= bool(term); truncated |= bool(trunc)
        frame = render_agent_pov(env, tile_size, grid_h, grid_w)
        if frame is None: return None
        save_png(frame, out_dir / rel_dir / f"step_{t+1:03d}.png")
        frames.append(str(Path(rel_dir) / f"step_{t+1:03d}.png"))
        states.append(extract_state(env.unwrapped, obs))
        if term or trunc: break

    return {
        "frames": frames, "state_seq": states, "reward": reward_sum,
        "success": bool(terminated and reward_sum > 0),
        "terminated": terminated, "truncated": truncated
    }

def mask_key_tiles_inplace(pov_img: np.ndarray, state_encoding: List, threshold_obj_name="key") -> Tuple[int, int]:
    if state_encoding is None: return 0, 0
    enc = np.array(state_encoding, dtype=int)
    Ht, Wt = enc.shape[0], enc.shape[1]
    H, W = pov_img.shape[0], pov_img.shape[1]
    tile_h, tile_w = H // Ht, W // Wt
    masked_count, target_count = 0, 0
    for i in range(Ht):
        for j in range(Wt):
            if IDX_TO_OBJECT.get(int(enc[i, j, 0])) == threshold_obj_name:
                target_count += 1
                # state_encoding uses (vx, vy); image tiles use (row=y, col=x)
                y1, y2 = j * tile_h, (j + 1) * tile_h
                x1, x2 = i * tile_w, (i + 1) * tile_w
                pov_img[y1:y2, x1:x2] = 0
                masked_count += 1
    return masked_count, target_count

def relocate_goal_cf(env_unwrapped, forbidden: Set[Tuple[int, int]], rng: np.random.Generator) -> Optional[Dict]:
    _, _, _, _, goal_pos = _find_key_door_goal(env_unwrapped)
    if not goal_pos: return None
    cands = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            if (x,y) != goal_pos and (x,y) not in forbidden and env_unwrapped.grid.get(x, y) is None:
                cands.append((x,y))
    if not cands: return None
    new_pos = cands[rng.integers(0, len(cands))]
    env_unwrapped.grid.set(goal_pos[0], goal_pos[1], None)
    env_unwrapped.grid.set(new_pos[0], new_pos[1], Goal())
    return {"cf_mode": "move_goal", "goal_from": list(goal_pos), "goal_to": list(new_pos), "intervention_step": 0}

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
        default=str(_repo_root() / "datasets" / "raw_data" / "doorkey"),
    )
    ap.add_argument("--num", "--n", dest="num", type=int, default=50)
    ap.add_argument("--env-id", type=str, default="MiniGrid-DoorKey-8x8-v0")
    ap.add_argument("--tile-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--nocue-max-visible", type=int, default=5)
    ap.add_argument("--alignment-min", type=float, default=0.7)
    ap.add_argument("--mask-ratio-max", type=float, default=0.35)
    ap.add_argument("--resume", action="store_true", help="Resume from existing directory (skip existing seeds)")
    args = ap.parse_args()

    out_dir = _resolve_path(Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "triplets.jsonl"

    prefix = "doorkey_s"

    # Resume Logic: Only delete jsonl if NOT resuming
    if not args.resume:
        if jsonl_path.exists(): jsonl_path.unlink()
        for child in out_dir.iterdir():
            if child.is_dir() and child.name.startswith(prefix):
                shutil.rmtree(child, ignore_errors=True)
        tmp_root = out_dir / "_tmp"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)

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

    made_count = len(existing_group_dirs) if args.resume else 0
    current_seed = (max(existing_seeds) + 1) if existing_seeds else args.seed_start

    while made_count < args.num:
        gid = f"{prefix}{made_count:06d}"
        final_root = out_dir / gid

        # Resume Check
        if final_root.exists():
            if args.resume:
                print(f"[RESUME] Skipping existing {gid}")
                made_count += 1
                continue
            else:
                shutil.rmtree(final_root)

        work_dir = out_dir / "_tmp" / f"{gid}"
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)

        env = None
        try:
            # 1. FULL
            env = gym.make(args.env_id, render_mode="rgb_array")
            obs0, _ = env.reset(seed=current_seed)
            gh, gw = obs0["image"].shape[0], obs0["image"].shape[1]

            actions = plan_doorkey_bfs(env.unwrapped)
            if not actions or len(actions) > args.max_steps: raise ValueError("Plan failed")

            full_traj = rollout_from_current(env, obs0, actions, args.tile_size, work_dir, "full", gh, gw)
            mission = env.unwrapped.mission
            env.close(); env = None # Release Resource

            if not full_traj or not full_traj["success"]: raise ValueError("Full rollout failed")

            pickup_step = next((t for t, a in enumerate(actions) if a == 3 and full_traj["state_seq"][t]["front_cell"]["type"] == "key"), None)
            toggle_step = next((t for t, a in enumerate(actions) if a == 5 and full_traj["state_seq"][t]["front_cell"]["type"] == "door"), None)
            if pickup_step is None or toggle_step is None: raise ValueError("Semantic events missing")

            # 2. CF
            forbidden = {tuple(st["agent"]["pos"]) for st in full_traj["state_seq"]} | \
                        {tuple(st["front_cell"]["pos"]) for st in full_traj["state_seq"]}

            cf_traj, cf_meta = None, None
            for k in range(20):
                # Clean CF dir before each attempt
                cf_path = work_dir / "cf"
                if cf_path.exists(): shutil.rmtree(cf_path)

                env_cf = gym.make(args.env_id, render_mode="rgb_array")
                try:
                    env_cf.reset(seed=current_seed)
                    rng = np.random.default_rng(current_seed + 1000 + k)
                    meta = relocate_goal_cf(env_cf.unwrapped, forbidden, rng)
                    if meta:
                        obs_cf = env_cf.unwrapped.gen_obs()
                        traj = rollout_from_current(env_cf, obs_cf, actions, args.tile_size, work_dir, "cf", gh, gw)

                        if traj is None: continue

                        # CF Gate: Reward 0, No Truncation, No Goal Visibility
                        goal_visible = any("goal" in st["visible_types_gt"] for st in traj["state_seq"])
                        if (not traj["success"] and
                            traj["reward"] == 0 and
                            not traj["truncated"] and
                            not goal_visible):
                            cf_traj, cf_meta = traj, meta
                            break
                finally:
                    env_cf.close() # Always close CF env

            if not cf_traj: raise ValueError("CF generation failed")

            # 3. NOCUE
            first_interaction_step = int(pickup_step)
            masked_indices, leak_indices, d5_excluded_indices = select_nocue_indices(
                state_seq=full_traj["state_seq"],
                first_interaction_step=first_interaction_step,
                max_visible=args.nocue_max_visible,
                is_target_visible=lambda st: ("key" in st.get("visible_types_gt", [])),
                is_d5_excluded=lambda st: st["front_cell"]["type"] == "key",
            )

            (work_dir / "nocue").mkdir(parents=True, exist_ok=True)
            nocue_states = copy.deepcopy(full_traj["state_seq"])
            per_frame_mask_ratios, alignment_ratios, masked_tiles_total = [], [], 0

            for i, fp_rel in enumerate(full_traj["frames"]):
                # FIX: Context Manager for FD Safety
                with Image.open(work_dir / fp_rel) as im:
                    img = np.array(im.convert("RGB"))

                st = nocue_states[i]
                if i in masked_indices:
                    cnt, tgt = mask_key_tiles_inplace(img, st["state_encoding"], "key")
                    masked_tiles_total += cnt
                    per_frame_mask_ratios.append(cnt / (gh*gw))
                    alignment_ratios.append(cnt/tgt if tgt > 0 else 0.0)
                    st["mask_applied"] = True
                    st["mask_type"] = "tile_suppression"
                    st["visible_types"] = [v for v in st.get("visible_types_gt", []) if v != "key"]
                else:
                    st["mask_applied"] = False; st["mask_type"] = None; st["visible_types"] = st["visible_types_gt"]
                save_png(img, work_dir / "nocue" / f"step_{i:03d}.png")

            mask_strength_target = 1.0 / float(gh * gw)
            metrics = compute_nocue_metrics(
                per_frame_mask_ratios=per_frame_mask_ratios,
                alignment_ratios=alignment_ratios,
                masked_tiles_total=int(masked_tiles_total),
                gh=int(gh),
                gw=int(gw),
                masked_count=len(masked_indices),
                mask_ratio_max=float(args.mask_ratio_max),
                alignment_min=float(args.alignment_min),
            )
            nocue_meta = build_nocue_meta(
                targets=["key"],
                first_interaction_step=int(first_interaction_step),
                masked_indices=masked_indices,
                leak_indices=leak_indices,
                d5_excluded_indices=d5_excluded_indices,
                mask_strength_target=float(mask_strength_target),
                mask_strength_actual=float(np.mean(per_frame_mask_ratios)) if per_frame_mask_ratios else 0.0,
                mask_strength_threshold=float(args.mask_ratio_max),
                alignment_threshold=float(args.alignment_min),
                metrics=metrics,
                extra={"max_visible_frames": int(args.nocue_max_visible)},
            )

            # Finalize
            if final_root.exists(): shutil.rmtree(final_root)
            shutil.move(str(work_dir), str(final_root))

            # Record
            base_rec = {
                "group_id": gid, "task": "doorkey", "env_id": args.env_id, "seed": int(current_seed),
                "actions_id": actions, "actions_text": [ACTION_ID2NAME[a] for a in actions],
                "mission": mission, "model_input_fields": ["frames", "mission"]
            }

            nocue_traj_struct = {
                "frames": full_traj["frames"], "state_seq": nocue_states,
                "success": True, "reward": full_traj["reward"],
                "terminated": full_traj["terminated"], "truncated": full_traj["truncated"]
            }

            def mk_rec(var, traj, meta_c=None, meta_n=None):
                return {**base_rec, "variant": var,
                        "frames": [f"{gid}/{var}/{Path(p).name}" for p in traj["frames"]],
                        "state_seq": traj["state_seq"], "success": traj["success"], "reward": traj["reward"],
                        "terminated": traj["terminated"], "truncated": traj["truncated"], "cf_meta": meta_c, "nocue_meta": meta_n}

            records = [
                mk_rec("full", full_traj),
                mk_rec("nocue", nocue_traj_struct, None, nocue_meta),
                mk_rec("cf", cf_traj, cf_meta, None)
            ]
            enforce_triplet_hard_gates(records)
            append_jsonl_batch(jsonl_path, records)

            print(f"[{made_count+1}/{args.num}] Generated {gid} (seed={current_seed})")
            made_count += 1

        except ValueError:
            if work_dir.exists(): shutil.rmtree(work_dir)
            if env: env.close()
        except Exception as e:
            if work_dir.exists(): shutil.rmtree(work_dir)
            if env: env.close()
            print(f"Error seed {current_seed}: {e}")

        current_seed += 1



# Private validation helpers must not shadow generator globals.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse as _check_argparse
import json as _check_json
from collections import defaultdict as _check_defaultdict
from pathlib import Path as _check_Path
from typing import Dict as _check_Dict, List as _check_List, Optional as _check_Optional

import numpy as _check_np
from PIL import Image as _check_Image
from minigrid.core.constants import OBJECT_TO_IDX as _check_OBJECT_TO_IDX, IDX_TO_OBJECT as _check_IDX_TO_OBJECT

# Clean model_input_fields: only what the model sees (no outcome leakage)
# SSOT §4.5 Physical Isolation: no terminated/truncated/reward/success
# Validate both supported model-input field sets.
_check_ALLOWED_INPUTS_CLEAN = {"frames", "mission"}
_check_ALLOWED_INPUTS_LEGACY = {"frames", "mission", "terminated", "truncated"}

# Union for backward compatibility during transition
_check_ALLOWED_INPUTS = _check_ALLOWED_INPUTS_CLEAN | _check_ALLOWED_INPUTS_LEGACY
_check_PROHIBITED_INPUTS = {"state_seq", "actions_id", "success", "reward", "cf_meta", "nocue_meta", "seed"}

def _check_load_jsonl(path: _check_Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                out.append(_check_json.loads(s))
            except _check_json.JSONDecodeError:
                # Crash tolerance: stop at partial line
                print("Warning: Stopped loading at corrupt JSON line.")
                break
    return out

def _check__agent_traj(rec):
    return [(tuple(st["agent"]["pos"]), int(st["agent"]["dir"])) for st in rec["state_seq"]]

def _check_validate_state_encoding(enc):
    if enc is None: return False
    arr = _check_np.array(enc)
    if len(arr.shape) != 3 or arr.shape[2] != 3: return False
    if not _check_np.issubdtype(arr.dtype, _check_np.integer): return False
    H, W, _ = arr.shape
    for i in range(H):
        for j in range(W):
            obj_idx, _, state = map(int, arr[i, j])
            if obj_idx not in _check_IDX_TO_OBJECT: return False
            name = _check_IDX_TO_OBJECT[obj_idx]
            if name == "door" and state not in (0, 1, 2): return False
            if name not in ("door", "empty", "unseen") and state != 0: return False
    return True

def _check_get_tile_mask(enc, target_name="key"):
    arr = _check_np.array(enc)
    H, W, _ = arr.shape
    mask = _check_np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            if _check_IDX_TO_OBJECT.get(arr[i, j, 0]) == target_name:
                mask[i, j] = True
    return mask

def _check__pixel_mask_from_tile_mask(tile_mask: _check_np.ndarray, H: int, W: int) -> _check_np.ndarray:
    # state_encoding uses (vx, vy); image tiles use (row=y, col=x)
    tile_mask = tile_mask.T
    Ht, Wt = tile_mask.shape
    th, tw = H // Ht, W // Wt
    return _check_np.kron(tile_mask, _check_np.ones((th, tw), dtype=bool))[:H, :W]

def _check_door_open_after_toggle(states, toggle_t):
    if toggle_t + 1 >= len(states): return False
    s = states[toggle_t + 1]["front_cell"].get("state", None)
    if s == 0: return True
    for o in states[toggle_t + 1].get("objects", []):
        if o.get("type") == "door" and o.get("state") == 0: return True
    return False

def _check_validate_group(gid, full, nocue, cf, root: _check_Path, audit_stats: _check_Dict):
    # D1 Action Identity
    if full["actions_id"] != nocue["actions_id"] or full["actions_id"] != cf["actions_id"]:
        return False, "action_mismatch"

    for v_name, v in [("full", full), ("nocue", nocue), ("cf", cf)]:
        if len(v["frames"]) != len(v["actions_id"]) + 1: return False, f"{v_name}_len_mismatch"
        if len(v["state_seq"]) != len(v["actions_id"]) + 1: return False, f"{v_name}_state_len_mismatch"

        mif = set(v.get("model_input_fields", []))
        # Accept either supported model-input field set.
        if not mif.issubset(_check_ALLOWED_INPUTS): return False, f"{v_name}_bad_contract_fields"
        if not all(k in v for k in mif): return False, f"{v_name}_missing_contract_data"
        if any(k in v.keys() and k in _check_PROHIBITED_INPUTS for k in mif): return False, f"{v_name}_info_leak"

        for st in v["state_seq"]:
            if not _check_validate_state_encoding(st.get("state_encoding")): return False, f"{v_name}_invalid_encoding"

    # Termination
    if full["terminated"] is not True: return False, "full_not_terminated"
    if cf["terminated"] is True: return False, "cf_terminated_error"
    if cf["truncated"] is True: return False, "cf_truncated_error"
    if nocue["terminated"] != full["terminated"]: return False, "nocue_term_mismatch"

    # Outcomes
    if not full["success"] or not nocue["success"] or cf["success"]: return False, "outcome_fail"
    if float(cf.get("reward", 0)) != 0.0: return False, "cf_reward_nonzero"

    # D6 Physics
    ft, nt, ct = _check__agent_traj(full), _check__agent_traj(nocue), _check__agent_traj(cf)
    if ft != nt or ft != ct: return False, "physics_drift"

    # E1 Semantics
    actions, states = full["actions_id"], full["state_seq"]
    pickup_t = next((t for t, a in enumerate(actions) if a == 3 and states[t]["front_cell"]["type"] == "key"), None)
    toggle_t = next((t for t, a in enumerate(actions) if a == 5 and states[t]["front_cell"]["type"] == "door"), None)
    if pickup_t is None or toggle_t is None or pickup_t > toggle_t: return False, "semantic_seq_fail"
    if states[toggle_t]["front_cell"].get("state") != 2: return False, "door_not_locked"
    if not _check_door_open_after_toggle(states, toggle_t): return False, "door_not_open"

    # CF Observability
    for st in cf["state_seq"]:
        if "goal" in st["visible_types_gt"]: return False, "cf_goal_visible"
    cfm = cf.get("cf_meta", {})
    goal_to = tuple(cfm.get("goal_to", []))
    if goal_to in {pos for pos, _ in ft}: return False, "cf_goal_on_path"

    # NoCue Hard Gates
    meta = nocue.get("nocue_meta", {})
    masked = meta.get("window_steps", [])
    if not masked: return False, "empty_mask"

    if int(meta.get("masked_tiles_total", 10**9)) > int(meta.get("mask_budget_tiles", -1)):
        return False, "budget_exceeded"

    # Diff-in-Mask
    for idx in masked:
        fp_f, fp_n = root / full["frames"][idx], root / nocue["frames"][idx]
        if not fp_f.exists() or not fp_n.exists(): return False, "missing_png"

        # FIX: FD Safety here too
        with _check_Image.open(fp_f) as imf, _check_Image.open(fp_n) as imn:
            arr_f, arr_n = _check_np.array(imf), _check_np.array(imn)

        diff = _check_np.any(arr_f != arr_n, axis=2)

        enc = full["state_seq"][idx]["state_encoding"]
        tile_mask_bool = _check_get_tile_mask(enc, "key")
        H, W = arr_f.shape[:2]
        pixel_mask = _check__pixel_mask_from_tile_mask(tile_mask_bool, H, W)

        if _check_np.any(diff & ~pixel_mask): return False, "diff_outside_target_tile"
        if not _check_np.any(diff & pixel_mask): return False, "no_diff_inside_target"

    audit_stats["mask_counts"].append(len(masked))
    return True, "pass"

def validate():
    ap = _check_argparse.ArgumentParser()
    repo_root = _check_Path(__file__).resolve().parents[3]
    default_root = repo_root / "datasets" / "raw_data" / "doorkey"
    ap.add_argument("--triplets", type=str, default=None, help="Path to triplets.jsonl (defaults to <root>/triplets.jsonl)")
    ap.add_argument("--root", type=str, default=str(default_root))
    ap.add_argument("--out-audit", type=str, default=None, help="Audit output path (defaults to <root>/audit_report.json)")
    args = ap.parse_args()

    root = _check_Path(args.root)
    if not root.is_absolute():
        root = (repo_root / root).resolve()
    triplets = _check_Path(args.triplets) if args.triplets else (root / "triplets.jsonl")
    if not triplets.is_absolute():
        triplets = (repo_root / triplets).resolve()
    out_audit = _check_Path(args.out_audit) if args.out_audit else (root / "audit_report.json")
    if not out_audit.is_absolute():
        out_audit = (repo_root / out_audit).resolve()

    data = _check_load_jsonl(triplets)
    groups = _check_defaultdict(dict)
    for r in data: groups[r["group_id"]][r["variant"]] = r

    audit = {"passed": 0, "failed": 0, "reasons": _check_defaultdict(int), "mask_counts": []}
    for gid, vs in groups.items():
        if not {"full", "nocue", "cf"}.issubset(vs.keys()):
            audit["failed"] += 1; audit["reasons"]["missing_variant"] += 1; continue
        ok, reason = _check_validate_group(gid, vs["full"], vs["nocue"], vs["cf"], root, audit)
        if ok: audit["passed"] += 1
        else: audit["failed"] += 1; audit["reasons"][reason] += 1; print(f"FAIL {gid}: {reason}")

    with open(out_audit, "w") as f: _check_json.dump(audit, f, indent=2)
    print(f"Audit: {audit['passed']} Passed, {audit['failed']} Failed")



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
