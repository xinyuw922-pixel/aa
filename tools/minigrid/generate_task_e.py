#!/usr/bin/env python3
"""
Generate solved MiniGrid object-manipulation trajectories for Task E.

Plan actions, render observations, record simulator states, and select
trajectories with valid object identity/status checkpoints.
"""

from __future__ import annotations

import argparse
import json
import time
import heapq
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from PIL import Image

from tools.minigrid.task_e import normalize_trace, extract_events, classify_status, assign_horizon_bucket, pick_identity_steps, pick_status_steps, build_v12_row


DIR_VEC = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


def save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def render_agent_pov(env: Any, tile_size: int = 32) -> np.ndarray:
    frame = env.unwrapped.get_frame(tile_size=tile_size, agent_pov=True)
    return np.asarray(frame, dtype=np.uint8)


def door_state(door_obj: Any) -> int:
    if getattr(door_obj, "is_locked", False):
        return 2
    if getattr(door_obj, "is_open", False):
        return 0
    return 1


def extract_state(env_unwrapped: Any, obs: Dict[str, Any]) -> Dict[str, Any]:
    img_enc = obs.get("image")
    visible_types_gt = set()
    if img_enc is not None:
        h, w = img_enc.shape[0], img_enc.shape[1]
        for i in range(h):
            for j in range(w):
                obj_idx = int(img_enc[i, j, 0])
                if obj_idx == 0:
                    continue
                name = {
                    1: "empty",
                    2: "wall",
                    4: "door",
                    5: "key",
                    6: "ball",
                    7: "box",
                    8: "goal",
                    10: "agent",
                }.get(obj_idx)
                if name and name not in ("empty", "wall", "agent"):
                    visible_types_gt.add(name)

    objects = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            cell = env_unwrapped.grid.get(x, y)
            if cell is None:
                continue
            t = getattr(cell, "type", "unknown")
            if t in ("wall", "floor"):
                continue
            obj = {"type": str(t), "pos": [int(x), int(y)]}
            if hasattr(cell, "color"):
                obj["color"] = str(getattr(cell, "color"))
            if t == "door":
                obj["state"] = door_state(cell)
            if t == "box" and getattr(cell, "contains", None) is not None:
                inner = cell.contains
                obj["contains"] = {
                    "type": str(getattr(inner, "type", "unknown")),
                    "color": str(getattr(inner, "color", "unknown")),
                }
            objects.append(obj)

    fx, fy = env_unwrapped.front_pos
    front = env_unwrapped.grid.get(int(fx), int(fy))
    front_type = getattr(front, "type", "empty") if front is not None else "empty"
    front_info = {"pos": [int(fx), int(fy)], "type": str(front_type)}
    if front is not None and front_type == "door":
        front_info["state"] = door_state(front)
        front_info["color"] = str(getattr(front, "color", "unknown"))
    elif front is not None and hasattr(front, "color"):
        front_info["color"] = str(getattr(front, "color"))

    carrying = None
    if env_unwrapped.carrying is not None:
        carrying = {
            "type": str(getattr(env_unwrapped.carrying, "type", "unknown")),
            "color": str(getattr(env_unwrapped.carrying, "color", "unknown")),
        }

    return {
        "agent": {
            "pos": [int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1])],
            "dir": int(env_unwrapped.agent_dir),
            "carrying": carrying,
        },
        "front_cell": front_info,
        "visible_types_gt": sorted(visible_types_gt),
        "visible_types": sorted(visible_types_gt),
        "state_encoding": img_enc.astype(int).tolist() if img_enc is not None else None,
        "objects": objects,
        "mask_applied": False,
        "mask_type": None,
    }


def carrying_label(obj: Any) -> str:
    if obj is None:
        return "none"
    if isinstance(obj, dict):
        return f"{obj.get('type', 'unknown')}_{obj.get('color', 'unknown')}"
    return str(obj)


def is_walkable(env_unwrapped: Any, pos: Tuple[int, int]) -> bool:
    x, y = pos
    if not (0 <= x < env_unwrapped.width and 0 <= y < env_unwrapped.height):
        return False
    obj = env_unwrapped.grid.get(int(x), int(y))
    if obj is None:
        return True
    t = getattr(obj, "type", None)
    if t in ("floor", "goal"):
        return True
    if t == "door":
        return not getattr(obj, "is_locked", False)
    return False


def plan_path(
    env_unwrapped: Any,
    start_pos: Tuple[int, int],
    start_dir: int,
    goal_pos: Tuple[int, int],
    max_nodes: int = 200000,
    timeout_s: float = 2.0,
) -> Optional[List[int]]:
    start = (int(start_pos[0]), int(start_pos[1]), int(start_dir))
    goal = (int(goal_pos[0]), int(goal_pos[1]))

    def heuristic(state: Tuple[int, int, int]) -> int:
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

    frontier: List[Tuple[int, int, Tuple[int, int, int]]] = []
    heapq.heappush(frontier, (heuristic(start), 0, start))
    parent: Dict[Tuple[int, int, int], Tuple[Optional[Tuple[int, int, int]], Optional[int]]] = {
        start: (None, None)
    }
    g_score: Dict[Tuple[int, int, int], int] = {start: 0}
    expanded = 0
    t0 = time.perf_counter()

    while frontier:
        if time.perf_counter() - t0 > timeout_s:
            return None
        _f, g, current = heapq.heappop(frontier)
        if g != g_score.get(current):
            continue
        expanded += 1
        if expanded > max_nodes:
            return None

        x, y, d = current
        if (x, y) == goal:
            actions: List[int] = []
            node = current
            while parent[node][0] is not None:
                prev, act = parent[node]
                assert act is not None
                actions.append(int(act))
                node = prev  # type: ignore[assignment]
            actions.reverse()
            return actions

        for nxt, act in (((x, y, (d - 1) % 4), 0), ((x, y, (d + 1) % 4), 1)):
            ng = g + 1
            if ng < g_score.get(nxt, 10**9):
                g_score[nxt] = ng
                parent[nxt] = (current, act)
                heapq.heappush(frontier, (ng + heuristic(nxt), ng, nxt))

        dx, dy = DIR_VEC[d]
        nx, ny = x + dx, y + dy
        if is_walkable(env_unwrapped, (nx, ny)):
            nxt = (nx, ny, d)
            ng = g + 1
            if ng < g_score.get(nxt, 10**9):
                g_score[nxt] = ng
                parent[nxt] = (current, 2)
                heapq.heappush(frontier, (ng + heuristic(nxt), ng, nxt))

    return None


def required_dir(src: Tuple[int, int], dst: Tuple[int, int]) -> Optional[int]:
    dx = int(dst[0] - src[0])
    dy = int(dst[1] - src[1])
    if dx == 1 and dy == 0:
        return 0
    if dx == 0 and dy == 1:
        return 1
    if dx == -1 and dy == 0:
        return 2
    if dx == 0 and dy == -1:
        return 3
    return None


@dataclass
class RolloutRecorder:
    env_name: str
    seed: int
    gid: str
    out_dir: Path
    tile_size: int = 32

    def __post_init__(self) -> None:
        self.actions: List[int] = []
        self.actions_text: List[str] = []
        self.frames: List[str] = []
        self.state_seq: List[Dict[str, Any]] = []
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False

    def record_initial(self, env: Any, obs: Dict[str, Any]) -> None:
        self._record_state(env, obs, step_idx=0)

    def step(self, env: Any, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = env.step(int(action))
        self.actions.append(int(action))
        self.actions_text.append({0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}[int(action)])
        self.total_reward += float(reward)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self._record_state(env, obs, step_idx=len(self.state_seq))
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _record_state(self, env: Any, obs: Dict[str, Any], step_idx: int) -> None:
        rel = f"pilot_frames/{self.env_name}/{self.gid}/step_{step_idx:03d}.png"
        save_image(render_agent_pov(env, tile_size=self.tile_size), self.out_dir / rel)
        self.frames.append(rel)
        self.state_seq.append(extract_state(env.unwrapped, obs))

    def as_triplet_row(self, env: Any) -> Dict[str, Any]:
        return {
            "task": self.env_name,
            "env_id": getattr(env.spec, "id", self.env_name),
            "seed": self.seed,
            "group_id": self.gid,
            "variant": "full",
            "mission": str(getattr(env.unwrapped, "mission", "")),
            "model_input_fields": ["frames", "mission"],
            "actions_id": self.actions,
            "actions_text": self.actions_text,
            "frames": self.frames,
            "state_seq": self.state_seq,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "reward": self.total_reward,
            "success": self.terminated and self.total_reward > 0.0,
        }


class Solver:
    def __init__(self, env: Any, recorder: RolloutRecorder):
        self.env = env
        self.r = recorder

    @property
    def u(self) -> Any:
        return self.env.unwrapped

    def turn_to(self, dst_dir: int) -> bool:
        cur_dir = int(self.u.agent_dir)
        turns = (int(dst_dir) - cur_dir) % 4
        if turns == 0:
            seq: List[int] = []
        elif turns == 1:
            seq = [1]
        elif turns == 2:
            seq = [1, 1]
        else:
            seq = [0]
        for act in seq:
            _obs, _r, term, trunc, _ = self.r.step(self.env, act)
            if term or trunc:
                return False
        return True

    def forward_with_door_fallback(self) -> bool:
        fx, fy = self.u.front_pos
        front = self.u.grid.get(int(fx), int(fy))
        if front is not None and getattr(front, "type", None) == "door" and not getattr(front, "is_open", False):
            _obs, _r, term, trunc, _ = self.r.step(self.env, 5)
            if term or trunc:
                return False
        prev_pos = tuple(int(x) for x in self.u.agent_pos)
        _obs, _r, term, trunc, _ = self.r.step(self.env, 2)
        if term or trunc:
            return False
        new_pos = tuple(int(x) for x in self.u.agent_pos)
        return new_pos != prev_pos

    def execute_path(self, path: List[int]) -> bool:
        for act in path:
            if int(act) == 2:
                if not self.forward_with_door_fallback():
                    return False
            else:
                _obs, _r, term, trunc, _ = self.r.step(self.env, int(act))
                if term or trunc:
                    return False
        return True

    def candidate_adj(self, target_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        sx, sy = (int(self.u.agent_pos[0]), int(self.u.agent_pos[1]))
        cands: List[Tuple[int, int, int]] = []
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            ap = (int(target_pos[0] + dx), int(target_pos[1] + dy))
            if is_walkable(self.u, ap):
                cands.append((abs(ap[0] - sx) + abs(ap[1] - sy), ap[0], ap[1]))
        cands.sort()
        return [(x, y) for _d, x, y in cands]

    def move_to_face(self, target_pos: Tuple[int, int], max_replans: int = 4) -> bool:
        for _ in range(max_replans):
            start_pos = (int(self.u.agent_pos[0]), int(self.u.agent_pos[1]))
            start_dir = int(self.u.agent_dir)
            best_path = None
            best_adj = None
            for adj in self.candidate_adj(target_pos):
                path = plan_path(self.u, start_pos, start_dir, adj)
                if path is None:
                    continue
                if best_path is None or len(path) < len(best_path):
                    best_path = path
                    best_adj = adj
            if best_path is None or best_adj is None:
                return False
            if not self.execute_path(best_path):
                continue
            cur = (int(self.u.agent_pos[0]), int(self.u.agent_pos[1]))
            if cur != best_adj:
                continue
            req = required_dir(cur, target_pos)
            if req is None:
                continue
            if not self.turn_to(req):
                return False
            if tuple(int(x) for x in self.u.front_pos) == tuple(int(x) for x in target_pos):
                return True
        return False

    def pickup_front(self, obj_type: Optional[str], obj_color: Optional[str]) -> Tuple[bool, bool]:
        fx, fy = self.u.front_pos
        front = self.u.grid.get(int(fx), int(fy))
        if front is None:
            return False, False
        if obj_type is not None and getattr(front, "type", None) != obj_type:
            return False, False
        if obj_color is not None and getattr(front, "color", None) != obj_color:
            return False, False
        _obs, reward, term, trunc, _ = self.r.step(self.env, 3)
        if trunc:
            return False, False
        if term:
            return bool(reward > 0.0), True
        return True, False

    def drop_front(self) -> bool:
        _obs, _r, term, trunc, _ = self.r.step(self.env, 4)
        if term or trunc:
            return False
        return self.u.carrying is None


def scan_objects(env_unwrapped: Any) -> List[Dict[str, Any]]:
    objs = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            cell = env_unwrapped.grid.get(x, y)
            if cell is None:
                continue
            objs.append(
                {
                    "type": str(getattr(cell, "type", "unknown")),
                    "color": str(getattr(cell, "color", "unknown")) if hasattr(cell, "color") else None,
                    "pos": (int(x), int(y)),
                    "state": door_state(cell) if getattr(cell, "type", None) == "door" else None,
                    "contains": getattr(cell, "contains", None),
                }
            )
    return objs


def choose_drop_cell(
    env_unwrapped: Any,
    avoid: set[Tuple[int, int]],
    near_target: Optional[Tuple[int, int]] = None,
    require_near_target: bool = False,
) -> Optional[Tuple[int, int]]:
    candidates = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            pos = (int(x), int(y))
            if pos in avoid:
                continue
            if env_unwrapped.grid.get(x, y) is not None:
                continue
            adj_ok = False
            for dx, dy in DIR_VEC.values():
                sx, sy = x + dx, y + dy
                if is_walkable(env_unwrapped, (sx, sy)):
                    adj_ok = True
                    break
            if not adj_ok:
                continue
            if near_target is not None:
                dist = abs(pos[0] - near_target[0]) + abs(pos[1] - near_target[1])
                if require_near_target and dist != 1:
                    continue
                candidates.append((dist, pos[0], pos[1]))
            else:
                candidates.append((0, pos[0], pos[1]))
    if not candidates:
        return None
    candidates.sort()
    return (candidates[0][1], candidates[0][2])


def solve_putnear(env: Any, recorder: RolloutRecorder) -> bool:
    s = Solver(env, recorder)
    u = env.unwrapped
    move_pos = tuple(int(x) for x in u.move_pos)
    target_pos = tuple(int(x) for x in u.target_pos)
    if not s.move_to_face(move_pos):
        return False
    ok, term = s.pickup_front(str(u.move_type), str(u.moveColor))
    if not ok or term:
        return False
    drop_cell = choose_drop_cell(
        u,
        avoid={move_pos, target_pos},
        near_target=target_pos,
        require_near_target=True,
    )
    if drop_cell is None:
        return False
    if not s.move_to_face(drop_cell):
        return False
    fx, fy = u.front_pos
    if (int(fx), int(fy)) != drop_cell:
        return False
    _obs, reward, term, trunc, _ = recorder.step(env, 4)
    return bool(term and reward > 0.0 and not trunc)


def solve_unlockpickup(env: Any, recorder: RolloutRecorder) -> bool:
    s = Solver(env, recorder)
    u = env.unwrapped
    objs = scan_objects(u)
    door = next(o for o in objs if o["type"] == "door" and o["state"] == 2)
    key = next(o for o in objs if o["type"] == "key" and o["color"] == door["color"])
    target = next(o for o in objs if o["type"] == "box")

    if not s.move_to_face(key["pos"]):
        return False
    ok, term = s.pickup_front("key", door["color"])
    if not ok or term:
        return False
    if not s.move_to_face(door["pos"]):
        return False
    fx, fy = u.front_pos
    front = u.grid.get(int(fx), int(fy))
    if front is None or getattr(front, "type", None) != "door":
        return False
    if not getattr(front, "is_open", False):
        _obs, _r, term, trunc, _ = recorder.step(env, 5)
        if term or trunc:
            return False
    if not s.forward_with_door_fallback():
        return False

    drop_cell = choose_drop_cell(u, avoid={target["pos"], door["pos"], key["pos"]})
    if drop_cell is None:
        return False
    if not s.move_to_face(drop_cell):
        return False
    fx, fy = u.front_pos
    if (int(fx), int(fy)) != drop_cell:
        return False
    if not s.drop_front():
        return False

    if not s.move_to_face(target["pos"]):
        return False
    ok, term = s.pickup_front("box", str(getattr(u.obj, "color", target["color"])))
    return bool(ok and term)


def solve_blockedunlockpickup(env: Any, recorder: RolloutRecorder) -> bool:
    s = Solver(env, recorder)
    u = env.unwrapped
    objs = scan_objects(u)
    door = next(o for o in objs if o["type"] == "door" and o["state"] == 2)
    key = next(o for o in objs if o["type"] == "key" and o["color"] == door["color"])
    target = next(o for o in objs if o["type"] == "box")
    blocker = next(
        o for o in objs
        if o["type"] == "ball" and abs(o["pos"][0] - door["pos"][0]) + abs(o["pos"][1] - door["pos"][1]) == 1
    )

    if not s.move_to_face(blocker["pos"]):
        return False
    ok, term = s.pickup_front("ball", blocker["color"])
    if not ok or term:
        return False
    blocker_drop = choose_drop_cell(u, avoid={door["pos"], key["pos"], target["pos"]})
    if blocker_drop is None:
        return False
    if not s.move_to_face(blocker_drop):
        return False
    fx, fy = u.front_pos
    if (int(fx), int(fy)) != blocker_drop:
        return False
    if not s.drop_front():
        return False

    if not s.move_to_face(key["pos"]):
        return False
    ok, term = s.pickup_front("key", door["color"])
    if not ok or term:
        return False
    if not s.move_to_face(door["pos"]):
        return False
    fx, fy = u.front_pos
    front = u.grid.get(int(fx), int(fy))
    if front is None or getattr(front, "type", None) != "door":
        return False
    if not getattr(front, "is_open", False):
        _obs, _r, term, trunc, _ = recorder.step(env, 5)
        if term or trunc:
            return False
    if not s.forward_with_door_fallback():
        return False

    key_drop = choose_drop_cell(u, avoid={door["pos"], target["pos"]})
    if key_drop is None:
        return False
    if not s.move_to_face(key_drop):
        return False
    fx, fy = u.front_pos
    if (int(fx), int(fy)) != key_drop:
        return False
    if not s.drop_front():
        return False

    if not s.move_to_face(target["pos"]):
        return False
    ok, term = s.pickup_front("box", str(getattr(u.obj, "color", target["color"])))
    return bool(ok and term)


def run_single_rollout(env_name: str, env_id: str, seed: int, gid: str, out_dir: Path) -> Optional[Dict[str, Any]]:
    env = gym.make(env_id, render_mode="rgb_array")
    try:
        obs, _info = env.reset(seed=seed)
        recorder = RolloutRecorder(env_name=env_name, seed=seed, gid=gid, out_dir=out_dir)
        recorder.record_initial(env, obs)
        if env_name == "putnear":
            ok = solve_putnear(env, recorder)
        elif env_name == "unlockpickup":
            ok = solve_unlockpickup(env, recorder)
        elif env_name == "blockedunlockpickup":
            ok = solve_blockedunlockpickup(env, recorder)
        else:
            ok = False
        if not ok:
            return None
        row = recorder.as_triplet_row(env)
        row["group_id"] = gid
        return row
    finally:
        env.close()


def summarize_env(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pickup_coverage = 0
    drop_coverage = 0
    checkpoint_queryable = 0
    dual_query_support = 0
    first_counter: Counter[str] = Counter()
    fate_counter: Counter[str] = Counter()
    terminal_counter: Counter[str] = Counter()
    horizon_counter: Counter[str] = Counter()
    query_rows = []

    for row in rows:
        trace = normalize_trace(row["state_seq"])
        events = extract_events(trace)
        pickups = [(s, l) for s, l in events["pickups"] if l != "none"]
        drops = list(events["drops"])
        n_steps = events["n_steps"]
        if pickups:
            pickup_coverage += 1
            first_pickup_step, first_label = pickups[0]
            first_counter[first_label] += 1
            first_drop_step = next((s for s, l in drops if l == first_label), None)
            if first_drop_step is not None:
                drop_coverage += 1
            fate = "retained" if first_drop_step is None else "dropped"
            fate_counter[fate] += 1
            id_steps = {k: v for k, v in pick_identity_steps(first_pickup_step, n_steps).items() if v is not None}
            st_steps = {k: v for k, v in pick_status_steps(first_pickup_step, first_drop_step, n_steps).items() if v is not None}
            has_id = bool(id_steps)
            has_st = bool(st_steps)
            if has_id or has_st:
                checkpoint_queryable += 1
            if has_id and has_st:
                dual_query_support += 1
            for qs in list(id_steps.values()) + list(st_steps.values()):
                gap = qs - first_pickup_step
                horizon_counter[assign_horizon_bucket(gap)] += 1
        terminal_counter[carrying_label(trace[-1]["carrying"])] += 1

    n = len(rows)
    terminal_dominant_rate = max(terminal_counter.values()) / n if n else 0.0
    if drop_coverage / n >= 0.4 and len(horizon_counter) >= 3 and terminal_dominant_rate < 0.8:
        shortcut_risk = "low"
    elif terminal_dominant_rate >= 0.8 or drop_coverage == 0:
        shortcut_risk = "high"
    else:
        shortcut_risk = "medium"

    verdict = "PROMOTE"
    if n == 0:
        verdict = "REJECT"
    elif pickup_coverage < n:
        verdict = "REJECT"
    elif checkpoint_queryable / n < 0.5:
        verdict = "REJECT"
    elif len(first_counter) < 3:
        verdict = "CONTROL_ONLY"
    elif dual_query_support / n < 0.5:
        verdict = "CONTROL_ONLY"
    elif shortcut_risk == "high":
        verdict = "CONTROL_ONLY"

    return {
        "n_rows": n,
        "pickup_coverage": pickup_coverage / n if n else 0.0,
        "drop_coverage": drop_coverage / n if n else 0.0,
        "checkpoint_queryable_rate": checkpoint_queryable / n if n else 0.0,
        "dual_query_support_rate": dual_query_support / n if n else 0.0,
        "first_object_diversity": len(first_counter),
        "first_object_counts": dict(first_counter),
        "fate_diversity": len(fate_counter),
        "fate_counts": dict(fate_counter),
        "terminal_counts": dict(terminal_counter),
        "horizon_coverage": dict(horizon_counter),
        "shortcut_risk": shortcut_risk,
        "verdict": verdict,
    }


def export_exam_for_env(env_name: str, rows: List[Dict[str, Any]], out_dir: Path, max_rows: int = 12) -> Dict[str, Any]:
    exam_rows = []
    review_rows = []
    row_idx = 0

    by_role: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        trace = normalize_trace(row["state_seq"])
        events = extract_events(trace)
        pickups = [(s, l) for s, l in events["pickups"] if l != "none"]
        if not pickups:
            continue
        first_pickup_step, first_label = pickups[0]
        first_drop_step = next((s for s, l in events["drops"] if l == first_label), None)
        n_steps = events["n_steps"]
        role_steps = {}
        role_steps.update(pick_identity_steps(first_pickup_step, n_steps))
        role_steps.update(pick_status_steps(first_pickup_step, first_drop_step, n_steps))
        for role, qs in role_steps.items():
            if qs is None or role in by_role:
                continue
            gap = qs - first_pickup_step
            horizon_bucket = assign_horizon_bucket(gap)
            query_type = "identity_of_first_picked_object" if role.startswith("identity") else "status_of_first_picked_object_at_checkpoint"
            answer = (
                first_label
                if query_type == "identity_of_first_picked_object"
                else classify_status(first_label, pickups, events["drops"], events["reacquires"], qs)
            )
            by_role[role] = {
                "row": row,
                "query_step": qs,
                "anchor_step": first_pickup_step,
                "horizon_gap": gap,
                "horizon_bucket": horizon_bucket,
                "query_type": query_type,
                "answer": answer,
                "first_label": first_label,
            }

    ordered_roles = [
        "identity_short",
        "identity_medium",
        "identity_long",
        "status_early_holding",
        "status_mid_holding",
        "status_first_post_drop",
        "status_late_post_drop",
    ]
    selected = [by_role[r] for r in ordered_roles if r in by_role][:max_rows]

    for item in selected:
        source = item["row"]
        built = build_v12_row(
            env=env_name,
            gid=source["group_id"],
            query_type=item["query_type"],
            query_step=item["query_step"],
            anchor_step=item["anchor_step"],
            horizon_gap=item["horizon_gap"],
            horizon_bucket=item["horizon_bucket"],
            answer=item["answer"],
            first_label=item["first_label"],
            frames=source["frames"],
            env_dir=out_dir,
            out_dir=out_dir,
            row_idx=row_idx,
            sampling_role=source.get("sampling_role", source["group_id"] + "_" + item["query_type"]),
        )
        built["source_seed"] = source["seed"]
        built["sampling_role"] = built["sampling_role"] if "sampling_role" in built else "pilot"
        exam_rows.append(built)
        review_rows.append(
            {
                "exam_id": built["exam_id"],
                "image_abs": str((out_dir / built["image"]).resolve()),
                "prompt": built["prompt"],
                "gold_answer": built["answer"],
                "sampling_role": built["sampling_role"],
                "env_task": env_name,
                "horizon_bucket": built["horizon_bucket"],
            }
        )
        row_idx += 1

    exam_path = out_dir / f"task_e_phase58_{env_name}_exam.jsonl"
    review_path = out_dir / f"task_e_phase58_{env_name}_review.jsonl"
    with exam_path.open("w", encoding="utf-8") as f:
        for row in exam_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with review_path.open("w", encoding="utf-8") as f:
        for row in review_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "exam_path": str(exam_path),
        "review_path": str(review_path),
        "n_exam_rows": len(exam_rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("runs/phase58_taske_env_screen_2026-03-31"))
    ap.add_argument("--target-per-env", type=int, default=24)
    ap.add_argument("--max-attempts-per-env", type=int, default=120)
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env_specs = {
        "putnear": "MiniGrid-PutNear-8x8-N3-v0",
        "unlockpickup": "MiniGrid-UnlockPickup-v0",
        "blockedunlockpickup": "MiniGrid-BlockedUnlockPickup-v0",
    }

    summary = {
        "phase": "58",
        "objective": "Task E environment screening pilot",
        "target_per_env": args.target_per_env,
        "envs": {},
        "passed_envs": [],
    }

    for env_name, env_id in env_specs.items():
        rows = []
        attempt_logs = []
        for attempt in range(args.max_attempts_per_env):
            if len(rows) >= args.target_per_env:
                break
            seed = attempt
            gid = f"{env_name}_s{seed:06d}"
            result = run_single_rollout(env_name, env_id, seed, gid, out_dir)
            attempt_logs.append({"seed": seed, "success": result is not None})
            if result is not None:
                rows.append(result)

        raw_path = out_dir / f"{env_name}_pilot_full.jsonl"
        with raw_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        metrics = summarize_env(rows)
        metrics["env_id"] = env_id
        metrics["pilot_rows_path"] = str(raw_path)
        metrics["attempts"] = len(attempt_logs)
        metrics["successful_rows"] = len(rows)

        if metrics["verdict"] == "PROMOTE":
            export_info = export_exam_for_env(env_name, rows, out_dir)
            metrics["exam_export"] = export_info
            summary["passed_envs"].append(env_name)
        summary["envs"][env_name] = metrics

    summary_path = out_dir / "phase58_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
