"""Leaf simulator-state BFS helpers; no workflow or file execution."""
from __future__ import annotations
import copy
from collections import deque
from typing import Any

ACTION_ID2NAME = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

def cell_signature(cell: Any) -> Any:
    if cell is None:
        return None
    cur_pos = getattr(cell, "cur_pos", None)
    if cur_pos is not None:
        try:
            cur_pos = tuple(int(v) for v in cur_pos)
        except TypeError:
            cur_pos = None
    return (
        getattr(cell, "type", None),
        getattr(cell, "color", None),
        getattr(cell, "is_open", None),
        getattr(cell, "is_locked", None),
        cur_pos,
    )

def env_signature(env: Any) -> tuple[Any, ...]:
    u = env.unwrapped
    carrying = getattr(u, "carrying", None)
    grid_sig = []
    for x in range(u.width):
        for y in range(u.height):
            grid_sig.append((x, y, cell_signature(u.grid.get(x, y))))
    return (
        tuple(int(v) for v in u.agent_pos),
        int(u.agent_dir),
        cell_signature(carrying),
        tuple(grid_sig),
    )

def object_summary(env: Any) -> list[dict[str, Any]]:
    u = env.unwrapped
    out = []
    for x in range(u.width):
        for y in range(u.height):
            cell = u.grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) not in ("wall", "floor"):
                out.append({
                    "x": int(x),
                    "y": int(y),
                    "type": getattr(cell, "type", None),
                    "color": getattr(cell, "color", None),
                    "is_open": getattr(cell, "is_open", None),
                    "is_locked": getattr(cell, "is_locked", None),
                })
    return out

def bfs_plan(env: Any, actions: list[int], max_nodes: int, max_depth: int) -> dict[str, Any]:
    start = copy.deepcopy(env)
    queue = deque([(start, [])])
    seen = {env_signature(start)}
    nodes = 0
    best_depth = 0
    while queue and nodes < max_nodes:
        cur, path = queue.popleft()
        nodes += 1
        best_depth = max(best_depth, len(path))
        if len(path) >= max_depth:
            continue
        for action in actions:
            nxt = copy.deepcopy(cur)
            try:
                _obs, reward, terminated, truncated, _info = nxt.step(action)
            except Exception as exc:  # noqa: BLE001 - probe
                continue
            new_path = [*path, int(action)]
            if reward > 0:
                return {
                    "found": True,
                    "actions_id": new_path,
                    "actions_text": [ACTION_ID2NAME.get(a, str(a)) for a in new_path],
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "nodes": nodes,
                    "max_depth_reached": best_depth,
                }
            if terminated or truncated:
                continue
            sig = env_signature(nxt)
            if sig in seen:
                continue
            seen.add(sig)
            queue.append((nxt, new_path))
    return {
        "found": False,
        "actions_id": [],
        "actions_text": [],
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "nodes": nodes,
        "max_depth_reached": best_depth,
    }
