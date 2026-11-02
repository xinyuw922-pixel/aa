"""Shared MiniGrid POV rendering and state queries."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def render_pov_with_fallback(env, tile_size: int, grid_h: int = 7, grid_w: int = 7) -> Optional[np.ndarray]:
    """
    Must return agent POV image with shape (grid_h*tile_size, grid_w*tile_size, 3).
    If cannot guarantee this, return None (hard reject).
    """
    for kwargs in (
        dict(tile_size=tile_size, agent_pov=True, highlight=False),
        dict(tile_size=tile_size, agent_pov=True),
    ):
        try:
            frame = env.unwrapped.get_frame(**kwargs)
            if isinstance(frame, np.ndarray) and frame.shape[:2] == (grid_h * tile_size, grid_w * tile_size):
                return frame
        except Exception:
            pass
    return None


def door_state(obj) -> int:
    """Return Minigrid Door state index: open=0, closed=1, locked=2."""
    if getattr(obj, "is_locked", False):
        return 2
    if getattr(obj, "is_open", False):
        return 0
    return 1


def room_index(env_unwrapped, pos: Tuple[int, int]) -> Optional[int]:
    """Find the room containing a world-grid position, if rooms are available."""
    rooms = getattr(env_unwrapped, "rooms", None)
    if rooms is None:
        return None

    x, y = pos
    for i, r in enumerate(rooms):
        top = getattr(r, "top", None)
        size = getattr(r, "size", None)

        if top is None and isinstance(r, dict):
            top = r.get("top")
            size = r.get("size")
        if top is None and isinstance(r, (list, tuple)) and len(r) >= 2:
            top, size = r[0], r[1]

        if top is None or size is None:
            continue

        rx, ry = int(top[0]), int(top[1])
        rw, rh = int(size[0]), int(size[1])
        if rx <= x < rx + rw and ry <= y < ry + rh:
            return i

    return None


def agent_trajectory(rec: Dict) -> List[Tuple[Tuple[int, int], int]]:
    """Read world-grid positions and directions from a trajectory record."""
    traj = []
    for st in rec["state_seq"]:
        pos = tuple(st["agent"]["pos"])
        d = int(st["agent"]["dir"])
        traj.append((pos, d))
    return traj



def front_cell_summary(env: Any) -> dict[str, Any]:
    u = env.unwrapped
    x, y = (int(v) for v in u.agent_pos)
    dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][int(u.agent_dir)]
    fx, fy = x + dx, y + dy
    if fx < 0 or fy < 0 or fx >= u.width or fy >= u.height:
        return {"pos": [fx, fy], "type": "out_of_bounds"}
    cell = u.grid.get(fx, fy)
    if cell is None:
        return {"pos": [fx, fy], "type": "empty"}
    out = {"pos": [fx, fy], "type": getattr(cell, "type", None)}
    color = getattr(cell, "color", None)
    if color is not None:
        out["color"] = color
    return out

