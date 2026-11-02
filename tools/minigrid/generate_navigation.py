#!/usr/bin/env python3
"""Generate Tier-2 MiniGrid navigation full/nocue/cf triplets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # Register simulator environments before gym.make.
import numpy as np
from PIL import Image

from utils.planning import ACTION_ID2NAME, bfs_plan, object_summary


ROOT = Path(__file__).resolve().parents[2]

NAV_ENVS = {
    "fourrooms": {
        "env_id": "MiniGrid-FourRooms-v0",
        "actions": [0, 1, 2],
        "max_nodes": 60000,
        "max_depth": 100,
    },
    "dynamicobstacles": {
        "env_id": "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "actions": [0, 1, 2],
        "max_nodes": 80000,
        "max_depth": 80,
    },
    "emptyrandom": {
        "env_id": "MiniGrid-Empty-Random-6x6-v0",
        "actions": [0, 1, 2],
        "max_nodes": 20000,
        "max_depth": 40,
    },
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


from utils.minigrid_state import front_cell_summary


def state_summary(env: Any, mask_applied: bool = False) -> dict[str, Any]:
    u = env.unwrapped
    return {
        "agent": {
            "pos": [int(v) for v in u.agent_pos],
            "dir": int(u.agent_dir),
            "carrying": None,
        },
        "front_cell": front_cell_summary(env),
        "objects": object_summary(env),
        "mask_applied": bool(mask_applied),
        "mask_type": "goal_tile_suppression" if mask_applied else None,
    }


def find_goal_pos(env: Any) -> tuple[int, int] | None:
    u = env.unwrapped
    for x in range(u.width):
        for y in range(u.height):
            cell = u.grid.get(x, y)
            if cell is not None and getattr(cell, "type", None) == "goal":
                return int(x), int(y)
    return None


def remove_goal(env: Any) -> dict[str, Any]:
    pos = find_goal_pos(env)
    if pos is None:
        return {"cf_mode": "remove_goal", "goal_from": None, "intervention_step": 0}
    env.unwrapped.grid.set(pos[0], pos[1], None)
    return {"cf_mode": "remove_goal", "goal_from": [pos[0], pos[1]], "intervention_step": 0}


def mask_goal_tile(frame: np.ndarray, env: Any) -> tuple[np.ndarray, bool]:
    pos = find_goal_pos(env)
    if pos is None:
        return frame, False
    u = env.unwrapped
    h, w = frame.shape[:2]
    tile_w = max(1, w // int(u.width))
    tile_h = max(1, h // int(u.height))
    x, y = pos
    out = frame.copy()
    out[y * tile_h : min((y + 1) * tile_h, h), x * tile_w : min((x + 1) * tile_w, w)] = 0
    return out, True


def rollout_variant(
    env_id: str,
    seed: int,
    actions: list[int],
    out_dir: Path,
    variant: str,
    mask_goal_until: int = -1,
    remove_goal_first: bool = False,
) -> dict[str, Any]:
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    cf_meta = remove_goal(env) if remove_goal_first else None
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    state_seq = []

    def save_frame(idx: int) -> None:
        frame = env.render()
        mask_applied = False
        if 0 <= idx <= mask_goal_until:
            frame, mask_applied = mask_goal_tile(frame, env)
        Image.fromarray(frame).save(out_dir / f"step_{idx:03d}.png")
        frames.append(f"{variant}/step_{idx:03d}.png")
        state_seq.append(state_summary(env, mask_applied=mask_applied))

    save_frame(0)
    reward = 0.0
    terminated = False
    truncated = False
    for idx, action in enumerate(actions, start=1):
        _obs, reward, terminated, truncated, _info = env.step(int(action))
        save_frame(idx)
        if terminated or truncated:
            break
    env.close()
    return {
        "frames": frames,
        "state_seq": state_seq,
        "reward": float(reward),
        "success": bool(reward > 0),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "cf_meta": cf_meta,
    }


def generate_one(env_key: str, spec: dict[str, Any], seed: int, group_idx: int, out_dir: Path) -> list[dict[str, Any]] | None:
    env = gym.make(spec["env_id"], render_mode="rgb_array")
    env.reset(seed=seed)
    plan = bfs_plan(
        env,
        actions=list(spec["actions"]),
        max_nodes=int(spec["max_nodes"]),
        max_depth=int(spec["max_depth"]),
    )
    mission = getattr(env.unwrapped, "mission", "")
    env.close()
    if not plan["found"]:
        return None

    group_id = f"{env_key}_tier2_s{group_idx:06d}"
    group_dir = out_dir / group_id
    if group_dir.exists():
        shutil.rmtree(group_dir)
    actions = [int(a) for a in plan["actions_id"]]
    mask_until = max(0, min(len(actions) // 2, len(actions) - 1))
    full = rollout_variant(spec["env_id"], seed, actions, group_dir / "full", "full")
    nocue = rollout_variant(spec["env_id"], seed, actions, group_dir / "nocue", "nocue", mask_goal_until=mask_until)
    cf = rollout_variant(spec["env_id"], seed, actions, group_dir / "cf", "cf", remove_goal_first=True)
    if not full["success"] or not nocue["success"] or cf["success"]:
        shutil.rmtree(group_dir, ignore_errors=True)
        return None

    def mk_record(variant: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "group_id": group_id,
            "task": env_key,
            "env_id": spec["env_id"],
            "seed": int(seed),
            "variant": variant,
            "mission": mission,
            "model_input_fields": ["frames", "mission"],
            "actions_id": actions,
            "actions_text": [ACTION_ID2NAME.get(a, str(a)) for a in actions],
            "frames": [f"{group_id}/{fp}" for fp in payload["frames"]],
            "state_seq": payload["state_seq"],
            "success": bool(payload["success"]),
            "reward": float(payload["reward"]),
            "terminated": bool(payload["terminated"]),
            "truncated": bool(payload["truncated"]),
            "cf_meta": payload.get("cf_meta"),
            "nocue_meta": {
                "mask_mode": "goal_tile_suppression",
                "mask_until_step": mask_until,
            } if variant == "nocue" else None,
        }

    return [mk_record("full", full), mk_record("nocue", nocue), mk_record("cf", cf)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--env-keys", default="fourrooms,emptyrandom,dynamicobstacles")
    ap.add_argument("--num-per-env", type=int, default=2)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-seed-attempts-per-env", type=int, default=200)
    args = ap.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = [x.strip() for x in args.env_keys.split(",") if x.strip()]
    records = []
    manifest = {
        "generator": 'generate_navigation.py',
        "num_per_env": args.num_per_env,
        "seed_start": args.seed_start,
        "envs": [],
    }
    group_idx = 0
    for env_key in selected_keys:
        spec = NAV_ENVS[env_key]
        accepted = 0
        attempted = 0
        seed = int(args.seed_start)
        while accepted < args.num_per_env and attempted < args.max_seed_attempts_per_env:
            attempted += 1
            rows = generate_one(env_key, spec, seed, group_idx, out_dir)
            seed += 1
            if rows is None:
                continue
            records.extend(rows)
            accepted += 1
            group_idx += 1
        manifest["envs"].append({
            "env_key": env_key,
            "env_id": spec["env_id"],
            "accepted": accepted,
            "attempted": attempted,
        })
    with (out_dir / "triplets.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "rows": len(records),
        "groups": len(records) // 3,
        "manifest": str(out_dir / "manifest.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
