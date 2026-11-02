#!/usr/bin/env python3
"""Generate Tier-2 MiniGrid interaction full/nocue/cf triplets."""

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

INTERACTION_ENVS = {
    "blockedunlockpickup": {
        "env_id": "MiniGrid-BlockedUnlockPickup-v0",
        "actions": [0, 1, 2, 3, 4, 5],
        "max_nodes": 120000,
        "max_depth": 120,
        "mask_object": ("box", "purple"),
        "cf_remove_object": ("box", "purple"),
    },
    "putnear": {
        "env_id": "MiniGrid-PutNear-8x8-N3-v0",
        "actions": [0, 1, 2, 3, 4],
        "max_nodes": 120000,
        "max_depth": 40,
        "mask_object": ("ball", None),
        "cf_remove_object": ("ball", None),
    },
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


from utils.minigrid_state import front_cell_summary


def carrying_summary(env: Any) -> dict[str, Any] | None:
    carrying = getattr(env.unwrapped, "carrying", None)
    if carrying is None:
        return None
    return {
        "type": getattr(carrying, "type", None),
        "color": getattr(carrying, "color", None),
    }


def state_summary(env: Any, mask_applied: bool = False, mask_type: str | None = None) -> dict[str, Any]:
    u = env.unwrapped
    return {
        "agent": {
            "pos": [int(v) for v in u.agent_pos],
            "dir": int(u.agent_dir),
            "carrying": carrying_summary(env),
        },
        "front_cell": front_cell_summary(env),
        "objects": object_summary(env),
        "mask_applied": bool(mask_applied),
        "mask_type": mask_type if mask_applied else None,
    }


def object_positions(env: Any, obj_type: str, color: str | None = None) -> list[tuple[int, int]]:
    u = env.unwrapped
    out = []
    for x in range(u.width):
        for y in range(u.height):
            cell = u.grid.get(x, y)
            if cell is None:
                continue
            if getattr(cell, "type", None) != obj_type:
                continue
            if color is not None and getattr(cell, "color", None) != color:
                continue
            out.append((int(x), int(y)))
    return out


def remove_objects(env: Any, obj_type: str, color: str | None = None) -> dict[str, Any]:
    removed = []
    for x, y in object_positions(env, obj_type, color):
        cell = env.unwrapped.grid.get(x, y)
        removed.append({
            "pos": [x, y],
            "type": getattr(cell, "type", None),
            "color": getattr(cell, "color", None),
        })
        env.unwrapped.grid.set(x, y, None)
    return {
        "cf_mode": "remove_object",
        "object_type": obj_type,
        "object_color": color,
        "removed": removed,
        "intervention_step": 0,
    }


def mask_object_tiles(frame: np.ndarray, env: Any, obj_type: str, color: str | None = None) -> tuple[np.ndarray, bool]:
    positions = object_positions(env, obj_type, color)
    if not positions:
        return frame, False
    u = env.unwrapped
    h, w = frame.shape[:2]
    tile_w = max(1, w // int(u.width))
    tile_h = max(1, h // int(u.height))
    out = frame.copy()
    for x, y in positions:
        out[y * tile_h : min((y + 1) * tile_h, h), x * tile_w : min((x + 1) * tile_w, w)] = 0
    return out, True


def rollout_variant(
    spec: dict[str, Any],
    seed: int,
    actions: list[int],
    out_dir: Path,
    variant: str,
    mask_until: int = -1,
    apply_cf: bool = False,
) -> dict[str, Any]:
    env = gym.make(spec["env_id"], render_mode="rgb_array")
    env.reset(seed=seed)
    cf_meta = None
    if apply_cf:
        cf_meta = remove_objects(env, *spec["cf_remove_object"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    state_seq = []

    def save_frame(idx: int) -> None:
        frame = env.render()
        mask_applied = False
        if 0 <= idx <= mask_until:
            frame, mask_applied = mask_object_tiles(frame, env, *spec["mask_object"])
        Image.fromarray(frame).save(out_dir / f"step_{idx:03d}.png")
        frames.append(f"{variant}/step_{idx:03d}.png")
        state_seq.append(
            state_summary(
                env,
                mask_applied=mask_applied,
                mask_type=f"{spec['mask_object'][0]}_tile_suppression",
            )
        )

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
    full = rollout_variant(spec, seed, actions, group_dir / "full", "full")
    nocue = rollout_variant(spec, seed, actions, group_dir / "nocue", "nocue", mask_until=mask_until)
    cf = rollout_variant(spec, seed, actions, group_dir / "cf", "cf", apply_cf=True)
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
                "mask_mode": f"{spec['mask_object'][0]}_tile_suppression",
                "mask_until_step": mask_until,
            } if variant == "nocue" else None,
        }

    return [mk_record("full", full), mk_record("nocue", nocue), mk_record("cf", cf)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--env-keys", default="blockedunlockpickup,putnear")
    ap.add_argument("--num-per-env", type=int, default=2)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--max-seed-attempts-per-env", type=int, default=200)
    args = ap.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    manifest = {
        "generator": 'generate_interactions.py',
        "num_per_env": args.num_per_env,
        "seed_start": args.seed_start,
        "envs": [],
    }
    group_idx = 0
    for env_key in [x.strip() for x in args.env_keys.split(",") if x.strip()]:
        spec = INTERACTION_ENVS[env_key]
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
