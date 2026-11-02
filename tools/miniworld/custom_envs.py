#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, List


def register_local_miniworld_envs(gym: Any) -> List[str]:
    """Register repo-local MiniWorld envs exactly once."""
    from gymnasium.envs.registration import register

    try:
        from tools.miniworld.keydoor_story_env import KeyDoorStoryEnv
    except Exception:
        from tools.miniworld.keydoor_story_env import KeyDoorStoryEnv

    env_specs = [
        ("MiniWorld-KeyDoorStory-v0", KeyDoorStoryEnv),
    ]

    registered: List[str] = []
    registry = getattr(gym.envs, "registry", {})
    for env_id, entry_point in env_specs:
        if env_id in registry:
            registered.append(env_id)
            continue
        register(id=env_id, entry_point=entry_point)
        registered.append(env_id)

    return registered
