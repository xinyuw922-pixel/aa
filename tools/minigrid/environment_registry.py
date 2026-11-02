"""MiniGrid production environment registry."""
from __future__ import annotations
from typing import Any

TASKS: list[dict[str, Any]] = [
    {
        "env_key": "doorkey",
        "module": 'tools/minigrid/environments/doorkey.py',
        "seed_start": 1000,
        "gen_args": ["--env-id", "MiniGrid-DoorKey-8x8-v0"],
        "validate_args": [],
    },
    {
        "env_key": "lavagap",
        "module": 'tools/minigrid/environments/lavagap.py',
        "seed_start": 3000,
        "gen_args": ["--env-id", "MiniGrid-LavaGapS7-v0"],
        "validate_args": [],
    },
    {
        "env_key": "keycorridor",
        "module": 'tools/minigrid/environments/keycorridor.py',
        "seed_start": 4000,
        "gen_args": ["--env-id", "MiniGrid-KeyCorridorS6R3-v0", "--log-every", "1"],
        "validate_args": [],
    },
    {
        "env_key": "memory",
        "module": 'tools/minigrid/environments/memory.py',
        "seed_start": 2000,
        "gen_args": ["--env-id", "MiniGrid-MemoryS13-v0", "--log-every", "1"],
        "validate_args": [],
    },
    {
        "env_key": "multiroom",
        "module": 'tools/minigrid/environments/multiroom.py',
        "seed_start": 5000,
        "gen_args": ["--env-id", "MiniGrid-MultiRoom-N6-v0"],
        "validate_args": [],
    },
    {
        "env_key": "redblue",
        "module": 'tools/minigrid/environments/redblue.py',
        "seed_start": 6000,
        "gen_args": ["--env-id", "MiniGrid-RedBlueDoors-8x8-v0"],
        "validate_args": [],
    },
]
