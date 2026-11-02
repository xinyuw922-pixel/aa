#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from utils.habitat_types import GeneratorConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Habitat-Sim Task A/C/D first-person assets."
    )
    parser.add_argument("--config", required=True, help="Generator JSON config")
    parser.add_argument(
        "--scene-glob",
        required=True,
        help="Glob matching .glb or .scene_instance.json scene files",
    )
    parser.add_argument(
        "--scene-dataset-config",
        default=None,
        help="Optional Habitat scene_dataset_config.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the plan without importing Habitat-Sim",
    )
    return parser.parse_args()


def resolve_scenes(pattern: str, max_scenes: int) -> list[str]:
    paths = sorted(
        str(Path(path).resolve())
        for path in glob.glob(pattern, recursive=True)
        if Path(path).is_file()
    )
    return paths[:max_scenes]


def main() -> int:
    args = parse_args()
    config = GeneratorConfig.from_json(args.config)
    scenes = resolve_scenes(args.scene_glob, config.max_scenes)
    if not scenes:
        print(f"No scene files matched: {args.scene_glob}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "scenes": scenes,
                    "scene_count": len(scenes),
                    "episodes_per_scene": config.episodes_per_scene,
                    "planned_episode_count": (
                        len(scenes) * config.episodes_per_scene
                    ),
                    "output_dir": str(Path(config.output_dir).resolve()),
                    "resolution": [config.width, config.height],
                },
                indent=2,
            )
        )
        return 0

    from tools.habitat.dataset.generator import generate_dataset

    manifest = generate_dataset(
        scenes=scenes,
        config=config,
        scene_dataset_config=args.scene_dataset_config,
    )
    print(json.dumps(manifest["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

