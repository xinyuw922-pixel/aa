"""Leaf Habitat configuration and trajectory types; standard-library dependencies only."""

from __future__ import annotations

import json


from dataclasses import asdict, dataclass, fields


from pathlib import Path


from typing import Any


from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorConfig:
    output_dir: str = "Habitat/outputs/pilot"
    base_seed: int = 20260725
    max_scenes: int = 10
    episodes_per_scene: int = 20
    width: int = 512
    height: int = 512
    sensor_height_m: float = 1.5
    agent_height_m: float = 1.5
    agent_radius_m: float = 0.1
    forward_step_m: float = 0.25
    turn_angle_deg: float = 30.0
    look_angle_deg: float = 30.0
    gpu_device_id: int = 0
    goal_radius_m: float = 0.35
    min_geodesic_distance_m: float = 2.0
    max_geodesic_distance_m: float = 12.0
    max_episode_steps: int = 256
    max_sampling_attempts: int = 100
    reference_sampling_retries: int = 8
    task_a_per_episode: int = 1
    generate_task_a: bool = True
    cf_yaw_offsets_deg: tuple[float, ...] = (
        30.0,
        -30.0,
        60.0,
        -60.0,
        90.0,
        -90.0,
        120.0,
        -120.0,
        150.0,
        -150.0,
        180.0,
    )
    enable_physics: bool = False
    save_task_c_images: bool = False
    navmesh_mode: str = "require"
    navmesh_cache_dir: str | None = None
    navmesh_include_static_objects: bool = False
    require_overhead_cover: bool = False
    overhead_cover_max_distance_m: float = 5.0
    overhead_point_pool_samples: int = 512

    @classmethod
    def from_json(cls, path: str | Path) -> "GeneratorConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        if "cf_yaw_offsets_deg" in payload:
            payload["cf_yaw_offsets_deg"] = tuple(payload["cf_yaw_offsets_deg"])
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        positive_ints = {
            "max_scenes": self.max_scenes,
            "episodes_per_scene": self.episodes_per_scene,
            "width": self.width,
            "height": self.height,
            "max_episode_steps": self.max_episode_steps,
            "max_sampling_attempts": self.max_sampling_attempts,
            "reference_sampling_retries": self.reference_sampling_retries,
            "task_a_per_episode": self.task_a_per_episode,
            "overhead_point_pool_samples": self.overhead_point_pool_samples,
        }
        bad = [name for name, value in positive_ints.items() if value <= 0]
        if bad:
            raise ValueError(f"These values must be positive: {', '.join(bad)}")
        if self.min_geodesic_distance_m <= 0:
            raise ValueError("min_geodesic_distance_m must be positive")
        if self.max_geodesic_distance_m <= self.min_geodesic_distance_m:
            raise ValueError(
                "max_geodesic_distance_m must exceed min_geodesic_distance_m"
            )
        if not self.cf_yaw_offsets_deg:
            raise ValueError("cf_yaw_offsets_deg must not be empty")
        if self.gpu_device_id < -1:
            raise ValueError("gpu_device_id must be -1 or a non-negative integer")
        if self.navmesh_mode not in {
            "require",
            "load_or_recompute",
            "recompute",
        }:
            raise ValueError(
                "navmesh_mode must be require, load_or_recompute, or recompute"
            )
        if (
            self.navmesh_mode != "require"
            and not self.navmesh_cache_dir
        ):
            raise ValueError(
                "navmesh_cache_dir is required when navmesh_mode can recompute"
            )
        if self.overhead_cover_max_distance_m <= 0:
            raise ValueError("overhead_cover_max_distance_m must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cf_yaw_offsets_deg"] = list(self.cf_yaw_offsets_deg)
        return payload


def episode_seed(base_seed: int, scene_index: int, episode_index: int) -> int:
    """Derive stable, non-overlapping seeds without Python's salted hash()."""
    if scene_index < 0 or episode_index < 0:
        raise ValueError("scene_index and episode_index must be non-negative")
    return int(base_seed + scene_index * 100_000 + episode_index)


NAVIGATION_ACTIONS = (
    "move_forward",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "stop",
)


RESERVED_INTERACTION_ACTIONS = ("open", "pickup")


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": list(self.position),
            "rotation_xyzw": list(self.rotation_xyzw),
        }


@dataclass(frozen=True)
class Transition:
    frame_index: int
    image_path: str
    action_from_previous: str | None
    pose: Pose
    collided: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "image_path": self.image_path,
            "action_from_previous": self.action_from_previous,
            "pose": self.pose.to_dict(),
            "collided": self.collided,
        }


@dataclass(frozen=True)
class Rollout:
    transitions: tuple[Transition, ...]
    actions: tuple[str, ...]
    final_distance_to_goal_m: float
    success: bool

