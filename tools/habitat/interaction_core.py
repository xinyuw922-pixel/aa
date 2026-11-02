"""Shared Habitat simulator operations; no batch-generation CLI."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from tools.habitat.dataset.simulator import yaw_quaternion
from utils.habitat_images import image_qc, visual_change
from utils.habitat_geometry import yaw_to_target


def set_agent(sim: Any, habitat_sim: Any, position: np.ndarray, yaw_deg: float) -> None:
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = yaw_quaternion(habitat_sim, yaw_deg)
    sim.get_agent(0).set_state(state, reset_sensors=True)


def observe(sim: Any) -> np.ndarray:
    return np.asarray(sim.get_sensor_observations()["rgb"][:, :, :3]).copy()


def set_joint(obj: Any, offset: int, value: float) -> None:
    positions = np.asarray(obj.joint_positions, dtype=float)
    positions[offset] = float(value)
    obj.joint_positions = positions
    obj.joint_velocities = np.zeros_like(positions)


def candidate_start_poses(
    sim: Any,
    target: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, float]]:
    candidates = []
    phase = float(rng.uniform(0.0, 30.0))
    # Two 0.25 m forward steps follow this pose. Keeping the initial radius
    # above 1.75 m prevents a swung door from filling or crossing the camera.
    for radius in (1.75, 2.00, 2.25):
        for angle_deg in np.arange(phase, phase + 360.0, 30.0):
            angle = math.radians(float(angle_deg))
            requested = np.array(
                [
                    target[0] + radius * math.sin(angle),
                    target[1],
                    target[2] + radius * math.cos(angle),
                ],
                dtype=float,
            )
            snapped = np.asarray(sim.pathfinder.snap_point(requested), dtype=float)
            if np.isnan(snapped).any():
                continue
            distance = float(np.linalg.norm((snapped - target)[[0, 2]]))
            if not 1.45 <= distance <= 2.60:
                continue
            target_yaw = yaw_to_target(snapped, target)
            candidates.append((snapped, target_yaw - 30.0))
    return candidates


def probe(
    habitat_sim: Any,
    sim: Any,
    obj: Any,
    component: dict[str, Any],
    target: np.ndarray,
    starts: list[tuple[np.ndarray, float]],
    min_difference: float,
    *,
    turn_action: str = "turn_left",
    avoid_image: np.ndarray | None = None,
    min_avoid_difference: float = 0.0,
) -> dict[str, Any] | None:
    if turn_action not in {
        "turn_left",
        "turn_right",
        "look_up",
        "look_down",
    }:
        raise ValueError(f"Unsupported probe view action: {turn_action}")
    original_positions = np.asarray(obj.joint_positions, dtype=float).copy()
    best = None
    for start, yaw in starts:
        obj.joint_positions = original_positions
        obj.joint_velocities = np.zeros_like(original_positions)
        set_agent(sim, habitat_sim, start, yaw)
        frames = [observe(sim)]
        positions = [np.asarray(sim.get_agent(0).get_state().position, dtype=float)]
        collisions = [False]
        sim.step(turn_action)
        frames.append(observe(sim))
        positions.append(np.asarray(sim.get_agent(0).get_state().position, dtype=float))
        for _ in range(2):
            before = np.asarray(sim.get_agent(0).get_state().position, dtype=float)
            sim.step("move_forward")
            after = np.asarray(sim.get_agent(0).get_state().position, dtype=float)
            collisions.append(float(np.linalg.norm(after - before)) < 0.15)
            frames.append(observe(sim))
            positions.append(after)
        pre_open = frames[-1]
        pre_qc = image_qc(pre_open)
        set_joint(
            obj,
            component["joint_offset"],
            component["open_position"],
        )
        opened = observe(sim)
        change = visual_change(pre_open, opened)
        avoid_change = (
            visual_change(pre_open, avoid_image)
            if avoid_image is not None
            else None
        )
        score = change["changed_pixel_fraction"]
        bbox = change["changed_bbox_xyxy"]
        border_touches = 0
        if bbox is not None:
            height, width = pre_open.shape[:2]
            border_touches = sum(
                (
                    bbox[0] <= 2,
                    bbox[1] <= 2,
                    bbox[2] >= width - 3,
                    bbox[3] >= height - 3,
                )
            )
        passes = (
            score >= min_difference
            and change["changed_bbox_width"] >= 64
            and change["changed_bbox_height"] >= 64
            and border_touches <= 2
            and pre_qc["mean_luminance"] >= 25.0
            and pre_qc["black_pixel_fraction"] <= 0.60
            and pre_qc["rgb_std"] >= 10.0
            and float(np.linalg.norm(positions[-1] - positions[0])) >= 0.20
            and (
                avoid_change is None
                or avoid_change["changed_pixel_fraction"]
                >= min_avoid_difference
            )
        )
        if passes and (best is None or score > best["visual_change"]["changed_pixel_fraction"]):
            best = {
                "start": start,
                "yaw_deg": yaw,
                "navigation_frames": frames,
                "positions": positions,
                "movement_collisions": collisions,
                "opened_image": opened,
                "visual_change": change,
                "image_qc": pre_qc,
                "avoid_image_change": avoid_change,
                "turn_action": turn_action,
            }
    obj.joint_positions = original_positions
    obj.joint_velocities = np.zeros_like(original_positions)
    return best

