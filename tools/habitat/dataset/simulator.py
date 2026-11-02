from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.habitat_types import GeneratorConfig
from utils.habitat_io import relative_path, save_rgb_png
from utils.habitat_types import Pose, Rollout, Transition


def require_habitat_sim() -> Any:
    try:
        import habitat_sim
    except ImportError as exc:
        raise RuntimeError(
            "habitat_sim is not installed. Create the Python 3.10 conda "
            "environment documented in Habitat/README.md."
        ) from exc
    return habitat_sim


def quaternion_xyzw(rotation: Any) -> tuple[float, float, float, float]:
    imag = rotation.imag
    return (
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
        float(rotation.real),
    )


def pose_from_agent_state(state: Any) -> Pose:
    position = tuple(float(value) for value in state.position)
    return Pose(position=position, rotation_xyzw=quaternion_xyzw(state.rotation))


def yaw_quaternion(habitat_sim: Any, yaw_deg: float) -> Any:
    return habitat_sim.utils.common.quat_from_angle_axis(
        math.radians(yaw_deg),
        np.array([0.0, 1.0, 0.0]),
    )


def scene_navmesh_cache_path(scene: str, cache_dir: str | Path) -> Path:
    """Return a stable cache path for GLB or scene-instance inputs."""
    name = Path(scene).name
    for suffix in (".scene_instance.json", ".glb"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return Path(cache_dir).expanduser().resolve() / f"{name}.navmesh"


def ensure_navmesh(
    habitat_sim: Any,
    simulator: Any,
    scene: str,
    config: GeneratorConfig,
) -> Path | None:
    """Load or build a navmesh according to the generator configuration."""
    mode = config.navmesh_mode
    cache_path: Path | None = None
    if config.navmesh_cache_dir:
        cache_path = scene_navmesh_cache_path(
            scene, config.navmesh_cache_dir
        )

    if mode != "recompute" and simulator.pathfinder.is_loaded:
        return None

    if (
        mode == "load_or_recompute"
        and cache_path is not None
        and cache_path.is_file()
    ):
        if simulator.pathfinder.load_nav_mesh(str(cache_path)):
            return cache_path

    if mode == "require":
        raise RuntimeError(
            f"No navigable navmesh was loaded for scene {scene}. "
            "Place the matching .navmesh beside the scene or set "
            "navmesh_mode=load_or_recompute with navmesh_cache_dir."
        )

    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_height = config.agent_height_m
    settings.agent_radius = config.agent_radius_m
    settings.include_static_objects = config.navmesh_include_static_objects
    if not simulator.recompute_navmesh(simulator.pathfinder, settings):
        raise RuntimeError(f"Could not recompute navmesh for scene {scene}")
    if not simulator.pathfinder.is_loaded:
        raise RuntimeError(
            f"Recomputed navmesh was not loaded for scene {scene}"
        )

    assert cache_path is not None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not simulator.pathfinder.save_nav_mesh(str(cache_path)):
        raise RuntimeError(f"Could not save navmesh cache: {cache_path}")
    return cache_path


def configure_simulator(
    scene: str,
    config: GeneratorConfig,
    scene_dataset_config: str | None,
) -> tuple[Any, Any]:
    habitat_sim = require_habitat_sim()

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene
    backend.enable_physics = config.enable_physics
    backend.gpu_device_id = config.gpu_device_id
    if scene_dataset_config:
        backend.scene_dataset_config_file = scene_dataset_config

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb.resolution = [config.height, config.width]
    rgb.position = [0.0, config.sensor_height_m, 0.0]

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.height = config.agent_height_m
    agent_config.radius = config.agent_radius_m
    agent_config.sensor_specifications = [rgb]
    agent_config.action_space = {
        "move_forward": habitat_sim.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=config.forward_step_m),
        ),
        "turn_left": habitat_sim.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=config.turn_angle_deg),
        ),
        "turn_right": habitat_sim.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=config.turn_angle_deg),
        ),
        "look_up": habitat_sim.ActionSpec(
            "look_up",
            habitat_sim.agent.ActuationSpec(amount=config.look_angle_deg),
        ),
        "look_down": habitat_sim.ActionSpec(
            "look_down",
            habitat_sim.agent.ActuationSpec(amount=config.look_angle_deg),
        ),
    }
    simulator = habitat_sim.Simulator(
        habitat_sim.Configuration(backend, [agent_config])
    )
    try:
        ensure_navmesh(habitat_sim, simulator, scene, config)
    except Exception:
        simulator.close()
        raise
    return habitat_sim, simulator


@dataclass(frozen=True)
class SampledEpisode:
    start: tuple[float, float, float]
    goal: tuple[float, float, float]
    initial_yaw_deg: float
    geodesic_distance_m: float
    actions: tuple[str, ...]


def _has_overhead_cover(
    habitat_sim: Any,
    simulator: Any,
    point: np.ndarray,
    max_distance_m: float,
) -> bool:
    origin = np.asarray(point, dtype=np.float32) + np.array(
        [0.0, 0.05, 0.0], dtype=np.float32
    )
    ray = habitat_sim.geo.Ray(
        origin, np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    return bool(simulator.cast_ray(ray, max_distance_m).has_hits())


def _path_has_overhead_cover(
    habitat_sim: Any,
    simulator: Any,
    points: list[Any],
    max_distance_m: float,
    sample_spacing_m: float,
) -> bool:
    """Reject routes that leave the furnished indoor shell."""
    if not points:
        return False
    samples = [np.asarray(points[0], dtype=float)]
    for raw_start, raw_end in zip(points, points[1:]):
        start = np.asarray(raw_start, dtype=float)
        end = np.asarray(raw_end, dtype=float)
        length = float(np.linalg.norm(end - start))
        count = max(1, int(math.ceil(length / sample_spacing_m)))
        samples.extend(
            start + (end - start) * (index / count)
            for index in range(1, count + 1)
        )
    return all(
        _has_overhead_cover(
            habitat_sim, simulator, point, max_distance_m
        )
        for point in samples
    )


class EpisodeSamplingError(RuntimeError):
    """No trajectory satisfied the configured sampling constraints."""


def sample_expert_episode(
    habitat_sim: Any,
    simulator: Any,
    config: GeneratorConfig,
    seed: int,
    fixed_goal: tuple[float, float, float] | None = None,
    excluded_start: tuple[float, float, float] | None = None,
) -> SampledEpisode:
    simulator.seed(seed)
    rng = np.random.default_rng(seed)
    agent = simulator.get_agent(0)

    indoor_points: list[np.ndarray] | None = None
    if config.require_overhead_cover:
        indoor_points = []
        for _ in range(config.overhead_point_pool_samples):
            point = np.asarray(
                simulator.pathfinder.get_random_navigable_point(),
                dtype=float,
            )
            if _has_overhead_cover(
                habitat_sim,
                simulator,
                point,
                config.overhead_cover_max_distance_m,
            ):
                indoor_points.append(point)
        if len(indoor_points) < 2:
            raise EpisodeSamplingError(
                "Could not build an indoor navigable-point pool: "
                f"found {len(indoor_points)} covered points from "
                f"{config.overhead_point_pool_samples} samples."
            )

    for _ in range(config.max_sampling_attempts):
        if indoor_points is None:
            start = np.asarray(
                simulator.pathfinder.get_random_navigable_point(), dtype=float
            )
            if fixed_goal is None:
                goal = np.asarray(
                    simulator.pathfinder.get_random_navigable_point(),
                    dtype=float,
                )
            else:
                goal = np.asarray(fixed_goal, dtype=float)
        else:
            start = indoor_points[int(rng.integers(len(indoor_points)))]
            if fixed_goal is None:
                goal = indoor_points[int(rng.integers(len(indoor_points)))]
            else:
                goal = np.asarray(fixed_goal, dtype=float)
        if (
            excluded_start is not None
            and np.linalg.norm(
                start - np.asarray(excluded_start, dtype=float)
            )
            < config.min_geodesic_distance_m
        ):
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        if not simulator.pathfinder.find_path(path):
            continue
        distance = float(path.geodesic_distance)
        if not (
            config.min_geodesic_distance_m
            <= distance
            <= config.max_geodesic_distance_m
        ):
            continue
        if config.require_overhead_cover and not _path_has_overhead_cover(
            habitat_sim=habitat_sim,
            simulator=simulator,
            points=list(path.points),
            max_distance_m=config.overhead_cover_max_distance_m,
            sample_spacing_m=config.forward_step_m,
        ):
            continue

        yaw = float(rng.uniform(-180.0, 180.0))
        state = habitat_sim.AgentState()
        state.position = start
        state.rotation = yaw_quaternion(habitat_sim, yaw)
        agent.set_state(state, reset_sensors=True)

        follower = simulator.make_greedy_follower(
            agent_id=0,
            goal_radius=config.goal_radius_m,
            stop_key="stop",
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )
        try:
            actions = follower.find_path(goal)
        except habitat_sim.errors.GreedyFollowerError:
            continue
        if not actions:
            continue
        normalized = tuple(str(action) for action in actions)
        if normalized[-1] != "stop":
            normalized = (*normalized, "stop")
        motion_count = sum(action != "stop" for action in normalized)
        if 0 < motion_count <= config.max_episode_steps:
            return SampledEpisode(
                start=tuple(float(value) for value in start),
                goal=tuple(float(value) for value in goal),
                initial_yaw_deg=yaw,
                geodesic_distance_m=distance,
                actions=normalized,
            )
    raise EpisodeSamplingError(
        f"Could not sample a valid expert episode after "
        f"{config.max_sampling_attempts} attempts (seed={seed})."
    )


def set_agent_start(
    habitat_sim: Any,
    simulator: Any,
    start: tuple[float, float, float],
    yaw_deg: float,
) -> None:
    state = habitat_sim.AgentState()
    state.position = np.asarray(start, dtype=float)
    state.rotation = yaw_quaternion(habitat_sim, yaw_deg)
    simulator.get_agent(0).set_state(state, reset_sensors=True)


def distance_to_goal(simulator: Any, goal: tuple[float, float, float]) -> float:
    current = np.asarray(simulator.get_agent(0).get_state().position, dtype=float)
    return float(np.linalg.norm(current - np.asarray(goal, dtype=float)))


def movement_collided(
    action: str,
    position_before: np.ndarray,
    position_after: np.ndarray,
    forward_step_m: float,
    tolerance_m: float = 1e-4,
) -> bool:
    """Infer navmesh collision from a clipped forward displacement."""
    if action != "move_forward":
        return False
    actual = float(np.linalg.norm(position_after - position_before))
    return actual + tolerance_m < forward_step_m


def run_rollout(
    habitat_sim: Any,
    simulator: Any,
    start: tuple[float, float, float],
    yaw_deg: float,
    goal: tuple[float, float, float],
    actions: tuple[str, ...],
    output_dir: Path,
    output_root: Path,
    goal_radius_m: float,
    forward_step_m: float,
) -> Rollout:
    set_agent_start(habitat_sim, simulator, start, yaw_deg)
    observations = simulator.get_sensor_observations()
    transitions: list[Transition] = []

    initial_path = output_dir / "frames" / "frame_0000.png"
    save_rgb_png(initial_path, observations["rgb"])
    transitions.append(
        Transition(
            frame_index=0,
            image_path=relative_path(initial_path, output_root),
            action_from_previous=None,
            pose=pose_from_agent_state(simulator.get_agent(0).get_state()),
            collided=False,
        )
    )

    executed: list[str] = []
    for action in actions:
        if action == "stop":
            executed.append(action)
            break
        if action in {"open", "pickup"}:
            raise NotImplementedError(
                f"{action} requires a physics interaction adapter"
            )
        position_before = np.asarray(
            simulator.get_agent(0).get_state().position, dtype=float
        )
        observations = simulator.step(action)
        current_state = simulator.get_agent(0).get_state()
        position_after = np.asarray(current_state.position, dtype=float)
        executed.append(action)
        frame_index = len(transitions)
        image_path = output_dir / "frames" / f"frame_{frame_index:04d}.png"
        save_rgb_png(image_path, observations["rgb"])
        transitions.append(
            Transition(
                frame_index=frame_index,
                image_path=relative_path(image_path, output_root),
                action_from_previous=action,
                pose=pose_from_agent_state(current_state),
                collided=movement_collided(
                    action,
                    position_before,
                    position_after,
                    forward_step_m,
                ),
            )
        )

    final_distance = distance_to_goal(simulator, goal)
    return Rollout(
        transitions=tuple(transitions),
        actions=tuple(executed),
        final_distance_to_goal_m=final_distance,
        success=final_distance <= goal_radius_m,
    )


def find_cf_yaw(
    habitat_sim: Any,
    simulator: Any,
    episode: SampledEpisode,
    offsets: tuple[float, ...],
    goal_radius_m: float,
) -> tuple[float, float, bool]:
    """Find a yaw intervention that makes the Full action replay fail."""
    motion_actions = tuple(action for action in episode.actions if action != "stop")
    fallback: tuple[float, float, bool] | None = None
    for offset in offsets:
        set_agent_start(
            habitat_sim,
            simulator,
            episode.start,
            episode.initial_yaw_deg + offset,
        )
        for action in motion_actions:
            simulator.step(action)
        distance = distance_to_goal(simulator, episode.goal)
        success = distance <= goal_radius_m
        candidate = (float(offset), distance, success)
        if fallback is None or distance > fallback[1]:
            fallback = candidate
        if not success:
            return candidate
    assert fallback is not None
    return fallback


def render_task_a_candidates(
    habitat_sim: Any,
    simulator: Any,
    start: tuple[float, float, float],
    yaw_deg: float,
    prefix_actions: tuple[str, ...],
    candidate_action_names: list[str],
    output_dir: Path,
    output_root: Path,
) -> list[str]:
    set_agent_start(habitat_sim, simulator, start, yaw_deg)
    for action in prefix_actions:
        simulator.step(action)
    anchor_state = simulator.get_agent(0).get_state()

    paths: list[str] = []
    for index, action in enumerate(candidate_action_names):
        # Expert PointNav prefixes contain body actions only. Resetting sensor
        # nodes here is essential because look_up/look_down distractors mutate
        # camera state independently of the body and would otherwise leak into
        # candidates rendered later in the loop.
        simulator.get_agent(0).set_state(anchor_state, reset_sensors=True)
        observation = simulator.step(action)
        path = output_dir / f"candidate_{index}.png"
        save_rgb_png(path, observation["rgb"])
        paths.append(relative_path(path, output_root))
    simulator.get_agent(0).set_state(anchor_state, reset_sensors=True)
    return paths


def render_action_sequence_candidates(
    habitat_sim: Any,
    simulator: Any,
    position: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    action_sequences: dict[str, tuple[str, ...]],
    output_dir: Path,
    output_root: Path,
    forward_step_m: float,
) -> dict[str, dict[str, Any]]:
    """Render macro-action branches from one exactly shared agent pose."""
    anchor = habitat_sim.AgentState()
    anchor.position = np.asarray(position, dtype=float)
    anchor.rotation = habitat_sim.utils.common.quat_from_coeffs(
        np.asarray(rotation_xyzw, dtype=float)
    )
    results: dict[str, dict[str, Any]] = {}
    for label, actions in action_sequences.items():
        simulator.get_agent(0).set_state(anchor, reset_sensors=True)
        collisions: list[bool] = []
        observation = simulator.get_sensor_observations()
        for action in actions:
            position_before = np.asarray(
                simulator.get_agent(0).get_state().position, dtype=float
            )
            observation = simulator.step(action)
            position_after = np.asarray(
                simulator.get_agent(0).get_state().position, dtype=float
            )
            collisions.append(
                movement_collided(
                    action,
                    position_before,
                    position_after,
                    forward_step_m,
                )
            )
        path = output_dir / f"{label}.png"
        save_rgb_png(path, observation["rgb"])
        final_state = simulator.get_agent(0).get_state()
        results[label] = {
            "image_path": relative_path(path, output_root),
            "action_sequence": list(actions),
            "final_pose": pose_from_agent_state(final_state).to_dict(),
            "collisions": collisions,
        }
    simulator.get_agent(0).set_state(anchor, reset_sensors=True)
    return results


def render_goal_image(
    habitat_sim: Any,
    simulator: Any,
    goal_position: tuple[float, float, float],
    rotation_xyzw: tuple[float, float, float, float],
    output_path: Path,
    output_root: Path,
) -> tuple[str, Pose]:
    """Render a goal view at the oracle target with the Full terminal yaw."""
    state = habitat_sim.AgentState()
    state.position = np.asarray(goal_position, dtype=float)
    state.rotation = habitat_sim.utils.common.quat_from_coeffs(
        np.asarray(rotation_xyzw, dtype=float)
    )
    simulator.get_agent(0).set_state(state, reset_sensors=True)
    observation = simulator.get_sensor_observations()
    save_rgb_png(output_path, observation["rgb"])
    rendered_pose = pose_from_agent_state(simulator.get_agent(0).get_state())
    return relative_path(output_path, output_root), rendered_pose
