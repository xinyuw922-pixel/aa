#!/usr/bin/env python3
"""Generate FourRooms successor-choice questions with a current image,
a navigation sequence and correct/alternative candidate images."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import subprocess
from pathlib import Path
from typing import Any
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from tools.miniworld.common import CANDIDATE_ENVS, concat_vertical_rgb, dump_json, ensure_output_dir, export_rgb, make_env, maybe_render_top_view, maybe_render_top_view_with_scale, runtime_context, try_import_stack, action_name
else:
    from tools.miniworld.common import CANDIDATE_ENVS, concat_vertical_rgb, dump_json, ensure_output_dir, export_rgb, make_env, maybe_render_top_view, maybe_render_top_view_with_scale, runtime_context, try_import_stack, action_name


# ---------------------------------------------------------------------------
# Human-facing macro actions.
# We expose 30°/60° turns and 5-step/10-step forward moves for readability,
# then compile them to the primitive MiniWorld actions internally.
MACRO_ACTIONS = {
    "turn_left_30": ["turn_left"] * 2,
    "turn_right_30": ["turn_right"] * 2,
    "turn_left_60": ["turn_left"] * 4,
    "turn_right_60": ["turn_right"] * 4,
    "forward_5": ["move_forward"] * 5,
    "forward_10": ["move_forward"] * 10,
}

# Candidate navigation action sequences.
# All navigation-focused (FourRooms environment)
# Each entry: (macro_correct_sequence, macro_distractor_sequence)
# Curated for human review with visibly different terminal views.
# ---------------------------------------------------------------------------
ACTION_SEQUENCE_POOL = [
    (["turn_left_30", "forward_5"], ["turn_right_30", "forward_5"]),
    (["turn_right_30", "forward_5"], ["turn_left_30", "forward_5"]),
    (["forward_5", "turn_left_30"], ["forward_5", "turn_right_30"]),
    (["forward_10"], ["turn_left_30", "forward_5"]),
    (["forward_5", "turn_left_30", "forward_5"], ["forward_5", "turn_right_30", "forward_5"]),
    (["turn_left_60", "forward_10"], ["turn_right_60", "forward_10"]),
    (["forward_5", "forward_5"], ["turn_left_30", "forward_10"]),
    (["turn_left_30", "forward_10"], ["turn_right_30", "forward_10"]),
    (["forward_10", "turn_left_30"], ["turn_right_30", "forward_10"]),
    (["turn_right_60", "forward_5"], ["turn_left_30", "forward_10"]),
    (["forward_10", "forward_5"], ["forward_5", "turn_left_60"]),
    (["turn_left_30", "forward_5", "forward_5"], ["turn_right_30", "forward_5", "forward_5"]),
    (["turn_right_30", "forward_5", "forward_5"], ["turn_left_30", "forward_5", "forward_5"]),
    (["turn_left_60", "forward_5"], ["turn_right_60", "forward_5"]),
    (["forward_5", "turn_left_30", "forward_10"], ["forward_5", "turn_right_30", "forward_10"]),
]

# Action name to ID mapping for FourRooms (matches MiniWorldEnv.Actions IntEnum)
ACTION_MAP = {
    "turn_left": 0,
    "turn_right": 1,
    "move_forward": 2,
}

# If both candidates keep the same cardinal direction and their terminal
# positions are too close, the question is usually too subtle for humans.
MIN_SAME_DIR_POS_DELTA = 0.12
MAX_CANDIDATE_EGO_MATCH_RATE = 0.80
DEFAULT_EASY_RATIO = 0.67


def expand_macro_sequence(sequence: list[str]) -> list[str]:
    primitive: list[str] = []
    for action_name_str in sequence:
        primitive.extend(MACRO_ACTIONS[action_name_str])
    return primitive


def get_room_index(env, pos_xz: list[float]) -> int | None:
    """Return the index of the room containing the given x/z position."""
    target = getattr(env, "unwrapped", env)
    rooms = getattr(target, "rooms", [])
    if not rooms:
        return None
    x, z = pos_xz
    for idx, room in enumerate(rooms):
        if room.min_x <= x <= room.max_x and room.min_z <= z <= room.max_z:
            return idx
    return None


def get_red_box_reference(env, current_state: dict[str, Any]) -> dict[str, Any]:
    """Collect metadata about the red box as a human reference object."""
    target = getattr(env, "unwrapped", env)
    box = getattr(target, "box", None)
    box_pos = None
    box_room_idx = None
    visible = False

    if box is not None and hasattr(box, "pos"):
        box_pos = [round(float(box.pos[0]), 3), round(float(box.pos[2]), 3)]
        box_room_idx = get_room_index(env, box_pos)

    current_room_idx = get_room_index(env, current_state["pos"])

    try:
        vis_ents = getattr(target, "get_visible_ents")()
        visible = box in vis_ents if box is not None else False
    except Exception:
        visible = False

    same_room = (
        current_room_idx is not None
        and box_room_idx is not None
        and current_room_idx == box_room_idx
    )

    return {
        "red_box_pos": box_pos,
        "red_box_room_idx": box_room_idx,
        "current_room_idx": current_room_idx,
        "red_box_visible_in_current_view": bool(visible),
        "current_room_has_red_box": bool(same_room),
        "difficulty_bucket": "easy" if same_room else "hard",
    }


def render_state_images(env, obs, img_dir, prefix):
    """Render current state as ego_topview_concat image."""
    top = maybe_render_top_view(env)
    if top is None:
        raise RuntimeError("Topview rendering failed")
    concat = concat_vertical_rgb(obs, top)
    img_path = img_dir / f"{prefix}.png"
    export_rgb(img_path, concat)
    return img_path


def get_motion_contract(env) -> dict[str, float]:
    """Return the deterministic per-step motion contract used for human review."""
    target = getattr(env, "unwrapped", env)
    params = getattr(target, "params", None)
    if params is None:
        return {"forward_step": 0.15, "turn_step_deg": 15.0}
    return {
        "forward_step": float(params.sample(None, "forward_step")),
        "turn_step_deg": float(params.sample(None, "turn_step")),
    }


def execute_sequence(env, sequence):
    """Execute an action sequence from current state. Returns final obs."""
    obs, info = env.reset()
    for action_name_str in sequence:
        action_id = ACTION_MAP[action_name_str]
        obs, reward, terminated, truncated, info = env.step(action_id)
    return obs


def get_agent_state(env) -> dict:
    """Extract agent position and direction from a MiniWorld env.

    Correct attribute path for FourRooms:
      - env.unwrapped.agent.pos       → numpy array [x, y, z]
      - env.unwrapped.agent.dir_vec   → numpy array [dx, 0, dz] (normalized)

    Returns {"pos": [x, z], "cardinal": "N|E|S|W|?"}.
    """
    target = getattr(env, "unwrapped", env)

    # --- position (x, z from the 3D pos vector; y is always 0 in FourRooms) ---
    try:
        agent = getattr(target, "agent", None)
        if agent is not None:
            pos = getattr(agent, "pos", None)
            if pos is not None and hasattr(pos, "__iter__"):
                px = float(pos[0])
                pz = float(pos[2])  # FourRooms is x-z plane
            else:
                px = pz = 0.0
        else:
            px = pz = 0.0
    except Exception:
        px = pz = 0.0

    # --- direction vector ---
    try:
        agent = getattr(target, "agent", None)
        if agent is not None:
            dir_angle = float(getattr(agent, "dir", 0.0))
            import math

            def angle_to_cardinal_4way(dir_rad: float) -> str:
                """Map direction angle to N/E/S/W.

                MiniWorld convention: dir=0 → facing +z (South on grid).
                FourRooms initial dirs are always N/E/S/W (no diagonals).
                We also accept diagonal-starting positions.
                """
                # Normalize to [0, 2π)
                angle = dir_rad % (2 * math.pi)
                if angle < 0:
                    angle += 2 * math.pi
                # Sector size = π/2.  sector 0=S, 1=W, 2=N, 3=E (counter-clockwise from +z)
                sector = round(angle / (math.pi / 2)) % 4
                return ["S", "W", "N", "E"][sector]

            cardinal = angle_to_cardinal_4way(dir_angle)
        else:
            cardinal = "?"
    except Exception:
        cardinal = "?"

    return {"pos": [round(px, 3), round(pz, 3)], "cardinal": cardinal}


EN_ACTION_LABELS = {
    # Macro names encode the human-facing total angle (30°=1-unit, 60°=2-unit).
    # Actual primitive count: turn_left_30 → 2× primitive turns, turn_left_60 → 4×.
    # turn_step_deg is 15° in FourRooms, so 1-unit=30° and 2-unit=60° — both correct.
    "turn_left_30": "turn left by one unit (30°)",
    "turn_right_30": "turn right by one unit (30°)",
    "turn_left_60": "turn left by two units (60°)",
    "turn_right_60": "turn right by two units (60°)",
    "forward_5": "move forward by one unit",
    "forward_10": "move forward by two units",
}


def _infer_strategy(seq: list[str], dist_seq: list[str]) -> str:
    """Determine which perturbation strategy was used for a question.

    - alternative_prefix: first action differs (preferred strategy A)
    - alternative_suffix_posN: first differing action is at position N > 0
    """
    if len(seq) != len(dist_seq):
        return "different_length"
    diffs = [(i, seq[i], dist_seq[i]) for i in range(len(seq)) if seq[i] != dist_seq[i]]
    if not diffs:
        return "identical_sequences"
    return (
        "alternative_prefix"
        if diffs[0][0] == 0
        else f"alternative_suffix_pos{diffs[0][0]}"
    )


def _make_distractor_source(seq: list[str], dist_seq: list[str]) -> str:
    """Build the distractor_source field value from the two sequences."""
    return f"{_infer_strategy(seq, dist_seq)}_rollout"


def _state_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.dist(a["pos"], b["pos"])


def _is_too_subtle_pair(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Reject pairs that are technically different but visually too similar."""
    same_cardinal = a["cardinal"] == b["cardinal"]
    return same_cardinal and _state_distance(a, b) < MIN_SAME_DIR_POS_DELTA


def _ego_panels_identical(path_a: Path, path_b: Path) -> bool:
    """Return True if the upper-half ego panels are pixel-identical."""
    with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
        a = im_a.convert("RGB")
        b = im_b.convert("RGB")
        ego_h = min(a.height, b.height) // 2
        return list(a.crop((0, 0, a.width, ego_h)).getdata()) == list(
            b.crop((0, 0, b.width, ego_h)).getdata()
        )


def _ego_match_rate(path_a: Path, path_b: Path) -> float:
    """Return the per-pixel RGB equality rate for the upper-half ego panels."""
    with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
        a = im_a.convert("RGB")
        b = im_b.convert("RGB")
        ego_h = min(a.height, b.height) // 2
        a_crop = a.crop((0, 0, a.width, ego_h))
        b_crop = b.crop((0, 0, b.width, ego_h))
        a_arr = list(a_crop.getdata())
        b_arr = list(b_crop.getdata())
        if not a_arr:
            return 1.0
        same = sum(1 for px_a, px_b in zip(a_arr, b_arr) if px_a == px_b)
        return same / len(a_arr)


def en_seq(sequence: list[str]) -> str:
    return " -> ".join(EN_ACTION_LABELS.get(action, action) for action in sequence)


def build_prompt(sequence: list[str], turn_step_deg: float = 15.0) -> str:
    """Build the question prompt.

    Args:
        sequence: list of macro action names (e.g. ["turn_left_30", "forward_5"])
        turn_step_deg: actual turn step in degrees from the environment.
            In FourRooms this is 15°.  Macro name encoding:
              - turn_left_30 / turn_right_30 → 2 primitive turns = 2×turn_step_deg
              - turn_left_60 / turn_right_60 → 4 primitive turns = 4×turn_step_deg
              - forward_5 → 5 primitive forward steps
              - forward_10 → 10 primitive forward steps
            The macro name's numeric suffix is always the total angle/steps (not a
            per-unit count), so it stays correct regardless of turn_step_deg.
    """
    # Describe the macro action system.
    # turn_step_deg is used to document what each primitive turn does.
    # turn_left_30 always means 30° total regardless of turn_step_deg.
    deg_per_primitive_turn = turn_step_deg
    deg_for_30_macro = 2 * deg_per_primitive_turn  # always = 30°
    deg_for_60_macro = 4 * deg_per_primitive_turn  # always = 60°

    actions_str = en_seq(sequence)
    return (
        "Task: Predict the true successor state.\n"
        "You are given the agent's current state and two candidate successor views.\n"
        "Action units used in this task:\n"
        f"- move forward by one unit = 5 primitive environment forward steps\n"
        f"- move forward by two units = 10 primitive environment forward steps\n"
        "- turn left/right by one unit = turn left/right by 30°\n"
        "- turn left/right by two units = turn left/right by 60°\n"
        "(Each primitive turn step changes the agent's heading by 1× the environment's "
        f"turn step; in this environment 1 turn step = {deg_per_primitive_turn:.1f}°, "
        f"so 2 turn steps = {deg_for_30_macro:.1f}° and 4 turn steps = {deg_for_60_macro:.1f}°.)\n"
        "The top-view grid uses the same distance scale: one grid cell equals one forward unit.\n"
        f"Starting from the current state, execute the following action sequence:\n[{actions_str}]\n"
        "Which candidate view, A or B, is the true successor state after the full sequence?\n"
        "Answer with a single letter: A or B."
    )


def make_question(
    env_id: str,
    seed: int,
    env_slug: str,
    obs_width: int,
    obs_height: int,
    correct_seq: list[str],
    distractor_seq: list[str],
    question_idx: int,
    img_dir: Path,
) -> dict[str, Any]:
    """Generate a question from post-action observations and terminal poses.

    Step through the environment wrapper and render the returned observations.
    Record current/correct/distractor poses independently of candidate placement.
    """
    gym, _ = try_import_stack()

    # ---- Current state (same env, same seed for both rollouts) ----
    env0, obs0, _ = make_env(
        gym, env_id, seed=seed,
        obs_width=obs_width, obs_height=obs_height,
        window_width=obs_width, window_height=obs_height,
    )
    current_img = render_state_images(env0, obs0, img_dir, f"q{question_idx:02d}_current")
    current_state = get_agent_state(env0)
    _, current_topview_scale = maybe_render_top_view_with_scale(env0)
    motion_contract = get_motion_contract(env0)
    red_box_reference = get_red_box_reference(env0, current_state)

    primitive_correct_seq = expand_macro_sequence(correct_seq)
    primitive_distractor_seq = expand_macro_sequence(distractor_seq)

    # ---- Correct candidate: execute correct_seq from current state ----
    # Use env.step() standard path so returned obs is always post-action.
    correct_env, correct_obs, _ = make_env(
        gym, env_id, seed=seed,
        obs_width=obs_width, obs_height=obs_height,
        window_width=obs_width, window_height=obs_height,
    )
    for action_name_str in primitive_correct_seq:
        correct_obs, reward, terminated, truncated, info = correct_env.step(
            ACTION_MAP[action_name_str]
        )
    correct_img = render_state_images(
        correct_env, correct_obs, img_dir, f"q{question_idx:02d}_candidate_correct"
    )
    correct_final_state = get_agent_state(correct_env)
    correct_env.close()

    # ---- Distractor candidate: execute distractor_seq from same initial state ----
    dist_env, dist_obs, _ = make_env(
        gym, env_id, seed=seed,
        obs_width=obs_width, obs_height=obs_height,
        window_width=obs_width, window_height=obs_height,
    )
    for action_name_str in primitive_distractor_seq:
        dist_obs, reward, terminated, truncated, info = dist_env.step(
            ACTION_MAP[action_name_str]
        )
    dist_img = render_state_images(
        dist_env, dist_obs, img_dir, f"q{question_idx:02d}_candidate_distractor"
    )
    dist_final_state = get_agent_state(dist_env)
    dist_env.close()

    env0.close()

    # ---- Degenerate state guard ----
    # Raise so main() can retry with a different seed.
    # Primary: candidates identical (no valid question).  Secondary: correct == current (no info).
    if (correct_final_state["pos"] == dist_final_state["pos"]
            and correct_final_state["cardinal"] == dist_final_state["cardinal"]):
        raise ValueError(
            f"Degenerate: correct_seq and distractor_seq reach identical terminal "
            f"state pos={correct_final_state['pos']} dir={correct_final_state['cardinal']} "
            f"(seed={seed}, correct_seq={correct_seq}, distractor_seq={distractor_seq})"
        )
    if (correct_final_state["pos"] == current_state["pos"]
            and correct_final_state["cardinal"] == current_state["cardinal"]):
        raise ValueError(
            f"Degenerate: correct_seq lands back at current state "
            f"(seed={seed}, correct_seq={correct_seq})"
        )
    if (dist_final_state["pos"] == current_state["pos"]
            and dist_final_state["cardinal"] == current_state["cardinal"]):
        raise ValueError(
            f"Degenerate: distractor_seq lands back at current state "
            f"(seed={seed}, distractor_seq={distractor_seq})"
        )
    if _is_too_subtle_pair(correct_final_state, dist_final_state):
        raise ValueError(
            f"Subtle: candidate states too close for human review "
            f"(seed={seed}, pos_dist={_state_distance(correct_final_state, dist_final_state):.3f}, "
            f"cardinal={correct_final_state['cardinal']}, correct_seq={correct_seq}, "
            f"distractor_seq={distractor_seq})"
        )
    if _ego_panels_identical(current_img, correct_img):
        raise ValueError(
            f"Subtle: current ego view identical to correct candidate ego "
            f"(seed={seed}, correct_seq={correct_seq})"
        )
    if _ego_panels_identical(current_img, dist_img):
        raise ValueError(
            f"Subtle: current ego view identical to distractor candidate ego "
            f"(seed={seed}, distractor_seq={distractor_seq})"
        )
    cand_ego_match = _ego_match_rate(correct_img, dist_img)
    if cand_ego_match >= MAX_CANDIDATE_EGO_MATCH_RATE:
        raise ValueError(
            f"Subtle: candidate ego panels too similar "
            f"(seed={seed}, match={cand_ego_match:.6f}, correct_seq={correct_seq}, "
            f"distractor_seq={distractor_seq})"
        )

    # Randomly assign A/B labels (correct candidate can be A or B with 50/50 chance)
    if random.random() < 0.5:
        candidate_a_path = correct_img.relative_to(img_dir.parent).as_posix()
        candidate_b_path = dist_img.relative_to(img_dir.parent).as_posix()
        correct_label = "A"
        label_idx = 0
        correct_final = correct_final_state
        distractor_final = dist_final_state
    else:
        candidate_a_path = dist_img.relative_to(img_dir.parent).as_posix()
        candidate_b_path = correct_img.relative_to(img_dir.parent).as_posix()
        correct_label = "B"
        label_idx = 1
        correct_final = correct_final_state
        distractor_final = dist_final_state

    current_rel = current_img.relative_to(img_dir.parent).as_posix()
    uid = f"A-MW-v2.fourrooms.s{seed:06d}.t{question_idx}"

    record = {
        "exam_id": uid,
        "uid": uid,
        "schema_version": "gridwm.mw_exam.v2",
        "task": "A-MW-v2",
        "env_task": "fourrooms",
        "subtask": "successor_choice",
        "seed": seed,
        "action_sequence": correct_seq,
        "distractor_sequence": distractor_seq,
        "primitive_action_sequence": primitive_correct_seq,
        "primitive_distractor_sequence": primitive_distractor_seq,
        "ground_truth": correct_label,
        "ground_truth_type": "successor_choice",
        "current_image_rel_path": current_rel,
        "candidate_a_rel_path": candidate_a_path,
        "candidate_b_rel_path": candidate_b_path,
        "correct_candidate": correct_label,
        "distractor_source": _make_distractor_source(correct_seq, distractor_seq),
        "distractor_strategy": _infer_strategy(correct_seq, distractor_seq),
        "prompt": build_prompt(correct_seq, motion_contract["turn_step_deg"]),
        "candidates": [
            {"label": "A", "value": "Candidate A is the true successor"},
            {"label": "B", "value": "Candidate B is the true successor"},
        ],
        "label": label_idx,
        "note": "prototype_validation_v2_fixed",
        # Terminal state metadata — verifiable via environment replay
        "current_agent_pos": current_state["pos"],
        "current_agent_cardinal": current_state["cardinal"],
        "current_topview_scale": current_topview_scale,
        "forward_step": round(motion_contract["forward_step"], 4),
        "turn_step_deg": round(motion_contract["turn_step_deg"], 4),
        "red_box_pos": red_box_reference["red_box_pos"],
        "red_box_room_idx": red_box_reference["red_box_room_idx"],
        "current_room_idx": red_box_reference["current_room_idx"],
        "red_box_visible_in_current_view": red_box_reference["red_box_visible_in_current_view"],
        "current_room_has_red_box": red_box_reference["current_room_has_red_box"],
        "difficulty_bucket": red_box_reference["difficulty_bucket"],
        "correct_candidate_agent_pos": correct_final["pos"],
        "correct_candidate_agent_cardinal": correct_final["cardinal"],
        "distractor_candidate_agent_pos": distractor_final["pos"],
        "distractor_candidate_agent_cardinal": distractor_final["cardinal"],
        # Same-as-current flags (QA gate: if True, topview of candidate == current)
        "correct_candidate_same_as_current": (
            correct_final["pos"] == current_state["pos"]
            and correct_final["cardinal"] == current_state["cardinal"]
        ),
        "distractor_candidate_same_as_current": (
            distractor_final["pos"] == current_state["pos"]
            and distractor_final["cardinal"] == current_state["cardinal"]
        ),
        # Identical-state gate (QA gate: both candidates have same pos+cardinal)
        "correct_distractor_states_identical": (
            correct_final_state["pos"] == dist_final_state["pos"]
            and correct_final_state["cardinal"] == dist_final_state["cardinal"]
        ),
    }
    return record


def isolated_question(worker_dir: Path, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Preserve the native algorithm/random stream, but bound renderer lifetime."""
    from utils.random_state import tuples
    worker_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"seed_{kwargs['seed']}_q{kwargs['question_idx']}"
    request_path = worker_dir / (prefix + '.request.json')
    result_path = worker_dir / (prefix + '.result.json')
    log_path = worker_dir / (prefix + '.log')
    with request_path.open('x') as handle:
        json.dump(dict(kwargs={**kwargs, 'img_dir': str(kwargs['img_dir'].resolve())}, random_state=random.getstate()), handle)
    root = Path(__file__).resolve().parents[2]
    command = [sys.executable, str(root / 'tools/miniworld/run_builder.py'),
               '--work-dir', str(Path.cwd()), 'render_worker.py', '--',
               '--request', str(request_path.resolve()), '--result', str(result_path.resolve())]
    with log_path.open('x') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    if process.returncode:
        raise RuntimeError(f'Isolated render failed ({process.returncode}); inspect {log_path}')
    result = json.loads(result_path.read_text())
    random.setstate(tuples(result['random_state']))
    if result['status'] == 'rejected':
        raise ValueError(result['reason'])
    return result['record']


def main() -> None:
    ap = argparse.ArgumentParser(description="Task A-MW v2 prototype generator")
    ap.add_argument(
        "--out_dir",
        default="experiments/generated/miniworld/taskA-MW-v2-prototype",
    )
    ap.add_argument("--obs_width", type=int, default=640)
    ap.add_argument("--obs_height", type=int, default=480)
    ap.add_argument("--num_questions", type=int, default=15)
    ap.add_argument("--seed_start", type=int, default=101)
    ap.add_argument("--easy_ratio", type=float, default=DEFAULT_EASY_RATIO)
    ap.add_argument('--isolate-attempts', action='store_true',
                    help='Render each candidate in a disposable process to bound software-renderer memory')
    args = ap.parse_args()

    out_dir = ensure_output_dir(Path(args.out_dir))
    img_dir = ensure_output_dir(out_dir / "images")

    manifest = {
        "schema_version": "gridwm.mw_exam.v2",
        "task": "A-MW-v2",
        "env_id": "MiniWorld-FourRooms-v0",
        "visual_contract": "ego_topview_concat",
        "render_size": {
            "obs_width": args.obs_width,
            "obs_height": args.obs_height,
            "concat_height": args.obs_height * 2,
            "concat_width": args.obs_width,
        },
        "runtime": runtime_context(),
        "num_questions": args.num_questions,
        "question_design": "successor_choice",
        "distractor_strategy": "mixed (alternative_prefix + alternative_suffix) — per-question field records exact strategy",
        "min_same_dir_pos_delta": MIN_SAME_DIR_POS_DELTA,
        "easy_ratio_target": args.easy_ratio,
        "records": [],
    }

    env_id = "MiniWorld-FourRooms-v0"
    env_slug = "fourrooms"

    print(f"Generating {args.num_questions} Task A-MW v2 prototype questions...")
    print(f"  Output: {out_dir}")
    print(f"  Resolution: {args.obs_width}x{args.obs_height}")

    # Pre-verified seeds: all seeds in 101-120 are non-degenerate for ALL 15 sequence
    # types in FourRooms (verified via exhaustive scan). Map each question directly.
    generated = 0
    base_seed = args.seed_start
    max_seed_attempts = 500  # safety cap to avoid infinite loops
    seed_attempted: set[int] = set()
    easy_target = max(1, round(args.num_questions * args.easy_ratio))
    hard_target = args.num_questions - easy_target
    easy_count = 0
    hard_count = 0

    while generated < args.num_questions:
        idx = generated
        seed = base_seed + idx
        seq_pair = ACTION_SEQUENCE_POOL[idx % len(ACTION_SEQUENCE_POOL)]
        correct_seq, distractor_seq = seq_pair
        rejected_attempts = 0

        attempt = 0
        while seed in seed_attempted and attempt < max_seed_attempts:
            seed += 1
            attempt += 1

        if attempt >= max_seed_attempts:
            raise RuntimeError(
                f"Could not find a non-degenerate seed after {max_seed_attempts} "
                f"attempts for sequence pair {correct_seq} vs {distractor_seq}. "
                f"Consider redesigning the distractor strategy."
            )

        seed_attempted.add(seed)

        print(
            f"  [{idx+1:02d}/{args.num_questions}] "
            f"seed={seed} seq={correct_seq}"
            + (f"  (retry #{attempt})" if attempt > 0 else "")
        )
        try:
            kwargs = dict(
                env_id=env_id,
                seed=seed,
                env_slug=env_slug,
                obs_width=args.obs_width,
                obs_height=args.obs_height,
                correct_seq=correct_seq,
                distractor_seq=distractor_seq,
                question_idx=idx + 1,
                img_dir=img_dir,
            )
            record = isolated_question(out_dir / 'worker_attempts', kwargs) if args.isolate_attempts else make_question(**kwargs)
        except ValueError as exc:
            if "Degenerate" in str(exc) or "Subtle" in str(exc):
                print(f"    [WARN] seed={seed} rejected: {exc} — retrying with next seed")
                rejected_attempts += 1
                if rejected_attempts >= max_seed_attempts:
                    raise RuntimeError(
                        f"Could not find an acceptable seed after {max_seed_attempts} "
                        f"rejections for sequence pair {correct_seq} vs {distractor_seq}"
                    ) from exc
                base_seed += 1  # shift base to try next seed
                continue
            raise

        bucket = record["difficulty_bucket"]
        if bucket == "easy" and easy_count >= easy_target:
            print(f"    [WARN] seed={seed} rejected: easy quota already full — retrying with next seed")
            base_seed += 1
            continue
        if bucket == "hard" and hard_count >= hard_target:
            print(f"    [WARN] seed={seed} rejected: hard quota already full — retrying with next seed")
            base_seed += 1
            continue

        manifest["records"].append(record)
        if bucket == "easy":
            easy_count += 1
        else:
            hard_count += 1
        generated += 1

    # Write JSONL
    jsonl_path = out_dir / "taskA-MW-v2-prototype.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in manifest["records"]:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write manifest summary
    manifest_summary = dict(manifest)
    manifest_summary.pop("records")
    manifest_summary["records_count"] = len(manifest["records"])
    manifest_summary["records_relpath"] = str(jsonl_path.relative_to(out_dir))
    manifest_path = out_dir / "taskA-MW-v2-prototype_manifest.json"
    dump_json(manifest_path, manifest_summary)

    print(f"\nDone. Output:")
    print(f"  Manifest: {manifest_path}")
    print(f"  JSONL:    {jsonl_path}")
    print(f"  Images:   {img_dir}/")
    print(f"\n  Total questions: {len(manifest['records'])}")
    print(f"  Action sequences used: {len(ACTION_SEQUENCE_POOL)} unique patterns")


if __name__ == "__main__":
    main()
