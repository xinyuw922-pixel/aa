from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


IMAGE_ACTIONS = {
    "move_forward",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
}
INTERACTION_ACTIONS = {"open", "close", "pickup", "drop", "unlock"}


TASK_A_PROMPT = """Task: Predict the next first-person observation.

Current observation:
[CURRENT IMAGE]

Action sequence:
{action_text}

Options A-D show four possible observations immediately after executing
exactly this action sequence.

Which option is the true next observation?

Answer with exactly one uppercase letter: A, B, C, or D."""


TASK_C_PROMPT = """Below is a time-ordered sparse storyboard from one Habitat
trajectory. Frames are ordered from left to right and top to bottom.

Task goal:
Navigate to the physical location depicted in the Goal View and issue STOP
inside the target region.

The storyboard contains the initial state, critical state transitions, and
terminal evidence. Exact pixels and camera orientation do not need to match
the Goal View; judge whether the same physical location was reached.

Did the trajectory satisfy the task goal?

Answer with exactly one token: Success or Fail."""


TASK_C_STATE_SEQUENCE_PROMPT_ZH = """下面是一段 Habitat 轨迹按时间顺序抽取的稀疏关键帧。
画面按从左到右、从上到下的顺序排列。

任务目标：
找到并操作“{target_description}”。必须对同一个“{component_description}”
依次完成：{event_text}，最后执行 STOP。

必须根据画面中物体的实际状态变化判断，而不能只根据动作名称判断。
如果操作了错误的物体或部件、动作没有真正生效、事件顺序错误、遗漏任一事件，
或者终止时最终状态不正确，均为失败。

这段轨迹是否完成了任务？

只能回答一个词：Success 或 Fail。"""


TASK_D_REFERENCE_PROMPT = """Below are two time-ordered Habitat trajectories
from the same scene and the same task.

Task goal:
Reach the physical location shown in the Goal View and issue STOP there.

Reference trajectory:
The reference is a known successful example. It may start from a different
location and follow a different path.

Query trajectory:
Determine whether the query also reaches the same physical target and stops
there. Small differences in viewpoint, path, and final camera orientation do
not imply failure.

Answer with exactly one token: Success or Fail."""


TASK_D_ARTICULATED_REFERENCE_PROMPT_ZH = """下面展示了同一 Habitat 场景、同一任务中的
两条按时间顺序排列的第一人称轨迹。每条轨迹都包含连续观察帧，以及相邻观察之间
执行的动作。

参考轨迹：
参考轨迹是一条已知成功完成任务的轨迹。请从参考轨迹中理解任务涉及的目标物体、
实际状态变化以及必要的事件顺序。

待判断轨迹：
判断待判断轨迹是否也完成了与参考轨迹相同的任务。

起始位置、移动路径、观察角度和动作数量可以与参考轨迹不同。不要要求两条轨迹
逐帧或像素一致。

必须根据画面中物体实际发生的状态变化及其时间顺序进行判断。仅出现 OPEN、CLOSE
或 STOP 动作名称，并不能证明动作实际生效。如果操作没有生效、事件缺失、顺序
错误、操作了错误目标，或者终止时物体状态不符合参考轨迹所体现的成功条件，
则判定为失败。

只能回答一个词：Success 或 Fail。"""


TASK_D_ARTICULATED_REFERENCE_PROMPT_EN = """Below are two time-ordered
first-person trajectories from the same Habitat scene and the same task. Each
trajectory contains consecutive observations and the actions executed between
them.

Reference trajectory:
The reference is a known successful completion. Infer the relevant target, the
required physical state changes, and their temporal order from this trajectory.

Query trajectory:
Determine whether the query trajectory also completes the same task.

The query may use a different starting position, path, viewpoint, and number of
navigation actions. Do not require frame-by-frame or pixel-level agreement.

Judge actual visible object-state changes and their temporal order. The
presence of OPEN, CLOSE, or STOP action names alone does not prove that an
interaction succeeded. A failed interaction, missing event, incorrect order,
wrong target, or incorrect terminal state must be judged as failure.

Answer with exactly one token: Success or Fail."""


TASK_D_ARTICULATED_PROMPT_STYLES = (
    "standard",
    "comparative",
    "event_checklist",
    "concise",
    "anti_shortcut",
)


TASK_D_ARTICULATED_PROMPTS_ZH = {
    "standard": TASK_D_ARTICULATED_REFERENCE_PROMPT_ZH,
    "comparative": """下面给出同一 Habitat 场景和同一任务的两条第一人称轨迹。
参考轨迹是已知成功的演示，待判断轨迹是需要评估的测试轨迹。

请比较两条轨迹中被操作的目标物体、物体真实发生的状态变化、变化顺序以及终止
状态。起点、路径、观察角度和导航动作数量可以不同，不需要逐帧或像素一致。

如果待判断轨迹完成了参考轨迹所展示的同一任务，则回答 Success；如果操作未
生效、目标错误、状态事件缺失或顺序错误、最终状态错误，或者没有正确停止，
则回答 Fail。

只能回答一个词：Success 或 Fail。""",
    "event_checklist": """参考轨迹是一条已知成功的 Habitat 任务演示。根据它确定
正确的目标物体和成功条件，然后检查待判断轨迹是否满足以下要求：

1. 操作的是正确目标及其正确部件；
2. 画面中确实出现了参考演示所要求的物体状态变化；
3. 所有状态事件以正确的时间顺序发生；
4. 终止时物体处于正确状态，并执行了 STOP。

两条轨迹的起点、路径、视角和动作数量可以不同。动作名称本身不能证明交互成功，
必须以第一人称画面中的实际变化为准。任一要求不满足即为失败。

只能回答一个词：Success 或 Fail。""",
    "concise": """参考轨迹是同一场景、同一任务的一次已知成功执行。判断待判断
轨迹是否对正确目标完成了相同的实际状态变化、事件顺序和最终停止条件。

允许起点、路径、视角和导航动作数量不同。不要仅凭 OPEN、CLOSE 或 STOP 等
动作名称判断，交互必须在画面中真实生效。

只能回答一个词：Success 或 Fail。""",
    "anti_shortcut": """你需要判断待判断轨迹是否完成了参考轨迹所示的 Habitat
任务。参考轨迹已知成功，但它不是需要逐帧复刻的固定动作脚本。

不要把动作文本当作成功证据：出现 OPEN、CLOSE 或 STOP 不代表动作真正生效。
也不要因为起点、路径、视角或导航动作数量不同就判为失败。

请只依据画面核对正确目标的实际状态变化、事件先后顺序和终止状态。错误目标、
无效交互、缺失或乱序事件以及错误的最终状态均为失败。

只能回答一个词：Success 或 Fail。""",
}


TASK_D_ARTICULATED_PROMPTS_EN = {
    "standard": TASK_D_ARTICULATED_REFERENCE_PROMPT_EN,
    "comparative": """Below are two first-person trajectories from the same
Habitat scene and task. The reference is a known successful demonstration; the
query is the trajectory to evaluate.

Compare the operated target, the physical object-state changes, their temporal
order, and the terminal state. Starting position, path, viewpoint, and number
of navigation actions may differ; frame-by-frame or pixel-level agreement is
not required.

Answer Success if the query completes the same task demonstrated by the
reference. Answer Fail if an interaction has no effect, the target is wrong,
an event is missing or out of order, the final state is wrong, or the agent
does not stop correctly.

Answer with exactly one token: Success or Fail.""",
    "event_checklist": """The reference trajectory is a known successful
Habitat task demonstration. Infer the correct target and success conditions,
then check whether the query satisfies all of the following:

1. It operates the correct target and component.
2. The required physical object-state changes are visibly realized.
3. All state events occur in the correct temporal order.
4. The object has the correct terminal state and the agent issues STOP.

The trajectories may differ in start, path, viewpoint, and action count.
Action names alone do not prove success; rely on actual visible changes. Any
unsatisfied requirement means failure.

Answer with exactly one token: Success or Fail.""",
    "concise": """The reference is a known successful execution of the same
task in the same scene. Decide whether the query completes the same real
object-state changes, event order, and stopping condition on the correct
target.

Start, path, viewpoint, and navigation-action count may differ. Do not infer
success from OPEN, CLOSE, or STOP action names unless the interaction visibly
takes effect.

Answer with exactly one token: Success or Fail.""",
    "anti_shortcut": """Judge whether the query completes the Habitat task
shown by the known-success reference. The reference is not a fixed action
script that must be copied frame by frame.

Do not treat action text as evidence of success: OPEN, CLOSE, or STOP can be
present even when an action has no effect. Conversely, differences in start,
path, viewpoint, or navigation-action count do not imply failure.

Use the images to verify the correct target's physical state changes, event
order, and terminal state. A wrong target, ineffective interaction, missing or
out-of-order event, or wrong final state is failure.

Answer with exactly one token: Success or Fail.""",
}


def task_d_articulated_prompts(style: str) -> tuple[str, str]:
    """Return the English and Chinese prompts for a Task D style."""
    if style not in TASK_D_ARTICULATED_PROMPT_STYLES:
        raise ValueError(f"Unsupported Task D prompt style: {style}")
    return (
        TASK_D_ARTICULATED_PROMPTS_EN[style],
        TASK_D_ARTICULATED_PROMPTS_ZH[style],
    )


ACTION_TEXT = {
    "move_forward": "MOVE_FORWARD — move forward by 0.25 meters.",
    "turn_left": "TURN_LEFT — rotate left by 30 degrees.",
    "turn_right": "TURN_RIGHT — rotate right by 30 degrees.",
    "look_up": "LOOK_UP — rotate the camera upward by 30 degrees.",
    "look_down": "LOOK_DOWN — rotate the camera downward by 30 degrees.",
}


STATE_EVENT_TEXT_ZH = {
    "open": "打开",
    "close": "关闭",
    "pickup": "拾取",
    "drop": "放下",
    "stop": "停止",
}


def task_c_state_sequence_prompt(
    target_description: str,
    component_description: str,
    required_events: Sequence[str] = ("open", "close", "stop"),
) -> str:
    """Build the model-visible prompt for an articulated-state Task C.

    Object names may vary (refrigerator, oven, wardrobe, cabinet, drawer),
    while the temporal and oracle contract remains identical.
    """
    events = [str(event).lower() for event in required_events]
    if not events or events[-1] != "stop":
        raise ValueError("Task C state sequences must terminate with stop")
    unsupported = [event for event in events if event not in STATE_EVENT_TEXT_ZH]
    if unsupported:
        raise ValueError(f"Unsupported Task C state events: {unsupported}")
    event_text = " → ".join(STATE_EVENT_TEXT_ZH[event] for event in events)
    return TASK_C_STATE_SEQUENCE_PROMPT_ZH.format(
        target_description=target_description,
        component_description=component_description,
        event_text=event_text,
    )


def normalized_joint_progress(
    position: float,
    closed_position: float,
    open_position: float,
) -> float:
    """Map either a revolute or prismatic joint to closed=0 and open=1."""
    span = float(open_position) - float(closed_position)
    if abs(span) <= 1e-9:
        raise ValueError("Open and closed joint positions must differ")
    return float(
        np.clip((float(position) - float(closed_position)) / span, 0.0, 1.0)
    )


def evaluate_state_cycle(
    actions: Sequence[str],
    target_progress: Sequence[float],
    *,
    interaction_succeeded: Sequence[bool] | None = None,
    interaction_targets: Sequence[str | None] | None = None,
    target_id: str | None = None,
    open_threshold: float = 0.60,
    closed_threshold: float = 0.05,
) -> dict[str, Any]:
    """Evaluate OPEN -> CLOSE -> STOP from actual target-joint state.

    ``target_progress`` contains one normalized joint value per rendered frame;
    action i leads to frame i+1. Merely emitting OPEN/CLOSE is insufficient:
    the selected target joint must cross the state thresholds after a
    successful interaction on the same target.
    """
    actions = [str(action).lower() for action in actions]
    progress = [float(value) for value in target_progress]
    if len(progress) != len(actions) + 1:
        raise ValueError("target_progress must contain one value per frame")
    if not 0.0 <= closed_threshold < open_threshold <= 1.0:
        raise ValueError("Expected 0 <= closed_threshold < open_threshold <= 1")
    if progress[0] > closed_threshold:
        return {
            "outcome": "Fail",
            "failure_reason": "target_not_initially_closed",
            "event_frames": [],
        }

    succeeded = (
        [True] * len(actions)
        if interaction_succeeded is None
        else [bool(value) for value in interaction_succeeded]
    )
    if len(succeeded) != len(actions):
        raise ValueError("interaction_succeeded must align with actions")
    targets = (
        [target_id] * len(actions)
        if interaction_targets is None
        else list(interaction_targets)
    )
    if len(targets) != len(actions):
        raise ValueError("interaction_targets must align with actions")

    def correct_target(index: int) -> bool:
        return target_id is None or targets[index] == target_id

    open_index: int | None = None
    close_index: int | None = None
    stop_index: int | None = None
    for index, action in enumerate(actions):
        if (
            open_index is None
            and action == "open"
            and succeeded[index]
            and correct_target(index)
            and progress[index + 1] >= open_threshold
        ):
            open_index = index
            continue
        if (
            open_index is not None
            and close_index is None
            and index > open_index
            and action == "close"
            and succeeded[index]
            and correct_target(index)
            and progress[index + 1] <= closed_threshold
        ):
            close_index = index
            continue
        if (
            close_index is not None
            and index > close_index
            and action == "stop"
        ):
            stop_index = index
            break

    failure_reason: str | None = None
    if open_index is None:
        failure_reason = "target_never_actually_opened"
    elif close_index is None:
        failure_reason = "target_not_closed_after_open"
    elif stop_index is None:
        failure_reason = "missing_stop_after_close"
    elif progress[-1] > closed_threshold:
        failure_reason = "target_not_closed_at_termination"

    event_frames = []
    for name, index in (
        ("open", open_index),
        ("close", close_index),
        ("stop", stop_index),
    ):
        if index is not None:
            event_frames.append(
                {
                    "name": name,
                    "action_index": index,
                    "pre_frame": index,
                    "post_frame": index + 1,
                }
            )
    return {
        "outcome": "Success" if failure_reason is None else "Fail",
        "failure_reason": failure_reason,
        "event_frames": event_frames,
        "open_action_index": open_index,
        "close_action_index": close_index,
        "stop_action_index": stop_index,
        "final_progress": progress[-1],
    }


def task_a_prompt(action: str | Sequence[str]) -> str:
    actions = [action] if isinstance(action, str) else list(action)
    if not actions or any(item not in ACTION_TEXT for item in actions):
        raise ValueError(f"Task A does not support action sequence: {actions}")
    runs: list[tuple[str, int]] = []
    for item in actions:
        if runs and runs[-1][0] == item:
            name, count = runs[-1]
            runs[-1] = (name, count + 1)
        else:
            runs.append((item, 1))
    descriptions = []
    for name, count in runs:
        base = ACTION_TEXT[name].rstrip(".")
        descriptions.append(
            f"{base} Repeat {count} times." if count > 1 else f"{base}."
        )
    return TASK_A_PROMPT.format(action_text="\n".join(descriptions))


def _collision_run_onsets(collisions: Sequence[bool]) -> list[int]:
    onsets = [
        index
        for index, collided in enumerate(collisions)
        if collided and (index == 0 or not collisions[index - 1])
    ]
    # Collision is supporting failure evidence, not a latent task-state
    # transition. One representative onset prevents a repeatedly blocked CF
    # replay from turning sparse Task C into dense Task D.
    return onsets[:1]


def _progress_indices(
    positions: Sequence[Sequence[float]],
    quantiles: Sequence[float],
) -> list[int]:
    if not positions:
        return []
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return []
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    total = float(cumulative[-1])
    if total <= 1e-9:
        return []
    return [
        int(np.argmin(np.abs(cumulative - total * float(quantile))))
        for quantile in quantiles
    ]


def critical_frame_selection(
    actions: Sequence[str],
    collisions: Sequence[bool],
    *,
    positions: Sequence[Sequence[float]] | None = None,
    goal: Sequence[float] | None = None,
    goal_radius_m: float = 0.35,
    min_frames: int = 6,
    event_frames: Sequence[dict[str, Any]] = (),
) -> tuple[list[int], dict[int, list[str]]]:
    """Select a sparse, event-complete Task C storyboard.

    Frame zero is the initial observation and action i leads to frame i + 1.
    Ordinary turns, forwards, and action-run boundaries are deliberately not
    events. Oracle reasons are returned for auditing and must not be exposed in
    a model-visible prompt.
    """
    if len(actions) != len(collisions):
        raise ValueError("actions and collisions must have equal length")
    if min_frames < 2:
        raise ValueError("min_frames must be at least two")

    terminal = len(actions)
    reasons: dict[int, set[str]] = {}

    def add(index: int, reason: str) -> None:
        bounded = max(0, min(terminal, int(index)))
        reasons.setdefault(bounded, set()).add(reason)

    add(0, "initial")
    add(max(0, terminal - 1), "pre_terminal")
    add(terminal, "terminal")

    for action_index, action in enumerate(actions):
        if action in INTERACTION_ACTIONS:
            add(action_index, f"pre_{action}")
            add(action_index + 1, f"post_{action}")

    for action_index in _collision_run_onsets(collisions):
        add(action_index, "pre_collision_run")
        add(action_index + 1, "collision_run_onset")

    for event in event_frames:
        name = str(event.get("name", "semantic_event"))
        add(int(event["pre_frame"]), f"pre_{name}")
        add(int(event["post_frame"]), f"post_{name}")

    if positions is not None:
        if len(positions) != terminal + 1:
            raise ValueError("positions must contain one entry per frame")
        if goal is not None:
            target = np.asarray(goal, dtype=float)
            distances = [
                float(np.linalg.norm(np.asarray(position, dtype=float) - target))
                for position in positions
            ]
            entries = [
                index
                for index, distance in enumerate(distances)
                if distance <= goal_radius_m
                and (index == 0 or distances[index - 1] > goal_radius_m)
            ]
            if entries:
                entry = entries[0]
                add(max(0, entry - 1), "pre_goal_entry")
                add(entry, "goal_entry")

    if len(reasons) < min_frames:
        if positions is not None:
            candidates = _progress_indices(
                positions, (0.25, 0.50, 0.75, 0.125, 0.375, 0.625, 0.875)
            )
        else:
            candidates = [
                round(slot * terminal / max(min_frames - 1, 1))
                for slot in range(1, min_frames - 1)
            ]
        for index in candidates:
            add(index, "progress_context")
            if len(reasons) >= min_frames:
                break

    indices = sorted(reasons)
    serializable = {
        index: sorted(values) for index, values in sorted(reasons.items())
    }
    return indices, serializable


def shared_critical_frame_selection(
    actions: Sequence[str],
    full_collisions: Sequence[bool],
    cf_collisions: Sequence[bool],
    *,
    full_positions: Sequence[Sequence[float]],
    cf_positions: Sequence[Sequence[float]],
    goal: Sequence[float],
    goal_radius_m: float = 0.35,
    min_frames: int = 6,
) -> tuple[list[int], dict[int, list[str]]]:
    """Use one shared index set for paired Full/CF Task C evidence."""
    full_indices, full_reasons = critical_frame_selection(
        actions,
        full_collisions,
        positions=full_positions,
        goal=goal,
        goal_radius_m=goal_radius_m,
        min_frames=2,
    )
    cf_indices, cf_reasons = critical_frame_selection(
        actions,
        cf_collisions,
        positions=cf_positions,
        goal=goal,
        goal_radius_m=goal_radius_m,
        min_frames=2,
    )
    indices = sorted(set(full_indices) | set(cf_indices))
    if len(indices) < min_frames:
        for index in _progress_indices(
            full_positions, (0.25, 0.50, 0.75, 0.125, 0.375, 0.625, 0.875)
        ):
            if index not in indices:
                indices.append(index)
                full_reasons.setdefault(index, []).append("progress_context")
            if len(indices) >= min_frames:
                break
        indices.sort()
    reasons: dict[int, list[str]] = {}
    for index in indices:
        values = {
            *(f"full:{reason}" for reason in full_reasons.get(index, [])),
            *(f"cf:{reason}" for reason in cf_reasons.get(index, [])),
        }
        reasons[index] = sorted(values)
    return indices, reasons


def action_chunks(
    actions: Sequence[str], frame_indices: Sequence[int]
) -> list[dict[str, Any]]:
    """Return compact action runs aligned between displayed keyframes."""
    chunks: list[dict[str, Any]] = []
    for start, end in zip(frame_indices, frame_indices[1:]):
        segment = list(actions[start:end])
        runs: list[dict[str, Any]] = []
        for action in segment:
            if runs and runs[-1]["action"] == action:
                runs[-1]["count"] += 1
            else:
                runs.append({"action": action, "count": 1})
        chunks.append(
            {
                "from_frame_index": int(start),
                "to_frame_index": int(end),
                "runs": runs,
            }
        )
    return chunks


def critical_frame_indices(
    actions: Sequence[str],
    collisions: Sequence[bool],
) -> list[int]:
    """Backward-compatible wrapper used by older package builders."""
    return critical_frame_selection(actions, collisions)[0]


def choose_task_a_steps(
    action_count: int,
    count: int | None,
    rng: Any,
) -> list[int]:
    if action_count <= 0:
        return []
    if count is None or count <= 0 or count >= action_count:
        return list(range(action_count))
    choices = rng.choice(action_count, size=count, replace=False)
    return sorted(int(value) for value in choices)


def candidate_actions(correct_action: str, rng: Any) -> list[str]:
    """Reject different-action distractors.
    
    Task A requires successor images from the same action at distinct states."""
    del rng
    if correct_action not in IMAGE_ACTIONS:
        raise ValueError(f"Task A does not support action: {correct_action}")
    raise RuntimeError(
        "Different-action Task A distractors are invalid. Sample four "
        "successor images produced by the same action at different states."
    )
