"""Task B/C/D instructions and shared Task C context."""
from __future__ import annotations

TASK_C_ENVIRONMENT_CONTEXT = {
    "doorkey": (
        "In DoorKey, the agent must use the key to unlock the door and then "
        "reach the green goal tile."
    ),
    "dynamicobstacles": (
        "In DynamicObstacles, the agent must avoid the moving obstacles and "
        "reach the green goal tile."
    ),
    "emptyrandom": (
        "In EmptyRandom, the agent must navigate the room and reach the green "
        "goal tile."
    ),
    "fourrooms": (
        "In FourRooms, the agent must navigate through the connected rooms "
        "and reach the green goal tile."
    ),
    "keycorridor": (
        "In KeyCorridor, the agent must find the key, unlock the door, and "
        "pick up the target object behind the locked door."
    ),
    "lavagap": (
        "In LavaGap, the agent must avoid the lava and reach the green goal "
        "tile."
    ),
    "memory": (
        "In Memory, the agent must remember the initial cue and go to the "
        "matching object at the end of the hallway."
    ),
    "multiroom": (
        "In MultiRoom, the agent must navigate through the rooms, opening "
        "doors as needed, and reach the final green goal tile."
    ),
    "putnear": (
        "In PutNear, the agent must move the specified object next to the "
        "specified target object."
    ),
    "redblue": (
        "In RedBlueDoors, the agent must open the red door before the blue "
        "door."
    ),
}


def build_task_c_context(env_task: str, mission: str) -> str:
    """Describe the same environment and objective under every output contract."""
    env = str(env_task).strip()
    goal = str(mission).strip()
    if env not in TASK_C_ENVIRONMENT_CONTEXT:
        raise ValueError(f"unsupported Task C environment: {env!r}")
    if not goal:
        raise ValueError(f"missing simulator mission for Task C environment {env!r}")
    return (
        "Below is a storyboard from one MiniGrid trajectory.\n"
        f"Environment task: {TASK_C_ENVIRONMENT_CONTEXT[env]}\n"
        f"Task goal for this trajectory: {goal}.\n"
        "Did the trajectory satisfy this goal?"
    )


def build_task_c_prompt(env_task: str, mission: str) -> str:
    """Return a Task C prompt with an explicit task goal and binary answer."""
    return build_task_c_context(env_task, mission) + "\nAnswer with ONLY: Yes or No."


TASK_D_REFERENCE_PROMPT = (
    "Below are two storyboard trajectories from the same MiniGrid environment.\n\n"
    "Reference trajectory:\n"
    "This trajectory is a known successful example of completing the task in "
    "this environment.\n\n"
    "Query trajectory:\n"
    "Determine whether this trajectory also completes the same task goal as "
    "the reference trajectory.\n\n"
    "Answer with ONLY: Success or Fail."
)


TASK_B_PROMPT = 'Return ONLY a JSON object with exactly these keys: agent, front_cell, objects.\n\nCOORDINATE SYSTEM:\n- Use 1-based image coordinates.\n- The origin is the BOTTOM-LEFT corner of the 7x7 image.\n- x increases to the RIGHT.\n- y increases UPWARD.\n- Report every position as [x=col, y=row].\n- Valid range: x in [1,7], y in [1,7].\n\nCOLOR VOCABULARY:\n- Only use: red, green, blue, yellow, purple, grey.\n- Use canonical colors only.\n- Lava must be red.\n- Goal must be green.\n- Do not invent color names.\n\nAGENT DIRECTION:\n- You MUST output agent.dir.\n- Use this exact encoding:\n  - 0 = east / right\n  - 1 = south / down\n  - 2 = west / left\n  - 3 = north / up\n- dir must be one of: 0, 1, 2, 3.\n- Do not omit dir.\n- Do not output null for dir.\n\nSTATE FIELD:\n- You MUST output a numeric `state` for front_cell and for every object.\n- For doors only, use:\n  - 0 = open\n  - 1 = closed\n  - 2 = locked\n- For all non-door types, use `state = null`.\n- Do not output 0 for non-door types unless the tile/object is a door.\n- Do not output strings such as "open" or "closed".\n\nRULES:\n- Judge only from the image.\n- Do not assume any fixed agent position.\n- front_cell must be reported as a separate field.\n- front_cell.type is the OCCUPANCY type of the tile directly in front of the agent.\n- Allowed front_cell.type values are: empty, wall, door, goal, ball, key, lava, box.\n- Do NOT use floor, grey, color words, or free-form descriptions as front_cell.type.\n- Do NOT include the front_cell tile in objects.\n- Only report NON-BACKGROUND objects: keys, balls, doors, goals, lava, boxes.\n- Do NOT report wall, floor, or empty as objects.\n- If an object is not clearly visible, do not report it.\n- Do not invent objects, colors, or states.\n\nOUTPUT:\n- Return raw JSON only.\n- No markdown.\n- No code blocks.\n- No explanation.\n\nRequired schema:\n{\n  "agent": {"pos": [x, y], "dir": d, "carrying": null_or_object},\n  "front_cell": {"pos": [x, y], "type": "TYPE", "state": null_or_door_state},\n  "objects": [{"type": "TYPE", "pos": [x, y], "color": "COLOR", "state": null_or_door_state}]\n}'
