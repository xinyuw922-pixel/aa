"""Base MiniGrid trajectory-goal prompt contracts."""

TASK_C_GOALS = {
    "doorkey": (
        "pick up the key, open the door, and reach the goal tile."
    ),
    "keycorridor": (
        "find the key, unlock the door, and pick up the colored target ball "
        "behind the locked door."
    ),
    "lavagap": (
        "cross the lava gap safely and reach the goal tile."
    ),
    "memory": (
        "use the colored cue visible at the start to identify and "
        "pick up the matching ball at the corridor end."
    ),
    "multiroom": (
        "navigate through the rooms (opening doors as needed) "
        "and reach the final green goal tile."
    ),
    "redblue": (
        "open the red door before the blue door. "
        "The red door must be opened first."
    ),
}


TASK_C_PROMPT_TEMPLATE = (
    "Below is a storyboard from one MiniGrid trajectory.\n"
    "Task goal: {goal}\n"
    "Did the trajectory satisfy this goal?\n"
    "Answer with ONLY: Yes or No."
)
