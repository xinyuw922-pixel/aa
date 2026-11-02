"""Final mission-hidden task ontology and response prompt contract."""

ENVIRONMENT_TASK_TYPE = {
    "doorkey": "UNLOCK_AND_REACH",
    "dynamicobstacles": "REACH_GOAL",
    "emptyrandom": "REACH_GOAL",
    "fourrooms": "REACH_GOAL",
    "keycorridor": "PICK_UP_OBJECT",
    "memory": "MATCH_CUE_OBJECT",
    "multiroom": "REACH_GOAL",
    "putnear": "PUT_OBJECT_NEAR_OBJECT",
    "redblue": "OPEN_DOORS_IN_ORDER",
}

ONTOLOGY_LINES = (
    "POSSIBLE_TASK_TYPES:",
    "- REACH_GOAL: Navigate until the agent occupies a designated terminal cell.",
    (
        "- UNLOCK_AND_REACH: Use an unlocking object to open a blocking "
        "barrier, then occupy a designated terminal cell."
    ),
    "- PICK_UP_OBJECT: End while carrying a specified object.",
    (
        "- MATCH_CUE_OBJECT: At the endpoint, select the object whose type "
        "matches an earlier displayed object."
    ),
    (
        "- PUT_OBJECT_NEAR_OBJECT: Move one specified object so it ends "
        "adjacent to another specified object."
    ),
    (
        "- OPEN_DOORS_IN_ORDER: Change specified doors to the open state in "
        "a specified order."
    ),
)

INTRO_LINES = (
    (
        "The REFERENCE is a completed demonstration. The task belongs to "
        "exactly one of the possible task types below."
    ),
    (
        "First infer the task type and the specific objective from the "
        "REFERENCE evidence. Then judge whether the QUERY completed that "
        "same objective."
    ),
    "Use only the evidence supplied in this condition.",
)

OUTPUT_LINES = (
    "Respond using exactly these four fields:",
    "TASK_TYPE: <one POSSIBLE_TASK_TYPES identifier>",
    "INFERRED_OBJECTIVE: <the specific objective inferred from REFERENCE>",
    "VERDICT: <Success or Fail>",
    (
        "EXPLANATION: <brief evidence from both REFERENCE and QUERY that "
        "supports the verdict>"
    ),
)
