"""Visible-image coordinates and state conventions for MiniGrid Task B."""
from typing import List, Tuple

from minigrid.core.constants import OBJECT_TO_IDX


def visible_state_answer(state: dict) -> dict:
    """Project simulator world state into the Task B seven-cell image frame."""
    agent = state['agent']
    ax, ay = agent['pos']
    direction = agent['dir']
    if direction not in (0, 1, 2, 3):
        raise ValueError('Task B requires a valid simulator agent direction')
    encoding = state['state_encoding']
    if len(encoding) != 7 or any(len(column) != 7 for column in encoding):
        raise ValueError('Task B requires a 7x7 semantic observation encoding')
    front = state['front_cell']
    front_pos = list(world_to_image(*front['pos'], ax, ay, direction))
    agent_pos = agent_to_image(ax, ay, direction)
    front_type = front.get('type', 'empty')
    if front_type == 'floor':
        front_type = 'empty'
    objects = []
    for obj in state.get('objects', []):
        pos = list(world_to_image(*obj['pos'], ax, ay, direction))
        if is_visible_object(*pos, encoding, object_type=obj.get('type')) and pos != front_pos:
            objects.append(dict(type=obj.get('type', 'unknown'), pos=pos,
                                color=obj.get('color'),
                                state=canonical_state_for_type(obj.get('type', ''), obj.get('state', 0))))
    return dict(agent=dict(pos=agent_pos, dir=compute_agent_dir_from_image(agent_pos, front_pos),
                           carrying=agent.get('carrying')),
                front_cell=dict(pos=front_pos, type=front_type,
                                state=canonical_state_for_type(front_type, front.get('state', 0))),
                objects=objects)

def world_to_image(world_x: int, world_y: int, agent_x: int, agent_y: int, agent_dir: int) -> Tuple[int, int]:
    dx = world_x - agent_x
    dy = world_y - agent_y
    if agent_dir == 0:
        x_local = dy
        y_local = dx
    elif agent_dir == 1:
        x_local = -dx
        y_local = dy
    elif agent_dir == 2:
        x_local = -dy
        y_local = -dx
    elif agent_dir == 3:
        x_local = dx
        y_local = -dy
    else:
        raise ValueError(f"Invalid dir: {agent_dir}")
    return x_local + 4, y_local + 1


def agent_to_image(agent_x: int, agent_y: int, agent_dir: int) -> List[int]:
    x, y = world_to_image(agent_x, agent_y, agent_x, agent_y, agent_dir)
    return [x, y]


def image_to_observe(x_img: int, y_img: int) -> Tuple[int, int]:
    x_local = x_img - 4
    y_local = y_img - 1
    return x_local + 3, 6 - y_local


def is_observe_cell_visible(obs_x: int, obs_y: int, state_encoding: List[List[List[int]]]) -> bool:
    if not (0 <= obs_x < 7 and 0 <= obs_y < 7):
        return False
    try:
        cell = state_encoding[obs_x][obs_y]
    except (IndexError, TypeError):
        return False
    return isinstance(cell, (list, tuple)) and not (cell[0] == 0 and cell[1] == 0)


def is_visible_object(x_img: int, y_img: int, state_encoding: List[List[List[int]]],
                      object_type: str | None = None) -> bool:
    """Check view visibility and, when supplied, the observed object identity."""
    if not (1 <= x_img <= 7 and 1 <= y_img <= 7):
        return False
    obs_x, obs_y = image_to_observe(x_img, y_img)
    if not is_observe_cell_visible(obs_x, obs_y, state_encoding):
        return False
    if object_type is None:
        return True
    # The view replaces the agent's underlying world tile with its carried
    # object or empty space. Visibility alone cannot establish object presence.
    expected = OBJECT_TO_IDX.get(object_type)
    return expected is not None and state_encoding[obs_x][obs_y][0] == expected


def canonical_state_for_type(obj_type: str, state: int | None) -> int | None:
    if obj_type == "door":
        return 0 if state is None else state
    return None


def compute_agent_dir_from_image(agent_pos_img: List[int], front_cell_pos_img: List[int]) -> int:
    dx = front_cell_pos_img[0] - agent_pos_img[0]
    dy = front_cell_pos_img[1] - agent_pos_img[1]
    if (dx, dy) == (1, 0):
        return 0
    if (dx, dy) == (0, -1):
        return 1
    if (dx, dy) == (-1, 0):
        return 2
    if (dx, dy) == (0, 1):
        return 3
    raise ValueError(f"agent.dir derivation failed: agent={agent_pos_img}, front={front_cell_pos_img}")
