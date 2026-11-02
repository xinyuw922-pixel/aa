"""Simulator-state normalization and evidence formatting for Oracle-State."""

from __future__ import annotations

import copy

from typing import Any

def fmt_value(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, list):
        return "(" + ",".join(str(x) for x in value) + ")"
    if isinstance(value, dict):
        parts = []
        for key in sorted(value):
            parts.append(f"{key}={fmt_value(value[key])}")
        return "{" + ", ".join(parts) + "}"
    return str(value)

def state_brief(state: dict[str, Any]) -> str:
    agent = state.get("agent", {})
    front = state.get("front_cell", {})
    visible = state.get("visible_types_gt") or state.get("visible_types") or []
    carrying = agent.get("carrying")
    return (
        f"agent_pos={fmt_value(agent.get('pos'))}; "
        f"agent_dir={fmt_value(agent.get('dir'))}; "
        f"carrying={fmt_value(carrying)}; "
        f"front_cell={fmt_value(front)}; "
        f"visible_types={fmt_value(visible)}"
    )

def object_brief(obj: dict[str, Any]) -> str:
    parts = [str(obj.get("type", "object"))]
    if obj.get("color") not in (None, "none"):
        parts.append(str(obj.get("color")))
    if obj.get("pos") is not None:
        parts.append(f"pos={fmt_value(obj.get('pos'))}")
    if obj.get("state") is not None:
        parts.append(f"state_code={obj.get('state')}")
    return " ".join(parts)

def object_inventory(state: dict[str, Any], limit: int = 16) -> str:
    objects = state.get("objects") or []
    if not objects:
        return "none listed"
    rendered = [object_brief(obj) for obj in objects[:limit]]
    if len(objects) > limit:
        rendered.append(f"... ({len(objects) - limit} additional objects)")
    return "; ".join(rendered)

def relation_brief(state: dict[str, Any]) -> str:
    agent = state.get("agent", {})
    agent_pos = pos_tuple(agent.get("pos"))
    carrying = agent.get("carrying")
    objects = state.get("objects") or []
    if agent_pos is None:
        return "agent_object_relations=agent position missing"

    relations: list[str] = []
    for obj in objects:
        obj_pos = pos_tuple(obj.get("pos"))
        if obj_pos is None:
            continue
        obj_name = object_brief(obj)
        manhattan = abs(agent_pos[0] - obj_pos[0]) + abs(agent_pos[1] - obj_pos[1])
        same_cell = "same_cell" if manhattan == 0 else f"manhattan={manhattan}"
        relations.append(f"{obj_name} -> {same_cell}")
    carry_text = f"carrying={fmt_value(carrying)}"
    if not relations:
        return carry_text + "; object_relations=none listed"
    return carry_text + "; object_relations=" + " | ".join(relations[:12])

def action_brief(actions: list[Any], head: int = 24, tail: int = 8) -> str:
    items = [str(a) for a in actions]
    if len(items) <= head + tail + 1:
        return ", ".join(items)
    return ", ".join(items[:head]) + f", ... ({len(items) - head - tail} omitted) ..., " + ", ".join(items[-tail:])

def normalize_raw_state_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize Tier-1 `pos` and Tier-2 `x/y` object schemas."""
    normalized = copy.deepcopy(raw)
    for state in normalized.get("state_seq") or []:
        for obj in state.get("objects") or []:
            if obj.get("pos") is None and obj.get("x") is not None and obj.get("y") is not None:
                obj["pos"] = [int(obj["x"]), int(obj["y"])]
            if obj.get("type") == "door" and obj.get("state") is None:
                if obj.get("is_locked") is True:
                    obj["state"] = "locked"
                elif obj.get("is_open") is True:
                    obj["state"] = "open"
                elif obj.get("is_open") is False:
                    obj["state"] = "closed"
    return normalized

def pos_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None

def object_text(obj: dict[str, Any] | None) -> str:
    if obj is None:
        return "not present"
    parts = []
    if obj.get("color") not in (None, "none"):
        parts.append(str(obj["color"]))
    parts.append(str(obj.get("type", "object")))
    if obj.get("pos") is not None:
        parts.append(f"at ({obj['pos'][0]},{obj['pos'][1]})")
    return " ".join(parts)

def carrying_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, dict):
        color = value.get("color")
        object_type = value.get("type", "object")
        return f"{color} {object_type}" if color else str(object_type)
    return str(value)

def carrying_events(raw: dict[str, Any]) -> list[str]:
    states = raw.get("state_seq") or []
    events = []
    previous = None
    for step, state in enumerate(states):
        current = (state.get("agent") or {}).get("carrying")
        if step and current != previous and current != "unobserved" and previous != "unobserved":
            events.append(
                f"step {step}: carrying changed "
                f"{carrying_text(previous)} -> {carrying_text(current)}"
            )
        previous = current
    return events

def door_events(raw: dict[str, Any]) -> list[str]:
    states = raw.get("state_seq") or []
    events = []
    previous: dict[tuple[str, tuple[int, int]], Any] = {}
    state_names = {0: "open", 1: "closed", 2: "locked"}
    for step, state in enumerate(states):
        current = {}
        for obj in state.get("objects") or []:
            pos = pos_tuple(obj.get("pos"))
            if obj.get("type") == "door" and pos is not None:
                current[(str(obj.get("color")), pos)] = obj.get("state")
        if step:
            for key in sorted(set(previous) & set(current)):
                if previous[key] != current[key]:
                    color, pos = key
                    before = state_names.get(previous[key], str(previous[key]))
                    after = state_names.get(current[key], str(current[key]))
                    events.append(
                        f"step {step}: {color} door at ({pos[0]},{pos[1]}) "
                        f"changed {before} -> {after}"
                    )
        previous = current
    return events

def final_object_pairs(raw: dict[str, Any]) -> list[str]:
    states = raw.get("state_seq") or []
    objects = (states[-1].get("objects") or []) if states else []
    relevant = [
        obj
        for obj in objects
        if obj.get("type") in {"ball", "box", "key", "goal"}
        and pos_tuple(obj.get("pos")) is not None
    ]
    output = []
    for index, first in enumerate(relevant):
        first_pos = pos_tuple(first["pos"])
        for second in relevant[index + 1 :]:
            second_pos = pos_tuple(second["pos"])
            dx = first_pos[0] - second_pos[0]
            dy = first_pos[1] - second_pos[1]
            chebyshev = max(abs(dx), abs(dy))
            if chebyshev <= 2:
                output.append(
                    f"{object_text(first)} <-> {object_text(second)}: "
                    f"delta=({dx},{dy}), chebyshev={chebyshev}, "
                    f"adjacent_8={str(chebyshev == 1).lower()}"
                )
    return output

def structural_layout_facts(raw: dict[str, Any]) -> list[str]:
    if raw.get("task") != "memory":
        return []
    states = raw.get("state_seq") or []
    objects = (states[0].get("objects") or []) if states else []
    positioned = [obj for obj in objects if pos_tuple(obj.get("pos")) is not None]
    if not positioned:
        return []
    min_x = min(pos_tuple(obj["pos"])[0] for obj in positioned)
    max_x = max(pos_tuple(obj["pos"])[0] for obj in positioned)
    leftmost = sorted(
        (obj for obj in positioned if pos_tuple(obj["pos"])[0] == min_x),
        key=lambda obj: tuple(obj["pos"]),
    )
    rightmost = sorted(
        (obj for obj in positioned if pos_tuple(obj["pos"])[0] == max_x),
        key=lambda obj: tuple(obj["pos"]),
    )
    final_front = (states[-1].get("front_cell") or {}) if states else {}
    front_obj = None
    if final_front.get("type") not in {None, "empty", "wall"}:
        front_obj = {
            "type": final_front.get("type"),
            "color": final_front.get("color"),
            "pos": final_front.get("pos"),
        }
    lines = ["- structural_layout:"]
    lines.extend(f"  - leftmost: {object_text(obj)}" for obj in leftmost)
    lines.extend(f"  - rightmost: {object_text(obj)}" for obj in rightmost)
    lines.append(f"  - final_front_object: {object_text(front_obj)}")
    return lines

def derived_facts(raw: dict[str, Any]) -> list[str]:
    lines = ["DERIVED_STATE_FACTS:"]
    carry = carrying_events(raw)
    doors = door_events(raw)
    pairs = final_object_pairs(raw)
    lines.append("- carrying_events:")
    lines.extend(f"  - {event}" for event in carry) if carry else lines.append("  - none observed")
    lines.append("- door_events:")
    lines.extend(f"  - {event}" for event in doors) if doors else lines.append("  - none observed")
    lines.append("- nearby_final_object_pairs:")
    lines.extend(f"  - {pair}" for pair in pairs) if pairs else lines.append("  - none listed")
    lines.extend(structural_layout_facts(raw))
    return lines

def trace_without_identifiers(title: str, raw: dict[str, Any]) -> str:
    states = raw.get("state_seq") or []
    actions = raw.get("actions_text") or []
    lines = [
        f"{title}:",
        f"- steps: {len(actions)}",
        f"- actions: {action_brief(actions)}",
        f"- initial_state: {state_brief(states[0]) if states else 'missing'}",
        f"- final_state: {state_brief(states[-1]) if states else 'missing'}",
        f"- initial_objects: {object_inventory(states[0]) if states else 'missing'}",
        f"- final_objects: {object_inventory(states[-1]) if states else 'missing'}",
        f"- final_agent_relations: {relation_brief(states[-1]) if states else 'missing'}",
        *derived_facts(raw),
    ]
    return "\n".join(lines)
