"""Event-complete keyframe selection and near-square storyboard layout."""
import math
from typing import Any, Dict, List

def parse_state_seq(state_seq: List[dict]) -> List[Dict[str, Any]]:
    """
    Parse state_seq into a clean per-step summary.
    Returns list of step dicts with only audit-safe fields.
    """
    steps = []
    prev_carrying = None
    for i, s in enumerate(state_seq):
        if not isinstance(s, dict):
            steps.append({'step': i, 'raw': s})
            continue

        agent = s.get('agent', {})
        carrying = agent.get('carrying')

        # Detect carrying change (key pickup / object pickup)
        carry_changed = False
        carry_type = None
        if carrying and carrying != prev_carrying:
            carry_changed = True
            if isinstance(carrying, dict):
                carry_type = f"{carrying.get('type','')}:{carrying.get('color','')}"
            else:
                carry_type = str(carrying)
        elif carrying and prev_carrying is None:
            carry_changed = True
            if isinstance(carrying, dict):
                carry_type = f"{carrying.get('type','')}:{carrying.get('color','')}"
            else:
                carry_type = str(carrying)

        # Detect door state
        objs = s.get('objects', [])
        door_states = {}
        for o in (objs if objs else []):
            if isinstance(o, dict) and o.get('type') == 'door':
                door_states[o.get('color', 'unknown')] = o.get('state', -1)

        # Detect visible target
        vis_types = s.get('visible_types', [])
        visible_set = set(vis_types) if vis_types else set()

        steps.append({
            'step': i,
            'carrying': carrying,
            'carrying_type': carry_type,
            'carry_changed': carry_changed,
            'door_states': door_states,
            'visible_types': list(visible_set),
            'terminated': s.get('terminated', False),
            'agent_room': s.get('agent_room'),  # room tracking for multiroom
        })
        prev_carrying = carrying
    return steps


def detect_event_steps(
    short_env: str,
    state_seq: List[dict],
    n_frames: int
) -> Dict[str, Any]:
    """
    Detect key event steps in a trajectory.

    Returns a dict with:
      - forced_steps: list of step indices that MUST be included
      - evidence_notes: list of strings describing what each step shows
      - coverage_flags: dict of coverage booleans
    """
    steps = parse_state_seq(state_seq)
    forced = []
    notes = {}
    n = len(steps)

    def add(step_idx, note):
        si = max(0, min(step_idx, n - 1))
        if si not in forced:
            forced.append(si)
        notes[si] = note

    # ── Step 0: Always include (start state) ──────────────────────────────────
    add(0, "initial state: agent at start")

    if short_env == "doorkey":
        # Events: key visible, key pickup, door closed (locked), door open, goal
        for i, st in enumerate(steps):
            c = st.get('carrying_type', '')
            if c and 'key' in c and i not in forced:
                add(i, f"key visible/pickup: {c}")
            ds = st.get('door_states', {})
            for color, state in ds.items():
                if state == 0 and i not in forced:  # door open
                    add(i, f"door {color} open")
                if state == 2 and i not in forced:  # door locked
                    add(i, f"door {color} locked")
        add(n - 1, "terminal frame")

    elif short_env == "keycorridor":
        # Events: key visible, key pickup, door visible, target visible
        has_key_seen = False
        has_carrying_key = False
        for i, st in enumerate(steps):
            vis = set(st.get('visible_types', []))
            ct = st.get('carrying_type', '')
            if 'key' in vis and not has_key_seen:
                add(i, "key first visible")
                has_key_seen = True
            if st.get('carry_changed') and 'key' in str(ct):
                add(i, f"key pickup: {ct}")
                has_carrying_key = True
        # Force: first step, key pickup, terminal
        add(0, "initial state")
        # Find key pickup
        for i, st in enumerate(steps):
            if st.get('carry_changed') and 'key' in str(st.get('carrying_type', '')):
                add(i, "key pickup")
                break
        add(n - 1, "terminal: agent at target / corridor end")

    elif short_env == "lavagap":
        # Events: lava visible, crossing, goal on far side
        for i, st in enumerate(steps):
            vis = set(st.get('visible_types', []))
            if 'lava' in vis and i not in forced[:1]:
                add(i, "lava visible")
            if 'goal' in vis and i not in forced:
                add(i, "goal visible")
        add(n - 1, "terminal: agent at goal")

    elif short_env == "memory":
        # 5 frames total — include all
        for i in range(n):
            add(i, f"step {i}")
        # No forced override needed; all frames are included

    elif short_env == "multiroom":
        # Events: front_cell type transitions (room zones), door state changes, goal
        # Door interaction = when door_state changes (e.g., locked→open)
        # front_cell type change = new room zone
        prev_front_type = None
        prev_door_states = None
        for i, st in enumerate(steps):
            # front_cell type change → entering new zone
            front_cell = st.get('front_cell', {})
            if isinstance(front_cell, dict):
                front_type = front_cell.get('type', '')
            else:
                front_type = str(front_cell)
            if front_type and front_type != prev_front_type:
                add(i, f"zone={front_type}")
                prev_front_type = front_type
            # Door state change → door interaction
            ds = st.get('door_states', {})
            if ds and ds != prev_door_states:
                for color, state in ds.items():
                    # Only flag actual transitions (not initial state)
                    if prev_door_states is not None:
                        prev_state = prev_door_states.get(color)
                        if prev_state != state:
                            door_note = f"door_{color}={'open' if state==0 else 'locked'}"
                            add(i, door_note)
                prev_door_states = ds
            elif prev_door_states is None and ds:
                prev_door_states = ds
        add(0, "initial room")
        add(n - 1, "terminal: goal reached")

    elif short_env == "redblue":
        # Events: red door, blue door, goal
        # Track door open sequence
        red_open = None
        blue_open = None
        for i, st in enumerate(steps):
            ds = st.get('door_states', {})
            if 'red' in ds and ds['red'] == 0 and red_open is None:
                add(i, "red door open")
                red_open = i
            if 'blue' in ds and ds['blue'] == 0 and blue_open is None:
                add(i, "blue door open")
                blue_open = i
        add(0, "initial state")
        if red_open is None:
            add(1, "red door interaction")
        if blue_open is None:
            add(n // 2 if n > 1 else 1, "blue door interaction")
        add(n - 1, "terminal: goal reached")

    # Sort forced steps
    forced = sorted(set(forced))

    # Compute coverage flags
    coverage = {
        'has_initial': 0 in forced,
        'has_terminal': (n - 1) in forced,
        'has_carrying_event': any(
            steps[i].get('carry_changed', False)
            for i in forced if i < len(steps)
        ),
        'n_forced': len(forced),
        'n_total': n,
        'n_available_slots': 8 - len(forced),
    }

    return {
        'forced_steps': forced,
        'step_notes': {i: notes.get(i, '') for i in forced},
        'coverage': coverage,
    }


def select_keyframes_variable_k(
    short_env: str,
    state_seq: List[dict],
    k_min: int = 6,
    k_max: int = 12,
) -> List[int]:
    """
    Variable-K event-complete keyframe selector.

    Strategy:
    1. Detect forced event steps (always included, event completeness first)
    2. Always include the final two frames (`n-2`, `n-1`) when available so
       terminal evidence is visible both immediately before and at episode end
    3. Include ALL forced events — do NOT truncate them
    4. Fill gaps between forced events only when:
       - Gap is small enough (gap_size <= 3)
       - AND current count < k_max
    5. K varies by sample: depends on number of critical events in the trajectory

    Critical events are NEVER dropped to fit a fixed K.
    """
    n = len(state_seq)
    if n <= 1:
        return [0] if n == 1 else []

    events = detect_event_steps(short_env, state_seq, n)
    forced = sorted(set(events['forced_steps']) | {n - 2, n - 1})

    n_forced = len(forced)

    # If forced events already exceed k_max, include all forced (event completeness > fixed K)
    if n_forced >= k_max:
        return forced

    # Start with all forced events
    filled = list(forced)

    # Build gap boundaries
    gap_boundaries = []
    if forced:
        gap_boundaries.append((0, forced[0]))
        for i in range(len(forced) - 1):
            gap_boundaries.append((forced[i] + 1, forced[i + 1]))
        gap_boundaries.append((forced[-1] + 1, n))
    else:
        gap_boundaries = [(0, n)]

    # Filter non-empty gaps
    gaps = [(start, end) for start, end in gap_boundaries if end > start]

    # Fill small gaps only
    for start, end in gaps:
        if len(filled) >= k_max:
            break
        gap_size = end - start
        # Only include gap frames if gap is small AND we still have room
        if gap_size <= 3 and len(filled) < k_max:
            # Include all steps in small gap
            for i in range(start, min(end, k_max)):
                if len(filled) >= k_max:
                    break
                if i not in filled:
                    filled.append(i)

    result = sorted(set(filled))

    # Soft floor: if below k_min but has forced events, include some extra context
    if len(result) < k_min and len(result) < n:
        # Try to add one step before first and one after last
        candidates = []
        if forced[0] > 0:
            candidates.append(forced[0] - 1)
        if forced[-1] < n - 1:
            candidates.append(forced[-1] + 1)
        for c in candidates:
            if len(result) < k_min and c not in result:
                result.append(c)
        result = sorted(result)

    # Keep chronological order; n-2/n-1 will naturally be the last two storyboard
    # frames whenever both exist.
    return sorted(set(result))


def select_keyframes_for_group(
    short_env: str,
    state_seq: List[dict],
    variant_map: Dict[str, dict],
    nocue_max_extra_hits: int = 2,
) -> List[int]:
    """
    Group-level selector used by Task C exam building.

    Final contract:
    - `full`, `nocue`, and `cf` share the SAME storyboard frame indices
    - the shared indices come from the event-complete selector on `full`
    - if `nocue` masks frames that are not currently visible, append a small
      number of those masked indices so the cue-removal perturbation is actually
      exposed in the final storyboard

    This preserves clean variant comparability:
    same frame count + same frame positions across variants, while still making
    the nocue perturbation visible to the model.
    """
    selected = list(select_keyframes_variable_k(short_env, state_seq))

    nocue_rec = variant_map.get("nocue")
    if not nocue_rec:
        return selected

    meta = nocue_rec.get("nocue_meta") or {}
    masked = sorted(int(x) for x in meta.get("masked_indices", []) if x is not None)
    if not masked:
        return selected

    extra = [m for m in masked if m not in selected][:nocue_max_extra_hits]
    if not extra:
        return selected

    return sorted(set(selected + extra))


def compute_layout(n_frames: int) -> Dict[str, Any]:
    """
    Compute adaptive near-square layout for variable-K frames.

    Layout contract:
    - Rows and cols are chosen to produce a near-square arrangement
    - Max frames shown: rows * cols >= n_frames
    - Empty cells are handled by the montage renderer (not the exam row)
    - Layout mode: "adaptive_near_square"

    Examples:
      4 frames → 2×2 (4 cells, 0 empty) — square
      5 frames → 2×3 (6 cells, 1 empty) — near-square
      6 frames → 2×3 (6 cells, 0 empty) — near-square
      7 frames → 3×3 (9 cells, 2 empty) — near-square
      9 frames → 3×3 (9 cells, 0 empty) — square
     10 frames → 2×5 or 3×4 (10 cells, 0 empty or 2 empty)
     12 frames → 3×4 (12 cells, 0 empty) — near-square
     14 frames → 2×7 (14 cells, 0 empty) vs 3×5 (15 cells, 1 empty) — prefer 3×5
     15 frames → 3×5 (15 cells, 0 empty) — near-square
    """
    if n_frames <= 1:
        return {"layout_mode": "adaptive_near_square", "rows": 1, "cols": 1, "n_cells": 1, "n_empty": 0}

    best_r, best_c = 1, n_frames
    best_score = float('inf')

    for r in range(1, n_frames + 1):
        c = (n_frames + r - 1) // r  # ceil division
        cells = r * c
        n_empty = cells - n_frames
        # Score: aspect ratio is primary, empty cells is secondary
        # aspect_ratio should be close to 1 (square)
        aspect_ratio = c / r if r > 0 else float('inf')
        # Use abs(log(aspect_ratio)) so deviation in either direction costs equally
        aspect_penalty = abs(math.log(aspect_ratio)) if aspect_ratio > 0 else float('inf')
        # Score = aspect_penalty * 2 + n_empty
        # (aspect is weighted more heavily to always prefer near-square)
        score = aspect_penalty * 2 + n_empty
        if score < best_score:
            best_score = score
            best_r, best_c = r, c

    # Ensure cols >= rows (wider than tall preferred for storyboard layout)
    if best_c < best_r:
        best_r, best_c = best_c, best_r

    return {
        "layout_mode": "adaptive_near_square",
        "rows": best_r,
        "cols": best_c,
        "n_cells": best_r * best_c,
        "n_empty": best_r * best_c - n_frames,
    }
