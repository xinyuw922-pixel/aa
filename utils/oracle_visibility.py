"""Construct mask-aware Oracle facts without changing simulator truth."""
from __future__ import annotations

import copy
from typing import Any, Callable

OBJECT_TYPES = {'key', 'door', 'ball', 'box', 'goal', 'lava'}
COLORS = {'red', 'green', 'blue', 'purple', 'yellow', 'grey'}
UNKNOWN = 'unobserved'


def nocue_targets(raw: dict[str, Any]) -> list[dict[str, str]]:
    """Resolve masking selectors from audit metadata, not from the mission."""
    meta = raw.get('nocue_meta') or {}
    cue = meta.get('cue')
    if isinstance(cue, dict):
        kind, color = cue.get('type'), cue.get('color')
        if kind not in OBJECT_TYPES or color not in COLORS:
            raise ValueError('NoCue cue requires a supported object type and color')
        return [{'type': kind, 'color': color}]
    targets = meta.get('targets')
    if not targets:
        mode = str(meta.get('mask_mode') or '')
        if not mode.endswith('_tile_suppression'):
            raise ValueError('NoCue requires explicit supported masking targets')
        targets = [mode.removesuffix('_tile_suppression')]
    if not isinstance(targets, list) or not targets:
        raise ValueError('NoCue masking targets must be a nonempty list')
    selectors = []
    for target in targets:
        if target in OBJECT_TYPES:
            selectors.append({'type': target})
        elif isinstance(target, str) and '_' in target:
            color, kind = target.split('_', 1)
            if color not in COLORS or kind not in OBJECT_TYPES:
                raise ValueError('Unsupported NoCue masking target: ' + target)
            selectors.append({'type': kind, 'color': color})
        else:
            raise ValueError('Unsupported NoCue masking target: ' + str(target))
    return selectors


def matches_target(obj: Any, selectors: list[dict[str, str]]) -> bool:
    """Match the type/color selectors used by the pixel suppression operation."""
    return isinstance(obj, dict) and any(
        all(obj.get(key) == value for key, value in selector.items())
        for selector in selectors
    )


def nocue_evidence(raw: dict[str, Any], observed: Callable[[dict, dict], bool]) -> dict[str, Any]:
    """Redact masked targets before object, relationship and event formatting.

    At unmasked steps, protected targets require an actual observation. Unknown
    front/carrying fields are not converted into assertions of empty space.
    The caller owns the environment-specific observation geometry.
    """
    result = copy.deepcopy(raw)
    if raw.get('variant') != 'nocue':
        return result
    selectors = nocue_targets(raw)
    states = result.get('state_seq')
    if not isinstance(states, list) or not states:
        raise ValueError('NoCue evidence requires recorded states')
    for state in states:
        if not isinstance(state.get('mask_applied'), bool):
            raise ValueError('NoCue state lacks an explicit mask_applied flag')
        masked = state['mask_applied']

        def hidden(obj: Any) -> bool:
            return matches_target(obj, selectors) and (masked or not observed(state, obj))

        state['objects'] = [obj for obj in state['objects'] if not hidden(obj)]
        front = state['front_cell']
        if hidden(front):
            state['front_cell'] = {'pos': front.get('pos'), 'type': UNKNOWN}
        agent = state['agent']
        carrying = agent.get('carrying')
        if isinstance(carrying, dict):
            candidate = {**carrying, 'pos': agent['pos']}
            if hidden(candidate):
                agent['carrying'] = UNKNOWN
        # Empty post-mask visibility is evidence, never a reason to restore GT.
        visible = state.get('visible_types', [])
        if not isinstance(visible, list):
            raise ValueError('NoCue visible_types must be a list when supplied')
        if masked:
            protected_tags = {s['type'] for s in selectors} | {
                s['color'] + '_' + s['type'] for s in selectors if 'color' in s}
            visible = [tag for tag in visible if tag not in protected_tags]
        state['visible_types'] = visible
        state.pop('visible_types_gt', None)
        state.pop('state_encoding', None)
    return result
