"""MiniWorld record path and action-enum helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def relpath_payload(payload: dict[str, Any], *, family_dir: Path, out_dir: Path) -> dict[str, Any]:
    out = dict(payload)
    family_rel = family_dir.relative_to(out_dir)
    for key in (
        "model_input_image_relpath",
        "frames_dir_relpath",
        "topviews_dir_relpath",
        "event_coverage_image_relpath",
        "prompt_relpath",
    ):
        value = out.get(key)
        if value:
            out[key] = (family_rel / value).as_posix()
    return out



def get_action_id_map(env: Any, action_names: list[str]) -> dict[str, int]:
    actions = getattr(env.unwrapped, "actions", None)
    if actions is None:
        raise RuntimeError("MiniWorld actions enum missing")
    unique_names = sorted(set(action_names))
    return {name: int(getattr(actions, name)) for name in unique_names}

