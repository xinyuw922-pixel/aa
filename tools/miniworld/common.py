#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

# Set headless rendering early so pyglet uses GLX with software Mesa fallback.
# Must happen before miniworld/pyglet is imported.
# WSL2 + NVIDIA: PYOPENGL_PLATFORM=egl segfaults with pyglet;
# using PYOPENGL_PLATFORM=glx + LIBGL_ALWAYS_SOFTWARE=1 forces Mesa
# software rendering, bypassing hardware EGL driver incompatibilities.
if os.environ.get("PYOPENGL_PLATFORM") != "glx":
    os.environ["PYOPENGL_PLATFORM"] = "glx"


# ---------------------------------------------------------------------------
# Headless fix: pyglet 1.5 needs GLU + libOpenGL, and creates a shadow window
# that fails without a display. Pre-load the libraries from the active conda
# runtime first so our scripts work without shell-level env setup.
# ---------------------------------------------------------------------------
_CONDA_PREFIX = sys.prefix
_LIB_DIRS = [
    Path.home() / ".local/lib",
    Path(_CONDA_PREFIX).resolve().parent.parent / "lib",
    Path(_CONDA_PREFIX) / "lib",
]


def _preload_opengl() -> None:
    """Pre-load GLU and libOpenGL so pyglet finds them on headless systems."""
    seen: set[Path] = set()
    for lib_dir in _LIB_DIRS:
        if not lib_dir.exists():
            continue
        if lib_dir in seen:
            continue
        seen.add(lib_dir)
        for lib_name in ("libGLU.so.1", "libOpenGL.so.0"):
            lib_path = lib_dir / lib_name
            if lib_path.exists():
                try:
                    ctypes.CDLL(str(lib_path), ctypes.RTLD_GLOBAL)
                except Exception:
                    pass


def _setup_pyglet_headless() -> None:
    """Set pyglet options for headless environments before importing pyglet."""
    try:
        import pyglet
        pyglet.options["headless"] = True
        pyglet.options["shadow_window"] = False
        pyglet.options["osmesa"] = True
    except Exception:
        pass


CANDIDATE_ENVS: List[str] = [
    "MiniWorld-FourRooms-v0",
    "MiniWorld-PickupObjects-v0",
    "MiniWorld-PutNext-v0",
]

LOCAL_CUSTOM_ENVS: List[str] = [
    "MiniWorld-KeyDoorStory-v0",
]


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def try_import_stack() -> tuple[Any, Any]:
    _preload_opengl()
    _setup_pyglet_headless()

    try:
        import gymnasium as gym
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "gymnasium 未安装，先执行："
            " python -m pip install gymnasium"
        ) from exc

    try:
        import miniworld  # noqa: F401
    except Exception as exc:
        msg = str(exc)
        if "GLU" in msg or "load_library" in msg:
            hint = (
                "pyglet 无法加载 GLU 库（OpenGL Utility Library）。\n"
                "请先手动执行以下命令安装 GLU 和 OpenGL 运行时库：\n"
                "  mkdir -p ~/.local/lib\n"
                "  # 下载 libglu1-mesa\n"
                "  wget -O /tmp/libglu1-mesa.deb "
                "http://archive.ubuntu.com/ubuntu/pool/main/libg/libglu/libglu1-mesa_9.0.2-1_amd64.deb\n"
                "  ar x /tmp/libglu1-mesa.deb && tar xf data.tar.zst -C ~/.local/\n"
                "  ln -sf ~/.local/usr/lib/x86_64-linux-gnu/libGLU.so.1 "
                "~/.local/lib/libGLU.so.1\n"
                "  # 下载 libopengl0\n"
                "  wget -O /tmp/libopengl0.deb "
                "http://archive.ubuntu.com/ubuntu/pool/main/libg/libglvnd/libopengl0_1.4.0-1_amd64.deb\n"
                "  ar x /tmp/libopengl0.deb && tar xf data.tar.zst -C ~/.local-extract/\n"
                "  ln -sf ~/.local-extract/usr/lib/x86_64-linux-gnu/libOpenGL.so.0 "
                "~/.local/lib/libOpenGL.so.0\n"
                "然后重新运行此脚本。"
            )
        elif "NoSuchDisplayException" in msg or "Cannot connect to" in msg:
            hint = (
                "pyglet 无法连接到 X display（NoSuchDisplayException）。\n"
                "包装脚本已自动设置 PYOPENGL_PLATFORM=glx + LIBGL_ALWAYS_SOFTWARE=1\n"
                "请使用包装脚本运行：\n"
                "  bash scripts/miniworld/run_with_headless_env.sh <script>.py\n"
                "如果仍有问题，请检查 LD_LIBRARY_PATH 是否包含 libGLU.so.1 和 libOSMesa.so.6。"
            )
        else:
            hint = (
                "miniworld 未安装，先执行："
                " python -m pip install -r requirements_miniworld.txt"
            )
        raise RuntimeError(hint) from exc

    return gym, miniworld


def register_local_miniworld_envs(gym: Any) -> List[str]:
    """Register repo-local MiniWorld envs exactly once per interpreter."""
    try:
        from tools.miniworld.custom_envs import register_local_miniworld_envs as _register
    except Exception:
        return []
    return _register(gym)


def list_registered_miniworld_envs(gym: Any) -> List[str]:
    ids: List[str] = []
    for spec in gym.envs.registry.values():
        env_id = getattr(spec, "id", "")
        if isinstance(env_id, str) and env_id.startswith("MiniWorld-"):
            ids.append(env_id)
    return sorted(ids)


def make_env(
    gym: Any,
    env_id: str,
    *,
    seed: int = 42,
    render_mode: str = "rgb_array",
    obs_width: int = 80,
    obs_height: int = 60,
    window_width: int | None = None,
    window_height: int | None = None,
    env_kwargs: Dict[str, Any] | None = None,
) -> tuple[Any, np.ndarray, Dict[str, Any]]:
    if window_width is None:
        window_width = obs_width
    if window_height is None:
        window_height = obs_height
    kwargs = dict(env_kwargs or {})
    env = gym.make(
        env_id,
        render_mode=render_mode,
        obs_width=obs_width,
        obs_height=obs_height,
        window_width=window_width,
        window_height=window_height,
        **kwargs,
    )
    obs, info = env.reset(seed=seed)
    return env, obs, info


def action_name(env: Any, action: int) -> str:
    actions = getattr(getattr(env, "unwrapped", env), "actions", None)
    if actions is None:
        return str(action)
    try:
        return actions(action).name
    except Exception:
        return str(action)


def export_rgb(path: Path, array: np.ndarray) -> None:
    img = Image.fromarray(np.asarray(array, dtype=np.uint8))
    img.save(path)


def maybe_render_top_view(env: Any, *, use_vis_fb: bool = True) -> np.ndarray | None:
    target = getattr(env, "unwrapped", env)
    fn = getattr(target, "render_top_view", None)
    if fn is None:
        return None
    try:
        frame_buffer = getattr(target, "vis_fb", None) if use_vis_fb else None
        arr = fn(frame_buffer=frame_buffer) if frame_buffer is not None else fn()
        if arr is None:
            return None
        return np.asarray(arr, dtype=np.uint8)
    except Exception:
        return None


def maybe_render_top_view_with_scale(
    env: Any, *, use_vis_fb: bool = True
) -> tuple[np.ndarray | None, dict[str, float] | None]:
    """Render top-view and return MiniWorld's world-to-pixel scale metadata."""
    target = getattr(env, "unwrapped", env)
    fn = getattr(target, "render_top_view", None)
    if fn is None:
        return None, None
    try:
        frame_buffer = getattr(target, "vis_fb", None) if use_vis_fb else None
        if frame_buffer is not None:
            arr, scale = fn(frame_buffer=frame_buffer, return_scale=True)
        else:
            arr, scale = fn(return_scale=True)
        if arr is None:
            return None, None
        return np.asarray(arr, dtype=np.uint8), dict(scale)
    except Exception:
        return None, None


from utils.miniworld_images import concat_vertical_rgb


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def runtime_context() -> Dict[str, Any]:
    return {
        "cwd": os.getcwd(),
        "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
    }
