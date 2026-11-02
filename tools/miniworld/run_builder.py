#!/usr/bin/env python3
"""Run a MiniWorld builder with software headless rendering."""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, help='External working directory for freshly generated source pools')
    parser.add_argument("builder", help="Python builder basename in tools/miniworld")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    directory = root / 'tools/miniworld'
    script = (directory / args.builder).resolve()
    if script.parent != directory or not script.is_file():
        parser.error("builder must name a Python file in tools/miniworld")
    os.chdir(args.work_dir.resolve() if args.work_dir else root)
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    import pyglet
    pyglet.options["headless"] = True
    # Initialize headless rendering before common.py configures graphics options.
    import miniworld  # noqa: F401
    sys.path.insert(0, str(directory))
    forwarded = args.arguments
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
