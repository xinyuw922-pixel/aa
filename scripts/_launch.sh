#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMPORT_ROOT="$ROOT"
if [[ "$ROOT" == *:* ]]; then
  if [[ "${TMPDIR:-}" == *:* ]]; then
    ALIAS_DIR="$(env -u TMPDIR mktemp -d -t gridwm-import.XXXXXXXX)"
  else
    ALIAS_DIR="$(mktemp -d -t gridwm-import.XXXXXXXX)"
  fi
  ln -s -- "$ROOT" "$ALIAS_DIR/source"
  IMPORT_ROOT="$ALIAS_DIR/source"
  trap 'rm -f -- "$ALIAS_DIR/source"; rmdir -- "$ALIAS_DIR"' EXIT
fi
env GRIDWM_IMPORT_ROOT="$IMPORT_ROOT" \
  PYTHONPATH="$IMPORT_ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 \
  "${PYTHON_BIN:-python3}" -m "$@"
