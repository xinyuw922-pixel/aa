#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT="structured-cot"
if [[ "${1:-}" == "oracle-state" || "${1:-}" == "structured-cot" ]]; then
  EXPERIMENT="$1"
  shift
fi
case "$EXPERIMENT" in
  oracle-state) MODULE="tools.experiments.oracle_state" ;;
  structured-cot) MODULE="tools.experiments.structured_cot" ;;
esac
exec bash "$ROOT/scripts/_launch.sh" "$MODULE" "$@"
