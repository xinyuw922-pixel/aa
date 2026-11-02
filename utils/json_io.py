"""Leaf structured-data IO helpers; no calls to project modules."""
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL records in their original order."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    """Write structured runtime output outside the source tree."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL records without changing order or field values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def write_jsonl_utf8(path: Path, rows: list[dict[str, Any]]) -> None:
    """Stream UTF-8 JSONL with ordinary overwrite, without atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
