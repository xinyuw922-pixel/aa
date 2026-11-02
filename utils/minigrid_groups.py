"""Ordered grouping of MiniGrid trajectory records."""
from __future__ import annotations
from collections import defaultdict
from typing import Any

def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("group_id"))].append(row)
    return dict(grouped)

