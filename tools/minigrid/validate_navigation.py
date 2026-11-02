#!/usr/bin/env python3
"""Validate Tier-2 MiniGrid navigation triplets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


from utils.minigrid_io import load_jsonl


def validate_group(root: Path, group_id: str, rows: list[dict[str, Any]]) -> tuple[bool, str]:
    by_variant = {str(row.get("variant")): row for row in rows}
    if set(by_variant) != {"full", "nocue", "cf"}:
        return False, "missing_variant"
    full, nocue, cf = by_variant["full"], by_variant["nocue"], by_variant["cf"]
    if full.get("actions_id") != nocue.get("actions_id") or full.get("actions_id") != cf.get("actions_id"):
        return False, "actions_mismatch"
    if not full.get("success") or not nocue.get("success"):
        return False, "full_or_nocue_not_success"
    if cf.get("success") or float(cf.get("reward", 0.0)) > 0:
        return False, "cf_not_failure"
    if not cf.get("cf_meta"):
        return False, "missing_cf_meta"
    if not nocue.get("nocue_meta"):
        return False, "missing_nocue_meta"
    for row in rows:
        if not row.get("frames"):
            return False, f"{row.get('variant')}_missing_frames"
        for rel in row["frames"]:
            if not (root / rel).exists():
                return False, f"missing_frame:{rel}"
        if len(row.get("frames", [])) != len(row.get("state_seq", [])):
            return False, f"{row.get('variant')}_frame_state_len_mismatch"
    return True, "pass"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-audit", default="")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    triplets = root / "triplets.jsonl"
    rows = load_jsonl(triplets)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("group_id"))].append(row)
    audit_rows = []
    ok_count = 0
    for group_id, group_rows in sorted(grouped.items()):
        ok, reason = validate_group(root, group_id, group_rows)
        ok_count += int(ok)
        audit_rows.append({"group_id": group_id, "ok": ok, "reason": reason})
    report = {
        "root": str(root),
        "rows": len(rows),
        "groups": len(grouped),
        "valid_groups": ok_count,
        "invalid_groups": len(grouped) - ok_count,
        "groups_detail": audit_rows,
    }
    out_audit = Path(args.out_audit) if args.out_audit else root / "validation_audit.json"
    if not out_audit.is_absolute():
        out_audit = (ROOT / out_audit).resolve()
    out_audit.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ok_count == len(grouped) and grouped else 1)


if __name__ == "__main__":
    main()
