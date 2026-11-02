#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared exam id/schema helpers for GridWM-Judge pipelines."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

EXAM_SCHEMA_VERSION = "gridwm.exam.v1"
REQUEST_SCHEMA_VERSION = "gridwm.request.v1"
RESPONSE_SCHEMA_VERSION = "gridwm.response.v1"

# Shared verdict patterns for scoring and paired metrics.
#
# Uses three separate branches so each word family matches its own suffixes:
#   succeed(?:ed)?   — matches succeed / succeeded (no ful/ure suffix)
#   success(?:ful(?:ly)?)? — matches success / successful / successfully
#   fail(?:ed|ure)?  — matches fail / failed / failure
#   yes|no           — Task C response contract after the D7-safe Yes/No switch
#
# Why three branches:
#   • "succeed"/"succeeded" cannot use the "success(ful)?" branch because
#     "success" + (?:ful(?:ly)?)? would need "ful" immediately after, so "succeed" fails.
#   • "successfully" needs "ful" followed by "ly" — the optional group makes "ful" greedy,
#     backtracks to find "fully", and the trailing \b checks the end boundary.
#   • "unsuccess" fails on both branches — no "ful" after "success", and "succeed"
#     branch ends at \b which can't match between 'd' and 'u'.
#   • "unsuccessful" correctly fails: "success" matches at pos 2, but "ful" fails at 'u'.
# Polarity word sets (exact membership — avoids .startswith() bugs with "succeed"/"succeeded")
_SUCCESS_WORDS = frozenset({
    "succeed", "succeeded",   # Accept verbal success forms as well as the noun.
    "success", "successful", "successfully",
    "yes",
})
_FAIL_WORDS = frozenset({"fail", "failed", "failure", "no"})

_VERDICT_RE = re.compile(
    r"\b(succeed(?:ed)?|success(?:ful(?:ly)?)?|fail(?:ed|ure)?|yes|no)\b",
    re.IGNORECASE
)


def _as_nonempty_str(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    s = v.strip()
    return s if s else None


def resolve_uid(record: Dict[str, Any]) -> Optional[str]:
    """Resolve canonical UID/exam_id from a record."""
    for key in ("uid", "exam_id", "item_id", "id"):
        value = _as_nonempty_str(record.get(key))
        if value:
            return value
    return None


def _to_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lstrip("-").isdigit():
            return int(s)
    return None


def parse_exam_id(exam_id: str) -> Dict[str, Any]:
    """
    Parse canonical exam id.

    Supported:
      A.<env_task>.<group_id>.t<step>
      B.<env_task>.<group_id>.t<frame>
      C.<env_task>.<group_id>.<variant>.<temporal>[.<visual>]
    """
    uid = _as_nonempty_str(exam_id)
    if not uid:
        return {}

    parts = uid.split(".")
    if len(parts) < 2:
        return {"uid": uid, "exam_id": uid}

    task = parts[0]
    out: Dict[str, Any] = {"uid": uid, "exam_id": uid, "task": task}

    if task in ("A", "B") and len(parts) >= 4:
        out["env_task"] = parts[1]
        out["group_id"] = parts[2]
        tpart = parts[3]
        if tpart.startswith("t"):
            tvalue = _to_int(tpart[1:])
            if tvalue is not None:
                out["t"] = tvalue
                out["frame_idx"] = tvalue
        return out

    if task == "C" and len(parts) >= 5:
        out["env_task"] = parts[1]
        out["group_id"] = parts[2]
        out["variant"] = parts[3]
        out["temporal"] = parts[4]
        if len(parts) >= 6:
            out["visual"] = parts[5]
        if len(parts) >= 7:
            out["framing"] = parts[6]
        return out

    return out


def build_exam_meta(
    uid: str,
    row: Optional[Dict[str, Any]] = None,
    schema_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Build normalized exam metadata used by requests/responses/scoring."""
    meta = parse_exam_id(uid)
    source = row if isinstance(row, dict) else {}

    for key in ("task", "env_task", "group_id", "variant", "temporal", "visual", "framing"):
        if key not in meta:
            value = source.get(key)
            if value is not None:
                meta[key] = value

    if "t" not in meta:
        tval = _to_int(source.get("t"))
        if tval is None:
            tval = _to_int(source.get("frame_idx"))
        if tval is not None:
            meta["t"] = tval
            meta["frame_idx"] = tval

    meta["exam_id"] = meta.get("exam_id", uid)
    meta["uid"] = meta.get("uid", uid)

    resolved_schema = (
        _as_nonempty_str(schema_version)
        or _as_nonempty_str(source.get("schema_version"))
        or EXAM_SCHEMA_VERSION
    )
    meta["schema_version"] = resolved_schema
    return meta


def parse_verdict(text: str) -> Tuple[str, str]:
    """
    Parse binary verdicts with shared normalization and recovery rules.

    Handles all morphological variants:
      success / successful / successfully / succeed / succeeded
      yes
      fail / failure / failed / unsuccessful / un-fail / unfail
      no

    Resolution priority when both "success" and "fail" appear:
      1. "unsuccess"  → Fail
      2. "un-fail"/"unfail" → Fail
      3. Compound phrases: "successfully failed" / "successful failure" → Fail
      4. Last keyword wins (model's final verdict when both appear far apart)
      5. Longest keyword when only one polarity present

    Returns (answer, mode):
      answer = "Success" | "Fail" | None
      mode   = "strict" | "recoverable" | "fail"
    """
    t = (text or "").strip()
    if not t:
        return None, "fail"

    tl = t.lower()

    if tl in _SUCCESS_WORDS:
        return "Success", "strict"
    if tl in _FAIL_WORDS:
        return "Fail", "strict"

    # Negation compounds (must check before regex to avoid partial-match confusion)
    if "unsuccess" in tl:
        return "Fail", "recoverable"
    if "un-fail" in tl or "unfail" in tl:
        return "Fail", "recoverable"

    # Extract all matching keywords with positions
    matches = []
    for m in _VERDICT_RE.finditer(tl):
        word = m.group(1).lower()
        matches.append((m.start(), m.end(), word))

    if not matches:
        return None, "fail"

    # Both polarities present → resolve via compound detection or first-occurrence heuristic
    has_success = any(w in _SUCCESS_WORDS for _, _, w in matches)
    has_fail = any(w in _FAIL_WORDS for _, _, w in matches)

    if has_success and has_fail:
        success_matches = sorted(
            [(s, e, w) for s, e, w in matches if w in _SUCCESS_WORDS],
            key=lambda x: len(x[2]), reverse=True
        )
        fail_matches = sorted(
            [(s, e, w) for s, e, w in matches if w in _FAIL_WORDS],
            key=lambda x: len(x[2]), reverse=True
        )
        success_word = success_matches[0][2] if success_matches else ""
        fail_word = fail_matches[0][2] if fail_matches else ""

        # Compound check: "successfully failed" / "successful failure" etc. (within 15 chars)
        for ss, se, sw in success_matches:
            for fs, fe, fw in fail_matches:
                if abs(ss - fs) <= 15 or abs(se - fe) <= 15:
                    if sw in _SUCCESS_WORDS and fw in _FAIL_WORDS:
                        return "Fail", "recoverable"

        # Both non-compound: first occurrence wins
        all_sorted = sorted(matches, key=lambda x: x[0])
        first_word = all_sorted[0][2]
        return ("Success" if first_word in _SUCCESS_WORDS else "Fail", "recoverable")

    # Only one polarity present → longest keyword wins
    longest = sorted([(len(w), w) for _, _, w in matches], key=lambda x: x[0], reverse=True)[0][1]
    if longest in _SUCCESS_WORDS:
        return "Success", "recoverable"
    if longest in _FAIL_WORDS:
        return "Fail", "recoverable"
    return None, "fail"
