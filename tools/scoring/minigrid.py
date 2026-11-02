#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MiniGrid answer parsing and per-question scoring.

Evaluates VLM performance on GridWM-Judge benchmark with capability-preserving scoring.
Directly scores raw inference outputs against ground truth, implementing robust answer
extraction to distinguish format noise from capability limitations.

Design Principles:
- SSOT: Scoring based ONLY on exam_dir (ground truth) + raw inference outputs
- Capability-Fidelity: Recover from harmless formatting noise (strict vs recoverable)
- Transparency: Report both strict and recoverable performance metrics
- Completeness: Every UID gets a score or documented failure reason

Usage:
    bash scripts/score.sh --protocol minigrid --exam-dir /external/exam \
        --responses /external/responses.jsonl --output /external/scores --per-row-only

Output: JSON report with micro/macro scores, per-task breakdown, failure analysis
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, Counter

from utils.exam_schema import (
    EXAM_SCHEMA_VERSION,
    build_exam_meta,
    parse_exam_id as parse_exam_id_ssot,
    parse_verdict as _parse_verdict_ssot,
    resolve_uid,
)

API_ERROR_PREFIX = "__GW_ERR__:"


# -------------------------
# Shared utilities (aligned with run_inference.py)
# -------------------------

def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Load JSONL file with error handling."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def parse_exam_id(uid: str) -> Dict[str, Any]:
    return parse_exam_id_ssot(uid)


# -------------------------
# Ground Truth Loading (SSOT)
# -------------------------

@dataclass
class Gold:
    uid: str
    task: str
    env_task: Optional[str]
    group_id: Optional[str]
    exam_schema_version: str = EXAM_SCHEMA_VERSION
    t: Optional[int] = None
    variant: Optional[str] = None
    temporal: Optional[str] = None
    visual: Optional[str] = None

    # Task-specific ground truth
    a_answer: Optional[str] = None          # "A"/"B"/"C"/"D"
    a_label: Optional[int] = None           # 0..3
    b_answer_json: Optional[Dict[str, Any]] = None
    c_answer: Optional[str] = None          # "Success"/"Fail"
    c_label: Optional[int] = None           # optional
    # Task E fields
    answer: Optional[str] = None            # exact-match gold label
    query_type: Optional[str] = None        # "final_carrying" | "object_fate"
    horizon_bucket: Optional[str] = None   # "short" | "medium" | "long"
    sampling_role: Optional[str] = None    # Object role for Task E queries.


def load_gold_from_exam_dir(exam_dir: Path) -> Dict[str, Gold]:
    """Load ground truth from exam directory (build_exam.py output)."""
    gold: Dict[str, Gold] = {}

    # Task A
    p = exam_dir / "task_a_exam.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            uid = resolve_uid(r)
            if not uid:
                continue
            info = build_exam_meta(uid, row=r)
            g = Gold(
                uid=uid,
                task="A",
                env_task=info.get("env_task"),
                group_id=info.get("group_id"),
                exam_schema_version=info.get("schema_version", EXAM_SCHEMA_VERSION),
                t=info.get("t"),
                a_answer=r.get("answer"),
                a_label=r.get("label"),
            )
            # Normalize answer format
            if isinstance(g.a_answer, str):
                g.a_answer = g.a_answer.strip().upper()
            if g.a_answer not in ("A", "B", "C", "D"):
                if isinstance(g.a_label, int):
                    letters = {0: "A", 1: "B", 2: "C", 3: "D"}
                    g.a_answer = letters.get(g.a_label)
            gold[uid] = g

    # Task B
    p = exam_dir / "task_b_exam.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            uid = resolve_uid(r)
            if not uid:
                continue
            info = build_exam_meta(uid, row=r)
            g = Gold(
                uid=uid,
                task="B",
                env_task=info.get("env_task"),
                group_id=info.get("group_id"),
                exam_schema_version=info.get("schema_version", EXAM_SCHEMA_VERSION),
                t=info.get("t"),
                b_answer_json=r.get("answer_json"),
            )
            gold[uid] = g

    # Task C / D: prefer the D filename when selecting reference/query rows.
    # Priority: task_d_exam.jsonl → task_c_exam.jsonl
    # Both files are loaded with task="D"/"C" respectively to prevent collision.
    for fname, task_label in [("task_d_exam.jsonl", "D"), ("task_c_exam.jsonl", "C")]:
        p = exam_dir / fname
        if not p.exists():
            continue
        for r in load_jsonl(p):
            uid = resolve_uid(r)
            if not uid:
                continue
            info = build_exam_meta(uid, row=r)
            g = Gold(
                uid=uid,
                task=task_label,
                env_task=info.get("env_task"),
                group_id=info.get("group_id"),
                exam_schema_version=info.get("schema_version", EXAM_SCHEMA_VERSION),
                variant=r.get("variant") or info.get("variant"),
                temporal=info.get("temporal"),
                visual=r.get("visual") or info.get("visual"),
                c_answer=r.get("answer"),
                c_label=r.get("label"),
            )
            if isinstance(g.c_answer, str):
                g.c_answer = g.c_answer.strip()
            gold[uid] = g

    # Task E (Long-Range Discrete State Tracking)
    # Support endpoint and checkpoint object-identity/status queries.
    for fname in ("task_e_exam_v13.jsonl", "task_e_exam_v12.jsonl", "task_e_exam_v11.jsonl", "task_e_exam_v1.jsonl"):
        p = exam_dir / fname
        if p.exists():
            for r in load_jsonl(p):
                uid = resolve_uid(r)
                if not uid:
                    continue
                info = build_exam_meta(uid, row=r)
                g = Gold(
                    uid=uid,
                    task="E",
                    env_task=info.get("env_task"),
                    group_id=info.get("group_id"),
                    exam_schema_version=info.get("schema_version", EXAM_SCHEMA_VERSION),
                    answer=r.get("answer"),
                    query_type=r.get("query_type"),
                    horizon_bucket=r.get("horizon_bucket"),
                    sampling_role=r.get("sampling_role"),
                )
                gold[uid] = g

    return gold


# -------------------------
# Response Loading
# -------------------------

@dataclass
class Resp:
    uid: str
    exam_id: Optional[str]
    schema_version: Optional[str]
    exam_schema_version: Optional[str]
    model: Optional[str]   # Model identifier extracted from response metadata.
    pred: str
    raw: str
    ok: bool
    error: Optional[str] = None


def discover_latest_responses(responses_dir: Path) -> Path:
    """Auto-discover the most recent experiment results."""
    if not responses_dir.exists():
        raise SystemExit(f"Responses directory not found: {responses_dir}")

    # Find experiment directories (exclude _requests_from_exam)
    exp_dirs = [d for d in responses_dir.iterdir()
               if d.is_dir() and not d.name.startswith('_')]
    if not exp_dirs:
        raise SystemExit(f"No experiment directories found in {responses_dir}")

    # Find the most recent experiment
    exp_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest_exp = exp_dirs[0]

    # Find response files in the latest experiment
    resp_files = list(latest_exp.glob("*.jsonl"))
    if not resp_files:
        raise SystemExit(f"No response files found in {latest_exp}")

    # Return the most recent response file
    resp_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return resp_files[0]


def load_responses(responses_path: Path) -> Tuple[Dict[str, Resp], Dict[str, Any]]:
    """
    Load responses from file or auto-discover from directory.

    Accepts:
      - Single JSONL file
      - Directory containing *.jsonl (sharded outputs supported)
    """
    files: List[Path] = []
    if responses_path.is_dir():
        files = sorted([p for p in responses_path.glob("*.jsonl") if p.is_file()])
    else:
        files = [responses_path]

    out: Dict[str, Resp] = {}
    duplicates = 0
    total_lines = 0
    for fp in files:
        for r in load_jsonl(fp):
            total_lines += 1
            uid = resolve_uid(r)
            if not uid:
                continue
            if uid in out:
                duplicates += 1
                continue

            raw = r.get("raw", r.get("pred", ""))
            pred = r.get("pred", "")
            meta = r.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            exam_meta = meta.get("exam", {}) if isinstance(meta, dict) else {}
            exam_id = r.get("exam_id") or (exam_meta.get("exam_id") if isinstance(exam_meta, dict) else None) or uid
            resp_schema = r.get("schema_version")
            exam_schema = (
                r.get("exam_schema_version")
                or (exam_meta.get("schema_version") if isinstance(exam_meta, dict) else None)
            )

            # Enhanced error detection
            ok = meta.get("ok", True)
            error = meta.get("error")

            # Check for API errors in raw text
            if isinstance(raw, str) and raw.startswith(API_ERROR_PREFIX):
                ok = False
                error = raw

            # Extract the model identifier for downstream analysis.
            model_name = r.get("model") or (meta.get("model") if isinstance(meta, dict) else None)

            out[uid] = Resp(
                uid=uid,
                exam_id=str(exam_id) if exam_id is not None else None,
                schema_version=str(resp_schema) if resp_schema else None,
                exam_schema_version=str(exam_schema) if exam_schema else None,
                model=str(model_name) if model_name else None,
                pred=str(pred),
                raw=str(raw),
                ok=ok,
                error=str(error) if error else None,
            )

    telemetry = {
        "n_files": len(files),
        "files": [str(p) for p in files],
        "total_lines": total_lines,
        "n_uids": len(out),
        "duplicates_dropped": duplicates,
        "response_schema_versions": dict(Counter([r.schema_version or "legacy" for r in out.values()])),
        "exam_schema_versions_in_responses": dict(Counter([r.exam_schema_version or "unknown" for r in out.values()])),
    }
    return out, telemetry


# -------------------------
# Task A/C Parsing (Capability-Fidelity)
# -------------------------

_A_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

def parse_task_a_answer(text: str) -> Tuple[Optional[str], str]:
    """
    Extract A/B/C/D answer with capability-preserving parsing.

    Returns (answer_letter, mode):
      mode = strict | recoverable | fail
    """
    t = (text or "").strip()
    if not t:
        return None, "fail"
    # Strict: exactly one char
    if len(t) == 1 and t.upper() in ("A","B","C","D"):
        return t.upper(), "strict"
    # Recoverable: find anywhere in text
    m = _A_LETTER_RE.search(t)
    if m:
        return m.group(1).upper(), "recoverable"
    return None, "fail"

def parse_task_c_answer(text: str) -> Tuple[Optional[str], str]:
    """Parse trajectory verdicts with the shared answer-schema helper."""
    ans, mode = _parse_verdict_ssot(text)
    return ans, mode


def parse_task_e_answer(text: str) -> Tuple[Optional[str], str]:
    """
    Parse Task E structured JSON answer.

    Task E output format: {"answer": "<one label from label_space>"}
    Returns (answer, mode):
      mode = "strict" | "recoverable" | "fail"
    """
    import json as _json
    t = (text or "").strip()
    if not t:
        return None, "fail"

    # Try direct JSON parse
    try:
        obj = _json.loads(t)
        if isinstance(obj, dict) and "answer" in obj:
            ans = str(obj["answer"]).strip()
            if ans:
                return ans, "strict"
    except Exception:
        pass

    # Try stripping markdown code fence
    if "```" in t:
        for part in t.split("```"):
            stripped = part.strip()
            if not stripped:
                continue
            try:
                obj = _json.loads(stripped)
                if isinstance(obj, dict) and "answer" in obj:
                    ans = str(obj["answer"]).strip()
                    if ans:
                        return ans, "recoverable"
            except Exception:
                continue

    # Fallback: try to find "answer" anywhere in text
    for line in t.split("\n"):
        line = line.strip()
        if '"answer"' in line or "'answer'" in line:
            try:
                # Try to extract {"answer": "value"} from a longer line
                for fragment in [line, line[max(0, line.find('"')):]]:
                    try:
                        obj = _json.loads(fragment)
                        if isinstance(obj, dict) and "answer" in obj:
                            ans = str(obj["answer"]).strip()
                            if ans:
                                return ans, "recoverable"
                    except Exception:
                        continue
            except Exception:
                continue

    return None, "fail"


# -------------------------
# Task B Parsing + Scoring (Perception IoU)
# -------------------------

def _strip_code_fence(s: str) -> str:
    t = (s or "").strip()
    if "```" not in t:
        return t
    parts = t.split("```")
    if len(parts) >= 3:
        return parts[1].strip()
    return t.replace("```", "").strip()

def _extract_json_object(text: str) -> Optional[str]:
    """
    Extract the most likely JSON object substring.
    - If whole string is JSON, return it.
    - Else take substring between first '{' and last '}'.
    """
    t = (text or "").strip()
    if not t:
        return None
    t = _strip_code_fence(t)
    # Quick path
    if t.startswith("{") and t.endswith("}"):
        return t
    l = t.find("{")
    r = t.rfind("}")
    if 0 <= l < r:
        return t[l:r+1]
    return None

def _min_fix_json_commas(s: str) -> str:
    # Remove trailing commas before } or ]
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s

def parse_task_b_json(text: str) -> Tuple[Optional[Dict[str, Any]], str, bool]:
    """
    Parse Task B JSON with recovery logic.

    Returns (obj, mode, has_markdown_fence):
      mode = strict | recoverable | fail_json | fail_type
      has_markdown_fence = True if raw text starts with markdown fence
    """
    t = (text or "").strip()
    if not t:
        return None, "fail_json", False

    has_fence = t.startswith("```")

    cand = _extract_json_object(t)
    if cand is None:
        return None, "fail_json", has_fence

    # Strict attempt
    try:
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj, ("strict" if cand.strip() == t.strip() else "recoverable"), has_fence
        return None, "fail_type", has_fence
    except Exception:
        # Minimal recovery: trailing commas
        try:
            obj = json.loads(_min_fix_json_commas(cand))
            if isinstance(obj, dict):
                return obj, "recoverable", has_fence
            return None, "fail_type", has_fence
        except Exception:
            return None, "fail_json", has_fence

def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    if isinstance(x, str) and x.strip().lstrip("-").isdigit():
        return int(x.strip())
    return None

def _to_pos(x: Any) -> Optional[Tuple[int,int]]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        a = _to_int(x[0]); b = _to_int(x[1])
        if a is not None and b is not None:
            return (a, b)
    return None

def _canon_state(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (int, float, bool)):
        return str(int(x)) if isinstance(x, bool) or (isinstance(x, float) and float(x).is_integer()) else str(x)
    if isinstance(x, str):
        return x.strip()
    return str(x)


def _semantic_state_for_type(typ: Any, state: Any) -> Optional[str]:
    """
    Task B semantic state contract:
    - door: state is meaningful and compared (0/1/2)
    - non-door: state is semantically undefined and ignored -> None
    """
    if not isinstance(typ, str):
        return _canon_state(state)
    if typ.strip() == "door":
        return _canon_state(state)
    return None

def _canon_obj(o: Any) -> Optional[Tuple[str, Tuple[int,int], Optional[str], Optional[str]]]:
    """
    Canonical object tuple: (type, pos, color, state)
    Returns None if critical fields missing.
    """
    if not isinstance(o, dict):
        return None
    typ = o.get("type")
    pos = _to_pos(o.get("pos"))
    if not isinstance(typ, str) or pos is None:
        return None
    color = o.get("color")
    color = color.strip() if isinstance(color, str) else None
    typ = typ.strip()
    state = _semantic_state_for_type(typ, o.get("state"))
    return (typ, pos, color, state)

def _canon_front(o: Any) -> Optional[Tuple[Tuple[int,int], str, Optional[str]]]:
    if not isinstance(o, dict):
        return None
    pos = _to_pos(o.get("pos"))
    typ = o.get("type")
    if pos is None or not isinstance(typ, str):
        return None
    typ = typ.strip()
    state = _semantic_state_for_type(typ, o.get("state"))
    return (pos, typ, state)

def _canon_carry(x: Any) -> Optional[Tuple[str, Optional[str]]]:
    """
    carrying: object|null
      - None => None
      - dict => (type, color)
      - str => (str, None)
    """
    if x is None:
        return None
    if isinstance(x, dict):
        typ = x.get("type")
        if not isinstance(typ, str):
            return ("<unknown>", None)
        color = x.get("color")
        color = color.strip() if isinstance(color, str) else None
        return (typ.strip(), color)
    if isinstance(x, str):
        return (x.strip(), None)
    return ("<unknown>", None)

def normalize_task_b(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Task B JSON to canonical format."""
    out: Dict[str, Any] = {}

    agent = obj.get("agent")
    if isinstance(agent, dict):
        out["agent_pos"] = _to_pos(agent.get("pos"))
        out["agent_dir"] = _to_int(agent.get("dir"))
        out["carrying"] = _canon_carry(agent.get("carrying"))
    else:
        out["agent_pos"] = None
        out["agent_dir"] = None
        out["carrying"] = None

    fc = obj.get("front_cell")
    out["front_cell"] = _canon_front(fc)

    objs = obj.get("objects")
    s = set()
    if isinstance(objs, list):
        for o in objs:
            co = _canon_obj(o)
            if co is not None:
                s.add(co)
    out["objects_set"] = s

    return out

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a and b:
        return 0.0
    if a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0

def score_task_b(
    pred_obj: Dict[str, Any],
    gold_obj: Dict[str, Any],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, Any], List[str]]:
    """
    Score Task B with weighted perception IoU.

    Returns:
      total_score in [0,1],
      component dict,
      failure_tags (schema/partial indicators)
    """
    p = normalize_task_b(pred_obj or {})
    g = normalize_task_b(gold_obj or {})

    failures: List[str] = []

    # Component scores
    agent_pos = 1.0 if (p["agent_pos"] is not None and g["agent_pos"] is not None and p["agent_pos"] == g["agent_pos"]) else 0.0
    if p["agent_pos"] is None:
        failures.append("B_missing_agent_pos")

    agent_dir = 1.0 if (p["agent_dir"] is not None and g["agent_dir"] is not None and p["agent_dir"] == g["agent_dir"]) else 0.0
    if p["agent_dir"] is None and weights.get("agent_dir", 0.0) > 0:
        failures.append("B_missing_agent_dir")

    carrying = 1.0 if (p["carrying"] == g["carrying"]) else 0.0
    if "carrying" not in (pred_obj.get("agent") or {}):
        failures.append("B_missing_carrying")

    front_cell = 1.0 if (p["front_cell"] is not None and g["front_cell"] is not None and p["front_cell"] == g["front_cell"]) else 0.0
    if p["front_cell"] is None:
        failures.append("B_missing_front_cell")

    objects = jaccard(p["objects_set"], g["objects_set"])
    if not isinstance(pred_obj.get("objects"), list):
        failures.append("B_missing_objects")

    comp = {
        "agent_pos": agent_pos,
        "agent_dir": agent_dir,
        "carrying": carrying,
        "front_cell": front_cell,
        "objects_jaccard": objects,
        "n_gold_objects": len(g["objects_set"]),
        "n_pred_objects": len(p["objects_set"]),
    }

    # Weighted sum
    wsum = sum(max(0.0, float(v)) for v in weights.values())
    if wsum <= 0:
        # Safe fallback weights
        weights = {"agent_pos": 0.30, "agent_dir": 0.10, "carrying": 0.15,
                  "front_cell": 0.25, "objects_jaccard": 0.20}
        wsum = sum(weights.values())

    total = (
        weights.get("agent_pos", 0.0) * agent_pos +
        weights.get("agent_dir", 0.0) * agent_dir +
        weights.get("carrying", 0.0) * carrying +
        weights.get("front_cell", 0.0) * front_cell +
        weights.get("objects_jaccard", 0.0) * objects
    ) / wsum

    # Mark partial schema if missing fields
    if failures:
        failures.append("B_partial_schema")

    return float(total), comp, failures


# -------------------------
# Scoring Logic
# -------------------------

def _default_weights_b() -> Dict[str, float]:
    """Return Task B component weights.
    
    Canonical image-frame direction is constant and has zero weight.
    A zero direction weight also excludes missing-direction penalties."""
    return {
        "agent_pos": 0.30,    # Most critical for spatial reasoning
        "agent_dir": 0.00,    # Constant image-frame direction has zero weight.
        "carrying": 0.15,     # Object manipulation state
        "front_cell": 0.25,   # Immediate environment perception
        "objects_jaccard": 0.20,  # Overall scene understanding
    }

def _parse_weights_b(s: Optional[str]) -> Dict[str, float]:
    """Parse custom weights string like 'agent_pos=0.4,objects_jaccard=0.3'."""
    if not s:
        return _default_weights_b()
    out = _default_weights_b()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except Exception:
            pass
    return out

def score_one(uid: str, g: Gold, r: Optional[Resp], weights_b: Dict[str, float], b_acc_threshold: float) -> Dict[str, Any]:
    """
    Score single example with detailed breakdown.

    Returns per-example row with score, correct, failure_code, parse_mode, details...
    """
    row: Dict[str, Any] = {
        "uid": uid,
        "task": g.task,
        "env_task": g.env_task,
        "group_id": g.group_id,
    }
    if g.task == "A":
        row["t"] = g.t
    if g.task == "C":
        row["variant"] = g.variant
        row["temporal"] = g.temporal
        if g.visual is not None:
            row["visual"] = g.visual
    if g.task == "D":
        row["variant"] = g.variant

    # Record Task E metadata before any early return,
    # so that missing_response / api_error rows are counted correctly in per_task_E breakdowns
    if g.task == "E":
        row["query_type"] = getattr(g, "query_type", None)
        row["horizon_bucket"] = getattr(g, "horizon_bucket", None)
        row["sampling_role"] = getattr(g, "sampling_role", None)

    # Missing response
    if r is None:
        row.update({"score": 0.0, "correct": False, "failure": "missing_response", "parse_mode": "fail"})
        return row

    # API error
    if not r.ok or (r.error is not None):
        row.update({"score": 0.0, "correct": False, "failure": "api_error", "parse_mode": "fail", "error": r.error})
        return row

    text = r.raw if r.raw else r.pred

    if g.task == "A":
        ans, mode = parse_task_a_answer(text)
        row["parse_mode"] = mode
        if ans is None:
            row.update({"score": 0.0, "correct": False, "failure": "A_no_letter"})
            return row
        gold_ans = g.a_answer
        if gold_ans is None:
            row.update({"score": 0.0, "correct": False, "failure": "A_missing_gold"})
            return row
        correct = (ans == gold_ans)
        row.update({"score": 1.0 if correct else 0.0, "correct": correct, "pred_norm": ans, "gold": gold_ans})
        if not correct:
            row["failure"] = "A_wrong"
        return row

    if g.task == "C":
        ans, mode = parse_task_c_answer(text)
        row["parse_mode"] = mode
        if ans is None:
            row.update({"score": 0.0, "correct": False, "failure": "C_no_label"})
            return row
        gold_ans = g.c_answer
        if isinstance(gold_ans, str):
            gold_ans = gold_ans.strip().lower()
        # Normalize supported gold labels before comparison.
        # Canonical form is "Success" or "Fail" (capitalized first letter).
        gold_norm = None
        if gold_ans == "success" or gold_ans == "succeed" or gold_ans == "succeeded":
            gold_norm = "Success"
        elif gold_ans == "fail" or gold_ans == "failure" or gold_ans == "failed":
            gold_norm = "Fail"
        if gold_norm is None:
            row.update({"score": 0.0, "correct": False, "failure": "C_missing_gold",
                        "gold_raw": g.c_answer})
            return row
        correct = (ans == gold_norm)
        row.update({"score": 1.0 if correct else 0.0, "correct": correct,
                    "pred_norm": ans, "gold": gold_norm})
        if not correct:
            row["failure"] = "C_wrong"
        return row

    # Task D uses identical verdict parsing as Task C (same full/nocue/cf variants, same "Success"/"Fail" gold)
    if g.task == "D":
        ans, mode = parse_task_c_answer(text)
        row["parse_mode"] = mode
        if ans is None:
            row.update({"score": 0.0, "correct": False, "failure": "D_no_label"})
            return row
        gold_ans = g.c_answer
        if isinstance(gold_ans, str):
            gold_ans = gold_ans.strip().lower()
        gold_norm = None
        if gold_ans == "success" or gold_ans == "succeed" or gold_ans == "succeeded":
            gold_norm = "Success"
        elif gold_ans == "fail" or gold_ans == "failure" or gold_ans == "failed":
            gold_norm = "Fail"
        if gold_norm is None:
            row.update({"score": 0.0, "correct": False, "failure": "D_missing_gold",
                        "gold_raw": g.c_answer})
            return row
        correct = (ans == gold_norm)
        row.update({"score": 1.0 if correct else 0.0, "correct": correct,
                    "pred_norm": ans, "gold": gold_norm})
        if not correct:
            row["failure"] = "D_wrong"
        return row

    if g.task == "B":
        obj, mode, has_fence = parse_task_b_json(text)
        row["parse_mode"] = mode
        row["has_markdown_fence"] = has_fence
        if obj is None:
            row.update({"score": 0.0, "correct": False, "failure": "B_json_parse_error"})
            return row
        if g.b_answer_json is None:
            row.update({"score": 0.0, "correct": False, "failure": "B_missing_gold"})
            return row

        score, comp, failures = score_task_b(obj, g.b_answer_json, weights_b)
        correct = (score >= b_acc_threshold)
        row.update({
            "score": score,
            "correct": correct,
            "b_components": comp,
            "b_acc_threshold": b_acc_threshold,
            "b_acc_at_threshold": correct,
        })
        if failures:
            # Primary failure bucket
            primary = next((f for f in failures if f != "B_partial_schema"), failures[-1])
            row["failure"] = primary
            row["failure_tags"] = failures
        return row

    if g.task == "E":
        # Task E: parse JSON answer field, exact match against gold.
        # Task E metadata (query_type, horizon_bucket, sampling_role) is already written
        # before the missing_response/api_error early-returns above.
        ans, mode = parse_task_e_answer(text)
        row["parse_mode"] = mode
        if ans is None:
            row.update({"score": 0.0, "correct": False, "failure": "E_parse_error"})
            return row
        gold_ans = g.answer
        if gold_ans is None:
            row.update({"score": 0.0, "correct": False, "failure": "E_missing_gold"})
            return row
        correct = (ans == gold_ans)
        row.update({
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "pred_norm": ans,
            "gold": gold_ans,
        })
        if not correct:
            row["failure"] = "E_wrong"
        return row

    row.update({"score": 0.0, "correct": False, "failure": "unknown_task", "parse_mode": "fail"})
    return row


def _agg_stats(rows: List[Dict[str, Any]], task: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate statistics for rows, optionally filtered by task."""
    if task is not None:
        rows = [r for r in rows if r.get("task") == task]
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "mean_score": 0.0,
            "acc": 0.0,
            "parse_mode": {},
            "failure_hist": {},
            "api_error_rate": 0.0,
            "missing_rate": 0.0,
        }

    scores = [float(r.get("score", 0.0)) for r in rows]
    corrects = [bool(r.get("correct", False)) for r in rows]
    parse_modes = Counter([r.get("parse_mode", "fail") for r in rows])
    failures = Counter([r.get("failure", "none") for r in rows if r.get("failure")])

    api_err = sum(1 for r in rows if r.get("failure") == "api_error")
    missing = sum(1 for r in rows if r.get("failure") == "missing_response")

    return {
        "n": n,
        "mean_score": sum(scores)/n,
        "acc": sum(corrects)/n,
        "parse_mode": dict(parse_modes),
        "failure_hist": dict(failures),
        "api_error_rate": api_err/n,
        "missing_rate": missing/n,
    }


def main():
    ap = argparse.ArgumentParser(description="GridWM-Judge SSOT Scorer")
    ap.add_argument("--exam_dir", default="datasets/exams",
                   help="Directory containing task_*_exam.jsonl (build_exam.py output)")
    ap.add_argument("--responses", default="runs/responses",
                   help="Responses jsonl file OR directory (auto-discovers latest experiment)")
    ap.add_argument("--out", default="runs/scores/score_report.json",
                   help="Output score report JSON path")
    ap.add_argument("--dump_rows", action="store_true",
                   help="Include per-example rows in output JSON (can be large)")
    ap.add_argument('--per-row-only', action='store_true', help='Write scored_rows.jsonl only')
    ap.add_argument("--max_bad_examples", type=int, default=50,
                   help="Store up to N failure examples per failure type")
    ap.add_argument("--b_acc_threshold", type=float, default=0.90,
                   help="Task B: score>=threshold counts as correct for acc metric")
    ap.add_argument("--b_weights", type=str, default=None,
                   help="Task B component weights, e.g. agent_pos=0.25,objects_jaccard=0.35,...")
    ap.add_argument("--dump_gold", type=str, default=None,
                   help="Output path for gold-enriched responses (for action ablation downstream scripts). "
                        "Each row includes gold label, model prediction, and score — enabling "
                        "score_action_ablation.py to work without needing the original exam dir. "
                        "Example: runs/scores/responses_with_gold.jsonl")
    args = ap.parse_args()

    exam_dir = Path(args.exam_dir)
    responses_path = Path(args.responses)

    # Load ground truth (SSOT)
    print(f"📚 Loading ground truth from {exam_dir}")
    gold = load_gold_from_exam_dir(exam_dir)
    print(f"   Found {len(gold)} ground truth examples")

    # Load responses
    if responses_path.is_dir():
        responses_path = discover_latest_responses(responses_path)
        print(f"🤖 Using latest experiment: {responses_path.parent.name}")

    print(f"📊 Loading responses from {responses_path}")
    responses, resp_tel = load_responses(responses_path)
    print(f"   Found {len(responses)} response examples")

    # Setup scoring parameters
    weights_b = _parse_weights_b(args.b_weights)

    # Score every gold UID (SSOT: gold is canonical set)
    print("🧮 Scoring examples...")
    rows: List[Dict[str, Any]] = []
    for uid, g in gold.items():
        r = responses.get(uid)
        rows.append(score_one(uid, g, r, weights_b, args.b_acc_threshold))

    if args.per_row_only:
        from utils.json_io import write_jsonl_utf8
        write_jsonl_utf8(Path(args.out).parent / 'scored_rows.jsonl', rows)
        return

    # Include gold labels and correctness when exporting enriched responses.
    if args.dump_gold:
        gold_output_path = Path(args.dump_gold)
        gold_output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_rows = []
        rows_by_uid = {row["uid"]: row for row in rows}
        for uid, g in gold.items():
            r_row = responses.get(uid)
            s_row = rows_by_uid.get(uid)
            if r_row is None or s_row is None:
                continue
            task = s_row.get("task", "")
            enriched = {
                "uid": uid,
                "exam_id": getattr(r_row, "exam_id", uid),
                "model": getattr(r_row, "model", ""),
                "pred_norm": s_row.get("pred_norm", ""),
                "gold": s_row.get("gold", ""),
                "gold_label": s_row.get("label"),
                "correct": s_row.get("correct", False),
                "score": s_row.get("score", 0.0),
                "failure": s_row.get("failure", ""),
                "parse_mode": s_row.get("parse_mode", ""),
                "has_markdown_fence": s_row.get("has_markdown_fence", False),
                "task": task,
                "env_task": s_row.get("env_task", ""),
                "group_id": s_row.get("group_id", ""),
            }
            if task == "B":
                enriched.update({
                    "b_score": s_row.get("score", 0.0),
                    "b_components": s_row.get("b_components", {}),
                })
            if task == "C":
                enriched.update({
                    "variant": s_row.get("variant", ""),
                    "temporal": s_row.get("temporal", ""),
                    "visual": s_row.get("visual", ""),
                })
            if task == "D":
                enriched.update({
                    "variant": s_row.get("variant", ""),
                })
            if task == "E":
                enriched.update({
                    "query_type": s_row.get("query_type", ""),
                    "horizon_bucket": s_row.get("horizon_bucket", ""),
                    "sampling_role": s_row.get("sampling_role", ""),
                })
            enriched_rows.append(enriched)
        with gold_output_path.open("w", encoding="utf-8") as f:
            for row in enriched_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"💾 Gold-enriched responses written to {gold_output_path} ({len(enriched_rows)} rows)")

    # Coverage analysis
    n_gold = len(gold)
    n_resp = len(responses)
    missing_uids = [r["uid"] for r in rows if r.get("failure") == "missing_response"]
    extra_uids = sorted([uid for uid in responses.keys() if uid not in gold])
    gold_schema_versions = dict(Counter([g.exam_schema_version or EXAM_SCHEMA_VERSION for g in gold.values()]))

    # Aggregate statistics
    per_task = {t: _agg_stats(rows, task=t) for t in ("A","B","C","D","E")}
    per_env_task: Dict[str, Dict[str, Any]] = {}
    groups = defaultdict(list)
    for r in rows:
        key = f'{r.get("task")}:{r.get("env_task")}'
        groups[key].append(r)
    for k, rs in groups.items():
        per_env_task[k] = _agg_stats(rs)

    # Global failure histogram
    global_fail_hist = Counter([r.get("failure","none") for r in rows if r.get("failure")])

    # Failure examples for debugging
    failure_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        f = r.get("failure")
        if not f:
            continue
        if len(failure_examples[f]) < args.max_bad_examples:
            ex = {k: r.get(k) for k in ("uid","task","env_task","parse_mode","failure","pred_norm","gold","error")}
            if r.get("task") == "B":
                ex["b_score"] = r.get("score")
                ex["b_components"] = r.get("b_components")
            failure_examples[f].append(ex)

    # Build final report
    report: Dict[str, Any] = {
        "ssot": {
            "gold_source": str(exam_dir),
            "scoring_policy": {
                "A_C": "accuracy with strict vs recoverable parsing",
                "B": {
                    "metric": "weighted perception IoU",
                    "weights": weights_b,
                    "acc_threshold": args.b_acc_threshold,
                },
            },
        },
        "coverage": {
            "n_gold": n_gold,
            "n_response_uids": n_resp,
            "n_scored": len(rows),
            "missing_responses": len(missing_uids),
            "extra_responses_not_in_gold": len(extra_uids),
            "gold_exam_schema_versions": gold_schema_versions,
            "responses_telemetry": resp_tel,
        },
        "overall_micro": _agg_stats(rows),
        "per_task": per_task,
        "per_env_task": per_env_task,
        "failure_hist": dict(global_fail_hist),
        "failure_examples": dict(failure_examples),
    }

    # Task E-specific breakdowns (query_type and horizon_bucket)
    # Support endpoint and checkpoint query fields.
    if "E" in per_task and per_task["E"]["n"] > 0:
        task_e_rows = [r for r in rows if r.get("task") == "E"]
        qt_stats = {}
        for qt in ("final_carrying", "object_fate",
                   "identity_of_first_picked_object", "status_of_first_picked_object_at_checkpoint",
                   "carrying_at_checkpoint"):
            qt_r = [r for r in task_e_rows if r.get("query_type") == qt]
            if qt_r:
                qt_stats[qt] = _agg_stats(qt_r)
        hb_stats = {}
        for hb in ("short", "medium", "long"):
            hb_r = [r for r in task_e_rows if r.get("horizon_bucket") == hb]
            if hb_r:
                hb_stats[hb] = _agg_stats(hb_r)
        sr_stats = {}
        for sr in ("identity_short", "identity_medium", "identity_long",
                   "status_early_holding", "status_mid_holding",
                   "status_first_post_drop", "status_late_post_drop",
                   "retained_control", "retained_control_medium"):
            sr_r = [r for r in task_e_rows if r.get("sampling_role") == sr]
            if sr_r:
                sr_stats[sr] = _agg_stats(sr_r)
        env_stats = {}
        task_e_envs = sorted({r.get("env_task") for r in task_e_rows if r.get("env_task")})
        for env in task_e_envs:
            env_r = [r for r in task_e_rows if r.get("env_task") == env]
            if env_r:
                env_stats[env] = _agg_stats(env_r)
        report["per_task_E"] = {
            "by_query_type": qt_stats,
            "by_horizon_bucket": hb_stats,
            "by_sampling_role": sr_stats,
            "by_env_task": env_stats,
        }

    if extra_uids:
        report["coverage"]["extra_uids_sample"] = extra_uids[:50]

    if args.dump_rows:
        report["rows"] = rows

    # Save report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary
    print("\n" + "="*60)
    print("🎯 SCORING COMPLETE")
    print("="*60)
    overall = report["overall_micro"]
    print(f"📊 Overall: {overall['n']} examples")
    print(f"   Mean Score: {overall['mean_score']:.3f}")
    print(f"   Accuracy: {overall['acc']:.3f}")
    print(f"   API Error Rate: {overall['api_error_rate']:.1%}")
    print(f"   Missing Rate: {overall['missing_rate']:.1%}")
    print(f"📋 Strict vs Recoverable: {overall['parse_mode']}")

    print(f"\n🔍 Per-Task Breakdown:")
    for task in ("A", "B", "C", "D", "E"):
        if task in per_task:
            tstats = per_task[task]
            print(f"   {task}: acc={tstats['acc']:.3f}, n={tstats['n']}")
            if task == "E" and tstats["n"] > 0:
    # Show Task E breakdowns by query type.
                all_qts = ("final_carrying", "object_fate",
                           "identity_of_first_picked_object", "status_of_first_picked_object_at_checkpoint",
                           "carrying_at_checkpoint")
                qt_rows = {qt: [r for r in rows if r.get("task")=="E" and r.get("query_type")==qt]
                           for qt in all_qts}
                for qt, qrows in qt_rows.items():
                    if qrows:
                        qt_acc = sum(1 for r in qrows if r.get("correct")) / len(qrows)
                        print(f"      {qt}: acc={qt_acc:.3f}, n={len(qrows)}")
                hb_rows = {hb: [r for r in rows if r.get("task")=="E" and r.get("horizon_bucket")==hb]
                           for hb in ("short", "medium", "long")}
                for hb, hrows in hb_rows.items():
                    if hrows:
                        hb_acc = sum(1 for r in hrows if r.get("correct")) / len(hrows)
                        print(f"      horizon={hb}: acc={hb_acc:.3f}, n={len(hrows)}")
                # Use report["per_task_E"] which now has by_sampling_role
                pE = report.get("per_task_E", {})
                for sr, sdata in pE.get("by_sampling_role", {}).items():
                    print(f"      role={sr}: acc={sdata['acc']:.3f}, n={sdata['n']}")
                for env, edata in pE.get("by_env_task", {}).items():
                    print(f"      env={env}: acc={edata['acc']:.3f}, n={edata['n']}")

    print(f"\n💾 Report saved: {out_path}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
