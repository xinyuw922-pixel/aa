#!/usr/bin/env python3
"""Build a selected MiniGrid raw triplet bank for the 10x expansion."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.minigrid_io import load_jsonl


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(os.environ.get("GRIDWM_REPORT_ROOT", str(ROOT / "rebuttal_packet/10_minigrid_10x_expansion"))).resolve()

PROFILES: dict[str, dict[str, int]] = {
    "smoke": {
        "tier1_target": 2,
        "tier1_pool": 6,
        "nav_pool": 6,
        "nav_target": 2,
        "putnear_pool": 6,
        "putnear_target": 2,
    },
    "pilot": {
        "tier1_target": 10,
        "tier1_pool": 30,
        "nav_pool": 20,
        "nav_target": 10,
        "putnear_pool": 20,
        "putnear_target": 20,
    },
    "scale1": {
        "tier1_target": 20,
        "tier1_pool": 100,
        "nav_pool": 80,
        "nav_target": 30,
        "putnear_pool": 80,
        "putnear_target": 60,
    },
    "scale2": {
        "tier1_target": 50,
        "tier1_pool": 260,
        "nav_pool": 260,
        "nav_target": 75,
        "putnear_pool": 260,
        "putnear_target": 150,
    },
    "formal10x": {
        "tier1_target": 135,
        "tier1_pool": 700,
        "nav_pool": 700,
        "nav_target": 135,
        "putnear_pool": 700,
        "putnear_target": 200,
    },
}


ACTIVE_WORK_ROOT: Path | None = None
STAGE_RECORDS: list[dict[str, Any]] = []


def run_cmd(cmd: list[str], timeout_s: int) -> dict[str, Any]:
    if ACTIVE_WORK_ROOT is None:
        raise RuntimeError("pipeline run directory was not initialized")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    logs = ACTIVE_WORK_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    stage = f"{len(STAGE_RECORDS) + 1:02d}_{Path(cmd[1]).stem}"
    stdout_path, stderr_path = logs / (stage + ".out.log"), logs / (stage + ".err.log")
    record = {"cmd": cmd, "status": "running", "stdout_log": str(stdout_path), "stderr_log": str(stderr_path)}
    STAGE_RECORDS.append(record)
    checkpoint = ACTIVE_WORK_ROOT / "stage_status.json"
    checkpoint.write_text(json.dumps(STAGE_RECORDS, indent=2))
    print(f"Starting {stage}", flush=True)
    timed_out = False
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    result = {"cmd": cmd, "returncode": proc.returncode, "stdout_tail": stdout_path.read_text(errors="replace")[-4000:], "stderr_tail": stderr_path.read_text(errors="replace")[-4000:], "timeout": timed_out, "stdout_log": str(stdout_path), "stderr_log": str(stderr_path)}
    record.update(result, status="passed" if proc.returncode == 0 and not timed_out else "failed")
    checkpoint.write_text(json.dumps(STAGE_RECORDS, indent=2))
    if record["status"] == "failed":
        raise RuntimeError(f"Stage {stage} failed; inspect {stderr_path}. All run outputs are retained.")
    return result


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def count_groups(root: Path) -> dict[str, Any]:
    triplets = root / "triplets.jsonl"
    rows = load_jsonl(triplets) if triplets.exists() else []
    groups = {}
    env_counts = Counter()
    for row in rows:
        gid = str(row.get("group_id"))
        groups.setdefault(gid, set()).add(str(row.get("variant")))
        env_counts[str(row.get("task") or row.get("env_task") or "unknown")] += 1
    return {
        "rows": len(rows),
        "groups": len(groups),
        "complete_groups": sum(1 for variants in groups.values() if variants == {"full", "nocue", "cf"}),
        "groups_by_env": {k: v // 3 for k, v in sorted(env_counts.items())},
        "png_count": sum(1 for _ in root.rglob("*.png")),
    }


def render_md(report: dict[str, Any]) -> str:
    summary = report["final_summary"]
    env_rows = [
        f"| `{env}` | {count} |"
        for env, count in sorted(summary["groups_by_env"].items(), key=lambda item: (-item[1], item[0]))
    ]
    step_rows = [
        f"| {step['name']} | {'PASS' if step['returncode'] == 0 else 'FAIL'} | `{step['returncode']}` | `{step['path']}` |"
        for step in report["steps"]
    ]
    return "\n".join([
        "# MiniGrid 10x Raw Bank Build Report",
        "",
        f"Generated UTC: `{report['generated_utc']}`",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Run tag: `{report['run_tag']}`",
        f"- Final raw bank: `{report['final_root']}`",
        f"- Rows: `{summary['rows']}`",
        f"- Complete groups: `{summary['complete_groups']}`",
        f"- PNG files: `{summary['png_count']}`",
        "",
        "## Steps",
        "",
        "| Step | Status | Return code | Path |",
        "| --- | --- | ---: | --- |",
        *step_rows,
        "",
        "## Groups By Environment",
        "",
        "| Env | Groups |",
        "| --- | ---: |",
        *env_rows,
        "",
        "The final bank is selected before merge and then validated/audited after merge.",
    ]) + "\n"


def main() -> None:
    global ACTIVE_WORK_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), default="pilot")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--seed-offset", type=int, default=830000,
                    help="Historical Scale-2 seed offset; pass an explicit value for new-sample validation")
    ap.add_argument("--work-root", default="tmp/minigrid_10x_raw_bank_runs")
    ap.add_argument("--final-root", default="")
    ap.add_argument("--timeout-tier1-s", type=int, default=7200)
    ap.add_argument("--timeout-tier2-s", type=int, default=7200)
    ap.add_argument("--timeout-validate-s", type=int, default=1200)
    ap.add_argument("--timeout-generate-per-env-s", type=int, default=1800)
    args = ap.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    cfg = PROFILES[args.profile]
    run_tag = args.run_tag or f"{args.profile}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    work_root = resolve(args.work_root) / run_tag
    final_root = resolve(args.final_root) if args.final_root else work_root / "raw_bank"
    if work_root.exists():
        raise FileExistsError(f"Run already exists and is preserved: {work_root}; choose a new --run-tag")
    if final_root.exists():
        raise FileExistsError(f"Final bank already exists and is preserved: {final_root}")
    work_root.mkdir(parents=True, exist_ok=True)
    ACTIVE_WORK_ROOT = work_root
    (work_root / "run_config.json").write_text(json.dumps({"profile": args.profile, "seed_offset": args.seed_offset, "config": cfg}, indent=2))

    steps: list[dict[str, Any]] = []

    tier1_root = work_root / "tier1_selected"
    tier1_pool = work_root / "tier1_pool"
    tier1_report_stem = f"{run_tag}_tier1_selected"
    tier1_cmd = [
        sys.executable,
        'tools/minigrid/generate_tier1.py',
        "--target-per-env",
        str(cfg["tier1_target"]),
        "--pool-per-env",
        str(cfg["tier1_pool"]),
        "--seed-offset",
        str(args.seed_offset),
        "--pool-root",
        str(tier1_pool),
        "--out-root",
        str(tier1_root),
        "--report-stem",
        tier1_report_stem,
        "--timeout-generate-s",
        str(args.timeout_generate_per_env_s),
        "--timeout-validate-s",
        str(args.timeout_validate_s),
    ]
    tier1 = run_cmd(tier1_cmd, args.timeout_tier1_s)
    steps.append({"name": "tier1_select", "path": str(tier1_root), **tier1})

    nav_pool = work_root / "tier2_navigation_pool"
    nav_selected = work_root / "tier2_navigation_selected"
    nav_gen = run_cmd([
        sys.executable,
        'tools/minigrid/generate_navigation.py',
        "--out-dir",
        str(nav_pool),
        "--num-per-env",
        str(cfg["nav_pool"]),
        "--seed-start",
        str(args.seed_offset + 1000000),
        "--max-seed-attempts-per-env",
        str(max(cfg["nav_pool"] * 10, 200)),
    ], args.timeout_tier2_s)
    steps.append({"name": "tier2_navigation_generate", "path": str(nav_pool), **nav_gen})

    nav_sel = run_cmd([
        sys.executable,
        'tools/minigrid/select_bank.py',
        "--source-root",
        str(nav_pool),
        "--out-root",
        str(nav_selected),
        "--target-per-env",
        str(cfg["nav_target"]),
        "--report-stem",
        f"{run_tag}_tier2_navigation_selected",
    ], args.timeout_validate_s)
    steps.append({"name": "tier2_navigation_select", "path": str(nav_selected), **nav_sel})

    putnear_pool = work_root / "putnear_pool"
    putnear_selected = work_root / "putnear_selected"
    putnear_gen = run_cmd([
        sys.executable,
        'tools/minigrid/generate_interactions.py',
        "--out-dir",
        str(putnear_pool),
        "--env-keys",
        "putnear",
        "--num-per-env",
        str(cfg["putnear_pool"]),
        "--seed-start",
        str(args.seed_offset + 2000000),
        "--max-seed-attempts-per-env",
        str(max(cfg["putnear_pool"] * 10, 200)),
    ], args.timeout_tier2_s)
    steps.append({"name": "putnear_generate", "path": str(putnear_pool), **putnear_gen})

    putnear_sel = run_cmd([
        sys.executable,
        'tools/minigrid/select_bank.py',
        "--source-root",
        str(putnear_pool),
        "--out-root",
        str(putnear_selected),
        "--target-per-env",
        str(cfg["putnear_target"]),
        "--report-stem",
        f"{run_tag}_putnear_selected",
    ], args.timeout_validate_s)
    steps.append({"name": "putnear_select", "path": str(putnear_selected), **putnear_sel})

    merge = run_cmd([
        sys.executable,
        'tools/minigrid/merge_banks.py',
        "--source",
        f"tier1={tier1_root}",
        "--source",
        f"tier2nav={nav_selected}",
        "--source",
        f"putnear={putnear_selected}",
        "--out-root",
        str(final_root),
        "--report-stem",
        f"{run_tag}_merged_raw_bank",
    ], args.timeout_validate_s)
    steps.append({"name": "merge", "path": str(final_root), **merge})

    validate = run_cmd([
        sys.executable,
        'tools/minigrid/validate_navigation.py',
        "--root",
        str(final_root),
    ], args.timeout_validate_s)
    steps.append({"name": "validate", "path": str(final_root / "validation_audit.json"), **validate})

    audit = run_cmd([
        sys.executable,
        'tools/minigrid/similarity.py',
        "--roots",
        str(final_root),
        "--report-stem",
        f"{run_tag}_similarity_audit",
    ], args.timeout_validate_s)
    steps.append({"name": "similarity_audit", "path": str(OUT_DIR / f"{run_tag}_similarity_audit.md"), **audit})

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "run_tag": run_tag,
        "config": cfg,
        "work_root": str(work_root),
        "final_root": str(final_root),
        "final_summary": count_groups(final_root),
        "steps": steps,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{run_tag}_raw_bank_report.json"
    md_path = OUT_DIR / f"{run_tag}_raw_bank_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_md(report), encoding="utf-8")
    print(json.dumps({
        "report": str(json_path),
        "markdown": str(md_path),
        "final_root": str(final_root),
        "final_summary": report["final_summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
