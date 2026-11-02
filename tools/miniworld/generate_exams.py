#!/usr/bin/env python3
"""Generate MiniWorld A/C/D source pools and exams without model requests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
WORK = Path("experiments/generated/miniworld")


def stages():
    def stage(name, builder, directory, *args):
        return (name, builder, directory, ["--out_dir", str(WORK / directory), *args])
    return [
        stage("a_fourrooms", 'generate_fourrooms.py', "taskA-MW-v2-candidate-50", "--num_questions", "50", '--isolate-attempts'),
        ("a_balance", 'balance_task_a_labels.py', "taskA-MW-v2-candidate-50-balanced",
         ["--src_dir", str(WORK / "taskA-MW-v2-candidate-50"), "--dst_dir", str(WORK / "taskA-MW-v2-candidate-50-balanced")]),
        stage("a_pickup", 'generate_pickup_objects.py', "taskA-MW-v2-PickupObjects", "--num_questions", "15"),
        stage("a_putnext", 'generate_put_next.py', "taskA-MW-v2-PutNext", "--num_questions", "15"),
        stage("a_exam", 'assemble_task_a.py', "taskA-MW-v2-crossenv"),
        stage("cd_migration", 'gen_miniworld_taskd_migration_showcase.py', "taskD-MW-migration-showcase-families-paired"),
        stage("cd_structural", 'gen_miniworld_taskd_structural_showcase.py', "taskD-MW-structural-showcase-v2"),
        stage("cd_round1", 'generate_templates.py', "taskD-MW-highdisc-expansion-round1-clean", '--template-set', 'round1'),
        stage("cd_round2", 'generate_templates.py', "taskD-MW-highdisc-expansion-round2", '--template-set', 'round2'),
        stage("cd_round3", 'generate_templates.py', "taskD-MW-highdisc-expansion-round3", '--template-set', 'round3'),
        stage("c_exam", 'gen_taskC_MW_formal_exam.py', "taskC-MW-formal-bank"),
        stage("d_exam", 'gen_taskD_MW_reference_family_exam50.py', "taskD-MW-reference-family-exam50"),
    ]


def main():
    available = stages()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT,
                        help='Working directory; generated files stay under experiments/generated/miniworld here')
    parser.add_argument("--stages", nargs="+", choices=[s[0] for s in available],
                        help="Default: all stages. Select pending stages to preserve existing outputs.")
    parser.add_argument("--timeout-per-stage", type=int, default=1200)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    if work_dir != ROOT and (work_dir == ROOT or ROOT in work_dir.parents):
        parser.error('Use an external work directory')
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.timeout_per_stage <= 0:
        parser.error("timeout must be positive")
    selected = [s for s in available if not args.stages or s[0] in args.stages]
    for name, _, directory, _ in selected:
        if (work_dir / WORK / directory).exists():
            parser.error(f"Stage {name} output already exists; select only pending stages. Nothing was removed.")
    log_root = work_dir / WORK / "pipeline_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    records = []
    stamp = time.time_ns()
    status = log_root / f"run_{stamp}.json"
    for name, builder, _, extra in selected:
        cmd = [sys.executable, str(ROOT / 'tools/miniworld/run_builder.py'), '--work-dir', str(work_dir), builder, "--", *extra]
        log_path = log_root / f"{stamp}_{name}.log"
        record = dict(stage=name, command=cmd, log=str(log_path), status="running")
        records.append(record)
        status.write_text(json.dumps(records, indent=2) + "\n")
        print(f"Running {name}; log={log_path}", flush=True)
        with log_path.open("w") as log:
            try:
                result = subprocess.run(cmd, cwd=work_dir, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=args.timeout_per_stage)
                record.update(returncode=result.returncode,
                              status="passed" if result.returncode == 0 else "failed")
            except subprocess.TimeoutExpired:
                record.update(status="timeout")
        status.write_text(json.dumps(records, indent=2) + "\n")
        if record["status"] != "passed":
            raise SystemExit(f"Stopped at {name}; intermediate files retained; see {log_path}")
    print(f"Selected builders completed. Verify counts/images using the generation audit; status={status}")


if __name__ == "__main__":
    main()
