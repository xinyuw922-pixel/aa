#!/usr/bin/env python3
"""Preserve Scale-2 A transitions/labels while replacing pixel-duplicate distractors."""
import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/minigrid'))
from utils.minigrid_panels import add_top_bar, compose_task_a_image, open_rgb, split_task_a_panels


from utils.json_io import load_jsonl as load


from utils.file_hash import pixel_sha256 as pixel_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('Output already exists and is preserved; use a new directory')
    rows = load(args.source)
    pools = defaultdict(list)
    for root in {Path(row['source_raw_root']) for row in rows}:
        for row in load(root / 'triplets.jsonl'):
            if row['variant'] == 'full' and row['success'] is True:
                for frame in row['frames'][1:]:
                    pools[(str(root), str(row['task']))].append((row['group_id'], frame))
    output.mkdir(parents=True)
    corrected, audit = [], []
    for row in rows:
        raw = Path(row['source_raw_root'])
        frames = row['candidate_frames']
        hashes = [pixel_hash(raw / frame) for frame in frames]
        right = 'ABCD'.index(row['answer'])
        truth = pixel_hash(raw / row['correct_next_frame'])
        if hashes[right] != truth:
            raise RuntimeError(f'Incorrect original ground truth candidate: {row["uid"]}')
        new_row = dict(row)
        if len(set(hashes)) != 4:
            revised = list(frames)
            seen = {truth}
            replacements = []
            rng = random.Random(int(hashlib.sha256(row['uid'].encode()).hexdigest()[:16], 16))
            pool = list(pools[(str(raw), row['env_task'])])
            rng.shuffle(pool)
            for index in range(4):
                if index == right:
                    continue
                fingerprint = pixel_hash(raw / revised[index])
                if fingerprint in seen:
                    candidate = next((frame for gid, frame in pool if gid != row['group_id']
                                      and pixel_hash(raw / frame) not in seen), None)
                    if candidate is None:
                        raise RuntimeError(f'Insufficient unique same-environment distractors: {row["uid"]}')
                    revised[index] = candidate
                    fingerprint = pixel_hash(raw / candidate)
                    replacements.append(index)
                seen.add(fingerprint)
            relative = Path('images/taskA') / row['env_task'] / row['group_id'] / f"step_{row['step_index']:03d}.png"
            compose_task_a_image(raw, row['current_frame'], revised, list('ABCD'), output / relative)
            new_row.update(candidate_frames=revised, image=str(relative), source_exam_root=str(output),
                           candidate_qc='pixel_unique_v1')
            audit.append(dict(uid=row['uid'], original_unique_candidates=len(set(hashes)),
                              original_correct_candidate_multiplicity=hashes.count(truth),
                              replaced_option_indices=replacements, original_candidate_frames=frames,
                              revised_candidate_frames=revised, original_source_root=row['source_exam_root'],
                              answer_preserved=True, simulator_transition_preserved=True))
        else:
            new_row['candidate_qc'] = 'pixel_unique_v1_unchanged'
        new_hashes = [pixel_hash(raw / frame) for frame in new_row['candidate_frames']]
        if len(set(new_hashes)) != 4 or new_hashes[right] != truth:
            raise RuntimeError(f'Candidate QC failed after correction: {row["uid"]}')
        corrected.append(new_row)
    manifest = output / 'taskA_expanded_manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in corrected))
    (output / 'candidate_repair_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    report = dict(status='PASS', checked=len(rows), repaired=len(audit), unchanged=len(rows)-len(audit),
                  originally_ambiguous_correct_options=sum(r['original_correct_candidate_multiplicity'] > 1 for r in audit),
                  all_candidates_pixel_unique=True, uids_labels_transitions_preserved=True,
                  source=str(args.source.resolve()), source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest())
    (output / 'candidate_qc_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))



from typing import Any, List, Tuple
import numpy as np
from utils.exam_schema import EXAM_SCHEMA_VERSION
SCHEMA_VERSION = "gridwm.exam.v1"
ACTION_NAMES = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}

def action_name(row: dict[str, Any], idx: int) -> str:
    names = row.get("actions_text") or []
    if idx < len(names):
        return str(names[idx])
    action_ids = row.get("actions_id") or []
    if idx < len(action_ids):
        return ACTION_NAMES.get(int(action_ids[idx]), str(action_ids[idx]))
    return "unknown"


def representation_prompt(row: dict[str, Any]) -> str:
    """Use one prompt for both single-image and ordered five-image evidence."""
    action = row.get('action')
    if not isinstance(action, str) or not action.strip():
        raise ValueError(f"Task A action missing: {row.get('uid')}")
    return (
        'A current MiniGrid state and four candidate next states (A, B, C, and D) are provided. '
        'The evidence may be packaged either as one combined image or as five images ordered '
        'as current state, candidate A, candidate B, candidate C, and candidate D.\n'
        f'Action: {action}\n'
        'Select the candidate that exactly represents the next state after applying the action once.\n'
        'Answer with ONLY one uppercase letter: A, B, C, or D.'
    )


def export_split_panels(row: dict[str, Any], source: Path, output_dir: Path) -> list[Path]:
    """Resolve known Task A layouts explicitly, checking raw-frame pixel identity."""
    if row.get('source_raw_root') and row.get('current_frame') and row.get('candidate_frames'):
        raw = Path(row['source_raw_root'])
        frames = [row['current_frame'], *row['candidate_frames']]
        if len(frames) != 5:
            raise ValueError('Task A needs exactly four candidate frames')
        tiles = [open_rgb(raw / frame, max_size=150) for frame in frames]
        boxes, x = [], 0
        with Image.open(source) as image:
            image = image.convert('RGB')
            expected = (sum(tile.width for tile in tiles) + 12 * 4,
                        max(tile.height + 34 for tile in tiles) + 36)
            if image.size != expected:
                raise ValueError(f'Unexpected Task A row layout: {image.size}, expected {expected}')
            for index, tile in enumerate(tiles):
                box = (x, 36, x + tile.width, 36 + tile.height + 34)
                expected_panel = add_top_bar(tile, 'Current' if index == 0 else 'ABCD'[index - 1],
                                             (49, 91, 143) if index == 0 else (86, 86, 86))
                if image.crop(box).tobytes() != expected_panel.tobytes():
                    raise ValueError('Task A composite pixels differ from ordered source frames')
                boxes.append(box)
                x += tile.width + 12
    elif str(row.get('uid', '')).startswith('A.'):
        with Image.open(source) as image:
            if image.width % 3 or image.height % 2:
                raise ValueError('Historical Task A requires the known 3x2 canvas layout')
            width, height = image.width // 3, image.height // 2
        boxes = [(0, height // 2, width, height + height // 2),
                 (width, 0, 2 * width, height), (2 * width, 0, 3 * width, height),
                 (width, height, 2 * width, 2 * height),
                 (2 * width, height, 3 * width, 2 * height)]
    else:
        raise ValueError('Unknown Task A layout; cannot safely derive split panels')
    return split_task_a_panels(source, boxes, output_dir)


def build_task_a(raw_root: Path, rows: list[dict[str, Any]], out_dir: Path, target_rows: int) -> list[dict[str, Any]]:
    full_rows = [row for row in rows if row.get("variant") == "full" and row.get("success")]
    frame_pool_by_env: dict[str, list[tuple[str, str]]] = defaultdict(list)
    transitions_by_env: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for row in full_rows:
        env = str(row.get("task") or "unknown")
        group_id = str(row.get("group_id"))
        frames = row.get("frames", [])
        for rel in frames[1:]:
            frame_pool_by_env[env].append((group_id, str(rel)))
        for step_idx in range(max(0, len(frames) - 1)):
            transitions_by_env[env].append((row, step_idx))

    rng = random.Random(20260526)
    out_rows = []
    letters = ["A", "B", "C", "D"]
    env_keys = [
        env for env in sorted(transitions_by_env)
        if len(frame_pool_by_env[env]) >= 4 and len(transitions_by_env[env]) > 0
    ]
    offsets = {env: 0 for env in env_keys}
    while len(out_rows) < target_rows:
        progressed = False
        for env in env_keys:
            transitions = transitions_by_env[env]
            while offsets[env] < len(transitions):
                row, step_idx = transitions[offsets[env]]
                offsets[env] += 1
                group_id = str(row.get("group_id"))
                frames = [str(x) for x in row.get("frames", [])]
                distractor_pool = [(gid, fp) for gid, fp in frame_pool_by_env[env] if gid != group_id]
                if len(distractor_pool) < 3:
                    continue
                correct = frames[step_idx + 1]
                distractors = rng.sample(distractor_pool, 3)
                candidates = [correct, *[fp for _gid, fp in distractors]]
                answer_idx = len(out_rows) % 4
                candidates.insert(answer_idx, candidates.pop(0))
                rel_img = Path("images/taskA") / env / group_id / f"step_{step_idx:03d}.png"
                compose_task_a_image(raw_root, frames[step_idx], candidates, letters, out_dir / rel_img)
                uid = f"A10X.{env}.{group_id}.step{step_idx:03d}"
                act = action_name(row, step_idx)
                out_rows.append({
                    "exam_id": uid,
                    "uid": uid,
                    "schema_version": SCHEMA_VERSION,
                    "task": "A",
                    "env_task": env,
                    "env_id": row.get("env_id"),
                    "group_id": group_id,
                    "step_index": step_idx,
                    "image": rel_img.as_posix(),
                    "source_exam_root": str(out_dir),
                    "prompt": (
                        f"The current observation and four candidate next observations are shown. "
                        f"The agent takes action: {act}. Which candidate is the next observation? "
                        "Answer with ONLY one uppercase letter: A, B, C, or D."
                    ),
                    "answer": letters[answer_idx],
                    "label": letters[answer_idx],
                    "action": act,
                    "action_id": (row.get("actions_id") or [None])[step_idx] if step_idx < len(row.get("actions_id") or []) else None,
                    "source_raw_root": str(raw_root),
                    "source_group_id": group_id,
                    "source_variant": "full",
                    "current_frame": frames[step_idx],
                    "correct_next_frame": correct,
                    "candidate_frames": candidates,
                    "generator": "rebuttal_packet/scripts/build_minigrid_10x_acd_manifests.py",
                })
                progressed = True
                break
            if len(out_rows) >= target_rows:
                break
        if not progressed:
            break
    return out_rows


from tools.minigrid.exam_source import compute_neg_score, image_fingerprint, load_frame, parse_traj, read_jsonl, stable_int, state_signature

def build_taskA(task_dirs: List[Path], out_dir: Path, args: Any) -> Tuple[List[dict], dict]:
    rows, stats = [], defaultdict(int)
    img_root = out_dir / "images" / "taskA"
    pools = defaultdict(list)

    # 1. Build Pools
    for td in task_dirs:
        for rec in read_jsonl(td / "triplets.jsonl"):
            tr = parse_traj(rec)
            if tr.variant == "full" and tr.n_frames > tr.n_steps:
                for t in range(min(tr.n_steps, tr.n_frames-1)):
                    pools[(tr.task, tr.actions_id[t])].append((td, tr, t))

    # 2. Generate Items
    for td in task_dirs:
        for rec in read_jsonl(td / "triplets.jsonl"):
            tr = parse_traj(rec)
            if tr.variant != "full" or tr.n_frames <= tr.n_steps: continue

            rng = np.random.default_rng(stable_int(f"{args.seed}|A|{tr.group_id}"))
            chosen_ts = rng.choice(range(tr.n_steps), size=min(tr.n_steps, args.a_per_traj), replace=False)

            for t in chosen_ts:
                if t+1 >= tr.n_frames or t+1 >= len(tr.state_seq): continue

                aid = tr.actions_id[t]
                sig_correct = state_signature(tr.state_seq[t+1])

                try:
                    curr_im = load_frame(td, tr.frames[t])
                    correct_im = load_frame(td, tr.frames[t+1])
                except FileNotFoundError:
                    stats[f"{tr.task}_skip_missing_source_frame"] += 1
                    continue

                correct_fp = image_fingerprint(correct_im)

                negs = []
                pool = pools.get((tr.task, aid), [])

                # Guard: Explicit logging for small pools
                if len(pool) < 3:
                    stats[f"{tr.task}_skip_small_pool"] += 1
                    continue

                # Strategy 1: Same trajectory, different steps (higher similarity)
                # This creates harder negatives by using states from the same trajectory
                same_traj_negs = []
                for t2 in range(tr.n_steps):
                    if t2 == t: continue  # Skip current step
                    if t2+1 >= len(tr.state_seq): continue
                    sig2 = state_signature(tr.state_seq[t2+1])
                    if sig2 != sig_correct:  # Different outcome state
                        # Score this negative for spatial similarity
                        neg_score = compute_neg_score(tr.state_seq[t+1], tr.state_seq[t2+1], aid)
                        try:
                            neg_im = load_frame(td, tr.frames[t2+1])
                            same_traj_negs.append((neg_im, neg_score, image_fingerprint(neg_im)))
                        except FileNotFoundError:
                            continue

                # Sort by score (lower = more similar = harder)
                same_traj_negs.sort(key=lambda x: x[1])
                seen_neg_fps = {correct_fp}
                for neg_im, _, neg_fp in same_traj_negs:
                    if neg_fp in seen_neg_fps:
                        continue
                    negs.append(neg_im)
                    seen_neg_fps.add(neg_fp)
                    if len(negs) >= 2:
                        break

                # Strategy 2: Different trajectories with same action (fallback)
                if len(negs) < 3:
                    step_rng = np.random.default_rng(stable_int(f"{args.seed}|A|pool|{tr.group_id}|{t}"))
                    pool_indices = step_rng.permutation(len(pool))

                    pool_negs = []
                    for idx in pool_indices:
                        src_td, tr2, t2 = pool[idx]
                        if tr2.group_id == tr.group_id: continue
                        if t2+1 >= len(tr2.state_seq): continue

                        if state_signature(tr2.state_seq[t2+1]) != sig_correct:
                            neg_score = compute_neg_score(tr.state_seq[t+1], tr2.state_seq[t2+1], aid)
                            try:
                                neg_im = load_frame(src_td, tr2.frames[t2+1])
                                pool_negs.append((neg_im, neg_score, image_fingerprint(neg_im)))
                            except FileNotFoundError:
                                continue
                            if len(pool_negs) >= 10: break  # Collect enough to pick from

                    # Sort by score and take the best ones
                    pool_negs.sort(key=lambda x: x[1])
                    for neg_im, _, neg_fp in pool_negs:
                        if neg_fp in seen_neg_fps:
                            continue
                        negs.append(neg_im)
                        seen_neg_fps.add(neg_fp)
                        if len(negs) >= 3:
                            break

                if len(negs) < 3:
                    stats[f"{tr.task}_skip_insufficient_negs"] += 1
                    continue

                # Take exactly 3 negatives (pad with last if needed)
                negs = negs[:3]
                cands = [correct_im] + negs

                order = [0,1,2,3]; rng.shuffle(order)
                w, h = curr_im.size
                comp = Image.new("RGB", (w*3, h*2), (20,20,20))
                draw = ImageDraw.Draw(comp)

                # Draw Current: Vertically centered in left column
                comp.paste(curr_im, (0, h//2))
                draw.text((10, h//2 - 15), "CURRENT (o_t)", fill=(255,255,255))

                letters = ['A', 'B', 'C', 'D']; correct_letter = ""

                for i, idx in enumerate(order):
                    col = (i % 2) + 1; row = i // 2
                    x, y = col * w, row * h
                    comp.paste(cands[idx], (x, y))
                    draw.rectangle([x+5, y+5, x+25, y+25], fill=(0,0,0))
                    draw.text((x+10, y+8), letters[i], fill=(255,255,255))
                    if idx == 0: correct_letter = letters[i]

                rel_path = f"{tr.task}/{tr.group_id}_t{t}.png"
                (img_root / rel_path).parent.mkdir(parents=True, exist_ok=True)
                comp.save(img_root / rel_path)

                action_str = tr.actions_text[t] if t < len(tr.actions_text) else f"action_{aid}"
                action_with_id = f"{action_str} (id={aid})"
                prompt = (
                    f"Action: {action_with_id}.\n"
                    "On the left is the current state. On the right is a 2x2 grid of candidates (A, B, C, D).\n"
                    "Select the correct next state o_{t+1}."
                )

                rows.append({
                    "exam_id": f"A.{tr.task}.{tr.group_id}.t{t}",
                    "uid": f"A.{tr.task}.{tr.group_id}.t{t}",
                    "schema_version": EXAM_SCHEMA_VERSION,
                    "task": "A", "env_task": tr.task, "action": action_with_id,
                    "image": f"images/taskA/{rel_path}", "prompt": prompt,
                    "answer": correct_letter, "label": order.index(0)
                })
                stats[tr.task] += 1
    return rows, dict(stats)


if __name__ == '__main__':
    main()
