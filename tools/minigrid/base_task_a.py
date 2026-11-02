"""Base Task A sampling and deterministic, pixel-distinct candidate assembly."""
from pathlib import Path
from typing import Any, Dict, List, Tuple
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw
from utils.exam_schema import EXAM_SCHEMA_VERSION
from tools.minigrid.exam_source import (
    compute_neg_score, image_fingerprint, load_frame, parse_traj, read_jsonl, stable_int, state_signature,
)

def _build_sampled_task_a(task_dirs: List[Path], out_dir: Path, args: Any) -> Tuple[List[dict], dict]:
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
                            same_traj_negs.append((load_frame(td, tr.frames[t2+1]), neg_score))
                        except FileNotFoundError:
                            continue

                # Sort by score (lower = more similar = harder)
                same_traj_negs.sort(key=lambda x: x[1])
                negs.extend([n[0] for n in same_traj_negs[:2]])  # Take top 2 from same trajectory

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
                                pool_negs.append((load_frame(src_td, tr2.frames[t2+1]), neg_score))
                            except FileNotFoundError:
                                continue
                            if len(pool_negs) >= 10: break  # Collect enough to pick from

                    # Sort by score and take the best ones
                    pool_negs.sort(key=lambda x: x[1])
                    negs.extend([n[0] for n in pool_negs[:3]])

                if len(negs) < 3:
                    stats[f"{tr.task}_skip_insufficient_negs"] += 1
                    continue

                # Take exactly 3 negatives (pad with last if needed)
                negs = negs[:3]

                try:
                    curr_im = load_frame(td, tr.frames[t])
                    cands = [load_frame(td, tr.frames[t+1])] + negs
                except FileNotFoundError:
                    stats[f"{tr.task}_skip_missing_source_frame"] += 1
                    continue

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

def detect_duplicate_options(img: Image.Image) -> List[List[str]]:
    """Heuristic duplicate detector over the Task A 3-column candidate grid."""
    W, H = img.size
    cell_w = W // 3
    cell_h = H // 2
    boxes = {
        "A": (cell_w, 0, 2 * cell_w, cell_h),
        "B": (2 * cell_w, 0, 3 * cell_w, cell_h),
        "C": (cell_w, cell_h, 2 * cell_w, 2 * cell_h),
        "D": (2 * cell_w, cell_h, 3 * cell_w, 2 * cell_h),
    }
    sigs: Dict[str, str] = {}
    for lab, box in boxes.items():
        crop = img.crop(box)
        cw, ch = crop.size
        # Ignore the option label badge and a thin outer border, but keep the
        # full MiniGrid state content for pixel-level duplicate detection.
        inner = crop.crop((max(28, cw // 10), max(28, ch // 10), cw - max(4, cw // 20), ch - max(4, ch // 20))).resize((64, 64))
        sigs[lab] = image_fingerprint(inner)
    inv: Dict[str, List[str]] = defaultdict(list)
    for lab, sig in sigs.items():
        inv[sig].append(lab)
    return [labs for labs in inv.values() if len(labs) > 1]

def select_unique_negatives(
    td: Path,
    tr: Any,
    t: int,
    pools: Dict[Tuple[str, int], List[Tuple[Path, Any, int]]],
    seed: int,
) -> Tuple[Image.Image, List[Image.Image]]:
    aid = tr.actions_id[t]
    correct_im = load_frame(td, tr.frames[t + 1])
    correct_fp = image_fingerprint(correct_im)
    sig_correct = state_signature(tr.state_seq[t + 1])
    negs: List[Image.Image] = []
    seen_neg_fps = {correct_fp}

    same_traj_negs = []
    for t2 in range(tr.n_steps):
        if t2 == t or t2 + 1 >= len(tr.state_seq):
            continue
        sig2 = state_signature(tr.state_seq[t2 + 1])
        if sig2 == sig_correct:
            continue
        neg_score = compute_neg_score(tr.state_seq[t + 1], tr.state_seq[t2 + 1], aid)
        try:
            neg_im = load_frame(td, tr.frames[t2 + 1])
        except FileNotFoundError:
            continue
        same_traj_negs.append((neg_im, neg_score, image_fingerprint(neg_im)))

    same_traj_negs.sort(key=lambda x: x[1])
    for neg_im, _, neg_fp in same_traj_negs:
        if neg_fp in seen_neg_fps:
            continue
        negs.append(neg_im)
        seen_neg_fps.add(neg_fp)
        if len(negs) >= 2:
            break

    pool = pools.get((tr.task, aid), [])
    if len(negs) < 3:
        step_rng = np.random.default_rng(stable_int(f"{seed}|A|pool|{tr.group_id}|{t}"))
        pool_indices = step_rng.permutation(len(pool))
        pool_negs = []
        for idx in pool_indices:
            src_td, tr2, t2 = pool[idx]
            if tr2.group_id == tr.group_id or t2 + 1 >= len(tr2.state_seq):
                continue
            if state_signature(tr2.state_seq[t2 + 1]) == sig_correct:
                continue
            neg_score = compute_neg_score(tr.state_seq[t + 1], tr2.state_seq[t2 + 1], aid)
            try:
                neg_im = load_frame(src_td, tr2.frames[t2 + 1])
            except FileNotFoundError:
                continue
            pool_negs.append((neg_im, neg_score, image_fingerprint(neg_im)))
            if len(pool_negs) >= 64:
                break
        pool_negs.sort(key=lambda x: x[1])
        for neg_im, _, neg_fp in pool_negs:
            if neg_fp in seen_neg_fps:
                continue
            negs.append(neg_im)
            seen_neg_fps.add(neg_fp)
            if len(negs) >= 3:
                break

    if len(negs) < 3:
        raise RuntimeError(f"{tr.group_id}.t{t}: insufficient unique negatives after repair search")

    return correct_im, negs[:3]

def render_taska_image(
    curr_im: Image.Image,
    correct_im: Image.Image,
    negs: List[Image.Image],
    order: List[int],
) -> Tuple[Image.Image, str, int]:
    cands = [correct_im] + negs
    w, h = curr_im.size
    comp = Image.new("RGB", (w * 3, h * 2), (20, 20, 20))
    draw = ImageDraw.Draw(comp)
    comp.paste(curr_im, (0, h // 2))
    draw.text((10, h // 2 - 15), "CURRENT (o_t)", fill=(255, 255, 255))

    letters = ["A", "B", "C", "D"]
    correct_letter = ""
    for i, idx in enumerate(order):
        col = (i % 2) + 1
        row = i // 2
        x, y = col * w, row * h
        comp.paste(cands[idx], (x, y))
        draw.rectangle([x + 5, y + 5, x + 25, y + 25], fill=(0, 0, 0))
        draw.text((x + 10, y + 8), letters[i], fill=(255, 255, 255))
        if idx == 0:
            correct_letter = letters[i]
    return comp, correct_letter, order.index(0)



# Final candidate-search configuration for the published base cohort.
_CANDIDATE_REFRESH_UIDS = frozenset((
    'A.redblue.redblue_s000001.t20', 'A.redblue.redblue_s000002.t8',
    'A.doorkey.doorkey_s000019.t12', 'A.lavagap.lavagap_s000003.t3',
    'A.redblue.redblue_s000000.t6', 'A.redblue.redblue_s000000.t5',
    'A.redblue.redblue_s000002.t6', 'A.redblue.redblue_s000002.t12',
    'A.lavagap.lavagap_s000011.t3', 'A.lavagap.lavagap_s000025.t3',
))


def build_base_task_a(task_dirs: list[Path], output: Path, seed: int = 42,
                      per_trajectory: int = 6, cohort_uids: list[str] | None = None) -> list[dict]:
    """Sample transitions first, then finalize candidates without shifting later RNG draws."""
    rows, _ = _build_sampled_task_a(task_dirs, output, SimpleNamespace(seed=seed, a_per_traj=per_trajectory))
    pools, groups = defaultdict(list), {}
    for directory in task_dirs:
        for record in read_jsonl(directory/'triplets.jsonl'):
            trajectory = parse_traj(record)
            if trajectory.variant != 'full':
                continue
            if trajectory.group_id in groups:
                raise ValueError(f'Duplicate Task A source group: {trajectory.group_id}')
            groups[trajectory.group_id] = (directory, trajectory)
            for step in range(min(trajectory.n_steps, trajectory.n_frames-1)):
                pools[trajectory.task, trajectory.actions_id[step]].append((directory, trajectory, step))
    for row in rows:
        path = output/row['image']
        with Image.open(path) as image:
            duplicates = detect_duplicate_options(image.convert('RGB'))
        if row['uid'] in _CANDIDATE_REFRESH_UIDS or duplicates:
            _, _, gid, timestep = row['uid'].split('.')
            step = int(timestep[1:])
            directory, trajectory = groups[gid]
            current = load_frame(directory, trajectory.frames[step])
            correct, negatives = select_unique_negatives(directory, trajectory, step, pools, seed)
            order = [0, 1, 2, 3]
            np.random.default_rng(stable_int(f'{seed}|A|{gid}')).shuffle(order)
            image, answer, label = render_taska_image(current, correct, negatives, order)
            if detect_duplicate_options(image):
                raise ValueError(f'Duplicate Task A candidates: {row["uid"]}')
            image.save(path)
            row.update(answer=answer, label=label)
        row.update(source_exam_root=str(output), prompt_contract='base_native')
    if cohort_uids is not None:
        if len(set(cohort_uids)) != len(cohort_uids):
            raise ValueError('Duplicate Task A cohort UID')
        by_uid = {row['uid']: row for row in rows}
        missing = set(cohort_uids)-set(by_uid)
        if missing:
            raise ValueError(f'Task A cohort is not generated by these inputs: {sorted(missing)[:10]}')
        rows = [by_uid[uid] for uid in cohort_uids]
    return rows
