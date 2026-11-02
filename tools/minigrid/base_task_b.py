"""Dual-step Task B construction with inherited anchors and strict image uniqueness."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from utils.minigrid_visible_state import (
    world_to_image, agent_to_image, is_visible_object,
    canonical_state_for_type, compute_agent_dir_from_image,
)
from utils.json_io import load_jsonl, write_jsonl

ENVS = ['doorkey', 'keycorridor', 'lavagap', 'memory', 'multiroom', 'redblue']
EXPANSION_VARIANTS = ('full', 'nocue')


def build_base_task_b(raw_root: Path, anchor_exam: Path, output: Path) -> list[dict]:
    """Require explicit anchor inputs and preserve native dual-step selection."""
    anchors = defaultdict(dict)
    for row in load_jsonl(anchor_exam):
        parts = row['uid'].split('.')
        env, gid, step = parts[1], parts[2], int(parts[-1][1:])
        if gid in anchors[env]:
            raise ValueError(f'Duplicate Task B anchor group: {env}/{gid}')
        anchors[env][gid] = AnchorRow(row['uid'], env, gid, step, row['image'], row)
    if set(anchors) != set(ENVS) or any(len(anchors[env]) != 12 for env in ENVS):
        raise ValueError('Task B requires 12 inherited full anchors per environment')
    triplets = {}
    for env in ENVS:
        entries = {}
        for row in load_jsonl(raw_root / env / 'triplets.jsonl'):
            key = (row['group_id'], row['variant'])
            if key in entries:
                raise ValueError(f'Duplicate Task B raw source: {env}/{key}')
            entries[key] = row
        triplets[env] = entries
        for gid, anchor in anchors[env].items():
            if (gid, 'full') not in entries or (gid, 'nocue') not in entries:
                raise ValueError(f'Missing full/nocue source for anchor {env}/{gid}')
            if env != 'doorkey':
                candidate = build_candidate(env, gid, 'full', entries[gid, 'full'], anchor.step, raw_root)
                if candidate.row['answer_json'] != anchor.row['answer_json']:
                    raise ValueError(f'Task B anchor/raw answer mismatch: {anchor.uid}')
                if candidate.image_hash != hash_image(anchor_exam.parent / anchor.image_rel):
                    raise ValueError(f'Task B anchor/raw image mismatch: {anchor.uid}')
    pool = build_candidate_pool(triplets, anchors, raw_root)
    extras = select_extras(anchors, pool, anchor_exam.parent)
    pairs = select_doorkey_pairs(build_doorkey_reanchor_pool(triplets, raw_root))
    rows = []
    for env in ENVS:
        for gid in sorted(anchors[env]):
            if env == 'doorkey':
                full, nocue = pairs[gid]
                selected = [(convert_full_candidate_to_anchor(full), Path(full.source_frame)),
                            (convert_nocue_candidate_to_extra(nocue), Path(nocue.source_frame))]
            else:
                anchor, extra = anchors[env][gid], extras[env][gid]
                anchor_row = {**anchor.row, '_slot': 'anchor', '_variant': 'full'}
                selected = [(anchor_row, anchor_exam.parent / anchor.image_rel),
                            (extra.row, Path(extra.source_frame))]
            for row, source in selected:
                destination = output / row['image']
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                rows.append({**row, 'source_exam_root': str(output)})
        hashes = [hash_image(output / row['image']) for row in rows if row['uid'].split('.')[1] == env]
        if len(hashes) != 24 or len(set(hashes)) != 24:
            raise ValueError(f'Task B requires 24 pixel-distinct questions for {env}')
    return rows


@dataclass(frozen=True)
class AnchorRow:
    uid: str
    env: str
    group_id: str
    step: int
    image_rel: str
    row: dict


@dataclass(frozen=True)
class Candidate:
    env: str
    group_id: str
    variant: str
    step: int
    image_hash: str
    image_name: str
    image_rel: str
    source_frame: str
    info_score: int
    visible_object_count: int
    front_type: str
    carrying: bool
    answer_signature: str
    row: dict

    @property
    def uid(self) -> str:
        return self.row["uid"]


def compute_info_score(state_entry: dict) -> int:
    world_agent = state_entry.get("agent", {})
    carrying = world_agent.get("carrying")
    triplet_front = state_entry.get("front_cell", {})
    front_type = triplet_front.get("type", "empty")
    if front_type == "floor":
        front_type = "empty"
    world_dir = world_agent.get("dir", 0)
    agent_pos = world_agent.get("pos")
    if agent_pos is None:
        return 0
    agent_x, agent_y = agent_pos
    fc_wp = triplet_front.get("pos")
    front_img_pos = None
    if fc_wp:
        front_img_pos = world_to_image(fc_wp[0], fc_wp[1], agent_x, agent_y, world_dir)

    world_objs = state_entry.get("objects", [])
    state_encoding = state_entry.get("state_encoding", [])
    has_carrying = 1 if carrying is not None else 0
    has_nonempty_front = 1 if front_type not in ("empty", "floor") else 0
    has_visible_objs = 0
    for obj in world_objs:
        wx, wy = obj["pos"]
        x_img, y_img = world_to_image(wx, wy, agent_x, agent_y, world_dir)
        if is_visible_object(x_img, y_img, state_encoding, object_type=obj.get('type')):
            if front_img_pos and (x_img, y_img) == front_img_pos:
                continue
            has_visible_objs = 1
            break
    return has_carrying + has_nonempty_front + 2 * has_visible_objs


def hash_image(path: Path) -> str:
    arr = np.array(Image.open(path))
    return hashlib.md5(arr.tobytes()).hexdigest()


def answer_signature(answer: dict) -> str:
    objs = sorted(
        (
            obj.get("type"),
            tuple(obj.get("pos", [])),
            obj.get("color"),
            obj.get("state"),
        )
        for obj in answer.get("objects", [])
    )
    payload = {
        "front": (
            answer["front_cell"].get("type"),
            tuple(answer["front_cell"].get("pos", [])),
            answer["front_cell"].get("state"),
        ),
        "carrying": answer["agent"].get("carrying"),
        "objects": objs,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def build_candidate(env: str, group_id: str, variant: str, triplet: dict, step: int, raw_root: Path) -> Candidate:
    state_seq = triplet.get("state_seq", [])
    if step >= len(state_seq):
        raise RuntimeError(f"{env}/{group_id}/{variant}: step {step} out of range")

    state = state_seq[step]
    world_agent = state.get("agent", {})
    world_front = state.get("front_cell", {})
    state_encoding = state.get("state_encoding", [])
    world_objects = state.get("objects", [])

    agent_pos = world_agent.get("pos")
    world_dir = world_agent.get("dir")
    if agent_pos is None or world_dir is None:
        raise RuntimeError(f"{env}/{group_id}/{variant}/t{step}: missing agent state")
    agent_x, agent_y = agent_pos

    front_type = world_front.get("type", "empty")
    if front_type == "floor":
        front_type = "empty"
    front_world_pos = world_front.get("pos")
    if front_world_pos is None:
        raise RuntimeError(f"{env}/{group_id}/{variant}/t{step}: missing front cell pos")

    agent_img = agent_to_image(agent_x, agent_y, world_dir)
    front_img = list(world_to_image(front_world_pos[0], front_world_pos[1], agent_x, agent_y, world_dir))
    objects_img = []
    for obj in world_objects:
        wx, wy = obj["pos"]
        x_img, y_img = world_to_image(wx, wy, agent_x, agent_y, world_dir)
        if is_visible_object(x_img, y_img, state_encoding, object_type=obj.get('type')):
            if [x_img, y_img] == front_img:
                continue
            objects_img.append(
                {
                    "type": obj.get("type", "unknown"),
                    "pos": [x_img, y_img],
                    "color": obj.get("color"),
                    "state": canonical_state_for_type(obj.get("type", ""), obj.get("state", 0)),
                }
            )

    answer = {
        "agent": {
            "pos": agent_img,
            "dir": compute_agent_dir_from_image(agent_img, front_img),
            "carrying": world_agent.get("carrying"),
        },
        "front_cell": {
            "pos": front_img,
            "type": front_type,
            "state": canonical_state_for_type(front_type, world_front.get("state", 0)),
        },
        "objects": objects_img,
    }
    image_name = f"{group_id}_{variant}_t{step}.png"
    image_rel = f"images/taskB/{image_name}"
    source_frame = raw_root / env / triplet["frames"][step]
    uid = f"B2.{env}.{group_id}.{variant}.t{step}"
    row = {
        "exam_id": uid,
        "uid": uid,
        "schema_version": "gridwm.exam.v1",
        "task": "B",
        "image": image_rel,
        "answer_json": answer,
        "answer": json.dumps(answer, ensure_ascii=False),
        "_variant": variant,
        "_slot": "extra",
        "_source_frame": str(source_frame),
    }
    return Candidate(
        env=env,
        group_id=group_id,
        variant=variant,
        step=step,
        image_hash=hash_image(source_frame),
        image_name=image_name,
        image_rel=image_rel,
        source_frame=str(source_frame),
        info_score=compute_info_score(state),
        visible_object_count=len(objects_img),
        front_type=front_type,
        carrying=world_agent.get("carrying") is not None,
        answer_signature=answer_signature(answer),
        row=row,
    )


def build_candidate_pool(
    triplets: Dict[str, Dict[Tuple[str, str], dict]],
    anchors: Dict[str, Dict[str, AnchorRow]],
    raw_root: Path,
) -> Dict[str, Dict[str, List[Candidate]]]:
    pool: Dict[str, Dict[str, List[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for env in ENVS:
        for (group_id, variant), triplet in triplets[env].items():
            if variant not in EXPANSION_VARIANTS:
                continue
            anchor = anchors[env].get(group_id)
            if anchor is None:
                continue
            for step in range(len(triplet.get("state_seq", []))):
                if variant == "full" and step == anchor.step:
                    continue
                pool[env][group_id].append(build_candidate(env, group_id, variant, triplet, step, raw_root))
    return pool


def candidate_sort_key(candidate: Candidate, anchor: AnchorRow) -> Tuple[int, int, int, int, int, int]:
    return (
        candidate.info_score,
        1 if candidate.variant == "nocue" else 0,
        1 if candidate.answer_signature != answer_signature(anchor.row["answer_json"]) else 0,
        candidate.visible_object_count,
        abs(candidate.step - anchor.step),
        -candidate.step,
    )


def select_extras(
    anchors: Dict[str, Dict[str, AnchorRow]],
    pool: Dict[str, Dict[str, List[Candidate]]],
    anchor_root: Path,
) -> Dict[str, Dict[str, Candidate]]:
    selected: Dict[str, Dict[str, Candidate]] = defaultdict(dict)
    for env in ENVS:
        used_hashes = set()
        for anchor in anchors[env].values():
            anchor_img = anchor_root / anchor.image_rel
            used_hashes.add(hash_image(anchor_img))

        pending = set(anchors[env].keys())
        while pending:
            availability = []
            for group_id in sorted(pending):
                candidates = [
                    c for c in pool[env][group_id]
                    if c.image_hash not in used_hashes
                ]
                availability.append((len(candidates), group_id, candidates))
            availability.sort(key=lambda x: (x[0], x[1]))
            count, group_id, candidates = availability[0]
            if count == 0:
                raise RuntimeError(f"{env}/{group_id}: no unique extra candidate left")

            anchor = anchors[env][group_id]
            best = sorted(
                candidates,
                key=lambda c: candidate_sort_key(c, anchor),
                reverse=True,
            )[0]
            selected[env][group_id] = best
            used_hashes.add(best.image_hash)
            pending.remove(group_id)
    return selected


def build_doorkey_reanchor_pool(triplets: Dict[str, Dict[Tuple[str, str], dict]], raw_root: Path) -> Dict[str, List[Candidate]]:
    env = "doorkey"
    pool: Dict[str, List[Candidate]] = defaultdict(list)
    for (group_id, variant), triplet in triplets[env].items():
        if variant not in {"full", "nocue"}:
            continue
        for step in range(len(triplet.get("state_seq", []))):
            pool[group_id].append(build_candidate(env, group_id, variant, triplet, step, raw_root))
    return pool


def select_doorkey_pairs(pool: Dict[str, List[Candidate]]) -> Dict[str, Tuple[Candidate, Candidate]]:
    cand_pairs: Dict[str, List[Tuple[Tuple[int, int, int], Candidate, Candidate]]] = {}
    for gid, cands in pool.items():
        fulls = [c for c in cands if c.variant == "full"]
        nocues = [c for c in cands if c.variant == "nocue"]
        pairs = []
        for full in fulls:
            for nocue in nocues:
                if full.image_hash == nocue.image_hash:
                    continue
                score = (
                    full.info_score + nocue.info_score,
                    full.visible_object_count + nocue.visible_object_count,
                    1 if answer_signature(full.row["answer_json"]) != answer_signature(nocue.row["answer_json"]) else 0,
                )
                pairs.append((score, full, nocue))
        pairs.sort(key=lambda x: x[0], reverse=True)
        cand_pairs[gid] = pairs

    order = sorted(cand_pairs, key=lambda gid: len(cand_pairs[gid]))
    selected: Dict[str, Tuple[Candidate, Candidate]] = {}
    used_hashes = set()

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        gid = order[i]
        for score, full, nocue in cand_pairs[gid][:800]:
            if full.image_hash in used_hashes or nocue.image_hash in used_hashes:
                continue
            used_hashes.add(full.image_hash)
            used_hashes.add(nocue.image_hash)
            selected[gid] = (full, nocue)
            if backtrack(i + 1):
                return True
            used_hashes.remove(full.image_hash)
            used_hashes.remove(nocue.image_hash)
            del selected[gid]
        return False

    ok = backtrack(0)
    if not ok:
        raise RuntimeError("Failed to find strict-unique doorkey re-anchor solution")
    return selected


def convert_full_candidate_to_anchor(candidate: Candidate) -> dict:
    row = dict(candidate.row)
    row["uid"] = f"B.doorkey.{candidate.group_id}.t{candidate.step}"
    row["exam_id"] = row["uid"]
    row["image"] = f"images/taskB/{candidate.group_id}_t{candidate.step}.png"
    row["_slot"] = "anchor"
    row["_variant"] = "full"
    row["_reanchor"] = True
    return row


def convert_nocue_candidate_to_extra(candidate: Candidate) -> dict:
    row = dict(candidate.row)
    row["_slot"] = "extra"
    row["_variant"] = "nocue"
    row["_reanchor"] = True
    return row
