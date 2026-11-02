"""Shared MiniGrid trajectory parsing, state queries and storyboard helpers.

Task builders construct transition choices, visible-state questions and
trajectory judgments from these simulator record utilities."""




import json


import hashlib








from dataclasses import dataclass


from pathlib import Path


from typing import Any, List, Tuple




import numpy as np


from PIL import Image, ImageEnhance


import sys as _sys


_scripts_dir = Path(__file__).resolve().parent


_repo_root = _scripts_dir.parent.parent


if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))


if str(_scripts_dir) not in _sys.path:
    _sys.path.insert(0, str(_scripts_dir))




PRIVILEGED_FIELDS = {
    'success', 'terminated', 'truncated', 'reward', 'done',
    'fail_reason', 'mss_seq', 'vis_target',
    'event_coverage', 'camera_fov', 'info', 'done_reason'
}


@dataclass
class CleanInferenceState:
    """
    PHYSICAL ISOLATION: This is the ONLY allowed data class for prompt building.

    All privileged fields have been stripped at the boundary between
    data generators and exam builders. This ensures no implicit leakage
    can occur through control flow or indirect access.
    """
    task: str
    group_id: str
    variant: str
    env_id: str
    seed: int
    mission: str
    actions_id: List[int]
    actions_text: List[str]
    frames: List[str]
    # NOTE: success/reward/terminated intentionally EXCLUDED
    # NOTE: state_seq EXCLUDED from prompt-facing interface

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_steps(self) -> int:
        return len(self.actions_id)


def create_clean_state(rec: dict) -> CleanInferenceState:
    """
    FACTORY: Creates CleanInferenceState from raw generator output.

    This is the ONLY entry point from generators to exam builder.
    All privileged fields are physically stripped here.
    """
    # Check for privileged fields and warn
    privileged_found = set(rec.keys()) & PRIVILEGED_FIELDS
    if privileged_found:
        print(f"[SECURITY] Stripping privileged fields: {privileged_found} from {rec.get('group_id', 'unknown')}")

    return CleanInferenceState(
        task=str(rec.get("task", "unknown")),
        group_id=str(rec.get("group_id", "unknown")),
        variant=str(rec.get("variant", "unknown")),
        env_id=str(rec.get("env_id", "")),
        seed=int(rec.get("seed", -1)),
        mission=str(rec.get("mission", "")),
        actions_id=list(rec.get("actions_id", [])),
        actions_text=list(rec.get("actions_text", [])),
        frames=list(rec.get("frames", []))
        # Intentionally EXCLUDED: success, reward, terminated, state_seq, etc.
    )


@dataclass
class Traj:
    """
    Internal use ONLY - contains privileged fields for answer computation.

    CRITICAL: This class is used for computing correct answers (state_seq),
    but the prompt builder must use CleanInferenceState to ensure no
    privileged information leaks into prompts.
    """
    task: str
    group_id: str
    variant: str
    env_id: str
    seed: int
    mission: str
    actions_id: List[int]
    actions_text: List[str]
    frames: List[str]
    state_seq: List[dict]  # Internal use only - for answer computation
    success: bool  # Internal use only - NOT for prompts

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_steps(self) -> int:
        return len(self.actions_id)


def parse_traj(rec: dict) -> Traj:
    """
    Parse trajectory records for answer computation.

    For prompt building, use create_clean_state() instead.
    """
    return Traj(
        task=str(rec.get("task", "unknown")),
        group_id=str(rec.get("group_id", "unknown")),
        variant=str(rec.get("variant", "unknown")),
        env_id=str(rec.get("env_id", "")),
        seed=int(rec.get("seed", -1)),
        mission=str(rec.get("mission", "")),
        actions_id=list(rec.get("actions_id", [])),
        actions_text=list(rec.get("actions_text", [])),
        frames=list(rec.get("frames", [])),
        state_seq=list(rec.get("state_seq", [])),
        success=bool(rec.get("success", False))
        # NOTE: reward/terminated intentionally NOT stored
    )


def _stable_seed(*parts: str) -> int:
    payload = "::".join(parts).encode("utf-8")
    digest = hashlib.md5(payload).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _apply_visual_variant(img: Image.Image, visual: str, seed: int,
                          profile: str = 'default', seed_material: str = '') -> Image.Image:
    if profile == 'matched-probes':
        from utils.minigrid_visual import matched_probe_visual
        return matched_probe_visual(img, visual, seed_material)
    if profile != 'default':
        raise ValueError('Unknown Task C visual profile: ' + profile)
    if visual == "clean":
        return img
    if visual == "noisy":
        rng = np.random.RandomState(seed)
        arr = np.asarray(img).astype(np.float32)
        sigma = 8.0
        noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    if visual == "style":
        rng = np.random.RandomState(seed)
        out = img
        # Mild, deterministic color/contrast/brightness adjustments.
        out = ImageEnhance.Color(out).enhance(1.05 + 0.10 * rng.rand())
        out = ImageEnhance.Contrast(out).enhance(1.05 + 0.10 * rng.rand())
        out = ImageEnhance.Brightness(out).enhance(0.98 + 0.08 * rng.rand())
        out = ImageEnhance.Sharpness(out).enhance(1.00 + 0.15 * rng.rand())
        return out
    return img


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists(): return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(obj, f, indent=2)


def stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def load_frame(task_dir: Path, rel_path: str) -> Image.Image:
    p = task_dir / rel_path
    if not p.exists():
        raise FileNotFoundError(f"Missing frame: {p}")
    return Image.open(p).convert("RGB")


def image_fingerprint(img: Image.Image) -> str:
    """Stable image identity used to guard against duplicate Task A candidates."""
    return hashlib.md5(img.tobytes()).hexdigest()


ACT_PICKUP, ACT_TOGGLE, ACT_DROP = 3, 5, 4


def state_signature(st: dict) -> Tuple:
    agent = st.get("agent", {})
    objs = st.get("objects", [])

    apos = tuple(agent.get("pos", [])) if agent.get("pos") else None
    # Fix: Sort dictionary items for deterministic signature
    carry_dict = agent.get("carrying")
    carry = tuple(sorted(carry_dict.items())) if carry_dict else None

    front = st.get("front_cell", {})

    obj_sigs = []
    for o in objs:
        op = tuple(o.get("pos", [])) if o.get("pos") else None
        obj_sigs.append((o.get("type"), o.get("color"), o.get("state"), op))

    return (
        apos, agent.get("dir"), carry,
        front.get("type"), front.get("state"),
        tuple(sorted(obj_sigs))
    )


def compute_neg_score(st_correct: dict, st_neg: dict, action_id: int) -> float:
    """
    Compute negative sample quality score.

    Lower score = more similar = harder negative.

    Key insight:
    - Agent position SHOULD match for non-move actions
    - Only the semantic effect of the action should differ
    """
    agent_correct = st_correct.get("agent", {})
    agent_neg = st_neg.get("agent", {})

    pos_correct = tuple(agent_correct.get("pos", [])) if agent_correct.get("pos") else None
    pos_neg = tuple(agent_neg.get("pos", [])) if agent_neg.get("pos") else None

    # Position penalty: heavily penalize if position differs for non-move actions
    if action_id not in (0, 1, 2):  # Not move actions
        if pos_correct != pos_neg:
            return 100.0  # Invalid negative - wrong agent position

    # Same action = position should be similar
    pos_penalty = 0 if pos_correct == pos_neg else 10.0

    # Direction should match
    dir_penalty = 0 if agent_correct.get("dir") == agent_neg.get("dir") else 2.0

    # Front cell: for toggle actions, can differ; for others should be similar
    front_correct = st_correct.get("front_cell", {})
    front_neg = st_neg.get("front_cell", {})
    front_penalty = 0 if front_correct == front_neg else 1.0

    # Object state: only the interacted object should differ
    obj_correct = {tuple(o.get("pos", [])): o for o in st_correct.get("objects", [])}
    obj_neg = {tuple(o.get("pos", [])): o for o in st_neg.get("objects", [])}

    # Find matching positions and compare
    common_pos = set(obj_correct.keys()) & set(obj_neg.keys())
    obj_penalty = 0
    for pos in common_pos:
        if obj_correct[pos] != obj_neg[pos]:
            obj_penalty += 0.5  # Minor penalty for object state diff

    # Extra positions in neg (shouldn't exist for good negative)
    obj_penalty += len(set(obj_neg.keys()) - common_pos) * 0.5
    # Missing positions in neg
    obj_penalty += len(set(obj_correct.keys()) - common_pos) * 0.5

    return pos_penalty + dir_penalty + front_penalty + obj_penalty


def canon_taskb_answer(st: dict) -> dict:
    objs = []
    for o in st.get("objects", []):
        objs.append({
            "type": o.get("type"),
            "pos": list(o.get("pos", [0,0])),
            "color": o.get("color"),
            "state": o.get("state")
        })
    objs.sort(key=lambda x: (str(x["type"]), x["pos"][0], x["pos"][1]))

    ag = st.get("agent", {})
    fc = st.get("front_cell", {})
    return {
        "agent": {"pos": list(ag.get("pos", [0,0])), "dir": ag.get("dir", 0), "carrying": ag.get("carrying")},
        "front_cell": {"pos": list(fc.get("pos", [0,0])), "type": fc.get("type"), "state": fc.get("state")},
        "objects": objs
    }


def visible_gt(st: dict) -> List[str]:
    v = st.get("visible_types_gt", None)
    if v is None: v = st.get("visible_types", [])
    return list(v) if isinstance(v, list) else []


def find_sight(tr: Traj, obj: str, start: int = 0) -> int:
    for t in range(max(0, start), len(tr.state_seq)):
        if obj in visible_gt(tr.state_seq[t]): return t
    return 0


def find_action(tr: Traj, aid: int, start: int = 0) -> int:
    for t in range(max(0, start), tr.n_steps):
        if tr.actions_id[t] == aid: return t
    return 0


def sb_doorkey(tr: Traj, K: int) -> List[int]:
    n = tr.n_frames
    t_key = find_sight(tr, "key")
    t_pick = find_action(tr, ACT_PICKUP)
    t_door = find_sight(tr, "door", t_pick)
    t_tog = find_action(tr, ACT_TOGGLE, t_door)
    return sorted(list(set([0, t_key, t_pick, min(n-1, t_pick+1), t_door, t_tog, min(n-1, t_tog+1), n-1])))


def sb_memory(tr: Traj, K: int) -> List[int]:
    n = tr.n_frames
    t_cue = max(find_sight(tr, "ball"), find_sight(tr, "key"))
    mid = np.linspace(t_cue, n-1, max(2, K-2)).astype(int).tolist()
    return sorted(list(set([0, t_cue] + mid + [n-1])))


def sb_lavagap(tr: Traj, K: int) -> List[int]:
    n = tr.n_frames
    t_lava = find_sight(tr, "lava")
    t_goal = find_sight(tr, "goal", t_lava)
    mid = int((t_lava + t_goal)/2)
    return sorted(list(set([0, t_lava, mid, t_goal, n-1])))


def sb_multiroom(tr: Traj, K: int) -> List[int]:
    n = tr.n_frames
    t_d1 = find_sight(tr, "door")
    t_tog1 = find_action(tr, ACT_TOGGLE)
    t_goal = find_sight(tr, "goal", t_tog1)
    return sorted(list(set([0, t_d1, t_tog1, min(n-1, t_tog1+1), t_goal, n-1])))


def sb_keycorridor(tr: Traj, K: int) -> List[int]:
    return sb_doorkey(tr, K)


def sb_redblue(tr: Traj, K: int) -> List[int]:
    n = tr.n_frames
    t_d1 = find_sight(tr, "door")
    t_tog1 = find_action(tr, ACT_TOGGLE)
    t_d2 = find_sight(tr, "door", t_tog1)
    t_tog2 = find_action(tr, ACT_TOGGLE, t_d2)
    return sorted(list(set([0, t_d1, t_tog1, min(n-1, t_tog1+1), t_d2, t_tog2, n-1])))


def sb_default(tr: Traj, K: int) -> List[int]:
    return np.linspace(0, tr.n_frames-1, K).astype(int).tolist()


def sb_dispatch(tr: Traj, K: int) -> List[int]:
    task = tr.task.lower()
    if "doorkey" in task: raw = sb_doorkey(tr, K)
    elif "memory" in task: raw = sb_memory(tr, K)
    elif "lavagap" in task: raw = sb_lavagap(tr, K)
    elif "multiroom" in task: raw = sb_multiroom(tr, K)
    elif "keycorridor" in task: raw = sb_keycorridor(tr, K)
    elif "redblue" in task: raw = sb_redblue(tr, K)
    else: raw = sb_default(tr, K)

    if len(raw) < 2: raw = [0, tr.n_frames-1]
    final = []
    indices = np.linspace(0, len(raw)-1, K)
    for i in indices: final.append(raw[int(i)])

    return [int(max(0, min(tr.n_frames-1, x))) for x in sorted(final)]
