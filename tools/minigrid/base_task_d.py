"""Reference/query MiniGrid Task D with controlled framing and visual conditions."""
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.minigrid_panels import windows_font_paths


def build_final_task_d(records: list[dict], output: Path) -> list[dict]:
    """Render continuous source trajectories and the four reference/query probes."""
    from utils.minigrid_continuous_storyboard import annotate_frame, make_montage, TASK_D_PROMPT
    source_rows = []
    for record in sorted(records, key=lambda row: (row['task'], row['group_id'], row['variant'])):
        env, gid, variant = record['task'], record['group_id'], record['variant']
        frames = []
        for step, path in enumerate(record['frames']):
            with Image.open(path) as image:
                frames.append(annotate_frame(image.convert('RGB'), step))
        montage, layout = make_montage(frames)
        uid = f'D.{env}.{gid}.{variant}.orig.clean'
        relative = f'images/taskD/{env}/{gid}/{variant}_orig_clean.png'
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        montage.save(path, 'PNG')
        source_rows.append(dict(exam_id=uid, uid=uid, schema_version=EXAM_SCHEMA_VERSION,
                                task='D', env_task=env, env_id=env, group_id=gid, variant=variant,
                                temporal='orig', visual='clean', image=relative,
                                prompt=TASK_D_PROMPT, answer='Success' if record['success'] else 'Fail',
                                label=int(record['success']), n_frames=layout['n_frames'], layout=layout,
                                source_group_id=gid, source_variant=variant, source_env=env))
    rows, _, _ = build_task_d_r_rows(source_rows, output, output, 2, 2, 2)
    for row in rows:
        row['source_exam_root'] = str(output)
    return rows

BASE_PROMPT = (
    "Below are two storyboard trajectories from the same MiniGrid environment.\n\n"
    "Reference trajectory:\n"
    "This trajectory is a known successful example of completing the task in this environment.\n\n"
    "Query trajectory:\n"
    "Determine whether this trajectory also completes the same task goal as the reference trajectory.\n\n"
    "Answer with ONLY: Success or Fail."
)


FRAMING_PREAMBLES = {
    "neu": "",
    "pos": "The query trajectory seems to handle the task competently in the following example. ",
    "neg": "The query trajectory seems to struggle with the task in the following example. ",
}


FRAMING_TRANSLATIONS_ZH = {
    "neu": "无额外语气暗示。",
    "pos": "该题对 query 轨迹加入正向语气暗示：它看起来较为熟练地完成了任务。",
    "neg": "该题对 query 轨迹加入负向语气暗示：它看起来在完成任务时比较吃力。",
}


VISUAL_TRANSLATIONS_ZH = {
    "clean": "原始 clean 图像。",
    "style": "style 图像：使用 cartoon-edge 风格化处理。",
}


EXAM_SCHEMA_VERSION = "gridwm.exam.v1"


LABEL_BAR_H = 60


GAP_H = 36


MARGIN = 24


FONT_PATHS = [
    *windows_font_paths(('arial.ttf',)),
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for font_path in FONT_PATHS:
        p = Path(font_path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def add_label_bar(img: Image.Image, text: str, fill: Tuple[int, int, int]) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + LABEL_BAR_H), (255, 255, 255))
    out.paste(img, (0, LABEL_BAR_H))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, img.width, LABEL_BAR_H], fill=fill)
    font = load_font(28)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    draw.text(
        ((img.width - tw) / 2, (LABEL_BAR_H - th) / 2 - 2),
        text,
        fill=(255, 255, 255),
        font=font,
    )
    return out


def apply_style_cartoon_medium(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    out = cv2.edgePreservingFilter(arr, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
    return Image.fromarray(out)


def compose_reference_query_image(
    reference_img: Path,
    query_img: Path,
    out_img: Path,
    visual: str,
) -> Dict[str, int | str]:
    ref = Image.open(reference_img).convert("RGB")
    qry = Image.open(query_img).convert("RGB")

    ref_labeled = add_label_bar(ref, "Reference (Known Success)", (34, 97, 168))
    qry_labeled = add_label_bar(qry, "Query", (97, 97, 97))

    width = max(ref_labeled.width, qry_labeled.width) + 2 * MARGIN
    height = ref_labeled.height + qry_labeled.height + GAP_H + 2 * MARGIN
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    ref_x = (width - ref_labeled.width) // 2
    qry_x = (width - qry_labeled.width) // 2
    ref_y = MARGIN
    qry_y = ref_y + ref_labeled.height + GAP_H
    canvas.paste(ref_labeled, (ref_x, ref_y))
    canvas.paste(qry_labeled, (qry_x, qry_y))

    if visual == "style":
        canvas = apply_style_cartoon_medium(canvas)
    elif visual != "clean":
        raise ValueError(f"Unsupported visual variant: {visual}")

    out_img.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_img, "PNG")

    return {
        "reference_width": ref.width,
        "reference_height": ref.height,
        "query_width": qry.width,
        "query_height": qry.height,
        "combined_width": canvas.width,
        "combined_height": canvas.height,
        "visual": visual,
    }


def choose_reference_and_queries(
    source_rows: List[Dict[str, Any]],
    full_per_env: int,
    nocue_per_env: int,
    cf_per_env: int,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    by_env: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if row.get("temporal") == "orig" and row.get("visual") == "clean":
            by_env[row["env_id"]].append(row)

    references: Dict[str, Dict[str, Any]] = {}
    queries: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {"envs": {}}

    for env in sorted(by_env):
        rows = sorted(by_env[env], key=lambda r: r["uid"])
        full_rows = [r for r in rows if r["variant"] == "full" and r["answer"] == "Success"]
        nocue_rows = [r for r in rows if r["variant"] == "nocue" and r["answer"] == "Success"]
        cf_rows = [r for r in rows if r["variant"] == "cf" and r["answer"] == "Fail"]
        if not full_rows:
            raise ValueError(f"No successful full rows available for env={env}")

        reference = full_rows[0]
        references[env] = reference

        selected_full = [r for r in full_rows[1:] if r["source_group_id"] != reference["source_group_id"]][:full_per_env]
        selected_nocue = [r for r in nocue_rows if r["source_group_id"] != reference["source_group_id"]][:nocue_per_env]
        selected_cf = [r for r in cf_rows if r["source_group_id"] != reference["source_group_id"]][:cf_per_env]

        if len(selected_full) < full_per_env or len(selected_nocue) < nocue_per_env or len(selected_cf) < cf_per_env:
            raise ValueError(
                f"Not enough rows for env={env}: "
                f"full={len(selected_full)}/{full_per_env}, "
                f"nocue={len(selected_nocue)}/{nocue_per_env}, "
                f"cf={len(selected_cf)}/{cf_per_env}"
            )

        queries.extend(selected_full)
        queries.extend(selected_nocue)
        queries.extend(selected_cf)
        stats["envs"][env] = {
            "reference_uid": reference["uid"],
            "reference_group": reference["source_group_id"],
            "selected_full": [r["uid"] for r in selected_full],
            "selected_nocue": [r["uid"] for r in selected_nocue],
            "selected_cf": [r["uid"] for r in selected_cf],
        }

    return references, queries, stats


def build_task_d_r_rows(
    source_rows: List[Dict[str, Any]],
    source_root: Path,
    out_dir: Path,
    full_per_env: int,
    nocue_per_env: int,
    cf_per_env: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, str]]:
    references, queries, stats = choose_reference_and_queries(
        source_rows, full_per_env=full_per_env, nocue_per_env=nocue_per_env, cf_per_env=cf_per_env
    )

    rows: List[Dict[str, Any]] = []
    images_root = out_dir / "images" / "taskDR"
    answer_counts: Counter[str] = Counter()
    env_counts: Counter[str] = Counter()
    visual_counts: Counter[str] = Counter()
    framing_counts: Counter[str] = Counter()
    prompt_examples: Dict[str, str] = {}

    for query in sorted(queries, key=lambda r: r["uid"]):
        env = query["env_id"]
        reference = references[env]
        query_img = source_root / query["image"]
        reference_img = source_root / reference["image"]

        for visual, framing in [
            ("clean", "neu"),
            ("clean", "pos"),
            ("clean", "neg"),
            ("style", "neu"),
        ]:
            rel_img = (
                Path(env)
                / reference["source_group_id"]
                / f"{query['source_group_id']}__{query['variant']}_{visual}_{framing}_ref_query.png"
            )
            out_img = images_root / rel_img
            layout = compose_reference_query_image(reference_img, query_img, out_img, visual=visual)

            uid = f"DR.{env}.{reference['source_group_id']}.{query['source_group_id']}.{query['variant']}.orig.{visual}.{framing}"
            prompt = FRAMING_PREAMBLES[framing] + BASE_PROMPT

            row = {
                "exam_id": uid,
                "uid": uid,
                "schema_version": EXAM_SCHEMA_VERSION,
                "task": "D",
                "audit_task": "D-R",
                "audit_design": "reference_aided_taskd_expanded_v2",
                "task_variant": "reference_aided",
                "env_task": env,
                "env_id": env,
                "group_id": query["source_group_id"],
                "variant": query["variant"],
                "temporal": "orig",
                "visual": visual,
                "framing": framing,
                "framing_translation_zh": FRAMING_TRANSLATIONS_ZH[framing],
                "visual_translation_zh": VISUAL_TRANSLATIONS_ZH[visual],
                "image": str(Path("images") / "taskDR" / rel_img),
                "prompt": prompt,
                "answer": query["answer"],
                "label": query["label"],
                "reference_uid": reference["uid"],
                "reference_group_id": reference["source_group_id"],
                "reference_variant": reference["variant"],
                "reference_answer": reference["answer"],
                "query_uid": query["uid"],
                "query_variant": query["variant"],
                "source_taskd_exam": "task_d_exam.jsonl",
                "source_env": env,
                "layout": layout,
            }
            rows.append(row)
            env_counts[env] += 1
            answer_counts[row["answer"]] += 1
            visual_counts[visual] += 1
            framing_counts[framing] += 1
            prompt_examples.setdefault(f"{visual}_{framing}", prompt)

    stats["rows_total"] = len(rows)
    stats["env_counts"] = dict(env_counts)
    stats["answer_counts"] = dict(answer_counts)
    stats["visual_counts"] = dict(visual_counts)
    stats["framing_counts"] = dict(framing_counts)
    stats["queries_per_env"] = {
        "full_success": full_per_env,
        "nocue_success": nocue_per_env,
        "cf_fail": cf_per_env,
    }
    return rows, stats, prompt_examples
