#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are judging an embodied agent from time-ordered
first-person observations. Use only the supplied images and action text.
Return exactly the requested label and no explanation."""


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Habitat A/C/D Evaluation · 550 Questions</title>
  <style>
    :root {
      --bg:#f3f1eb; --panel:#fffdf9; --ink:#202826; --muted:#66716e;
      --line:#d8d5cc; --accent:#116d63; --soft:#dcece8; --dark:#172522;
    }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:var(--bg);
      font:15px/1.5 Inter,system-ui,sans-serif; }
    header { position:sticky; top:0; z-index:10; padding:12px 18px;
      background:rgba(243,241,235,.96); border-bottom:1px solid var(--line);
      backdrop-filter:blur(12px); }
    .bar, main { max-width:1480px; margin:auto; }
    .bar { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
    h1 { margin:0 auto 0 0; font-size:19px; }
    button,select,input { min-height:38px; border:1px solid var(--line);
      border-radius:9px; background:var(--panel); color:var(--ink);
      padding:7px 11px; font:inherit; }
    button { cursor:pointer; }
    button.primary { color:white; background:var(--accent);
      border-color:var(--accent); }
    main { padding:20px; }
    .status { display:flex; align-items:center; gap:10px; margin-bottom:13px;
      color:var(--muted); flex-wrap:wrap; }
    .badge { display:inline-flex; padding:4px 9px; border-radius:999px;
      background:var(--soft); color:#0b554d; font-size:12px; font-weight:750; }
    .card { overflow:hidden; background:var(--panel); border:1px solid var(--line);
      border-radius:15px; box-shadow:0 12px 32px rgba(30,40,38,.08); }
    .head { display:flex; gap:10px; align-items:center; padding:16px 19px;
      border-bottom:1px solid var(--line); }
    .head h2 { margin:0; font-size:18px; }
    .uid { margin-left:auto; color:var(--muted); font-family:monospace; }
    .prompt { margin:18px 20px 0; padding:16px 18px; border-radius:11px;
      white-space:pre-wrap; color:#eff8f5; background:var(--dark);
      font:14px/1.55 ui-monospace,monospace; }
    .system { color:#3d4946; background:#e8e5dd; }
    .prompt-label { display:block; margin-bottom:7px; color:#8fd7ca;
      font:700 11px/1.2 Inter,sans-serif; letter-spacing:.08em; }
    .system .prompt-label { color:#66716e; }
    .content { padding:20px; }
    h3 { margin:22px 0 10px; font-size:15px; }
    h3:first-child { margin-top:0; }
    .single { max-width:512px; }
    .grid4 { display:grid; grid-template-columns:repeat(4,minmax(170px,1fr));
      gap:12px; }
    .story { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr));
      gap:12px; }
    .dense { display:grid; grid-template-columns:repeat(auto-fill,minmax(135px,1fr));
      gap:8px; }
    figure { margin:0; overflow:hidden; border:1px solid var(--line);
      border-radius:10px; background:#ebe8e0; }
    figure img { display:block; width:100%; aspect-ratio:1; object-fit:cover;
      cursor:zoom-in; }
    figcaption { display:flex; justify-content:space-between; min-height:34px;
      padding:7px 9px; color:var(--muted); background:var(--panel);
      font-size:12px; }
    details { margin:12px 0; padding:9px 11px; background:#faf8f3;
      border:1px solid var(--line); border-radius:9px; }
    summary { cursor:pointer; font-weight:700; }
    .actions { overflow-wrap:anywhere; font-family:monospace; }
    .segment { margin-top:7px; padding:7px 9px; border-radius:7px;
      background:#e9f0ed; color:#31504a; font:12px/1.4 monospace; }
    .role { margin-top:25px; padding-top:18px; border-top:2px solid var(--line); }
    .answer { margin-top:24px; padding:16px; border:1px solid var(--line);
      border-radius:11px; background:#f6f3ed; }
    .choices { display:flex; gap:18px; margin-top:9px; flex-wrap:wrap; }
    .choices label { font-weight:750; cursor:pointer; }
    .save-state { color:var(--accent); font-weight:700; }
    dialog { max-width:96vw; max-height:96vh; padding:0; border:0;
      border-radius:10px; background:#111; }
    dialog::backdrop { background:rgba(0,0,0,.82); }
    dialog img { display:block; max-width:94vw; max-height:92vh; }
    .error { padding:30px; color:#9b2727; white-space:pre-wrap; }
    @media(max-width:900px) {
      .grid4 { grid-template-columns:repeat(2,1fr); }
      .uid { display:none; } main { padding:10px; } header { padding:9px; }
    }
  </style>
</head>
<body>
  <header><div class="bar">
    <h1>Habitat A/C/D Evaluation</h1>
    <span class="badge">ENGLISH PROMPTS</span>
    <select id="task">
      <option value="A">Task A · 150</option>
      <option value="C">Task C · 200</option>
      <option value="D">Task D-R · 200</option>
    </select>
    <button id="prev">← Previous</button>
    <input id="index" type="number" min="1" value="1" style="width:88px">
    <span id="total">/ …</span>
    <button id="next">Next →</button>
    <button id="random">Random</button>
    <button id="export" class="primary">Export responses.csv</button>
  </div></header>
  <main>
    <div class="status">
      <span id="progress">Loading exam…</span>
      <span id="saved" class="save-state"></span>
    </div>
    <section id="question" class="card"></section>
  </main>
  <dialog id="lightbox"><img alt="Enlarged observation"></dialog>
  <script>
    const SYSTEM_PROMPT = __SYSTEM_PROMPT__;
    const STORAGE_KEY = "habitat_exam_en_550_v1";
    const state = {questions:[], task:"A", index:0, responses:{}};
    const qnode = document.querySelector("#question");
    const taskSelect = document.querySelector("#task");
    const esc = value => String(value).replace(/[&<>"']/g, c =>
      ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
    function card(path,label,extra="") {
      return `<figure><img src="${esc(path)}" alt="${esc(label)}" loading="lazy">
        <figcaption><span>${esc(label)}</span>${extra}</figcaption></figure>`;
    }
    function actions(sequence) {
      const labels={move_forward:"MOVE_FORWARD",turn_left:"TURN_LEFT",
        turn_right:"TURN_RIGHT",look_up:"LOOK_UP",look_down:"LOOK_DOWN",
        open:"OPEN",close:"CLOSE",stop:"STOP"};
      const runs=[];
      for(const item of sequence||[]) {
        if(runs.length && runs.at(-1).name===item) runs.at(-1).count++;
        else runs.push({name:item,count:1});
      }
      return runs.map(x => `${labels[x.name]||x.name}${x.count>1?` × ${x.count}`:""}`).join(" → ");
    }
    function prompt(row) {
      return `<div class="prompt system"><span class="prompt-label">SYSTEM PROMPT</span>${esc(SYSTEM_PROMPT)}</div>
        <div class="prompt"><span class="prompt-label">USER PROMPT</span>${esc(row.prompt)}</div>`;
    }
    function answer(row) {
      const labels=row.task==="A"?["A","B","C","D"]:["Success","Fail"];
      const selected=state.responses[row.question_id]||"";
      return `<div class="answer"><strong>Response</strong><div class="choices">
        ${labels.map(x=>`<label><input type="radio" name="response" value="${x}"
          ${selected===x?"checked":""}> ${x}</label>`).join("")}</div></div>`;
    }
    function head(row,title) {
      return `<div class="head"><span class="badge">Task ${row.task}</span>
        <h2>${esc(title)}</h2><span class="uid">${esc(row.question_id)}</span></div>`;
    }
    function renderA(row) {
      const opts=row.candidate_images.map((p,i)=>card(p,String.fromCharCode(65+i))).join("");
      qnode.innerHTML=head(row,"Next-observation prediction")+prompt(row)+
        `<div class="content"><h3>Current observation</h3>
        <div class="single">${card(row.current_image,"Current observation")}</div>
        <details><summary>Action sequence (${row.action_sequence.length} actions)</summary>
        <p class="actions">${esc(actions(row.action_sequence))}</p></details>
        <h3>Options A–D</h3><div class="grid4">${opts}</div>${answer(row)}</div>`;
    }
    function renderC(row) {
      const chunkMap=new Map((row.action_chunks||[]).map(x=>[x.to_frame_index,x]));
      const frames=row.keyframe_images.map((p,i)=>{
        const source=row.keyframe_indices[i], chunk=chunkMap.get(source);
        const segment=i&&chunk?`<div class="segment">Since previous frame:<br>
          ${esc(chunk.runs.map(x=>`${x.action.toUpperCase()}${x.count>1?` × ${x.count}`:""}`).join(" → "))}</div>`:"";
        return `<div>${card(p,`Storyboard frame ${i+1}`,`<span>source ${source}</span>`)}${segment}</div>`;
      }).join("");
      qnode.innerHTML=head(row,"Sparse trajectory judgment")+prompt(row)+
        `<div class="content"><h3>Goal View</h3><div class="single">
        ${card(row.goal_image,"Goal View")}</div>
        <details><summary>Complete action sequence (${row.action_sequence.length} actions)</summary>
        <p class="actions">${esc(actions(row.action_sequence))}</p></details>
        <h3>Time-ordered sparse storyboard</h3><div class="story">${frames}</div>
        ${answer(row)}</div>`;
    }
    function role(name,payload) {
      const frames=payload.images.map((p,i)=>card(p,`Frame ${i}`,`<span>${i?"t="+i:"initial"}</span>`)).join("");
      return `<section class="role"><h3>${name} trajectory</h3>
        <details><summary>Action sequence (${payload.action_sequence.length} actions)</summary>
        <p class="actions">${esc(actions(payload.action_sequence))}</p></details>
        <div class="dense">${frames}</div></section>`;
    }
    function renderD(row) {
      qnode.innerHTML=head(row,"Reference-conditioned trajectory judgment")+prompt(row)+
        `<div class="content">${role("Known-success reference",row.reference)}
        ${role("Query",row.query)}${answer(row)}</div>`;
    }
    function currentRows(){ return state.questions.filter(x=>x.task===state.task); }
    function save() {
      localStorage.setItem(STORAGE_KEY,JSON.stringify(state.responses));
      document.querySelector("#saved").textContent="Saved locally";
      setTimeout(()=>document.querySelector("#saved").textContent="",900);
    }
    function updateProgress() {
      const answered=Object.keys(state.responses).filter(k=>state.responses[k]).length;
      document.querySelector("#progress").textContent=
        `Answered ${answered} / ${state.questions.length} · responses are stored in this browser`;
    }
    function render() {
      const rows=currentRows();
      state.index=Math.max(0,Math.min(state.index,rows.length-1));
      const row=rows[state.index];
      document.querySelector("#index").value=state.index+1;
      document.querySelector("#index").max=rows.length;
      document.querySelector("#total").textContent=`/ ${rows.length}`;
      if(row.task==="A") renderA(row); else if(row.task==="C") renderC(row); else renderD(row);
      qnode.querySelectorAll('input[name="response"]').forEach(input=>
        input.addEventListener("change",event=>{
          state.responses[row.question_id]=event.target.value; save(); updateProgress();
        }));
      qnode.querySelectorAll("img").forEach(img=>img.addEventListener("click",()=>{
        const d=document.querySelector("#lightbox"); d.querySelector("img").src=img.src; d.showModal();
      }));
      window.scrollTo({top:0,behavior:"instant"}); updateProgress();
    }
    function exportCsv() {
      const rows=[["question_id","task","response"]];
      for(const q of state.questions) rows.push([q.question_id,q.task,state.responses[q.question_id]||""]);
      const csv=rows.map(row=>row.map(value=>`"${String(value).replaceAll('"','""')}"`).join(",")).join("\n");
      const link=document.createElement("a");
      link.href=URL.createObjectURL(new Blob([csv],{type:"text/csv;charset=utf-8"}));
      link.download="habitat_exam_en_550_responses.csv"; link.click();
      URL.revokeObjectURL(link.href);
    }
    async function load() {
      try {
        state.responses=JSON.parse(localStorage.getItem(STORAGE_KEY)||"{}");
        const response=await fetch("questions.json?v=1",{cache:"no-store"});
        if(!response.ok) throw new Error(`questions.json: HTTP ${response.status}`);
        state.questions=await response.json(); render();
      } catch(error) {
        qnode.innerHTML=`<div class="error">Open this page through the supplied HTTP URL.\n\n${esc(error)}</div>`;
      }
    }
    taskSelect.addEventListener("change",e=>{state.task=e.target.value;state.index=0;render();});
    document.querySelector("#prev").addEventListener("click",()=>{state.index--;render();});
    document.querySelector("#next").addEventListener("click",()=>{state.index++;render();});
    document.querySelector("#random").addEventListener("click",()=>{
      state.index=Math.floor(Math.random()*currentRows().length);render();
    });
    document.querySelector("#index").addEventListener("change",e=>{state.index=Number(e.target.value)-1;render();});
    document.querySelector("#export").addEventListener("click",exportCsv);
    document.querySelector("#lightbox").addEventListener("click",e=>e.currentTarget.close());
    document.addEventListener("keydown",e=>{
      if(e.key==="ArrowLeft"){state.index--;render();}
      if(e.key==="ArrowRight"){state.index++;render();}
    });
    load();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an English-only 150/200/200 Habitat exam."
    )
    parser.add_argument(
        "--task-a-root",
        type=Path,
        default=Path("Habitat/outputs/hssd-template-candidate-pool"),
    )
    parser.add_argument(
        "--interaction-root",
        type=Path,
        default=Path("Habitat/outputs/hssd-taskc-articulated-full"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Habitat/outputs/hssd-exam-en-550"),
    )
    parser.add_argument("--task-a-count", type=int, default=150)
    parser.add_argument("--task-c-count", type=int, default=200)
    parser.add_argument("--task-d-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


from utils.habitat_records import read_jsonl


def prefixed(prefix: str, path: str) -> str:
    return f"{prefix}/{path}"


def stratified_task_a(
    rows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"Task A needs {count} rows, found {len(rows)}")
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        buckets[row["action_sequence_label"]][row["answer_label"]].append(
            row
        )
    for action_buckets in buckets.values():
        for bucket in action_buckets.values():
            rng.shuffle(bucket)

    actions = sorted(buckets)
    available = {
        action: sum(len(bucket) for bucket in buckets[action].values())
        for action in actions
    }
    quotas = Counter()
    while sum(quotas.values()) < count:
        progressed = False
        for action in actions:
            if (
                quotas[action] < available[action]
                and sum(quotas.values()) < count
            ):
                quotas[action] += 1
                progressed = True
        if not progressed:
            break

    selected: list[dict[str, Any]] = []
    selected_by_action = Counter()
    selected_by_answer = Counter()
    tie_order = list("ABCD")
    rng.shuffle(tie_order)
    tie_rank = {label: index for index, label in enumerate(tie_order)}
    while len(selected) < count:
        progressed = False
        for action in actions:
            if selected_by_action[action] >= quotas[action]:
                continue
            labels = [
                label
                for label, bucket in buckets[action].items()
                if bucket
            ]
            if not labels:
                continue
            label = min(
                labels,
                key=lambda value: (
                    selected_by_answer[value],
                    tie_rank.get(value, 99),
                ),
            )
            selected.append(buckets[action][label].pop())
            selected_by_action[action] += 1
            selected_by_answer[label] += 1
            progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise RuntimeError("Could not fill the stratified Task A sample")
    rng.shuffle(selected)
    return selected


def task_c_prompt(row: dict[str, Any]) -> str:
    category = str(row.get("target", {}).get("category", "articulated object"))
    category_text = {
        "refrigerator": "refrigerator",
        "oven": "oven or microwave",
        "wardrobe": "wardrobe",
        "cabinet": "cabinet or storage unit",
        "drawer": "drawer unit",
    }.get(category, category.replace("_", " "))
    component_name = str(
        row.get("target", {}).get("component_id", "")
    ).lower()
    component = "drawer" if "drawer" in component_name else "door"
    return f"""Below is a sparse, time-ordered storyboard from a Habitat
trajectory. Frames are ordered from left to right and top to bottom.

Task goal:
Find and operate the target {category_text} depicted in the Goal View. On the
same target {component}, complete OPEN → CLOSE → STOP in that order.

Judge the physical object-state changes visible in the images, not action names
alone. The trajectory fails if it operates the wrong object or component, an
interaction has no effect, an event is missing or out of order, or the terminal
object state is incorrect.

Did the trajectory complete the task?

Answer with exactly one token: Success or Fail."""


def normalize_gold(value: str) -> str:
    return "Success" if str(value).lower() == "success" else "Fail"


def main() -> int:
    args = parse_args()
    task_a_root = args.task_a_root.resolve()
    interaction_root = args.interaction_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows_a = stratified_task_a(
        read_jsonl(task_a_root / "taska" / "task_a.jsonl"),
        args.task_a_count,
        rng,
    )
    rows_c_all = read_jsonl(interaction_root / "taskc" / "task_c.jsonl")
    rows_d_all = read_jsonl(interaction_root / "taskd" / "task_d.jsonl")
    if len(rows_c_all) < args.task_c_count:
        raise ValueError(
            f"Task C needs {args.task_c_count}, found {len(rows_c_all)}"
        )
    if len(rows_d_all) < args.task_d_count:
        raise ValueError(
            f"Task D needs {args.task_d_count}, found {len(rows_d_all)}"
        )
    rows_c = rows_c_all[:]
    rows_d = rows_d_all[:]
    rng.shuffle(rows_c)
    rng.shuffle(rows_d)
    rows_c = rows_c[: args.task_c_count]
    rows_d = rows_d[: args.task_d_count]

    questions: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    for index, row in enumerate(rows_a, start=1):
        question_id = f"A{index:03d}"
        questions.append(
            {
                "question_id": question_id,
                "task": "A",
                "source_uid": row["uid"],
                "prompt": row["prompt"],
                "current_image": prefixed(
                    "taska_source", row["current_image"]
                ),
                "candidate_images": [
                    prefixed("taska_source", path)
                    for path in row["candidate_images"]
                ],
                "action_sequence": row["action_sequence"],
            }
        )
        answers.append(
            {
                "question_id": question_id,
                "task": "A",
                "gold": row["answer_label"],
                "source_uid": row["uid"],
                "stratum": {
                    "action_sequence_label": row["action_sequence_label"],
                    "answer_label": row["answer_label"],
                },
            }
        )
    for index, row in enumerate(rows_c, start=1):
        question_id = f"C{index:03d}"
        questions.append(
            {
                "question_id": question_id,
                "task": "C",
                "source_uid": row["uid"],
                "prompt": task_c_prompt(row),
                "goal_image": prefixed(
                    "interaction_source", row["goal_image"]
                ),
                "keyframe_indices": row["keyframe_indices"],
                "keyframe_images": [
                    prefixed("interaction_source", path)
                    for path in row["keyframe_images"]
                ],
                "action_sequence": row["action_sequence"],
                "action_chunks": row["action_chunks"],
            }
        )
        answers.append(
            {
                "question_id": question_id,
                "task": "C",
                "gold": normalize_gold(row["gold"]),
                "source_uid": row["uid"],
                "condition": row["condition"],
            }
        )
    for index, row in enumerate(rows_d, start=1):
        question_id = f"D{index:03d}"
        questions.append(
            {
                "question_id": question_id,
                "task": "D",
                "task_variant": "D-R",
                "source_uid": row["uid"],
                "prompt": row["prompt"],
                "prompt_style": row["prompt_style"],
                "reference": {
                    "images": [
                        prefixed("interaction_source", path)
                        for path in row["reference"]["images"]
                    ],
                    "action_sequence": row["reference"][
                        "action_sequence"
                    ],
                },
                "query": {
                    "images": [
                        prefixed("interaction_source", path)
                        for path in row["query"]["images"]
                    ],
                    "action_sequence": row["query"]["action_sequence"],
                },
            }
        )
        answers.append(
            {
                "question_id": question_id,
                "task": "D",
                "gold": normalize_gold(row["gold"]),
                "source_uid": row["uid"],
                "condition": row["condition"],
                "prompt_style": row["prompt_style"],
            }
        )

    for link_name, target in (
        ("taska_source", task_a_root),
        ("interaction_source", interaction_root),
    ):
        link = output_dir / link_name
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Unexpected existing exam asset link: {link}")
        link.symlink_to(target, target_is_directory=True)

    (output_dir / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "answer_key.json").write_text(
        json.dumps(
            {
                "schema_version": "habitat-exam-answer-key-v1",
                "answers": answers,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "answer_sheet.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["question_id", "task", "response"])
        for row in questions:
            writer.writerow([row["question_id"], row["task"], ""])

    counts = Counter(row["task"] for row in questions)
    gold = {
        task: dict(
            Counter(row["gold"] for row in answers if row["task"] == task)
        )
        for task in ("A", "C", "D")
    }
    a_actions = dict(
        Counter(
            row["stratum"]["action_sequence_label"]
            for row in answers
            if row["task"] == "A"
        )
    )
    d_styles = dict(
        Counter(
            row["prompt_style"]
            for row in answers
            if row["task"] == "D"
        )
    )
    missing_images = []
    for row in questions:
        paths: list[str] = []
        if row["task"] == "A":
            paths = [row["current_image"], *row["candidate_images"]]
        elif row["task"] == "C":
            paths = [row["goal_image"], *row["keyframe_images"]]
        else:
            paths = [
                *row["reference"]["images"],
                *row["query"]["images"],
            ]
        for path in paths:
            if not (output_dir / path).is_file():
                missing_images.append(path)
    if missing_images:
        raise FileNotFoundError(
            f"Exam has {len(missing_images)} missing images"
        )
    if counts != Counter(
        {
            "A": args.task_a_count,
            "C": args.task_c_count,
            "D": args.task_d_count,
        }
    ):
        raise RuntimeError(f"Unexpected exam counts: {counts}")
    if any(
        any("\u4e00" <= char <= "\u9fff" for char in row["prompt"])
        for row in questions
    ):
        raise RuntimeError("A participant-visible prompt contains Chinese")

    manifest = {
        "schema_version": "habitat-exam-en-550-v1",
        "seed": args.seed,
        "english_prompts_only": True,
        "answer_blind_questions": True,
        "total_questions": len(questions),
        "task_counts": dict(counts),
        "gold_balance": gold,
        "task_a_action_counts": a_actions,
        "task_d_prompt_style_counts": d_styles,
        "image_references_checked": sum(
            5 if row["task"] == "A"
            else 1 + len(row["keyframe_images"])
            if row["task"] == "C"
            else len(row["reference"]["images"]) + len(row["query"]["images"])
            for row in questions
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_text = HTML.replace(
        "__SYSTEM_PROMPT__", json.dumps(SYSTEM_PROMPT)
    )
    (output_dir / "questions.html").write_text(
        html_text, encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(output_dir / "questions.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
