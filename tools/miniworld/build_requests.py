"""Export MiniWorld A/C/reference-D exams using their native request contracts."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from utils.exam_schema import resolve_uid
from utils.json_io import load_jsonl, write_jsonl_utf8


def task_a_prompt(record: dict) -> str:
    """Preserve the native three-image ordering and strict A/B output contract."""
    base_prompt = str(record.get('prompt', '')).strip()
    base_prompt = re.sub(
        r'\n\nAnswer with exactly one uppercase letter: A or B\.\nDo not output any other words\.\s*$',
        '', base_prompt, flags=re.DOTALL).strip()
    order_note = (
        "The input consists of three images shown in this exact order:\n"
        "1. the current state image\n"
        "2. candidate view A\n"
        "3. candidate view B\n\n"
        "The current state image contains the agent's ego view together with a top-view map.\n"
        "Each candidate image shows only the candidate successor ego view.\n"
    )
    output_contract = (
        "\n\nOutput rule:\n"
        "Your entire response must be exactly one uppercase letter: A or B.\n"
        "Do not output any other character, word, explanation, punctuation, or newline.\n"
        "If you think the answer is candidate A, output exactly: A\n"
        "If you think the answer is candidate B, output exactly: B"
    )
    return f'{order_note}{base_prompt}{output_contract}'


def build(exam: Path, task: str, output: Path) -> list[dict]:
    """Export without changing exam IDs, candidate order, answers or image layout."""
    exam = exam.resolve(strict=True)
    rows, seen, requests = load_jsonl(exam), set(), []
    for row in rows:
        uid = resolve_uid(row)
        if not uid or uid in seen:
            raise ValueError('Missing or duplicate MiniWorld exam UID')
        seen.add(uid)
        if task == 'A':
            paths = [exam.parent / row[key] for key in
                     ('current_image_rel_path', 'candidate_a_rel_path', 'candidate_b_rel_path')]
            prompt = task_a_prompt(row)
        else:
            root = Path(row.get('image_root') or exam.parent)
            if not root.is_absolute():
                root = exam.parent / root
            paths, prompt = [root / row['image']], row['prompt']
        images = [str(path.resolve(strict=True)) for path in paths]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Missing MiniWorld model-facing prompt: ' + uid)
        requests.append(dict(uid=uid, exam_id=row.get('exam_id', uid),
                             schema_version=row.get('schema_version', 'gridwm.exam.v1'),
                             images=images, prompt=prompt, exam=dict(row)))
    if not requests:
        raise ValueError('Empty MiniWorld exam')
    if output.exists():
        raise ValueError('Choose a new request output file')
    write_jsonl_utf8(output, requests)
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exam-file', type=Path, required=True)
    parser.add_argument('--task', choices=('A', 'C', 'D'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.exam_file, args.task, args.output)
    print(f'Built {len(rows)} requests -> {args.output}')


if __name__ == '__main__':
    main()
