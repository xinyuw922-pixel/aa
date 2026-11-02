"""Build mission-hidden Oracle-State exams from reference/query Task D."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil

from tools.minigrid.task_d import episode_identity
from utils.json_io import load_jsonl, write_jsonl_utf8
from utils.oracle_state_facts import normalize_raw_state_fields, trace_without_identifiers
from utils.oracle_state_prompt import ENVIRONMENT_TASK_TYPE, INTRO_LINES, ONTOLOGY_LINES, OUTPUT_LINES
from utils.oracle_task_types import TASK_TYPES
from utils.oracle_visibility import nocue_evidence
from utils.minigrid_visible_state import world_to_image, image_to_observe, is_visible_object
from minigrid.core.constants import COLOR_TO_IDX

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ('pixel_only', 'oracle_state_only', 'pixel_plus_oracle_state')
VARIANTS = ('full', 'nocue', 'cf')
EVIDENCE_POLICY = 'nocue_mask_aware_v1'
FORBIDDEN = ('mission', 'success', 'reward', 'terminated', 'truncated', 'fail_reason',
             'done', 'target_object', 'cue_object', 'selected_endpoint')


def observed_object(state: dict, obj: dict, *, full_grid: bool = False) -> bool:
    """Check recorded POV encoding; Tier-2 records use full-grid images."""
    encoding = state.get('state_encoding')
    if encoding is None:
        if not full_grid:
            raise ValueError('POV NoCue record lacks its observation encoding')
        return True
    pos = obj.get('pos')
    if not isinstance(pos, list) or len(pos) != 2:
        raise ValueError('Protected Oracle object lacks its world position')
    agent = state['agent']
    x, y = world_to_image(*pos, *agent['pos'], agent['dir'])
    if not is_visible_object(x, y, encoding, object_type=obj.get('type')):
        return False
    ox, oy = image_to_observe(x, y)
    color = obj.get('color')
    return color is None or encoding[ox][oy][1] == COLOR_TO_IDX.get(color)


def evidence_record(raw: dict) -> dict:
    """Apply the source family's observation geometry before rendering facts."""
    full_grid = (raw.get('task') or raw.get('env_task')) in {
        'dynamicobstacles', 'emptyrandom', 'fourrooms', 'putnear', 'blockedunlockpickup',
    }
    return nocue_evidence(raw, lambda state, obj: observed_object(state, obj, full_grid=full_grid))


def evidence_for(condition: str, reference: dict, query: dict) -> str:
    """Expose simulator facts, never the audit-only objective or outcome."""
    lines = ['EVIDENCE:']
    if condition in ('pixel_only', 'pixel_plus_oracle_state'):
        lines.append('- The attached image contains the REFERENCE and QUERY storyboards.')
    if condition in ('oracle_state_only', 'pixel_plus_oracle_state'):
        reference = evidence_record(reference)
        query = evidence_record(query)
        lines.extend(['- Structured simulator facts are supplied below.', '',
                      trace_without_identifiers('REFERENCE_TRAJECTORY', reference), '',
                      trace_without_identifiers('QUERY_TRAJECTORY', query)])
    return '\n'.join(lines)


def make_prompt(condition: str, reference: dict, query: dict) -> str:
    """Combine the task ontology, evidence and hierarchical response contract."""
    evidence = evidence_for(condition, reference, query)
    for token in FORBIDDEN:
        if re.search(rf'\b{re.escape(token)}\b', evidence, re.IGNORECASE):
            raise ValueError('Privileged or unsupported token in Oracle evidence: ' + token)
    prompt = '\n'.join([*INTRO_LINES, '', *ONTOLOGY_LINES, '', evidence, '', *OUTPUT_LINES])
    for raw in (reference, query):
        mission = str(raw.get('mission') or '').strip()
        env = str(raw.get('task') or raw.get('env_task') or '')
        if mission and mission.lower() in prompt.lower():
            raise ValueError('Exact mission leaked into Oracle prompt')
        if env and re.search(rf'\b{re.escape(env)}\b', prompt, re.IGNORECASE):
            raise ValueError('Environment identifier leaked into Oracle prompt')
    return prompt


def load_bank(root: Path) -> tuple[Path, dict]:
    """Read one explicit triplet bank and reject ambiguous raw record keys."""
    path = root.resolve(strict=True)
    path = path / 'triplets.jsonl' if path.is_dir() else path
    lookup, seen = defaultdict(list), set()
    for row in load_jsonl(path):
        key = (str(row.get('task') or row.get('env_task')), str(row['group_id']), str(row['variant']))
        identity = (key, episode_identity(row, path.parent))
        if identity in seen:
            raise ValueError('Duplicate raw episode record: ' + str(identity))
        seen.add(identity)
        lookup[key].append(normalize_raw_state_fields(row))
    return path.parent, lookup


def find_raw(banks: list[tuple[Path, dict]], env: str, group: str, variant: str,
             mission: str, seed=None) -> tuple[Path, dict]:
    """Select the exact episode, failing instead of guessing across conflicting banks."""
    matches = {}
    for root, bank in banks:
        for row in bank.get((env, group, variant), []):
            if str(row.get('mission') or '').strip() != mission:
                continue
            if seed is not None and row.get('seed') != seed:
                continue
            # Copied views of the same episode may legitimately occur in explicit banks.
            signature = json.dumps({name: row.get(name) for name in
                                    ('env_id', 'seed', 'state_seq', 'actions_id', 'actions_text', 'success')},
                                   sort_keys=True)
            matches.setdefault(signature, (root, row))
    if len(matches) != 1:
        raise ValueError(f'Expected one exact Oracle raw episode for {env}/{group}/{variant}, found {len(matches)}')
    return next(iter(matches.values()))


def validate_state(raw: dict) -> None:
    """Do not quietly create a symbolic condition from missing simulator states."""
    states = raw.get('state_seq')
    if not isinstance(states, list) or not states:
        raise ValueError('Oracle source must include a nonempty state_seq')
    for state in states:
        agent = state.get('agent') if isinstance(state, dict) else None
        if not isinstance(agent, dict) or agent.get('pos') is None or agent.get('dir') is None:
            raise ValueError('Oracle source state lacks agent position/direction')
        if not isinstance(state.get('objects'), list) or not isinstance(state.get('front_cell'), dict):
            raise ValueError('Oracle source state lacks object/front-cell records')


def build(source: Path, output: Path, raw_roots: list[Path], reference_roots: list[Path]) -> dict:
    """Build three conditions per source row using complete Full/NoCue/CF groups."""
    source, output = source.resolve(strict=True), output.resolve()
    if output.exists() or output == ROOT or ROOT in output.parents or output in ROOT.parents:
        raise ValueError('Choose a fresh output directory outside the source repository')
    rows = load_jsonl(source)
    grouped, seen = defaultdict(dict), set()
    for row in rows:
        if row.get('task') not in ('D', 'DR', 'D-R'):
            continue
        if (row.get('variant') not in VARIANTS or row.get('temporal', 'orig') != 'orig'
                or row.get('visual', 'clean') != 'clean' or row.get('framing', 'neu') not in ('', 'neu')):
            continue
        env, group = str(row['env_task']), str(row['group_id'])
        if env not in ENVIRONMENT_TASK_TYPE:
            raise ValueError('No Oracle v4 task ontology for environment: ' + env)
        if not row.get('reference_group_id') or row['reference_group_id'] == group:
            raise ValueError('Oracle requires a distinct reference episode')
        if row['uid'] in seen or row['variant'] in grouped[(env, group)]:
            raise ValueError('Duplicate Oracle source UID/group variant')
        seen.add(row['uid'])
        grouped[(env, group)][row['variant']] = row
    if not grouped or any(set(group) != set(VARIANTS) for group in grouped.values()):
        raise ValueError('Oracle requires complete Full/NoCue/CF groups')
    discovered_queries = {Path(row[field]) for group in grouped.values() for row in group.values()
                          for field in ('source_raw_root', 'query_raw_root') if row.get(field)}
    discovered_refs = {Path(row['reference_raw_root']) for group in grouped.values() for row in group.values()
                       if row.get('reference_raw_root')}
    queries = [load_bank(path) for path in (raw_roots or sorted(discovered_queries))]
    if not queries:
        raise ValueError('Supply --raw-root for source Task D rows without raw-bank paths')
    references = queries + [load_bank(path) for path in (reference_roots or sorted(discovered_refs))]
    requests, image_sources = [], {}
    for (env, group), variants in sorted(grouped.items()):
        fixed_reference = None
        for variant in VARIANTS:
            row = variants[variant]
            mission = str(row.get('goal_signature') or row.get('query_mission') or row.get('mission') or '').strip()
            if not mission:
                raise ValueError('Oracle requires an audit-only exact mission for raw matching')
            query_root, query = find_raw(queries, env, str(row.get('source_group_id') or group), variant, mission)
            ref_root, reference = find_raw(references, env, str(row['reference_group_id']), 'full',
                                           mission, row.get('reference_seed'))
            reference_key = (str(ref_root), str(reference['group_id']), reference.get('seed'))
            if fixed_reference is not None and reference_key != fixed_reference:
                raise ValueError('Reference changes across the same Full/NoCue/CF group')
            fixed_reference = reference_key
            if episode_identity(reference, ref_root) == episode_identity(query, query_root):
                raise ValueError('Reference and query are the same simulator episode')
            expected = 'Fail' if variant == 'cf' else 'Success'
            if reference.get('success') is not True or query.get('success') is not (variant != 'cf') or row.get('answer') != expected:
                raise ValueError('Simulator outcome disagrees with Oracle exam labels')
            validate_state(reference)
            validate_state(query)
            image = Path(row.get('source_exam_root') or source.parent) / row['image']
            image = image.resolve(strict=True)
            relative = Path('images') / (hashlib.sha256(row['uid'].encode()).hexdigest() + '.png')
            image_sources[relative] = image
            for condition in CONDITIONS:
                uid = row['uid'] + '.oracle_scale2_v4_' + EVIDENCE_POLICY + '_' + condition
                prompt = make_prompt(condition, reference, query)
                exam = {**row, 'uid': uid, 'exam_id': uid, 'source_uid': row['uid'], 'prompt': prompt,
                        'image': relative.as_posix(), 'source_exam_root': str(output),
                        'oracle_control_condition': condition, 'oracle_prompt_version': 'scale2_v4',
                        'oracle_evidence_policy': EVIDENCE_POLICY,
                        'task_ontology_version': 'minigrid_task_types_v1', 'goal_signature': mission,
                        'gold_task_type': ENVIRONMENT_TASK_TYPE[env], 'gold_specific_objective': mission,
                        'oracle_prompt_uses_mission': False, 'oracle_prompt_uses_outcome_fields': False,
                        'model_facing_mission': False,
                        'oracle_reference_raw_key': {'group_id': reference['group_id'], 'variant': 'full'},
                        'oracle_query_raw_key': {'group_id': query['group_id'], 'variant': variant},
                        'reference_raw_root': str(ref_root), 'query_raw_root': str(query_root),
                        'response_contract': {'fields': ['TASK_TYPE', 'INFERRED_OBJECTIVE', 'VERDICT', 'EXPLANATION'],
                                              'task_type_values': list(TASK_TYPES), 'verdict_values': ['Success', 'Fail']}}
                requests.append(dict(uid=uid, exam_id=uid, schema_version=exam['schema_version'],
                                     images=[] if condition == 'oracle_state_only' else [str(output / relative)],
                                     prompt=prompt, exam=exam, api_content_order='text_first'))
    output.mkdir(parents=True)
    for relative, image in image_sources.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)
    write_jsonl_utf8(output / 'exam/task_d_exam.jsonl', [request['exam'] for request in requests])
    write_jsonl_utf8(output / 'requests_oracle_state_v4.jsonl', requests)
    for condition in CONDITIONS:
        write_jsonl_utf8(output / 'requests_by_condition' / f'requests_{condition}.jsonl',
                        [request for request in requests if request['exam']['oracle_control_condition'] == condition])
    return dict(status='PASS', groups=len(grouped), source_rows=len(grouped) * 3,
                evidence_policy=EVIDENCE_POLICY,
                requests=len(requests), conditions=list(CONDITIONS), output=str(output))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True, help='Generated mission-matched Task D rows')
    parser.add_argument('--raw-root', action='append', type=Path, default=[], help='Query triplet bank; repeat for multiple banks')
    parser.add_argument('--reference-root', action='append', type=Path, default=[], help='Reference triplet bank; defaults to query banks')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_manifest, args.output_dir, args.raw_root, args.reference_root), indent=2))


if __name__ == '__main__':
    main()
