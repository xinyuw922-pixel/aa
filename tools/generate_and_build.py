#!/usr/bin/env python3
"""Generate simulator images and assemble exams; no model or synthetic scoring."""
import argparse
import fcntl
import hashlib
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from utils.json_io import load_jsonl, write_json, write_jsonl
from utils.minigrid_prompts import TASK_B_PROMPT

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    """Hash artifacts incrementally, including large generated files."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_contract() -> dict:
    """Fingerprint executable code, configs and dependency specifications."""
    return {str(p.relative_to(ROOT)): file_hash(p)
            for directory in ('tools', 'utils', 'scripts', 'requirements', 'environments')
            for p in sorted((ROOT/directory).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


class GenerationPipeline:
    """Own stage execution and exam assembly within one fresh output root."""

    def __init__(self, output: Path, seed: int, profile: str, habitat_python: Path | None = None,
                 habitat_renderer: str = 'software', resume: bool = False):
        self.output, self.seed, self.profile = output, seed, profile
        self.habitat_python = habitat_python
        self.habitat_renderer = habitat_renderer
        self.resume = resume
        self.active_dataset = ''
        self.base_exam_options = {}
        self.state = dict(status='RUNNING', seed=seed, profile=profile, stages=[], datasets={},
                          scope='Simulator generation and exam assembly only; no inference or scoring')
        if resume:
            previous = json.loads((output/'generation_status.json').read_text())
            if (previous['seed'], previous['profile']) != (seed, profile):
                raise ValueError('Resume requires the original seed and profile')
            if previous.get('habitat_renderer', 'software') != habitat_renderer:
                raise ValueError('Resume cannot change the Habitat renderer')
            for stage in previous['stages']:
                if stage['status'] == 'RUNNING':
                    stage.update(status='INTERRUPTED',
                                 interruption_reason='Previous execution did not record completion before resume')
            previous.pop('error', None)
            self.state = previous
            self.state['status'] = 'RUNNING'
        self.state['habitat_renderer'] = habitat_renderer
        import_root = os.environ.get('GRIDWM_IMPORT_ROOT', str(ROOT))
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONPATH=import_root,
                        SDL_VIDEODRIVER='dummy', LIBGL_ALWAYS_SOFTWARE='1', EGL_PLATFORM='surfaceless',
                        MESA_LOADER_DRIVER_OVERRIDE='llvmpipe', GALLIUM_DRIVER='llvmpipe',
                        OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                        MPLCONFIGDIR=str(output/'cache/matplotlib'), XDG_CACHE_HOME=str(output/'cache'))

    def _save(self) -> None:
        write_json(self.output/'generation_status.json', self.state)

    def _run(self, name: str, target: str, *args, habitat: bool = False, allowed=(0,)) -> int:
        """Run an internal tool with isolated logs and explicit failure handling."""
        executable = str(self.habitat_python) if habitat else sys.executable
        if habitat and not self.habitat_python:
            raise ValueError('Habitat requires --habitat-python')
        command = [executable, str(ROOT/target), *map(str,args)]
        stage_name = self.active_dataset+':'+name
        previous = next((row for row in self.state['stages']
                         if row['name'] == stage_name and row['status'] == 'PASS'), None)
        if previous:
            if previous['command'] != command:
                raise ValueError('Resume command changed for '+name)
            self._verify_artifacts(previous['artifacts'])
            print('Completed stage retained: '+name, flush=True)
            return previous['returncode']
        interrupted = next((row for row in reversed(self.state['stages'])
                            if row['name'] == stage_name), None)
        if interrupted and not habitat:
            before = interrupted.get('artifacts_before', {})
            current = self._inventory()
            if current != before:
                raise ValueError('Stage has partial artifacts and no sample-level recovery: '+stage_name+
                                 '. All files retained; choose a fresh output root.')
        before = self._inventory()
        log = self.output/'logs'/f'{len(self.state["stages"]):03d}_{name}.log'
        log.parent.mkdir(exist_ok=True)
        record = dict(name=stage_name, command=command, log=str(log), status='RUNNING',
                      artifacts_before=before)
        self.state['stages'].append(record); self._save()
        print('Starting '+name, flush=True)
        with log.open('x') as handle:
            env = self.env.copy()
            if habitat and self.habitat_renderer == 'gpu':
                for key in ('LIBGL_ALWAYS_SOFTWARE', 'MESA_LOADER_DRIVER_OVERRIDE', 'GALLIUM_DRIVER'):
                    env.pop(key, None)
            elif habitat and self.habitat_renderer == 'd3d12':
                for key in ('LIBGL_ALWAYS_SOFTWARE', 'MESA_LOADER_DRIVER_OVERRIDE', '__EGL_VENDOR_LIBRARY_DIRS'):
                    env.pop(key, None)
                vendor = Path('/usr/share/glvnd/egl_vendor.d/50_mesa.json')
                if not vendor.is_file() or not Path('/usr/lib/wsl/lib').is_dir():
                    raise ValueError('D3D12 requires WSL graphics libraries and the system Mesa EGL vendor')
                env.update(GALLIUM_DRIVER='d3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA',
                           __EGL_VENDOR_LIBRARY_FILENAMES=str(vendor))
                env['LD_LIBRARY_PATH'] = ':'.join(filter(None, (
                    '/usr/lib/wsl/lib', '/usr/lib/x86_64-linux-gnu', env.get('LD_LIBRARY_PATH'))))
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if (habitat and self.habitat_renderer == 'd3d12' and result.returncode == 0
                and name in ('habitat_navigation', 'habitat_interactions', 'habitat_action_candidates')):
            text = log.read_text(errors='replace')
            renderers = [line for line in text.splitlines() if line.startswith('Renderer:')]
            if not renderers or any('D3D12 (NVIDIA' not in line for line in renderers):
                record.update(status='FAIL', returncode=result.returncode,
                              error='D3D12 NVIDIA hardware renderer was not confirmed')
                self._save()
                raise RuntimeError('Hardware renderer verification failed: '+str(log))
        record.update(status='PASS' if result.returncode in allowed else 'FAIL', returncode=result.returncode)
        if record['status'] == 'PASS':
            after = self._inventory()
            record['artifacts'] = {k:v for k,v in after.items() if before.get(k) != v}
            record.pop('artifacts_before', None)
        self._save()
        if result.returncode not in allowed:
            raise RuntimeError('Generation stage failed; all outputs retained: '+str(log))
        return result.returncode

    def minigrid(self, scale2: bool = False) -> dict:
        """Build A/C/D from fresh banks, plus B/E for the base MiniGrid path."""
        folder = self.output/('scale2' if scale2 else 'minigrid')
        folder.mkdir(exist_ok=self.resume)
        self.env['GRIDWM_REPORT_ROOT'] = str(folder/'generation_reports')
        formal_scale2 = scale2 and self.profile == 'formal'
        formal_base = not scale2 and self.profile == 'formal'
        profile = 'scale2' if formal_scale2 else ('pilot' if formal_base else 'smoke')
        self._run('raw_bank', 'tools/minigrid/generate_bank.py', '--profile', profile,
                  '--seed-offset', self.seed, '--work-root', folder/'work', '--run-tag', 'fresh')
        run = folder/'work/fresh'
        if formal_base:
            return self._formal_minigrid(folder, run/'raw_bank')
        if formal_scale2:
            self._run('formal_exams','tools/exams/assemble_formal.py','scale2','--run-root',run)
            exam_root = run/'requests_validated/exam'
        else:
            raw, base = run/'raw_bank', folder/'base'
            self._run('base_exams','tools/minigrid/build_exams.py','--raw-root',raw,'--out-dir',base,'--task-a-target',30)
            self._run('check_a','tools/minigrid/task_a.py','--source',base/'taskA_expanded_manifest.jsonl','--output-dir',folder/'a')
            self._run('prompt_c','tools/minigrid/task_c.py','--source',base/'taskC_expanded_manifest.jsonl','--output-dir',folder/'c')
            self._run('references','tools/minigrid/reference_pool.py','--base-root',base,'--run-root',folder/'work',
                      '--out-dir',folder/'references','--seed-start',self.seed+3000000)
            self._run('match_d','tools/minigrid/task_d.py','--scale2-root',base,'--output-dir',folder/'d',
                      '--raw-source','fresh='+str(folder/'references/existing'), '--generated-root',folder/'references/generated',
                      '--generated-source-tag','fresh','--expected-query-groups',len(load_jsonl(base/'taskD_expanded_manifest.jsonl'))//3)
            manifests=folder/'manifests'; manifests.mkdir(exist_ok=self.resume)
            for task in ('A','C','D'):
                shutil.copyfile(folder/task.lower()/f'task{task}_expanded_manifest.jsonl',manifests/f'task{task}_expanded_manifest.jsonl')
            self._run('assemble_acd','tools/exams/build_requests.py','--manifest-dir',manifests,'--out-dir',folder/'bundle','--mode','full')
            exam_root=folder/'bundle/exam'
            if scale2:
                return self._exam_index(exam_root)
            from tools.minigrid.task_b import ENVIRONMENTS, build_taskB
            views=folder/'views'; grouped=defaultdict(list)
            for row in load_jsonl(raw/'triplets.jsonl'):
                if row['task'] not in ENVIRONMENTS:
                    continue
                row['frames']=[str((raw/p).resolve()) for p in row['frames']]
                grouped[row['task']].append(row)
            for env,rows in grouped.items():write_jsonl(views/env/'triplets.jsonl',rows)
            rows,_=build_taskB(sorted(views.iterdir()),exam_root,SimpleNamespace(seed=self.seed,b_per_traj=1))
            for row in rows:row.update(prompt=TASK_B_PROMPT,source_exam_root=str(exam_root))
            write_jsonl(exam_root/'task_b_exam.jsonl',rows)
            e=folder/'task_e'
            target=2 if self.profile=='smoke' else 24
            self._run('task_e_trajectories','tools/minigrid/generate_task_e.py','--out-dir',e/'trajectories',
                      '--target-per-env',target,'--max-attempts-per-env',120)
            self._run('task_e_exam','tools/minigrid/task_e.py','--phase58-root',e/'trajectories','--out-dir',e/'exam')
            # Keep the native file/version and source image root; never relabel the UID prefix.
            e_rows=load_jsonl(e/'exam/task_e_exam_v13.jsonl')
            for row in e_rows:row['source_exam_root']=str(e/'exam')
            write_jsonl(exam_root/'task_e_exam_v13.jsonl',e_rows)
            self._complete_base_requests(folder/'bundle', rows, e_rows)
        return self._exam_index(exam_root)

    def _inventory(self) -> dict:
        """Snapshot generated artifacts, excluding mutable progress and caches."""
        result = {}
        for path in self.output.rglob('*'):
            relative = path.relative_to(self.output)
            if (any(part in ('logs', 'cache', '__pycache__', 'pipeline_logs', 'formal_assembly_logs')
                    for part in relative.parts) or path.name in (
                    'generation_status.json', '.generation.lock', 'stage_status.json')
                    or path.suffix in ('.log', '.pyc') or not path.is_file()):
                continue
            result[str(relative)] = file_hash(path)
        return result

    def _verify_artifacts(self, artifacts: dict) -> None:
        for relative, expected in artifacts.items():
            path = self.output/relative
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError('Completed artifact missing or changed; nothing regenerated: '+str(path))

    def execute_dataset(self, dataset: str, assets: Path | None = None) -> dict:
        """Resume only validated stages; protect completed dataset artifacts."""
        self.active_dataset = dataset
        completed = self.state.get('dataset_artifacts', {}).get(dataset)
        if dataset in self.state['datasets'] and completed is not None:
            self._verify_artifacts(completed)
            print('Completed dataset retained: '+dataset, flush=True)
            return self.state['datasets'][dataset]
        if dataset == 'minigrid':
            result = self.minigrid()
        elif dataset == 'scale2':
            result = self.minigrid(scale2=True)
        elif dataset == 'miniworld':
            result = self.miniworld()
        else:
            result = self.habitat(assets)
        prefix = dataset+'/'
        self.state.setdefault('dataset_artifacts', {})[dataset] = {
            k:v for k,v in self._inventory().items() if k.startswith(prefix)}
        self.state['datasets'][dataset] = result
        self._save()
        return result

    def _formal_minigrid(self, folder: Path, raw: Path) -> dict:
        """Assemble formal A/B/C/D cohorts and all valid object-tracking checkpoints."""
        formal = folder/'formal'
        lava_pool = folder/'work/fresh/tier1_pool/lavagap/lavagaps7'
        self._run('formal_validate_lavagap_pool','tools/minigrid/environments/lavagap.py',
                  'validate','--root',lava_pool)
        options = self.base_exam_options
        a_source = options.get('task_a_raw_root')
        if not a_source:
            self._run('formal_base_a_sources', 'tools/minigrid/generate_base_a.py',
                      '--output-dir', folder/'base_a_sources')
            a_source = folder/'base_a_sources/sources'
        arguments = ['--task-b-raw-root', options.get('task_b_raw_root') or folder/'work/fresh/tier1_pool',
                     '--task-a-raw-root', a_source,
                     '--task-a-per-traj', options['task_a_per_traj'], '--exam-seed', options['exam_seed']]
        if options['task_a_cohort']:
            arguments.extend(['--task-a-cohort', options['task_a_cohort']])
        self._run('formal_base_sources','tools/minigrid/build_formal.py','--raw-root',raw,
                  '--output-dir',formal,'--seed',self.seed,'--lavagap-pool',lava_pool, *arguments)
        manifests = folder/'manifests'
        manifests.mkdir(exist_ok=self.resume)
        for task, source in (('A',formal),('C',formal),('D',formal)):
            shutil.copyfile(source/f'task{task}_expanded_manifest.jsonl',
                            manifests/f'task{task}_expanded_manifest.jsonl')
        bundle = folder/'bundle'
        self._run('formal_assemble_acd','tools/exams/build_requests.py','--manifest-dir',manifests,
                  '--out-dir',bundle,'--mode','full')
        exam = bundle/'exam'
        b_rows = load_jsonl(formal/'task_b_exam.jsonl')
        write_jsonl(exam/'task_b_exam.jsonl', b_rows)
        e = folder/'task_e'
        self._run('formal_e_trajectories','tools/minigrid/generate_task_e.py','--out-dir',e/'trajectories',
                  '--target-per-env',24,'--max-attempts-per-env',120)
        for env in ('unlockpickup','blockedunlockpickup'):
            if len(load_jsonl(e/'trajectories'/f'{env}_pilot_full.jsonl')) != 24:
                raise ValueError('Task E formal trajectory quota not met for '+env)
        self._run('formal_e_exam','tools/minigrid/task_e.py','--phase58-root',e/'trajectories',
                  '--out-dir',e/'exam')
        e_rows = load_jsonl(e/'exam/task_e_exam_v13.jsonl')
        if not e_rows or {row['env_task'] for row in e_rows} != {'unlockpickup','blockedunlockpickup'}:
            raise ValueError('Task E v1.3 requires both promoted environments')
        for row in e_rows:
            row['source_exam_root'] = str(e/'exam')
        write_jsonl(exam/'task_e_exam_v13.jsonl',e_rows)
        self._complete_base_requests(bundle,b_rows,e_rows)
        result = self._exam_index(exam)
        expected = {'task_a_exam.jsonl':103,'task_b_exam.jsonl':144,
                    'task_c_exam.jsonl':486,'task_d_exam.jsonl':144,
                    'task_e_exam_v13.jsonl':len(e_rows)}
        if result['counts'] != expected:
            raise ValueError('Formal MiniGrid count mismatch: '+str(result['counts']))
        result.update(cohort='fresh_base_minigrid_formal', historical_uid_replay=False,
                      task_e_protocol='v1.3: all valid checkpoints from 24 trajectories per promoted environment')
        return result

    def _complete_base_requests(self, bundle: Path, b_rows: list, e_rows: list) -> None:
        """Retain the native offline request contract for the B/E exam rows."""
        request_root = bundle/'_requests_from_manifest'
        requests = load_jsonl(request_root/'requests_ACD_full.jsonl')
        for task, exam_rows in (('B', b_rows), ('E', e_rows)):
            task_requests = []
            for row in exam_rows:
                image = (Path(row['source_exam_root'])/row['image']).resolve(strict=True)
                task_requests.append(dict(uid=row['uid'], exam_id=row.get('exam_id', row['uid']),
                                          schema_version=row['schema_version'], images=[str(image)],
                                          prompt=row['prompt'], exam=row, api_content_order='text_first'))
            write_jsonl(request_root/f'requests_{task}.jsonl', task_requests)
            requests.extend(task_requests)
        exam_rows = [row for path in (bundle/'exam').glob('task_*_exam*.jsonl') for row in load_jsonl(path)]
        if len(requests) != len(exam_rows) or len({row['uid'] for row in requests}) != len(requests):
            raise ValueError('Duplicate or incomplete offline request coverage')
        if {row['uid'] for row in requests} != {row['uid'] for row in exam_rows}:
            raise ValueError('Request and exam UID sets differ')
        write_jsonl(request_root/'requests_ABCDE_full.jsonl', requests)

    def miniworld(self) -> dict:
        """Render MiniWorld sources and build its native A/C/D exams."""
        work=self.output/'miniworld'; work.mkdir(exist_ok=self.resume)
        bank=work/'experiments/generated/miniworld'
        if self.profile=='formal':
            from tools.miniworld.generate_exams import stages
            for stage, _, _, _ in stages():
                self._run('miniworld_'+stage,'tools/miniworld/generate_exams.py',
                          '--work-dir',work,'--stages',stage)
            roots=[bank/'taskA-MW-v2-crossenv',bank/'taskC-MW-formal-bank',bank/'taskD-MW-reference-family-exam50']
        else:
            def native(name,builder,*extra):
                self._run(name,'tools/miniworld/run_builder.py','--work-dir',work,builder,'--',*extra)
            native('mw_fourrooms','generate_fourrooms.py','--out_dir',bank/'a_fourrooms','--num_questions',4,'--isolate-attempts')
            native('mw_balance','balance_task_a_labels.py','--src_dir',bank/'a_fourrooms','--dst_dir',bank/'a_balanced')
            native('mw_pickup','generate_pickup_objects.py','--out_dir',bank/'a_pickup','--num_questions',3)
            native('mw_putnext','generate_put_next.py','--out_dir',bank/'a_putnext','--num_questions',3)
            native('mw_a_exam','assemble_task_a.py','--out_dir',bank/'a_exam',
                   '--fourrooms_jsonl',bank/'a_balanced/taskA-MW-v2-prototype.jsonl',
                   '--pickupobjects_jsonl',bank/'a_pickup/taskA-MW-v2-pickupobjects.jsonl',
                   '--putnext_jsonl',bank/'a_putnext/taskA-MW-v2-putnext.jsonl')
            native('mw_pairs','gen_miniworld_taskd_migration_showcase.py','--out_dir',bank/'taskD-MW-migration-showcase-families-paired')
            native('mw_c_exam','gen_taskC_MW_formal_exam.py','--out_dir',bank/'c_exam','--source-summary',
                   bank/'taskD-MW-migration-showcase-families-paired/migration_showcase_summary.json','--max-groups',4)
            native('mw_d_exam','gen_taskD_MW_reference_family_exam50.py','--out_dir',bank/'d_exam','--source-bucket','paired','--max-pairs',5)
            roots=[bank/'a_exam',bank/'c_exam',bank/'d_exam']
        requests=[]
        for task, root in zip('ACD', roots):
            output=work/'requests'/f'requests_{task}.jsonl'
            self._run('mw_requests_'+task,'tools/miniworld/build_requests.py',
                      '--exam-file',root/f'task_{task.lower()}_exam.jsonl',
                      '--task',task,'--output',output)
            requests.append(str(output))
        return dict(status='PASS',exams=[self._exam_index(p) for p in roots],requests=requests)

    def habitat(self, assets: Path) -> dict:
        """Generate fresh navigation/interactions/references, then assemble Habitat A/C/D."""
        folder=self.output/'habitat'; folder.mkdir(exist_ok=self.resume)
        formal=self.profile=='formal'
        config=json.loads((ROOT/'tools/habitat/config.hssd-pilot.json').read_text())
        config.update(output_dir=str(folder/'navigation'),base_seed=self.seed,
                      max_scenes=10 if formal else 1,episodes_per_scene=20 if formal else 2,
                      navmesh_cache_dir=str(folder/'navmeshes'))
        config['gpu_device_id'] = 0 if self.habitat_renderer == 'gpu' else -1
        write_json(folder/'navigation_config.json',config)
        self._run('habitat_navigation','tools/habitat/generate_navigation.py','--config',folder/'navigation_config.json',
                  '--scene-glob',str(assets/'scenes/*.scene_instance.json'),'--scene-dataset-config',assets/'hssd-hab.scene_dataset_config.json',habitat=True)
        config['output_dir']=str(folder/'interactions'); write_json(folder/'interaction_config.json',config)
        self._run('habitat_interactions','tools/habitat/generate_interactions.py','--config',folder/'interaction_config.json',
                  '--scene-glob',str(assets/'scenes-articulated/*.scene_instance.json'),
                  '--scene-dataset-config',assets/'hssd-hab-articulated.scene_dataset_config.json','--output',folder/'interactions',
                  '--max-scenes',0 if formal else 4,'--max-episodes',110 if formal else 6,'--max-targets-per-scene',6 if formal else 3,
                  '--max-probes-per-scene',48 if formal else 24,'--seed',self.seed,habitat=True)
        self._run('habitat_references','tools/habitat/generate_references.py','--root',folder/'interactions',
                  '--scene-root',assets/'scenes-articulated','--scene-dataset-config',assets/'hssd-hab-articulated.scene_dataset_config.json',
                  '--config',folder/'interaction_config.json','--reference-phase-attempts',8,'--seed',self.seed+1,habitat=True,allowed=(0,2))
        self._run('habitat_select','tools/habitat/select_pairs.py','--source',folder/'interactions','--output',folder/'selected','--targets',100 if formal else 5)
        self._run('habitat_candidates','tools/habitat/build_candidates.py','--source',folder/'navigation','--output',folder/'candidates','--seed',self.seed)
        self._run('habitat_action_candidates','tools/habitat/build_action_candidates.py',
                  '--source',folder/'candidates','--output',folder/'action_candidates','--seed',self.seed,habitat=True)
        self._run('habitat_exam','tools/habitat/build_exams.py','--task-a-root',folder/'action_candidates','--interaction-root',folder/'selected',
                  '--output-dir',folder/'exam','--task-a-count',150 if formal else 10,'--task-c-count',200 if formal else 10,
                  '--task-d-count',200 if formal else 10,'--seed',self.seed)
        self._run('habitat_requests','tools/habitat/build_requests.py','--exam-dir',folder/'exam','--output-dir',folder/'requests','--task-d-montage')
        return dict(status='PASS',exam=str(folder/'exam'),counts=json.loads((folder/'exam/manifest.json').read_text()).get('task_counts'))

    def _exam_index(self, folder: Path) -> dict:
        """Index native exam files without changing their row contracts."""
        files=sorted(folder.glob('task_*_exam*.jsonl'))
        if not files:raise ValueError('No exams generated: '+str(folder))
        counts={p.name:len(load_jsonl(p)) for p in files}
        return dict(status='PASS',exam_root=str(folder),counts=counts)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--datasets',nargs='+',choices=['minigrid','scale2','miniworld','habitat'],default=['minigrid'])
    parser.add_argument('--profile',choices=['smoke','formal'],default='smoke')
    parser.add_argument('--seed',type=int,default=830000)
    parser.add_argument('--task-b-raw-root', type=Path)
    parser.add_argument('--task-a-cohort', type=Path)
    parser.add_argument('--task-a-raw-root', type=Path,
                        help='Override native Task A sources; defaults to fresh base-cohort generation')
    parser.add_argument('--task-a-per-traj', type=int, default=6)
    parser.add_argument('--exam-seed', type=int, default=42)
    parser.add_argument('--habitat-python',type=Path)
    parser.add_argument('--habitat-assets',type=Path)
    parser.add_argument('--habitat-renderer', choices=['software', 'gpu', 'd3d12'], default='software',
                        help='software: Mesa CPU; gpu: native NVIDIA EGL; d3d12: WSL NVIDIA hardware OpenGL')
    parser.add_argument('--resume', action='store_true',
                        help='Resume validated stages with the original code/configuration; partial stages may require a fresh run')
    args=parser.parse_args()
    if args.task_b_raw_root and not args.task_b_raw_root.is_dir():
        parser.error('Task B raw source does not exist')
    if args.task_a_raw_root and not args.task_a_raw_root.is_dir():
        parser.error('Task A raw source does not exist')
    if args.task_a_per_traj < 1:
        parser.error('--task-a-per-traj must be positive')
    output=args.output.resolve()
    if (output.exists() and not args.resume) or output==ROOT or ROOT in output.parents or output in ROOT.parents:
        parser.error('Use a fresh output directory outside the repository')
    if args.resume and not (output/'generation_status.json').is_file():
        parser.error('--resume requires an existing generation_status.json')
    if not 0<=args.seed<2**31-4010000:parser.error('Seed outside supported range')
    if 'habitat' in args.datasets and (not args.habitat_python or not args.habitat_assets):
        parser.error('Habitat requires --habitat-python and --habitat-assets')
    contract = dict(version=1, datasets=list(dict.fromkeys(args.datasets)), seed=args.seed,
                    profile=args.profile, habitat_renderer=args.habitat_renderer,
                    habitat_python=str(args.habitat_python.resolve()) if args.habitat_python else None,
                    habitat_assets=str(args.habitat_assets.resolve()) if args.habitat_assets else None,
                    core_python=str(Path(sys.executable).resolve()), source=source_contract(),
                    base_exam_options=dict(task_b_protocol='full_only_144',
                        task_b_raw_root=str(args.task_b_raw_root.resolve()) if args.task_b_raw_root else None,
                        task_a_cohort=str(args.task_a_cohort.resolve()) if args.task_a_cohort else None,
                        task_a_raw_root=str(args.task_a_raw_root.resolve()) if args.task_a_raw_root else None,
                        task_a_per_traj=args.task_a_per_traj, exam_seed=args.exam_seed))
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output/'.generation.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another generation process owns this output directory')
        if args.resume:
            previous=json.loads((output/'generation_status.json').read_text())
            if previous.get('resume_contract') != contract:
                parser.error('Resume code/configuration differs, or this run predates stage checkpoints; files preserved')
        pipeline=GenerationPipeline(output,args.seed,args.profile,args.habitat_python,args.habitat_renderer,args.resume)
        pipeline.base_exam_options = contract['base_exam_options']
        pipeline.state['resume_contract'] = contract
        pipeline._save()
        try:
            for dataset in contract['datasets']:
                pipeline.execute_dataset(dataset,args.habitat_assets.resolve() if args.habitat_assets else None)
            pipeline.state['status']='PASS'
        except BaseException as exc:
            pipeline.state.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAIL',
                                  error=type(exc).__name__+': '+str(exc))
            raise
        finally:pipeline._save()
    print(json.dumps(pipeline.state['datasets'],indent=2))


if __name__=='__main__':main()
