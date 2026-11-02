# GridWM-Judge: Generation, Inference And Scoring

Generate fresh simulator images, assemble MiniGrid, MiniWorld and Habitat
exams, run remote model APIs from request bundles, and score recorded model
responses. Images, requests and responses remain outside the source repository.

```text
scripts/*.sh -> tools/ workflows -> tools/ generators/builders
                                 -> utils/ leaf helpers
                                 -> installed simulator dependencies
```

## Task And Experiment Modules

MiniGrid task ownership is explicit. Simulator trajectories, source parsing,
quality gates and image composition are shared, rather than copied into each
task. Base and expanded cohorts use explicitly named builders.

| Module | Responsibility |
| --- | --- |
| `tools/minigrid/task_a.py` | Transition-choice construction and candidate uniqueness checks; its CLI takes a source manifest. |
| `tools/minigrid/task_b.py` | Native visible-state B construction; standalone CLI takes an environment triplet root. |
| `tools/minigrid/task_c.py` | Storyboard judgment construction, task prompts and optional matched presentation conditions. |
| `tools/minigrid/task_d.py` | Base/final D construction and exact-mission independent-reference matching. |
| `tools/minigrid/task_e.py` | Object identity/status question assembly and event/label helpers. |
| `tools/minigrid/base_task_a.py` | Base A action-matched, state-ranked distractors; optional explicit cohort UID selection. |
| `tools/minigrid/base_task_b.py` | Frame selection, visible-state answers and DoorKey view re-anchoring. |
| `tools/minigrid/base_task_c.py` | Base C event-anchored evidence, six temporal/visual conditions and three framing conditions. |
| `tools/minigrid/base_task_d.py` | Base D reference/query assembly with four presentation conditions per query. |
| `tools/minigrid/exam_source.py` | Shared trajectory parsing, state/negative-choice helpers and storyboard selection. No task batch CLI. |
| `tools/minigrid/build_exams.py` | Thin A/C/D base assembly orchestration, delegates task logic to the modules above. |
| `tools/experiments/structured_cot.py` | Three-arm C/D prompt experiment request construction only. No API calls or scoring. |
| `tools/miniworld/generate_templates.py` | Selectable round1/round2/round3 template packs. |
| `tools/miniworld/build_requests.py` | Native A/C/reference-D request export; also called after MiniWorld generation. |

Leaf helpers under `utils/` implement MiniGrid image panels, B/C/D prompts,
NoCue calculations, Habitat I/O and configuration/trajectory data types.
They import no project modules and run no other files. Environment-specific
CF/NoCue semantics and renderer process isolation remain separate.

## MiniGrid Environments

Environment adapters are separate from the Task A/B/C/D/E exam builders:

```text
tools/minigrid/environments/
├── doorkey.py
├── keycorridor.py
├── lavagap.py
├── memory.py
├── multiroom.py
└── redblue.py
```

Each module provides `generate` and `validate` commands. The generated
trajectory pool is then consumed by the task builders. For example:

```bash
bash scripts/_launch.sh tools.minigrid.environments.doorkey generate \
  --out-dir /external/new-doorkey --num 2 --seed-start 1000 \
  --env-id MiniGrid-DoorKey-8x8-v0
bash scripts/_launch.sh tools.minigrid.environments.doorkey validate \
  --root /external/new-doorkey
```

Use `generate --help` or `validate --help` for environment-specific options.
`tools/minigrid/environment_registry.py` defines the adapter paths, environment
IDs and default generation/validation arguments used by the bank workflow.
Validation summaries retain environment-specific formats; check the reported
failure counts, not only the process exit status.

For example, construct the three-arm experiment from generated C/D requests:

```bash
bash scripts/build_experiment.sh \
  --source-requests /external/run/requests_ACD_full.jsonl \
  --output-dir /external/new-three-arm
```

The arms are `direct_original`, `structured_direct`, and
`structured_cot_v1`. Default expected group counts are C101/D98.
Bundles use `task_context_matched_v2`: all Task C
arms share the same environment description and simulator mission, including
the direct arm; only the response/evidence instructions differ. Missing Task C
missions are rejected. Task D continues to infer the objective from its known
successful reference, without exposing its scoring-only mission. For a
different cohort, pass `--expected-c-groups` and `--expected-d-groups` explicitly.

## Model Inference

### Oracle-State Control

Build a three-condition Oracle exam from generated mission-matched
Task D rows and the simulator triplet banks used for those reference/query
episodes:

```bash
bash scripts/build_experiment.sh oracle-state \
  --source-manifest /external/minigrid/manifests/taskD_expanded_manifest.jsonl \
  --raw-root /external/minigrid/work/fresh/raw_bank \
  --reference-root /external/minigrid/references/existing \
  --reference-root /external/minigrid/references/generated \
  --output-dir /external/new-oracle-v4
```

Repeat the bank arguments if needed. Query banks also supply eligible
references; extra reference banks supply independently generated examples.
Input episodes must resolve uniquely, have matching missions and complete
Full/NoCue/CF groups, and include the required state fields. Each group uses a
fixed reference. Invalid inputs raise an error. Rows with explicit
`query_raw_root`/`reference_raw_root` may resolve their banks without overrides.

Each source group produces nine requests: Full/NoCue/CF crossed with
`pixel_only`, `oracle_state_only`, and `pixel_plus_oracle_state`. The state
conditions use the recorded simulator `state_seq`, actions, initial/final
objects and derived carrying/door/spatial facts. They are not captions inferred
from images. State-only requests have no images; both pixel conditions retain
the original reference/query composite pixels. The NoCue trace uses its own
recorded state sequence rather than the Full trace. The `nocue_mask_aware_v1`
evidence policy removes suppressed targets before formatting inventories,
relations and events. Protected targets can reappear only at unmasked steps
where the recorded observation supports them. An empty post-mask visibility
list is never replaced by world truth; unknown front/carrying observations are
described as unobserved rather than empty space. Simulator outcomes remain unchanged.
This is targeted suppression of NoCue evidence, not a full conversion of all
world-state facts to pixel-only visibility. Success/Fail labels and native
scoring remain simulator-grounded; missing evidence is not a failure label.
Request UIDs include the evidence policy. Responses must match the exact
request bundle and evidence policy being scored.

Every condition presents the same six possible task types and asks for
`TASK_TYPE`, `INFERRED_OBJECTIVE`, `VERDICT`, and `EXPLANATION`. The model must
infer the specific objective from the completed reference. The actual mission,
environment identity, success flags and answer labels are kept in local exam
metadata for matching/scoring and are not added to model-facing evidence.

Outputs are `exam/task_d_exam.jsonl`, `requests_oracle_state_v4.jsonl`, and
three files under `requests_by_condition/`, plus copied images outside the
source tree. Run inference using the combined or condition-specific requests.
Use the hierarchical Oracle parser, not the ordinary binary Task D parser,
for these four-field responses. To write only per-question scores:

```bash
bash scripts/score.sh --protocol oracle_state_v4_native_v1 --per-row-only \
  --requests /external/new-oracle-v4/requests_oracle_state_v4.jsonl \
  --responses /external/oracle-run/responses.jsonl --model your-model-id \
  --output /external/new-oracle-scores
```

This writes `native/scored_rows.jsonl`, covering parsed task type, objective,
verdict, explanation-presence flags and their existing automatic scores.
It does not bootstrap or generate statistical tables. Statistical reports
are available when `--per-row-only` is absent.

Select the three-arm builder with
`bash scripts/build_experiment.sh structured-cot ...`; omitting the experiment
name selects the same builder.

### Optional Task C Presentation Profile

Task C keeps its ordinary generation behavior by default. To export matched
presentation requests explicitly, use the same task module:

```bash
bash scripts/_launch.sh tools.minigrid.task_c --profile matched-probes \
  --source /external/taskC_expanded_manifest.jsonl \
  --output-dir /external/new-taskc-probes
```

The source must have complete clean-neutral Full/CF pairs and access to their
raw triplets via `source_raw_root` (or an explicit `--raw-root`). NoCue is not
included. Each pair produces eight requests: Full/CF crossed with clean,
noise, style and temporal reversal. Every condition uses the same selected
frames, action/outcome labels and source prompt; reversal adds an explicit
ordering note and reverses the timestamp labels with the frames. Clean is
exported as a request too, to be evaluated in the same run. The clean pixels
are checked against the raw-frame
storyboard before export. Images, exam rows and requests are written outside
the source tree.

The optional profile uses these transform settings:
Gaussian noise with sigma 20 and deterministic per-frame seeds, and the
edge-preserving style filter. The ordinary Task C profile uses separate mild
noise/color transforms.
`source_uid` and `presentation_arm` identify each condition without altering
the original exam. Use the regular inference and SSOT scoring entries on
the generated `_requests_from_manifest/requests_C.jsonl` and `exam/` directory.

MiniGrid Task A request construction exports both representations by default.
The normal `exam/task_a_exam.jsonl` and `_requests_from_manifest/requests_A.jsonl`
are Composite. `split/exam/task_a_exam.jsonl` and `split/requests_A.jsonl`
contain the matching five-image condition. Both have the same prompt, UID,
action, candidate order and answer; keep their response streams in separate
run directories. Split crops retain the exact panel pixels and dimensions,
without resampling. Each request records `representation` and `source_uid`.
The combined A/C/D or A/B/C/D/E bundle retains only the Composite A arm, so
question counts and response UID uniqueness are unchanged. Run and score
the Split A bundle separately with the same decoding settings.

`scripts/infer.sh` consumes an external request JSONL with `uid`, `prompt`, an
ordered `images` list, and optional `exam` metadata. Use `images: []` for a
text-only condition. Relative image paths resolve from the request file's
directory. Only prompt text and images are sent; answer keys and audit state
in `exam` are retained locally for scoring, not added to the API payload.

| Location | Responsibility |
| --- | --- |
| `scripts/infer.sh` | Repository-independent shell launcher. |
| `tools/inference/run.py` | Input validation, bounded concurrency, sharding, resume, progress and response writing. |
| `tools/inference/client.py` | HTTP transport, shared rate limit and transient-error retries. |
| `utils/inference_payload.py` | Leaf PNG encoding, payload construction and unparsed provider-text extraction. |

Choose `--protocol chat`, `responses`, or `gemini` explicitly. Supply the exact
provider model ID and API base URL; no model aliases or alternate routes are
substituted automatically. Chat/Responses base URLs normally include `/v1`;
Gemini accepts a service root or a root ending in `/v1beta`. This entry covers
remote APIs, not local model-weight inference or native Anthropic messages.

Validate without a key, API calls, or output files:

```bash
bash scripts/infer.sh --requests /external/requests.jsonl \
  --protocol chat --base-url https://your-provider.example/v1 \
  --model your-exact-model-id --max-tokens 8192 \
  --output /external/new-run --dry-run
```

For a real run, set `VLM_API_KEY` in your environment, or pass an external
`--env-file` and the variable name with `--api-key-env`. Never place credentials
in CLI arguments, request bundles or provider extra-body JSON.

```bash
bash scripts/infer.sh --requests /external/requests.jsonl \
  --protocol chat --base-url https://your-provider.example/v1 \
  --model your-exact-model-id --max-tokens 8192 --temperature 0 \
  --workers 2 --rps 1 --output /external/new-run

bash scripts/score.sh --protocol historical_ssot_v1 \
  --exam-dir /external/exam --responses /external/new-run/responses.jsonl \
  --output /external/new-scores
```

Decoding parameters must match the experiment configuration: the example
budget is not a universal setting. Temperature and seed are omitted
unless explicitly supplied. Images use lossless RGB PNG, without resizing by
default; `--max-side` records an explicit resize. Per-row `api_content_order`
and `api_image_detail` override the corresponding CLI defaults. Split images
retain their order and are never automatically fused. Provider-specific options
may be supplied through `--extra-body /external/options.json`, but cannot
override model, prompt/images, token budget, temperature, seed or streaming.
Environment proxy settings are ignored unless `--trust-env` is supplied.

Each output directory contains `responses.jsonl`, `run_config.json`, and
`run_summary.json`. Response `raw`/`pred`, `meta.ok`, `meta.exam`, finish reason
and usage follow the offline scorer contracts; complete provider JSON and retry
attempts are also retained. Transport/429/selected 5xx errors receive bounded
retries; authorization, quota-denied and model-name errors are not retried
indefinitely. Truncated, refused, malformed or incorrect answers are never
automatically retried to improve scores. Provider finish reasons are preserved.
Runs containing API errors exit with code 2 and keep all records for inspection.

Repeat the same command with `--resume` after an interruption. Request/image
hashes and effective configuration must match; changing only the API key is
allowed. All saved UIDs are skipped, including saved API errors. Failed rows
must be rerun as an explicit, separate request subset/output; they are not
silently replaced. Duplicate/unexpected IDs and incomplete JSONL tails fail
without truncating the file. Only one writer may own an output directory.
For independent processes, use `--num-shards N --shard-id I` and distinct output
directories; merge only disjoint response shards before scoring.

## Offline Scoring

Add `--per-row-only` to write per-question results to `native/scored_rows.jsonl`,
together with the scoring receipt and log. Result fields follow the selected
task protocol. This mode checks UID coverage and applies the protocol's parsing
and correctness rules. Bootstrap overrides cannot be combined with
`--per-row-only`; omit it to generate the protocol's standard reports.

Scoring implementations are in `tools/scoring/`. Supply external exam and
response files, and select the task protocol with `--protocol`. Each protocol
defines its parser, scoring denominator and statistical defaults.

| Protocol | Inputs | Parsing And Statistics |
| --- | --- | --- |
| `historical_ssot_v1` | MiniGrid A/B/C/D/E exam directory and response JSONL | SSOT formatting recovery; B weights, denominator .90 and Acc threshold .90. Optional A/C/D cluster bootstrap with `--ci`. |
| `miniworld_a_native_v1` | A exam JSONL and responses | Native successor-option parser and metrics. |
| `miniworld_c_native_v1` | C exam JSONL and responses | Native verdict parser and metrics. |
| `miniworld_d_native_v1` | Trajectory D exam JSONL and responses | Native trajectory-D parser, distinct from reference-family D. |
| `miniworld_reference_native_v1` | Reference-family D exam JSONL and responses | Native reference/query D parser and metrics. |
| `structured_native_v1` | Three-arm requests, responses and model ID | Native strict/recoverable parsing; 20,000 bootstrap iterations, seed 20260724. |
| `oracle_state_v4_native_v1` | Oracle-state v4 requests, responses and model ID | Native four-field parsing/recovery; 20,000 bootstrap iterations, seed 20260725. |
| `habitat_native_v1` | Exam directory with answer_key.json, responses and model ID | Last-option/last-verdict parser with explicit missing/error handling. |

```bash
bash scripts/score.sh --protocol historical_ssot_v1 \
  --exam-dir /external/exam --responses /external/responses.jsonl \
  --output /external/new-scores

bash scripts/score.sh --protocol historical_ssot_v1 \
  --exam-file /external/task_d_r_exam.jsonl --exam-task D \
  --responses /external/requests_D.jsonl --output /external/new-task-dr-scores

bash scripts/score.sh --protocol miniworld_reference_native_v1 \
  --exam-file /external/task_d_exam.jsonl \
  --responses /external/responses.jsonl --output /external/new-mw-d-scores

bash scripts/score.sh --protocol structured_native_v1 \
  --requests /external/requests_structured_cot_ablation.jsonl \
  --responses /external/responses.jsonl --model gpt-5.4 \
  --output /external/new-cot-scores

bash scripts/score.sh --protocol habitat_native_v1 \
  --exam-dir /external/habitat-exam --responses /external/responses.jsonl \
  --model gpt-5.4 --output /external/new-habitat-scores
```

MiniGrid Task D uses a successful reference trajectory and a query
trajectory. D-R is an alias for this reference/query design. Both use the same SSOT
scorer: `--exam-task D`, `DR`, and `D-R` select the same task. With `--exam-dir`,
either `task_d_exam.jsonl` or `task_d_r_exam.jsonl` is accepted.
If both filenames exist, explicitly select one using `--exam-file --exam-task D`.
Filename aliases are created only in the external runtime output; original
exam bytes and response UIDs are never rewritten. The response filename alone
does not identify its matching exam: `DR.*` responses require
the corresponding `DR.*` exam rows, not unrelated newly generated questions or `D.*`
single-trajectory questions. Missing predictions retain the selected native
scorer's denominator and handling.
Duplicate or unexpected response IDs are rejected before scoring;
resolve retry selection explicitly rather than silently taking the last row.
Supply one response JSONL per invocation, merging only disjoint shards first.

Results are in `native/` using the native report formats. The external runtime
output also contains `scoring_receipt.json` with the protocol, effective
parameters, coverage and input/source hashes, and `scoring.log`. Existing
output directories are never overwritten. Model IDs affect native statistical
seeds; use the actual model ID. Changing `--iterations` or `--seed` for the
structured/oracle protocols is recorded and is not an exact-default replay.

Strict-only sensitivity analyses are not silently substituted for these
protocols: several native scorers deliberately accept recoverable answers.
Add `--ci` to the SSOT scoring command for base MiniGrid or Scale-2 A/C/D:

```bash
bash scripts/score.sh --protocol historical_ssot_v1 \
  --exam-dir /external/exam --responses /external/responses.jsonl \
  --ci --iterations 20000 --seed 20261103 --output /external/scores-with-ci
```

This writes `native/confidence_intervals.json` and
`native/confidence_intervals.csv` alongside the native scores. A/C/D accuracy
uses every gold question, including missing responses and API/parse failures
under the native scoring rules. C/D also report accuracy by variant and
complete-triplet diagnostics: `p_active` is the fraction predicting Success
on Full; `IDR_raw` is the fraction predicting Success on Full and Fail on CF;
`IDR_cond` divides that joint fraction by `p_active`.

Both 95% percentile intervals resample whole clusters: group bootstrap keeps
all questions/styles/variants of an `(environment, group_id)` together;
environment bootstrap keeps each environment intact. Estimates are pooled
ratios weighted by question counts, not unweighted environment averages.
Diagnostics match Full/NoCue/CF within environment, group, temporal style,
visual style, framing and reference identity; incomplete triplets are counted
separately and excluded only from these paired diagnostics. Zero-active conditional replicates are counted as
undefined. Reports flag slices with fewer than two clusters. Defaults are
20,000 iterations and seed 20261103; `--ci` cannot be combined with
`--per-row-only`. Score each model and representation separately using its
matching exam and response file.

## Run

Activate the core environment or set `PYTHON_BIN` explicitly:

```bash
bash scripts/generate_and_build.sh --datasets minigrid --profile smoke \
  --seed 930000 --output /external/new-generation

bash scripts/generate_and_build.sh --datasets minigrid --profile formal \
  --output /external/new-minigrid-formal

bash scripts/generate_and_build.sh --datasets scale2 --profile formal \
  --output /external/new-scale2

bash scripts/generate_miniworld.sh --profile formal \
  --output /external/new-miniworld

bash scripts/generate_habitat.sh --profile smoke \
  --habitat-python /path/to/habitat-python --habitat-assets /path/to/hssd-hab \
  --output /external/new-habitat
```

Use `--datasets scale2 --profile smoke` to generate a small A/C/D exam set and
Task A Composite/Split requests. Use `--datasets scale2 --profile formal` for
the formal configuration with A1030/C1326/D1296 questions.

Habitat defaults to software rendering. On WSL with working NVIDIA D3D12
graphics, add `--habitat-renderer d3d12` to `generate_habitat.sh`. This uses
system Mesa and the WSL graphics libraries, and verifies the NVIDIA D3D12
renderer in navigation and interaction logs rather than silently accepting
software fallback. Native Linux NVIDIA EGL is selected with
`--habitat-renderer gpu`; it is a different backend from WSL D3D12.
Rendering acceleration does not move path sampling or PNG encoding to CUDA.
Use a fresh output directory when changing renderer: `--resume` requires the
original seed, profile and renderer to avoid mixing rendering backends.

Generation-level `--resume` retains validated completed stages for MiniGrid,
Scale-2, MiniWorld and Habitat. Repeat the exact generation command with
`--resume`: the dataset list, seed, profile, interpreter paths, renderer,
source/configuration fingerprints and completed artifact hashes must match.
A per-output lock rejects concurrent generators. MiniWorld formal generation
checkpoints each native builder separately, in the original order.

This is stage-level recovery, not universal sample-level recovery. An incomplete
MiniGrid/Scale-2/MiniWorld stage that has already written partial artifacts is
preserved and rejected with an actionable error; use a new output root rather
than deleting or silently overwriting its files. Habitat's native episode/scene
recovery remains available. Stale stage records are marked `INTERRUPTED`.
Runs created before these fingerprints/checkpoints cannot be resumed automatically.
Completed datasets are hash-checked and skipped without regeneration.
API inference has its separate UID-based `--resume` contract described above.
Scoring and supplemental exam builders still require fresh output directories.

Choose a new `--output` directory outside the repository. MiniGrid `smoke`
uses a small expanded-bank cohort; `formal` uses the base task builders.
Task A uses action-matched, state-ranked distractors, with `--exam-seed 42`
and `--task-a-per-traj 6` by default. The formal entry requires 103 questions
and reports an error if the selected source cohort yields a different count.
Use `--task-a-cohort` to provide a JSONL of generated UIDs in the desired order.

Formal base Task A generates its own three Full trajectories per environment
(18 total), separately from the expanded bank. Sampling precedes final candidate
assembly so candidate deduplication cannot shift later answer permutations.
The default exam seed is 42 with six sampled transitions per trajectory;
source generation uses the native base-cohort environment and seed-start settings.
Use `--task-a-raw-root` to explicitly supply an alternative six-environment source
cohort.

Task B generates 144 Full-only questions by default, selecting 24 pixel-distinct
frames per environment across trajectory groups. Answers describe the visible
simulator observations. Formal generation uses its generated trajectory pool;
use `--task-b-raw-root` to select another pool. Generation requires enough
distinct Full frames to meet the requested quota. The standalone builder uses
the same defaults:

```bash
python -m tools.minigrid.task_b --root /external/full-trajectory-pool \
  --out-dir /external/taskb-exam --seed 42
```

Use `--per-environment` for an explicitly smaller exam, or `--b-per-traj` for
variable-size per-trajectory sampling. The smoke profile remains a small test.

Task C selects six complete groups per KeyCorridor, MultiRoom and RedBlueDoor
environment. It uses shared event-anchored variable-keyframe indices and near-square
storyboards, with six temporal/visual conditions plus three framing overlays on
original-clean evidence: 486 rows. Task D uses one fixed successful reference and
two query groups per environment, each with Full/NoCue/CF and clean-neutral,
clean-positive, clean-negative and style-neutral conditions: 144 rows.
This base D protocol is separate from Scale-2 exact-mission reference matching.
Task E generates 24 successful trajectories in each of UnlockPickup and
BlockedUnlockPickup and constructs identity/status questions at every valid
event checkpoint. The question count depends on the events in those trajectories.
Formal Scale-2 retains A1030/C1326/D1296; native
formal MiniWorld retains A80/C84/D50; formal Habitat targets A150/C200/D200.
Habitat Task A renders each candidate from the exact same agent pose, using
forward-2/3/4 and left/right-turn action branches. Each question has four
distinct action/distance outcomes; every candidate pair must exceed the 0.20
changed-pixel fraction at an RGB-channel difference threshold of 24. The
pipeline writes these candidates under `action_candidates/` before selecting
the final exam. The lower-level `build_candidates.py` state-context pool is
not the final Task A exam protocol.
The default runtime seed is 830000. Task E uses its native
deterministic search and v1.3 construction, including native UID prefixes.

Generation produces images, trajectory records, native exam JSON/JSONL,
answer files and request bundles where supported. A runtime
`generation_status.json` records stage progress and failures. Existing native
quality metadata is used by downstream selection and assembly; it is
generated outside the source tree, not shipped.

## Dependencies

Use the separate core and Habitat environments described in `environments/`
and `requirements/`. HSSD scene assets remain external. No upstream simulator
source is copied into this stage. All project code calls resolve internally.
Keep these environments separate: Habitat-Sim 0.3.3 requires Pillow 10.4.0,
whereas the core environment uses Pillow 12.1.1. Do not install the core
requirements into the Habitat environment. Run `python -m pip check` in each
environment after installation.

```bash
conda env create -f environments/core.yml
conda run -n gridwm-repro-core python -m pip install -r requirements/core.lock.txt
conda env create -f environments/habitat-linux.yml
conda run -n gridwm-repro-habitat python -m pip install -r requirements/habitat.lock.txt
conda run -n gridwm-repro-core python -m pip check
conda run -n gridwm-repro-habitat python -m pip check
```

These specifications pin direct dependencies; they are not a complete
transitive, platform-independent lockfile. OpenGL/EGL/GLU runtime libraries,
working graphics drivers and externally licensed HSSD assets are host
prerequisites, not installed by pip. WSL D3D12 rendering additionally requires
a working WSLg graphics stack.

Run launchers from any working directory with `PYTHON_BIN` pointing to the
core interpreter. Habitat uses its separate interpreter through
`--habitat-python`.

Optional Windows fonts are discovered from `WINDIR` and mounted Windows font
directories. Set `GRIDWM_FONT_DIR` to supply an explicit font directory.
Installed Linux fonts and Pillow's default font provide fallbacks. Font choice
affects text rendering; use the same font files when matching rendered images.
Temporary launcher directories use `TMPDIR`; paths containing a colon fall
back to the system default temporary directory for Python import compatibility.

Native MiniWorld A generators randomize candidate A/B placement without
fixing Python's global random seed. Correct/distractor content is preserved,
but repeated runs need not produce the same answer letter.
The `correct_candidate_agent_*` and `distractor_candidate_agent_*` metadata
describe the semantic action branches, independent of A/B placement. Swapping
the displayed options changes their paths and answer letter, not those poses.
