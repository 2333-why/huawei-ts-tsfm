# Real-Weight CUDA Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to execute this plan task by task. If a real run fails, stop mutation work and return the exact failure to the controller; the controller must load `superpowers:systematic-debugging` and `superpowers:test-driven-development` before dispatching a focused fix.

**Goal:** Provision a reproducible modern runtime without modifying the user's existing environments, cache all five exact pinned checkpoints, and prove the repository's complete 68-task smoke matrix runs with real weights on both local CUDA GPUs.

**Architecture:** Keep `/opt/data/private/penv/time` as the already-satisfied Python 3.8 legacy profile. Seed a clean Python 3.11 virtual environment at `.venv/tsfm-modern` from `/opt/conda/envs/pcd/bin/python`, but do not inherit or mutate `pcd`. Put the Hugging Face cache under the already-ignored `checkpoints_huggingface/` tree and all run artifacts under the already-ignored `results_real_weight_smoke/` tree. Use an exact, driver-compatible package set for this verification, then force offline model loading during every smoke command so no run can silently replace a pinned revision.

**Tech Stack:** Python 3.11, PyTorch 2.4.1 with CUDA 12.1 runtime, Transformers 5.3.0, PEFT 0.18.1, Chronos Forecasting 2.3.1, TiRex 1.4.2, Hugging Face Hub, two RTX 4090 GPUs, pytest, Bash.

**Spec:** `task.md`, `README.md`, `models/registry.py`, and `scripts/smoke_all_foundation_models_2gpu.sh`.

## Global Constraints

- Preserve the user's dirty `时序基础模型.md` and untracked `task.md`; never stage, edit, delete, or overwrite either file.
- Do not mutate `/opt/data/private/penv/time` beyond an idempotent `pip install -r requirements.txt`, and do not mutate `/opt/conda/envs/pcd`; it is only the Python 3.11 executable used to create the clean venv.
- Never print or persist Hugging Face credentials. All five repositories are public; do not request a token.
- Use `HF_HOME=$PWD/checkpoints_huggingface` consistently. After download, every preflight and smoke run must additionally set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Load only the registry identities and immutable revisions below:
  - `thuml/sundial-base-128m@3212e42564493f520593e5414af4367fc4b49226`
  - `Maple728/TimeMoE-50M@446753ee48ff3726d0606a81d0092d54acee995e`
  - `amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c`
  - `NX-AI/TiRex@63c740922493f5fbe60b277609ec62babfba2762`
  - `google/timesfm-2.5-200m-transformers@5a9806b9b291fad9233b5249d88263f1846304d3`
- The minimal gate is five real zero-shot runs, one per model. Completion requires the repository's canonical two-GPU wrapper to emit 68 unique `PASS` rows; a smaller matrix is diagnostic evidence only.
- A successful command is not sufficient by itself. Verify package identity, CUDA availability, exact cached revisions, offline preflight, summary cardinality, artifact hashes, pinned model identity, step counts, and finite predictions.
- Runtime caches, virtual environments, and smoke results stay ignored. Only focused compatibility fixes, their tests, this plan, and the final truthful README evidence may be committed.

### Task 1: Provision and audit the two dependency profiles

**Files:**
- Verify: `requirements.txt`
- Verify: `requirements-modern.txt`
- Create (ignored runtime state): `.venv/tsfm-modern/`
- Create (ignored evidence): `results_real_weight_smoke/environment/`

**Step 1: Record the immutable baseline**

Run `git status --short --branch`, `git rev-parse HEAD`, both interpreters' versions, `nvidia-smi`, and disk availability. Confirm only the two known user-owned worktree entries are dirty before continuing.

**Step 2: Confirm the specified legacy profile is installed**

Run:

```bash
/opt/data/private/penv/time/bin/python -m pip install -r requirements.txt
/opt/data/private/penv/time/bin/python -c 'import torch, transformers, peft; print(torch.__version__, torch.version.cuda, transformers.__version__, peft.__version__)'
```

Expected: the install is already satisfied; Torch is `2.3.1+cu118`, CUDA is available, Transformers is `4.46.2`, and PEFT is `0.13.2`. Record but do not repair unrelated pre-existing `pip check` conflicts.

**Step 3: Create the clean modern profile**

Run:

```bash
/opt/conda/envs/pcd/bin/python -m venv .venv/tsfm-modern
.venv/tsfm-modern/bin/python -m pip install --upgrade pip setuptools wheel
.venv/tsfm-modern/bin/python -m pip install \
  -r requirements-modern.txt \
  'torch==2.4.1' \
  'transformers==5.3.0' \
  'peft==0.18.1' \
  'chronos-forecasting==2.3.1' \
  'tirex-ts==1.4.2'
```

The exact additions constrain the broader repository ranges to a mutually compatible set and avoid a newer CUDA wheel that the installed NVIDIA 525 driver may not support.

**Step 4: Verify the modern environment independently**

Run `pip check`, import every required backend package, import `TimesFm2_5ModelForPrediction`, and assert:

```text
Python == 3.11.x
torch == 2.4.1
transformers == 5.3.0
peft == 0.18.1
chronos-forecasting == 2.3.1
tirex-ts == 1.4.2
torch.cuda.is_available() == True
torch.cuda.device_count() == 2
```

Write the command output and `pip freeze` under `results_real_weight_smoke/environment/`. Do not continue if `pip check` fails or a CUDA tensor cannot be allocated separately on device 0 and device 1.

### Task 2: Cache and verify every exact checkpoint revision

**Files:**
- Read: `models/registry.py`
- Read: `scripts/check_foundation_environment.py`
- Create (ignored runtime state): `checkpoints_huggingface/`
- Create (ignored evidence): `results_real_weight_smoke/downloads/`

**Step 1: Download complete snapshots, not selected files**

With `HF_HOME=$PWD/checkpoints_huggingface`, run `.venv/tsfm-modern/bin/hf download` once for each of the five model/revision pairs in Global Constraints. Capture each resolved snapshot path and the CLI exit status without logging credentials.

**Step 2: Prove the cache is exact and complete**

For each repository, resolve the Hub cache `snapshots/<revision>` directory, require that it is a real directory, and require the preflight's named files. Also reject a snapshot path whose terminal revision differs from the registry revision. Keep full snapshots so backend-specific metadata and sharded-weight indexes remain available.

**Step 3: Run the repository's offline preflight**

Run:

```bash
HF_HOME="$PWD/checkpoints_huggingface" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
.venv/tsfm-modern/bin/python scripts/check_foundation_environment.py --offline --json
```

Expected: exit `0`; all five models report dependency-ready and checkpoint-ready, both Parquet datasets exist, and both CUDA GPUs are available. Save the JSON under `results_real_weight_smoke/environment/preflight.json`.

### Task 3: Run a five-model real-weight CUDA gate

**Files:**
- Execute: `run.py`
- Read: `models/*.py`
- Create (ignored artifacts): `results_real_weight_smoke/gate/`

**Step 1: Launch one real zero-shot smoke per model**

Use the `skippd_luoyang`, `seq_len=48`, `pred_len=1` row. Run Sundial, Chronos2, and TimesFM serially on physical GPU 0; run TimeMoE and TiRex serially on physical GPU 1. Each command must set the three offline/cache variables, set `CUDA_VISIBLE_DEVICES` to its physical ordinal, pass `--device cuda:0`, `--mode zero_shot`, `--smoke`, and a distinct directory under `results_real_weight_smoke/gate/`.

**Step 2: Inspect the gate artifacts**

For each model, require `best.pt`, `predictions.csv`, `metrics.json`, and `completion.tsv`; validate the three recorded SHA-256 values; require the exact model ID/revision; require `run_mode=smoke`, `epochs=0`, `train_steps=0`, `val_steps=0`, `test_steps=1`; and require all serialized predictions to be finite. Confirm the logs show real backend construction and no network access or fake/test backend.

**Step 3: Stop on the first real incompatibility**

Do not paper over warnings or retry with an unpinned revision. Return the exact command, package versions, traceback, and last GPU-memory snapshot to the controller. A focused fix must begin with a reproducing failing test and retain exact identity/offline guarantees.

### Task 4: Run the canonical complete two-GPU smoke matrix

**Files:**
- Execute: `scripts/smoke_all_foundation_models_2gpu.sh`
- Execute: `scripts/run_all_foundation_models_2gpu.sh`
- Create (ignored artifacts): `results_real_weight_smoke/full/`

**Step 1: Launch the official wrapper**

Run from the repository root with:

```bash
HF_HOME="$PWD/checkpoints_huggingface" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHON="$PWD/.venv/tsfm-modern/bin/python" \
CONFIG_PYTHON="$PWD/.venv/tsfm-modern/bin/python" \
GPUS="0 1" \
RESUME=0 \
OUTPUT_ROOT="$PWD/results_real_weight_smoke/full" \
SUMMARY_PATH="$PWD/results_real_weight_smoke/full/smoke_summary.tsv" \
bash scripts/smoke_all_foundation_models_2gpu.sh
```

Use a managed background job because this command may outlast one interactive turn. Keep both queues alive until completion; do not start unrelated GPU work.

**Step 2: Validate all 68 outcomes**

Require exactly 68 unique data rows plus the header, exactly 68 `PASS` statuses, no `FAIL` statuses, launch GPU values drawn from both `0` and `1`, and the expected set of 4 dataset/window settings × 17 supported model/mode combinations. Re-run the script with `RESUME=1` and require all 68 validated artifacts to resume without executing model work.

**Step 3: Deep-check every artifact set**

For every row, validate its completion manifest, SHA-256 hashes, data fingerprint, exact model ID/revision, lifecycle counts, trainability mode, finite prediction rows, and output directory identity. For adapter/full/last-layer runs require `(train_steps, val_steps, test_steps) == (1, 1, 1)`; for zero-shot require `(0, 0, 1)`. Verify each trainability report is non-empty where training is supported and zero-shot exposes no trainable parameters.

### Task 5: Regression verification and truthful handoff

**Files:**
- Modify only after real success: `README.md`
- Verify: `tests/`
- Preserve: `时序基础模型.md`, `task.md`

**Step 1: Run regression suites**

Run the full repository suite with the specified legacy interpreter and again with the modern interpreter. Both runs must pass. Also rerun the modern offline preflight after the full matrix to prove no cache drift.

**Step 2: Update the stale environment-limitation evidence**

Replace only README's obsolete claim that modern real checkpoint/CUDA smoke was not run. Record the verified date, exact modern package versions, five pinned revisions, two-GPU CUDA execution, and `68/68 PASS`. Keep the Python 3.8 limitation explicit: it remains the legacy profile and cannot host all five modern backends.

**Step 3: Audit repository integrity**

Confirm no checkpoint, venv, result, log, credential, dataset, or cache file is tracked or staged. Confirm the two user-owned dirty paths are byte-for-byte untouched. Review any focused compatibility diff, commit only scoped source/tests/README/plan files, and run `git status --short --branch` plus `git diff --check`.

**Step 4: Report evidence**

Report the modern interpreter path, package/CUDA versions, cache root, five exact revisions, preflight status, five-model gate status, canonical matrix status, resume status, full-test counts, result-summary path, commits, and any remaining non-blocking warnings. Do not claim completion unless every acceptance criterion above is evidenced by fresh command output.

**Step 5: Send the requested completion email**

Only after every acceptance criterion is proven, use the connected Gmail app to send one message to `757425789@qq.com`. Use subject `tsfm-ts 真实权重 CUDA smoke 已完成` and summarize the verified environment, `68/68 PASS`, test count, and local summary path. Do not send a success email for partial progress or a failed run.
