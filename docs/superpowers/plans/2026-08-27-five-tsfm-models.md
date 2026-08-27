# 五模型 TSFM-only 重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把仓库重构为 TSFM-only，迁移现有两个 backend 到 `models/`，新增 Chronos2、TiRex、TimesFM，统一 `run.py` 和脚本，并诚实验证五模型能力与当前环境限制。

**Architecture:** `models/registry.py` 是模型身份和 capability 的唯一来源；factory 懒加载每个独立模型文件；runner 只消费统一 tensor/backend 契约。TiRex 仅 zero-shot，其余四模型支持 zero-shot、LoRA、full、last-layer。实验矩阵按 capability 生成 68 项。

**Tech Stack:** Python 3.8.18（基础测试环境）、PyTorch 2.3.1、Transformers/PEFT、Chronos2、TiRex、pytest、Bash；新模型真实运行需要 Python 3.10+ modern optional environment。

**Spec:** `docs/superpowers/specs/2026-08-27-five-tsfm-models-design.md`

## Global Constraints

- 生产实现全部由 luna_worker 完成；controller 只提供设计、任务边界和审查。
- 使用 `/opt/data/private/penv/time/bin/python` 运行代码级测试；不得修改或自动安装到外部环境。
- 保留 `task.md`、`时序基础模型.md` 及用户未提交改动，不修改参考仓库。
- `models/` 只包含 TSFM；最终不存在 `foundation_models/`、纯时序模型、baseline 或它们的支撑代码。
- 五个 checkpoint/revision、supported modes、last-layer selector 以 spec 表格为准；registry/factory/list 操作必须 lazy 且离线安全。
- 输入/输出严格为 `[B,L,1]` / `[B,H,1]`，训练 loss 必须 scalar、有限、requires-grad 并尊重 `target_mask`。
- TiRex 只支持 `zero_shot`，unsupported mode 必须在导入包/下载权重前失败。
- 不允许用 mock/fake 结果宣称真实 checkpoint 可运行；当前 Python 3.8 incompatibility 必须由预检和 README 明示。
- 采用严格 TDD：每个行为先运行聚焦测试观察预期 RED，再写最小实现，最后运行聚焦测试和一次全套测试；报告保存 RED/GREEN 命令与输出。

---

### Task 1: 迁移现有 TSFM 核心并清除纯时序代码

**Ownership:** `models/` 中现有 Sundial/TimeMoE 与共享核心，`data_provider/power_only.py`，`utils/artifacts.py`，纯时序/baseline/layers 文件及对应旧测试/脚本/文档的删除。不要改新三模型、runner CLI、foundation 环境/批处理脚本。

**Files:**
- Create: `models/base.py`, `models/common.py`, `models/factory.py`, `models/registry.py`, `models/tasks.py`, `models/trainability.py`, `models/Sundial.py`, `models/TimeMoE.py`, `utils/artifacts.py`
- Modify: `models/__init__.py`, `data_provider/power_only.py`
- Modify tests: `tests/test_foundation_{registry,tasks,backends,trainability}.py`, `tests/test_power_only_dataset.py`
- Delete: `foundation_models/`, `baselines/`, `layers/`, eight pure model files, `models/tslib_*`, `run_time_series.py`, pure-time-series scripts/tests/docs and obsolete 2026-08-26 plans/specs

**Interfaces:** preserves `build_backend`, `get_model_spec`, `iter_experiment_tasks`, `configure_trainable`, `power_only_batch`, artifact helpers; registry entrypoints become `models.Sundial:SundialBackend` and `models.TimeMoE:TimeMoEBackend`.

- [ ] **Step 1 — RED:** first change focused imports/tests to the desired `models.*`, `data_provider.power_only` and `utils.artifacts` boundaries; run `/opt/data/private/penv/time/bin/python -m pytest -q tests/test_foundation_registry.py tests/test_foundation_tasks.py tests/test_foundation_backends.py tests/test_foundation_trainability.py tests/test_power_only_dataset.py` and record the expected missing-module/import failure.
- [ ] **Step 2 — minimal migration:** move shared validation/trainability/factory/task behavior without changing observable Sundial/TimeMoE forecasts, training losses, revisions or artifact serialization. Split the old 613-line backend by model; do not duplicate shared validation blocks.
- [ ] **Step 3 — cleanup:** migrate `PowerBatch/power_only_batch` and the three artifact helpers, then delete every listed pure/classical file. Do not retain a `foundation_models` forwarding package.
- [ ] **Step 4 — GREEN:** run the focused command above. Each retained test must exercise a consumer-visible contract; remove tests whose only purpose was pure models or source-text presence.
- [ ] **Step 5 — audit:** run `git status --short`, module import smoke, and a filesystem scan proving the prohibited implementation files are absent. Do not add `task.md` or the modified `时序基础模型.md` to the commit.
- [ ] **Step 6 — full suite and commit:** run `/opt/data/private/penv/time/bin/python -m pytest -q`, address any regression in this task's scope, then commit `refactor: move foundation models into models package`.

### Task 2: 新增 Chronos2、TiRex、TimesFM 与 capability 契约

**Ownership:** `models/Chronos2.py`, `models/TiRex.py`, `models/TimesFM.py` and capability-related edits in `models/{registry,tasks,trainability,common,base}.py`; related model tests only. Do not edit runner/scripts/docs/dependency files.

**Files:**
- Create: `models/Chronos2.py`, `models/TiRex.py`, `models/TimesFM.py`
- Modify: `models/registry.py`, `models/tasks.py`, `models/trainability.py`, optionally `models/common.py` / `models/base.py`
- Modify tests: `tests/test_foundation_registry.py`, `tests/test_foundation_tasks.py`, `tests/test_foundation_backends.py`, `tests/test_foundation_trainability.py`

**Interfaces:** `MODEL_NAMES == ("Sundial", "TimeMoE", "Chronos2", "TiRex", "TimesFM")`; `get_model_modes(name)` and `validate_model_mode(name, mode)`; task iterator yields 68 unique supported tasks.

- [ ] **Step 1 — RED catalog/capabilities:** add literal, hand-derived expectations for all five model specs, their pinned identities/selectors/modes, early TiRex rejection and 68-task ordering/count. Run registry/task tests and capture failures before implementation.
- [ ] **Step 2 — GREEN catalog/capabilities:** implement immutable capability fields and make task generation iterate `spec.supported_modes`. Unknown model/mode errors list valid choices. Imports of registry/tasks must leave `chronos`, `tirex` and TimesFM-specific modules unloaded.
- [ ] **Step 3 — RED backends:** using model-specific complete fakes at the external loader boundary, write failures that catch wrong model ID/revision/device, wrong point-output extraction, wrong horizon crop, NaN/wrong shape, eager optional imports and unsupported training calls.
- [ ] **Step 4 — implement inference:** Chronos2 calls the official pipeline point prediction, TiRex calls `forecast(context=..., prediction_length=...)`, TimesFM calls `TimesFm2_5ModelForPrediction(...).mean_predictions`. Keep optional imports inside loader functions and issue directed install/Python-floor errors.
- [ ] **Step 5 — RED then GREEN training:** Chronos2 native forward must receive context, target, binary mask and enough output patches; TimesFM must produce a differentiable point forecast and masked MSE; both losses pass scalar/finite/gradient validation. Add optimizer-step tests proving full/last-layer/adapter change only expected state. TiRex training remains rejected.
- [ ] **Step 6 — trainability:** allow per-spec LoRA targets, including modern `all-linear`, while retaining base-parameter leak audits and strict last-layer subtree matching.
- [ ] **Step 7 — verification and commit:** run the four focused model test files, then full pytest once. Commit `feat: add Chronos2 TiRex and TimesFM backends` and write the report with RED/GREEN evidence.

### Task 3: 统一 run.py、能力感知双 GPU 编排和简单训练脚本

**Ownership:** `run.py`, removal of `run_foundation_model.py`, all retained foundation runner/script/environment code and their tests. Do not edit model implementations, README, requirements or design docs.

**Files:**
- Create: `run.py`, `scripts/train_model.sh`
- Modify/rename as appropriate: `scripts/check_foundation_environment.py`, `scripts/run_all_foundation_models_2gpu.sh`, `scripts/smoke_all_foundation_models_2gpu.sh`
- Modify tests: `tests/test_foundation_runner.py`, `tests/test_foundation_scripts.py`, `tests/test_foundation_environment.py`
- Delete: `run_foundation_model.py`

**Interfaces:** canonical runner flags plus the six normalized Time-Series-Library-style aliases from the spec; four stable artifacts; 68 supported tasks; two process-level GPU queues.

- [ ] **Step 1 — RED runner:** change tests to import `run.py`; add tests for five-model list output, per-model early capability rejection, alias normalization/conflicts, `task_name/is_training` consistency, and `debug` boolean parsing. Run `tests/test_foundation_runner.py` and record RED.
- [ ] **Step 2 — runner migration:** make `run.py` use `models.*`, `data_provider.power_only` and `utils.artifacts`; preserve checkpoint/completion schemas and historical output directory layout. Validate model/mode before `build_backend`, and remove completion marker on every failure path.
- [ ] **Step 3 — runner GREEN:** run zero-shot and three training-mode fake integrations, checking loaders, optimizer, best-state restore, metrics identity and artifacts. Ensure TiRex training fake is never constructed.
- [ ] **Step 4 — RED scripts:** replace hard-coded 2/32/33 assertions with behavior derived from a fake runner and registry/task output. Tests must execute scripts in a controlled harness without copying the whole repository or inspecting source text; assert 68 unique supported invocations, serial work per GPU, cross-GPU concurrency, resume validity and failure propagation.
- [ ] **Step 5 — scripts GREEN:** update the two-GPU scripts so all counts derive from emitted tasks. Add simple `scripts/train_model.sh` in the requested shell style; its debug branch performs one TimesFM adapter smoke task, normal branch loops the four real dataset/window combinations and invokes `run.py` with normalized aliases plus explicit `--mode adapter`.
- [ ] **Step 6 — environment check:** make dependency/cache checks registry-driven for all five models. On current Python 3.8, Chronos2/TiRex/TimesFM must be reported as blocked without traceback, token output or model import; catalog-only checks still pass.
- [ ] **Step 7 — verification and commit:** run runner/script/environment focused tests, `bash -n` on all retained shell scripts, a fake-runner 68-task smoke, and full pytest once. Commit `feat: add capability-aware TSFM runner and scripts`.

### Task 4: 依赖、TSFM-only README 与最终清理

**Ownership:** `README.md`, requirements files, obsolete foundation docs/plans/specs and any stale naming/import cleanup. Do not change model or runner behavior except for a test-proven documentation/packaging defect discovered here.

**Files:**
- Rewrite: `README.md`
- Modify: `requirements.txt`
- Create: `requirements-modern.txt`
- Delete: `FOUNDATION_MODELS.md`, `PURE_TIME_SERIES.md`, obsolete/conflicting old superpowers plan/spec files

**Interfaces:** one authoritative README with model table, capabilities, task counts, install/preflight/download/run commands, artifacts and limitations.

- [ ] **Step 1 — dependency cleanup:** remove `einops`/`pvlib` only after confirming no retained import. Keep a Python-3.8-compatible base test/runtime set; put modern-model requirements and Python floor in `requirements-modern.txt` with bounded/traceable versions or revisions.
- [ ] **Step 2 — README:** document the five pinned model IDs/revisions, TiRex zero-shot-only status, 68-task matrix derivation, two datasets/windows, `run.py` canonical and alias examples, simple/2-GPU scripts, cache/download flow, artifacts/resume and current Python 3.8 limitation. Do not claim real smoke that was not run.
- [ ] **Step 3 — cleanup scan:** remove duplicate/stale docs and scan retained Python/shell/tests for imports of `foundation_models`, baselines, pure model names, `run_time_series` and obsolete 32-task constants. Distinguish the intentionally retained result-path string from package imports.
- [ ] **Step 4 — verification and commit:** run full pytest, `run.py --list-models`, a task-count command, all `bash -n` checks and current-environment preflight. Commit `docs: document five-model TSFM workflows` without staging user-owned files.

## Plan Self-Review

- Task 1 creates the final TSFM-only package boundary before new code depends on it; Task 2 adds models without runner/script conflicts; Task 3 integrates behavior; Task 4 owns prose/packaging only.
- Every production behavior has a named RED test and hand-derived expected contract. File absence and human prose are verified by delivery audits, not brittle source-text unit tests.
- The plan preserves existing artifact semantics and user files, while removing only code demonstrably unrelated to TSFM.
- The unavoidable Python 3.8/upstream incompatibility is a first-class preflight result, not hidden behind mocks or automatic environment mutation.

