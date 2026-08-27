# Foundation Models：独立运行与复现实验指南

本文档说明 `Sundial` 和 `TimeMoE` 两个时序基础模型的独立 runner、固定
模型快照和双 GPU 实验矩阵。基础模型实验不属于原有纯时序模型/经典基线的
144 项矩阵，也不会改变 `run_time_series.py` 的参数或输出契约。

## 1. 范围与数据泄漏边界

基础模型只使用历史归一化功率，模型输入严格为 `[B, seq_len, 1]`。每个窗口的
最后一个历史点是 issue time，预测目标在模型调用之后才用于评估或训练 loss。

基础模型 runner `run_foundation_model.py` 不向模型传入天气、图像、站点静态信息、
日历、太阳位置、未来目标、未来 mask 以外的协变量，也不把测试区间统计量用于训练。
缺失的未来目标由 `target_mask` 排除；这不改变模型只能看到历史功率的边界。

配置文件和 Parquet 只负责构造确定的 power-only 窗口。基础模型、模式、revision、
输出目录和任务汇总均由独立的 foundation 路径维护，不应与纯时序 144 项汇总混用。

## 2. 固定模型目录与 revision

所有运行都必须使用 registry 中的完整 commit hash。表中的链接仅用于阅读上游资料；
预检和 runner 不会因为列出模型而下载权重。

| 模型 | 固定 ID 与完整 revision | 许可证与上游入口 | 精度、上下文、原生预测范围 | 本地 `last_layer` selector |
| --- | --- | --- | --- | --- |
| `Sundial` | [`thuml/sundial-base-128m`](https://huggingface.co/thuml/sundial-base-128m) @ [`3212e42564493f520593e5414af4367fc4b49226`](https://huggingface.co/thuml/sundial-base-128m/commit/3212e42564493f520593e5414af4367fc4b49226) | [固定 model card（Apache-2.0）](https://huggingface.co/thuml/sundial-base-128m/blob/3212e42564493f520593e5414af4367fc4b49226/README.md)；[官方仓库/LICENSE](https://github.com/thuml/Sundial/blob/main/LICENSE)；[上游用法](https://github.com/thuml/Sundial) | FP32；上下文最多 2880 点；原生多 patch 预测最多 720 点 | `flow_loss` |
| `TimeMoE` | [`Maple728/TimeMoE-50M`](https://huggingface.co/Maple728/TimeMoE-50M) @ [`446753ee48ff3726d0606a81d0092d54acee995e`](https://huggingface.co/Maple728/TimeMoE-50M/commit/446753ee48ff3726d0606a81d0092d54acee995e) | [固定 model card（Apache-2.0）](https://huggingface.co/Maple728/TimeMoE-50M/blob/446753ee48ff3726d0606a81d0092d54acee995e/README.md)；[官方仓库/LICENSE](https://github.com/Time-MoE/Time-MoE/blob/main/LICENSE)；[上游用法](https://github.com/Time-MoE/Time-MoE) | BF16（仅 pinned config metadata；loader runtime dtype/VRAM 未验证）；最大位置 4096；原生 heads `[1, 8, 32, 64]`，本项目最大裁剪 horizon 为 64 | `lm_heads` |

模型目录和 revision 的唯一代码来源是 `foundation_models/registry.py`。如果 model
card、缓存目录或命令行出现其他 revision，应视为不匹配，不能当作本项目的可复现实验。

## 3. 参考环境、安装与兼容性

验收解释器固定为：

```text
/opt/data/private/penv/time/bin/python
Python 3.8.18
```

需要同时满足以下运行依赖：

| 组件 | 验收范围 |
| --- | --- |
| PyTorch distribution/import | `>=2.3,<2.4`，CUDA build 必须为 `11.8` |
| Transformers | `>=4.46.2,<4.47` |
| PEFT | `>=0.13.2,<0.14` |
| CUDA | `torch.cuda.is_available()` 为真，至少 2 张卡；每张卡名称非空、总显存字节数为正 |

注意：仓库现有 `requirements.txt` 的 `torch>=2.0` 只是历史项目的宽松下限，
不足以表达本基础模型验收的 `>=2.3,<2.4` 与 CUDA 11.8 要求。应在独立环境中
额外执行有界安装，再用预检确认实际 distribution/import 和 CUDA build：

```bash
/opt/data/private/penv/time/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  "torch>=2.3,<2.4"
/opt/data/private/penv/time/bin/python -m pip install \
  "transformers>=4.46.2,<4.47" "peft>=0.13.2,<0.14"
/opt/data/private/penv/time/bin/python scripts/check_foundation_environment.py \
  --offline --json
```

在仓库根目录安装：

```bash
cd /opt/data/private/code/tsfm-ts
/opt/data/private/penv/time/bin/python -m pip install -r requirements.txt
/opt/data/private/penv/time/bin/python -m pip install pytest
```

基础模型依赖和缓存是独立前置条件，即使旧的纯时序测试通过，也必须单独运行预检：

```bash
/opt/data/private/penv/time/bin/python scripts/check_foundation_environment.py
/opt/data/private/penv/time/bin/python scripts/check_foundation_environment.py --json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /opt/data/private/penv/time/bin/python \
  scripts/check_foundation_environment.py --offline --json
```

预检只检查环境、文件、精确缓存和无凭据的固定 revision `HEAD` 可达性；它不会加载
模型或执行 forward/backward。没有发明最低 VRAM 阈值：显存是否适合特定 batch、模式和
horizon，必须由真实 smoke 验证。

上游当前推荐 Python 3.10+ 和 Transformers 4.40.1（Sundial model card 也给出类似
建议）。本项目验收组合是 Python 3.8.18、Transformers 4.46.x；在真实模型加载和
单 batch smoke 之前，不把这组组合描述为已获上游或本项目数值兼容性证明。

## 4. 四种运行模式

四种模式是固定字符串：`zero_shot`、`adapter`、`full`、`last_layer`。

| 模式 | 参数策略 | 可观察不变量 | 训练/验证/测试阶段 |
| --- | --- | --- | --- |
| `zero_shot` | 冻结全部参数，不创建 optimizer | `trainable=0` | `0 / 0 / 正数` |
| `adapter` | 只允许名称包含 `lora_` 的 LoRA 参数可训练 | `0 < trainable < total`，每个可训练名均为 LoRA | 三阶段均为正数（smoke 时各 1） |
| `full` | 全部基础参数可训练 | `trainable == total` | 三阶段均为正数（smoke 时各 1） |
| `last_layer` | 仅解冻 registry selector 选中的最后层 | `0 < trainable < total`，selector 非空且不能命中全部模型 | 三阶段均为正数（smoke 时各 1） |

每个训练模式都会审计真实 `requires_grad` 参数、总数、可训练数、比例、名称前 50
项和名称摘要。审计失败会在写 `completion.tsv` 前退出。`zero_shot` 不应因为传入
`--epochs` 而创建训练阶段；runner 会强制 epoch 为 0。

Sundial 的点预测来自原生 `generate` 的 20 个样本平均；这只是本地 point forecast
边界，不改变输入只能是历史功率的约束。TimeMoE 的训练 loss 见第 9 节限制说明。

## 5. 四组任务设置与 32 项矩阵

基础模型只使用以下四个 dataset-setting 组合：

| setting | 数据集 | 历史 `seq_len` | 目标 | `pred_len` |
| --- | --- | ---: | --- | ---: |
| `seq48_pred1` | `skippd_luoyang`（SKIPP'D） | 48 | 未来 1 个 5 分钟点 | 1 |
| `seq48_pred1` | `pvod_station00_ylj`（PVOD） | 48 | 未来 1 个 15 分钟点 | 1 |
| `seq96_h4` | `skippd_luoyang`（SKIPP'D） | 96 | 未来 4 小时 | 48 |
| `seq96_h4` | `pvod_station00_ylj`（PVOD） | 96 | 未来 4 小时 | 16 |

每组设置运行 2 个模型 × 4 个模式：`4 × 2 × 4 = 32` 个唯一 identity。特别注意：
SKIPP'D 的 `96→48` 和 PVOD 的 `96→16` 都是 4 小时；`pred_len` 是采样点数，不是
小时数。任务顺序和 model/mode 名称来自 lazy registry 与 `foundation_models.tasks`，
不要手工改写成旧的 `seq24` 或 `seq48_h4` 基础模型任务。

完整 identity（也是 summary 中去重的主键）如下；每行一个 dataset-setting/model/mode：

```text
seq48_pred1/skippd_luoyang/Sundial/zero_shot
seq48_pred1/skippd_luoyang/Sundial/adapter
seq48_pred1/skippd_luoyang/Sundial/full
seq48_pred1/skippd_luoyang/Sundial/last_layer
seq48_pred1/skippd_luoyang/TimeMoE/zero_shot
seq48_pred1/skippd_luoyang/TimeMoE/adapter
seq48_pred1/skippd_luoyang/TimeMoE/full
seq48_pred1/skippd_luoyang/TimeMoE/last_layer
seq48_pred1/pvod_station00_ylj/Sundial/zero_shot
seq48_pred1/pvod_station00_ylj/Sundial/adapter
seq48_pred1/pvod_station00_ylj/Sundial/full
seq48_pred1/pvod_station00_ylj/Sundial/last_layer
seq48_pred1/pvod_station00_ylj/TimeMoE/zero_shot
seq48_pred1/pvod_station00_ylj/TimeMoE/adapter
seq48_pred1/pvod_station00_ylj/TimeMoE/full
seq48_pred1/pvod_station00_ylj/TimeMoE/last_layer
seq96_h4/skippd_luoyang/Sundial/zero_shot
seq96_h4/skippd_luoyang/Sundial/adapter
seq96_h4/skippd_luoyang/Sundial/full
seq96_h4/skippd_luoyang/Sundial/last_layer
seq96_h4/skippd_luoyang/TimeMoE/zero_shot
seq96_h4/skippd_luoyang/TimeMoE/adapter
seq96_h4/skippd_luoyang/TimeMoE/full
seq96_h4/skippd_luoyang/TimeMoE/last_layer
seq96_h4/pvod_station00_ylj/Sundial/zero_shot
seq96_h4/pvod_station00_ylj/Sundial/adapter
seq96_h4/pvod_station00_ylj/Sundial/full
seq96_h4/pvod_station00_ylj/Sundial/last_layer
seq96_h4/pvod_station00_ylj/TimeMoE/zero_shot
seq96_h4/pvod_station00_ylj/TimeMoE/adapter
seq96_h4/pvod_station00_ylj/TimeMoE/full
seq96_h4/pvod_station00_ylj/TimeMoE/last_layer
```

## 6. 可复制命令

### 6.1 列表与无下载预检

```bash
cd /opt/data/private/code/tsfm-ts

/opt/data/private/penv/time/bin/python run_foundation_model.py --list-models
/opt/data/private/penv/time/bin/python run_foundation_model.py --list-modes
/opt/data/private/penv/time/bin/python scripts/check_foundation_environment.py \
  --network-timeout 3
```

列表命令必须分别只输出 `Sundial`、`TimeMoE`，以及 `zero_shot`、`adapter`、`full`、
`last_layer`；列表不会导入 Transformers/PEFT 或下载权重。

### 6.2 固定 revision 的 cache warm-up 与离线预检

两个公开仓库不要求 token。下面的 warm-up 会下载文件，因此只有在确认网络和磁盘
策略后才执行；它不是本次代码验收步骤。命令关闭隐式 token，并固定完整 revision：

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /opt/data/private/penv/time/bin/hf download \
  thuml/sundial-base-128m \
  --revision 3212e42564493f520593e5414af4367fc4b49226

HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /opt/data/private/penv/time/bin/hf download \
  Maple728/TimeMoE-50M \
  --revision 446753ee48ff3726d0606a81d0092d54acee995e

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /opt/data/private/penv/time/bin/python \
  scripts/check_foundation_environment.py --offline --json
```

预检看到完整精确快照时会跳过网络；缺少任一 load-critical 文件时，`--offline` 必须
失败。在线模式最多对每个不完整模型发送一次无凭据 `HEAD` 到固定 revision 的
`config.json`，只表示 source reachable，不表示权重已下载或运行时可用。在线检查依赖
系统中的绝对路径 `curl`：每个不完整模型只启动一个带 `--disable --head --location
--max-redirs 3 --max-time 3.0` 的进程，最多跟随 3 次 HTTPS-only HEAD redirect，并且
只有最终 HTTP 2xx 才算 reachable；curl 与 Python subprocess 都有硬超时边界。stdout/
stderr 只在进程内捕获并转换为固定状态，不会写入报告。

### 6.3 单任务真实 smoke

只有当预检显示对应模型 `cache_complete=true` 时才执行真实权重 smoke。以下命令测试
SKIPP'D 的 `96→48` Sundial 路径：

```bash
CUDA_VISIBLE_DEVICES=0 /opt/data/private/penv/time/bin/python \
  run_foundation_model.py \
  --dataset skippd_luoyang \
  --model Sundial \
  --mode zero_shot \
  --seq_len 96 \
  --pred_len 48 \
  --epochs 0 \
  --smoke \
  --max_test_steps 1 \
  --device cuda:0 \
  --output_dir \
  results_foundation_real_smoke/seq96_h4/skippd_luoyang/Sundial/zero_shot
```

`CUDA_VISIBLE_DEVICES=1` 时进程内部仍使用 `--device cuda:0`；此时内部 `cuda:0` 是
物理 GPU 1。`completion.tsv` 写入前会校验 `metrics.json`、预测和 checkpoint 的 hash。

### 6.4 两个后端的八个 per-backend/per-mode smoke

先运行两个 zero-shot，再按每个后端运行 `adapter`、`full`、`last_layer`。以下两组
命令共 8 个真实单任务 smoke；每个训练模式都限制为每阶段一个 batch：

```bash
# 1. Sundial zero-shot，SKIPP'D 96 -> 48
CUDA_VISIBLE_DEVICES=0 /opt/data/private/penv/time/bin/python run_foundation_model.py \
  --dataset skippd_luoyang --model Sundial --mode zero_shot \
  --seq_len 96 --pred_len 48 --epochs 0 --smoke --max_test_steps 1 \
  --device cuda:0 --output_dir \
  results_foundation_real_smoke/seq96_h4/skippd_luoyang/Sundial/zero_shot

# 2. TimeMoE zero-shot，PVOD 96 -> 16
CUDA_VISIBLE_DEVICES=1 /opt/data/private/penv/time/bin/python run_foundation_model.py \
  --dataset pvod_station00_ylj --model TimeMoE --mode zero_shot \
  --seq_len 96 --pred_len 16 --epochs 0 --smoke --max_test_steps 1 \
  --device cuda:0 --output_dir \
  results_foundation_real_smoke/seq96_h4/pvod_station00_ylj/TimeMoE/zero_shot

# 3--5. Sundial 的三个训练模式，SKIPP'D 48 -> 1
for MODE in adapter full last_layer; do
  CUDA_VISIBLE_DEVICES=0 /opt/data/private/penv/time/bin/python run_foundation_model.py \
    --dataset skippd_luoyang --model Sundial --mode "$MODE" \
    --seq_len 48 --pred_len 1 --epochs 1 --smoke \
    --max_train_steps 1 --max_eval_steps 1 --max_test_steps 1 --batch_size 2 \
    --device cuda:0 --output_dir \
    "results_foundation_real_smoke/seq48_pred1/skippd_luoyang/Sundial/$MODE"
done

# 6--8. TimeMoE 的三个训练模式，PVOD 48 -> 1
for MODE in adapter full last_layer; do
  CUDA_VISIBLE_DEVICES=1 /opt/data/private/penv/time/bin/python run_foundation_model.py \
    --dataset pvod_station00_ylj --model TimeMoE --mode "$MODE" \
    --seq_len 48 --pred_len 1 --epochs 1 --smoke \
    --max_train_steps 1 --max_eval_steps 1 --max_test_steps 1 --batch_size 2 \
    --device cuda:0 --output_dir \
    "results_foundation_real_smoke/seq48_pred1/pvod_station00_ylj/TimeMoE/$MODE"
done
```

八个命令只证明命名的模型加载和一条受限执行路径。真实 smoke 后应检查：精确
`model_id`/revision、模式、有限 metrics、阶段步数和参数计数；不能把一次 smoke 当作
收敛性、准确率或完整训练证明。

### 6.5 双 GPU 的 32 项 smoke/正式运行

两个 worker 各自绑定一张物理卡，worker 内部串行。确认预检、单任务 smoke 和缓存都
满足要求后运行：

```bash
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" \
OUTPUT_ROOT=results_foundation_models_smoke \
  bash scripts/smoke_all_foundation_models_2gpu.sh
```

完整运行使用相同的固定 32 项矩阵：

```bash
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" \
OUTPUT_ROOT=results_foundation_models \
  bash scripts/run_all_foundation_models_2gpu.sh
```

通过标准是 summary 恰好包含 32 个唯一 identity 且每行 `PASS`；脚本失败、被中断或
发现产物不完整时不会发布成功 summary。

## 7. 产物、身份与安全恢复

单任务目录固定为：

```text
<OUTPUT_ROOT>/<setting>/<dataset>/<model>/<mode>/
```

直接调用 `run_foundation_model.py` 时，每个任务目录的核心产物是：

| 核心产物 | 作用 |
| --- | --- |
| `best.pt` | zero-shot 的生命周期 checkpoint，或训练模式的最佳可恢复 state |
| `predictions.csv` | issue time、target time、horizon 和真实/预测功率 |
| `metrics.json` | 模式、精确 ID/revision、参数审计、阶段步数和有限指标 |
| `completion.tsv` | 最终身份、运行限制、阶段步数以及三个产物的 SHA-256 |

批处理脚本额外写入每个任务目录：

| 批处理产物 | 作用 |
| --- | --- |
| `data_fingerprint.txt` | 配置与 Parquet 内容的批处理指纹 |
| `run.log` | 双 GPU 脚本重定向的 runner 输出 |

`<OUTPUT_ROOT>/run_summary.tsv` 或
`<OUTPUT_ROOT>/smoke_summary.tsv` 是 OUTPUT_ROOT 层级的 32 项原子汇总，不是单任务
runner 的核心产物。

`completion.tsv` 总是在前三个产物和 metrics 校验成功后最后写入。`RESUME=1` 时，脚本
验证 summary 行、数据 fingerprint、规范输出路径、模式、seq/pred 长度、模型 ID 和
完整 revision、阶段步数、completion identity 以及三个 SHA-256；缺失、截断、篡改、
重复 identity、数据变化或参数变化的任务会重新运行。恢复时改变 `GPUS` 不会改写已经
通过任务的原始 `launch_gpu` provenance。

继续一次完整 32 项运行时使用同一个输出根目录和固定参数：

```bash
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" \
RESUME=1 \
OUTPUT_ROOT=results_foundation_models \
  bash scripts/run_all_foundation_models_2gpu.sh
```

## 8. 证据等级

不同证据只支持不同强度的结论，不能互相替代：

| 证据 | 能证明 | 不能证明 |
| --- | --- | --- |
| fake backend / fake runner 单测 | CLI、shape、矩阵、调度、manifest、参数策略逻辑 | remote code、真实权重、CUDA kernel、数值兼容性 |
| preflight | 解释器、包/CUDA、Parquet 路径、精确 cache 或 source reachability | 模型 forward/backward、显存适配、准确率 |
| 真实单 batch smoke | 指定卡上的 pinned model 加载和一条受限执行路径 | 完整训练稳定性、收敛和模型质量 |
| 32 项真实 smoke | 每个 identity 能运行受限 batch | 完整实验指标或泛化能力 |
| 完整 32 项运行 | 配置实验和产物完整结束 | 数据集之外的泛化能力 |

`network_status=pass` 且 `download_required=true` 只表示固定 metadata `HEAD` 可达；
它不是 cache 完整、权重已下载或真实模型通过的证据。

### 8.1 当前验证记录（2026-08-27）

本记录只描述已实际执行的 focused/fake/preflight/full 检查；真实模型 smoke 仍需
根据 cache 条件单独判断。

```text
/opt/data/private/penv/time/bin/python -m pytest -q tests/test_foundation_environment.py
58 passed in 0.25s

/opt/data/private/penv/time/bin/python -m pytest -q tests/test_foundation_scripts.py::test_two_gpu_smoke_expands_32_tasks_with_serial_queues_and_contained_outputs
1 passed in 3.05s; fake runner: 32/32 PASS

/opt/data/private/penv/time/bin/python -m pytest -q
406 passed in 424.20s (0:07:04)

/opt/data/private/penv/time/bin/python scripts/check_foundation_environment.py --network-timeout 3
exit=1; status=FAIL
Sundial: exact pinned cache absent; network_status=timeout; download_required=true
TimeMoE: exact pinned cache absent; network_status=connection_error; download_required=true

/opt/data/private/penv/time/bin/python scripts/check_foundation_environment.py --json --network-timeout 3
exit=1; ok=false; Sundial.network_status=timeout; TimeMoE.network_status=timeout

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/data/private/penv/time/bin/python scripts/check_foundation_environment.py --offline --json --network-timeout 3
exit=1; ok=false; Sundial.network_status=offline; TimeMoE.network_status=offline
```

预检还确认了解释器、三个包（Torch CUDA build 11.8）、两张 GPU 和两个配置后的
Parquet 文件；它没有加载真实权重。由于两个 exact cache 都缺失且在线 source 只得到
sanitized `timeout`/`connection_error` reachability failures，本环境没有 real model
load、predict/forward、backward 或 real 32-task smoke 证据，也没有下载权重。

## 9. 已知限制

- 两个后端都使用 `trust_remote_code=True`。完整 commit pin 降低代码漂移，但仍会在
  本机执行 model repository 的 Python，运行前必须审阅并信任该快照。
- 上游文档推荐 Python 3.10+/Transformers 4.40.1；本项目验收的 Python 3.8.18/
  Transformers 4.46.x 组合只有在真实 smoke 后才可称为本地验证过。
- Sundial 官方仓库当时尚未发布正式 fine-tuning recipe。这里的 `flow_loss` 最后一
  个 patch 训练路径是根据 pinned remote code 派生的本地实现，不应冒充官方 recipe。
- Sundial 的点预测使用 20 个随机生成样本的平均；随机种子能帮助复现，但不等于确定性
  的上游单样本输出。
- TimeMoE 训练目标是有意的本地边界：模型只接收历史，选择能覆盖目标的最小原生
  `lm_heads` horizon，裁剪后使用模型 pointwise masked Huber loss。它不同于 stock
  `outputs.loss` 的 teacher-forced future-target 路径，也不启用 history-wide router
  auxiliary loss；这不是上游默认训练目标。
- 表中 TimeMoE 的 BF16 只来自 pinned config metadata；loader 的实际 dtype、显存占用
  和 VRAM 适配必须由真实 smoke 验证，目前均未验证。
- TimeMoE 原生 heads 最大为 64；Sundial context/native horizon 也有限，超出范围的
  任务不应通过简单 padding 冒充支持。
- 当前 pipeline 是单变量 power-only，不支持 covariates、动态特征或分类任务。
- cache、磁盘和网络属于外部条件；预检不会下载权重。没有 exact cache 且 DNS/HTTP
  不可达时，真实模型 smoke 被阻断是诚实的环境结果，不是 fake PASS。
- 没有预先承诺 VRAM 最低值。batch、horizon、模式和模型结构的真实 backward 才能
  证明显存适配。

## 10. 故障排查

### 依赖缺失或版本不匹配

使用验收解释器重新安装 requirements，并检查 preflight 的 metadata/import 两列。
只安装了 distribution metadata 但 import 失败仍是 `FAIL`；不要用系统 Python 运行
基础模型命令。

### CUDA 不可用或只有一张卡

确认驱动可见、`torch.version.cuda` 为 `11.8`，且 `torch.cuda.device_count() >= 2`。
单卡只能做不依赖双卡编排的诊断；双 GPU 脚本要求两个不同的物理编号。

### Parquet 路径错误

预检只接受两个配置的非空 regular file。临时路径可通过这两个变量覆盖，并且覆盖必须
是非空值：

```bash
SKIPPD_PARQUET=/absolute/path/skippd_luoyang.parquet \
PVOD_PARQUET=/absolute/path/station00_ylj.parquet \
  /opt/data/private/penv/time/bin/python scripts/check_foundation_environment.py
```

预检只读取配置并检查路径/大小，不读取 Parquet 内容；窗口连续性、列名和 mask 仍由
runner smoke 检查。

### cache 不完整、offline 或网络超时

逐项查看 `missing_files` 和固定的 `network_status`（例如 `timeout`、`dns`、
`http_404`、`connection_error`、`missing_curl`）。offline 模式不会调用网络；在线 HEAD 可达也只会
显示 `download_required=true`。先按第 6.2 节用完整 revision warm-up，再重新运行
offline preflight。

### online preflight 找不到 curl

在线模式需要系统可执行的绝对路径 `curl`。如果状态为 `missing_curl`，安装或提供
系统 curl 后重试；不要把 token、代理密码或自定义 header 通过环境变量传给预检。离线
模式不需要 curl，但 exact cache 不完整时仍会失败。

### OOM 或真实 smoke 失败

先保留失败日志和不带 `completion.tsv` 的目录；减小 batch 或换有足够显存的卡后重跑。
不要删除或手工伪造 completion 来绕过恢复审计。OOM、remote-code import failure、
forward/backward 错误都只能记录为真实 smoke 未通过。

### 中断、部分输出和继续运行

保留同一个 `OUTPUT_ROOT`，确认 data fingerprint、模式、完整 revision 和参数没有变化，
再设置 `RESUME=1`。只有完整 hash/identity 校验成功的任务才会跳过；其余任务会重新
执行。不要把旧的纯时序 summary 复制到 foundation 输出目录。

### 凭据与日志安全

预检不接受 token 参数，不枚举或输出 credential-bearing 环境变量，不发送 Authorization
header，也不会打印 proxy/password、请求 header、URL 或原始异常；它会有意报告已解析的
dataset Parquet 路径和 Hub cache 路径，便于操作者修复文件位置。公开模型不需要 token；若本地网络策略
要求凭据，应由外部缓存流程安全处理，不能把 token 写入命令行、summary、metrics 或
issue 日志。
