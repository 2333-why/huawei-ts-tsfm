# 仅 TSFM 的功率预测

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库提供仅使用功率数据的预测，支持五个时序基础模型：Sundial、TimeMoE、Chronos2、TiRex
和 TimesFM。每次运行都使用形状为 `[B, L, 1]` 的单一归一化功率通道，并返回
`[B, H, 1]`。实现位于 `models/` 下；可选模型包采用惰性导入，因此模型目录和任务列表
命令可以离线工作。
不存在 `foundation_models/` 实现包。

## 模型与能力

registry 是检查点身份、revision、模式和最后一层选择器的事实来源。

| 模型 | 检查点 ID | 固定 revision | 模式 | 最后一层选择器 |
| --- | --- | --- | --- | --- |
| `Sundial` | `thuml/sundial-base-128m` | `3212e42564493f520593e5414af4367fc4b49226` | `zero_shot`、`adapter`、`full`、`last_layer` | `flow_loss` |
| `TimeMoE` | `Maple728/TimeMoE-50M` | `446753ee48ff3726d0606a81d0092d54acee995e` | `zero_shot`、`adapter`、`full`、`last_layer` | `lm_heads` |
| `Chronos2` | `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` | `zero_shot`、`adapter`、`full`、`last_layer` | `output_patch_embedding` |
| `TiRex` | `NX-AI/TiRex` | `63c740922493f5fbe60b277609ec62babfba2762` | `zero_shot` | — |
| `TimesFM` | `google/timesfm-2.5-200m-transformers` | `5a9806b9b291fad9233b5249d88263f1846304d3` | `zero_shot`、`adapter`、`full`、`last_layer` | `output_projection_point` |

模式是明确的可训练策略：

- `zero_shot` 冻结模型，在没有优化器的情况下运行推理。
- `adapter` 注入 LoRA 模块，只训练适配器参数。
- `full` 训练模型的全部参数。
- `last_layer` 只训练指定选择器子树。

TiRex 有意只支持 zero-shot；不支持的模式会在构造其可选包或检查点加载器之前被拒绝。

## 数据与任务矩阵

运行器只接受仓库中的两个功率配置：

- `configs/datasets/skippd_luoyang.json`（`SKIPPD_PARQUET` 可覆盖其中配置的 Parquet 路径；5 分钟采样）
- `configs/datasets/pvod_station00_ylj.yaml`（`PVOD_PARQUET` 可覆盖其中配置的 Parquet 路径；15 分钟采样）

支持的数据集/窗口行是固定的：

| 设置 | 数据集 | `seq_len` | `pred_len` |
| --- | --- | ---: | ---: |
| `seq48_pred1` | `skippd_luoyang` | 48 | 1 |
| `seq48_pred1` | `pvod_station00_ylj` | 48 | 1 |
| `seq96_h4` | `skippd_luoyang` | 96 | 48 |
| `seq96_h4` | `pvod_station00_ylj` | 96 | 16 |

共有四种数据集/窗口设置。每种设置都有四个支持训练的模型，各支持四种模式，另加一个
TiRex zero-shot 模式：

`4 settings × (4 trainable models × 4 modes + 1 TiRex mode) = 68 tasks`。

## 运行环境约定

运行本仓库前请先激活您已有的 Python 环境。本仓库不会创建、激活或修改任何环境。
所有实验脚本默认直接使用当前 shell 中的 `python`；只有确实需要指定其他解释器时，
才设置 `PYTHON=/path/to/python`。

### 当前 Python 低于 3.10 时的终端命令

以下命令由用户在华为服务器终端中手动执行。`--override-channels` 会绕过 `.condarc`
中失效的清华 Conda 源，并且只访问华为内网仓库，不会回退到 `repo.anaconda.com`；后续
pip 安装仍继承服务器当前配置的华为 pip 源。如果服务器管理员提供的 Conda 仓库地址
不同，只需替换 `HUAWEI_CONDA_CHANNEL` 的值。

先查看当前 Conda 配置和相关环境变量：

```bash
conda config --show-sources
conda config --show channels
conda config --show default_channels
conda config --show custom_channels
conda config --show channel_alias

env | grep -Ei 'conda|pip|proxy|repo|mirror'
```

扫描华为网络中可能存在的 Conda 仓库。该命令只测试地址，不会修改 Conda 配置：

```bash
for channel in \
  "http://repo.myhuaweicloud.com/repository/anaconda/pkgs/main" \
  "https://repo.myhuaweicloud.com/repository/anaconda/pkgs/main" \
  "http://repo.huaweicloud.com/repository/anaconda/pkgs/main" \
  "https://repo.huaweicloud.com/repository/anaconda/pkgs/main" \
  "http://mirrors.huaweicloud.com/repository/anaconda/pkgs/main" \
  "https://mirrors.huaweicloud.com/repository/anaconda/pkgs/main"
do
  url="${channel}/linux-64/current_repodata.json"
  code=$(curl -L -sS \
    --connect-timeout 5 \
    --max-time 20 \
    --range 0-0 \
    -o /dev/null \
    -w '%{http_code}' \
    "$url" 2>/dev/null)

  case "$code" in
    200|206) echo "[可下载] HTTP $code  $channel" ;;
    401|403) echo "[可连接但需要权限] HTTP $code  $channel" ;;
    404)     echo "[路径不存在] HTTP $code  $channel" ;;
    000)     echo "[无法连接] HTTP $code  $channel" ;;
    *)       echo "[需要检查] HTTP $code  $channel" ;;
  esac
done
```

把扫描结果中标记为“可下载”的地址填入下面变量，再确认该源确实包含 Python 3.10：

```bash
export HUAWEI_CONDA_CHANNEL="http://repo.myhuaweicloud.com/repository/anaconda/pkgs/main"

conda search 'python=3.10' \
  --override-channels \
  -c "$HUAWEI_CONDA_CHANNEL"
```

查询成功后，再创建环境并继续安装：

```bash
cd /你的实际路径/huawei-ts-tsfm

source "$(conda info --base)/etc/profile.d/conda.sh"

export HUAWEI_CONDA_CHANNEL="http://repo.myhuaweicloud.com/repository/anaconda/pkgs/main"

conda create -n huawei-ts-tsfm-py310 \
  python=3.10 pip -y \
  --override-channels \
  -c "$HUAWEI_CONDA_CHANNEL"

conda activate huawei-ts-tsfm-py310

python --version
python -m pip config list

git pull origin main

bash scripts/setup_foundation_runtime.sh

HF_HOME="$PWD/checkpoints_huggingface" \
python scripts/check_foundation_environment.py --offline --json
```

环境检查成功后，启动完整的八卡训练和测试：

```bash
cd /你的实际路径/huawei-ts-tsfm
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate huawei-ts-tsfm-py310

GPUS='0 1 2 3 4 5 6 7' \
SMOKE=0 \
RESUME=1 \
OUTPUT_ROOT=results_foundation_models \
bash scripts/run_all_foundation_models_8gpu.sh
```

## 安装依赖并下载权重

以下命令直接安装到当前已激活环境，不会创建或切换虚拟环境。它随后按照
`models/registry.py` 中的固定 revision 下载 Sundial、TimeMoE、Chronos2、TiRex 和
TimesFM 权重：

```bash
bash scripts/setup_foundation_runtime.sh
```

默认缓存目录是仓库下的 `checkpoints_huggingface/`，实验脚本会自动复用该目录。如果依赖
已经安装，只下载权重：

```bash
python scripts/download_foundation_weights.py
```

### Xet/CAS 下载失败后使用 Token 续传

如果日志包含 `cas-server.xethub.hf.co`、`File reconstruction error` 或
`CAS Client Error`，说明失败发生在 Hugging Face Xet/CAS 文件传输阶段。Token 可以提高
请求限额，但仍需禁用 Xet，才能绕过无法访问的 CAS 地址并改用普通 HTTP。以下命令会复用
已经下载完成的缓存；输入 Token 时不会在终端回显，也不会把 Token 写入仓库：

```bash
cd /你的实际路径/huawei-ts-tsfm
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py3_10

export HF_HOME="$PWD/checkpoints_huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

read -rsp '请输入 Hugging Face 只读 Token: ' HF_TOKEN
echo
export HF_TOKEN

python - <<'PY'
from huggingface_hub import whoami
info = whoami()
print("Hugging Face authenticated as:", info.get("name", "unknown"))
PY

# 先续传失败的 Sundial；成功后继续下载/校验其余固定权重。
python scripts/download_foundation_weights.py --model Sundial
python scripts/download_foundation_weights.py

python scripts/check_foundation_environment.py --offline --json

unset HF_TOKEN
```

如果关闭 Xet 后 Sundial 仍因已有临时分片而失败，只清理该模型未完成的分片，再重新下载：

```bash
cd /你的实际路径/huawei-ts-tsfm
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py3_10

export HF_HOME="$PWD/checkpoints_huggingface"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

read -rsp '请输入 Hugging Face 只读 Token: ' HF_TOKEN
echo
export HF_TOKEN

SUNDIAL_CACHE="$HF_HOME/hub/models--thuml--sundial-base-128m"
find "$SUNDIAL_CACHE" -type f -name '*.incomplete' -print -delete

python scripts/download_foundation_weights.py --model Sundial
python scripts/download_foundation_weights.py
python scripts/check_foundation_environment.py --offline --json

unset HF_TOKEN
```

如服务器需要使用其他共享缓存，可在安装、下载和实验命令前统一设置
`HF_HOME=/path/to/huggingface-cache`。下载完成后可离线检查：

```bash
HF_HOME="$PWD/checkpoints_huggingface" \
python scripts/check_foundation_environment.py --offline --json
```

## 运行器

从仓库根目录运行。目录命令不会加载可选模型后端：

```bash
python run.py --list-models
python run.py --list-modes
```

单次运行示例（Sundial）：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --dataset skippd_luoyang \
  --model Sundial \
  --mode zero_shot \
  --seq_len 48 \
  --pred_len 1 \
  --device cuda:0 \
  --output_dir results_foundation_models/example
```

也支持 Time-Series-Library 风格的别名：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --data skippd_luoyang \
  --model TimesFM \
  --model_id timesfm_skippd_adapter_example \
  --mode adapter \
  --seq_len 48 \
  --pred_len 1 \
  --train_epochs 1 \
  --debug True \
  --task_name long_term_forecast \
  --is_training 1 \
  --device cuda:0 \
  --output_dir results_foundation_models/example-adapter
```

`--data` 是 `--dataset` 的别名；`--train_epochs` 是 `--epochs` 的别名；`--debug` 是
`--smoke` 的别名。`--debug` 必须显式提供 `True` 或 `False`。只有两个拼写的值一致时，
才允许同时提供它们；冲突会提前失败。`--model_id` 只用于实验标签元数据，不会替换
registry 中固定的检查点 ID 或 revision。`--task_name` 和 `--is_training` 是可选的一致性
断言：`zero_shot` 对应 `zero_shot_forecast` / `0`，训练模式对应
`long_term_forecast` / `1`。未知选项和缩写选项都会被拒绝。

## 简单训练脚本

`scripts/train_model.sh` 遵循要求的 shell 循环风格。其默认模型是 `TimesFM`，模式是
`adapter`。设置 `DEBUG_MODE=1` 时，它会启动恰好一个有边界的 adapter smoke 运行；否则
会启动四个真实数据集/窗口行。可用变量包括 `PYTHON`、`CUDA_VISIBLE_DEVICES`、
`MODEL_NAME`（此示例必须保持为 `TimesFM`）、`SEQ_LEN`、`DEBUG_MODE` 和
`RESULTS_ROOT`。

```bash
DEBUG_MODE=1 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_model.sh
```

普通分支运行四个真实数据集/窗口行：

```bash
DEBUG_MODE=0 CUDA_VISIBLE_DEVICES=0 \
  bash scripts/train_model.sh
```

## 双 GPU 批处理脚本

`scripts/smoke_all_foundation_models_2gpu.sh` 是
`scripts/run_all_foundation_models_2gpu.sh` 的有边界包装器；后者会执行 68 个支持的任务。
两者默认使用 `GPUS="0 1"`，并创建两个进程级队列。每个队列内部串行执行，而两个队列
可以重叠运行。队列会将一个物理序号作为
`CUDA_VISIBLE_DEVICES=<ordinal>` 启动，运行器使用进程内的
`--device cuda:0`；不使用 DataParallel。

相关变量包括：

- `PYTHON` 和 `CONFIG_PYTHON`：运行以及目录/配置检查所用的解释器。
- `GPUS`：恰好两个不同的物理 GPU 序号。
- `SMOKE`：`0` 表示完整运行，`1` 表示单步 smoke 运行。
- `RESUME`：`1` 启用经过验证的恢复，否则重新运行任务。
- `OUTPUT_ROOT`：结果根目录；`SUMMARY_PATH`：汇总 TSV 目标路径。
- `SKIPPD_PARQUET` 和 `PVOD_PARQUET`：可选的 Parquet 路径覆盖。

Smoke 示例：

```bash
GPUS="0 1" \
RESUME=0 \
OUTPUT_ROOT=results_foundation_models_smoke \
  bash scripts/smoke_all_foundation_models_2gpu.sh
```

完整批处理或恢复已验证任务时，使用非 smoke 脚本：

```bash
GPUS="0 1" \
RESUME=0 \
OUTPUT_ROOT=results_foundation_models \
  bash scripts/run_all_foundation_models_2gpu.sh

GPUS="0 1" \
RESUME=1 \
OUTPUT_ROOT=results_foundation_models \
SUMMARY_PATH=results_foundation_models/run_summary.tsv \
  bash scripts/run_all_foundation_models_2gpu.sh
```

## 八 GPU 批处理脚本

`scripts/run_all_foundation_models_8gpu.sh` 是运行全部 68 个支持任务的固定八 GPU 批量入口。
它默认使用 `GPUS='0 1 2 3 4 5 6 7'`，并严格要求八个不同且规范的物理 GPU 序号。任务按
`ordinal % 8` 分配；每张卡的队列串行执行，而八个队列可以并行运行。每个进程将一个物理
序号作为 `CUDA_VISIBLE_DEVICES=<ordinal>` 启动，运行器使用进程内的
`--device cuda:0`；不使用 DataParallel。

该入口支持 `SMOKE=0` 或 `SMOKE=1`，以及 `RESUME=0` 或 `RESUME=1`。它与
`scripts/run_all_foundation_models_2gpu.sh` 共用相同的校验和产物契约：恢复任务前必须校验
身份、schema、数据指纹、产物 hash、checkpoint/metrics 字段、模型 ID 和固定 revision。
相关变量包括 `PYTHON`、`CONFIG_PYTHON`、`GPUS`、`SMOKE`、`RESUME`、`OUTPUT_ROOT`、
`SUMMARY_PATH`、`SKIPPD_PARQUET` 和 `PVOD_PARQUET`。

八 GPU smoke 调用示例：

```bash
GPUS='0 1 2 3 4 5 6 7' \
SMOKE=1 \
RESUME=0 \
OUTPUT_ROOT=results_foundation_models_smoke \
  bash scripts/run_all_foundation_models_8gpu.sh
```

完整的八 GPU 批处理，或恢复已验证任务：

```bash
GPUS='0 1 2 3 4 5 6 7' \
SMOKE=0 \
RESUME=0 \
OUTPUT_ROOT=results_foundation_models \
  bash scripts/run_all_foundation_models_8gpu.sh

GPUS='0 1 2 3 4 5 6 7' \
SMOKE=0 \
RESUME=1 \
OUTPUT_ROOT=results_foundation_models \
SUMMARY_PATH=results_foundation_models/run_summary.tsv \
  bash scripts/run_all_foundation_models_8gpu.sh
```

## 产物与恢复

每个任务目录包含四个最终产物：

- `best.pt`：最佳/恢复的模型状态和生命周期元数据；
- `predictions.csv`：原始功率单位下的有效功率预测；
- `metrics.json`：指标、身份、可训练性和阶段计数；
- `completion.tsv`：最后写入的完成清单。

批处理还会为每个任务写入 `data_fingerprint.txt` 和 `run.log`，并在 `SUMMARY_PATH` 写入
TSV 汇总。设置 `RESUME=1` 时，只有在身份、schema、数据指纹、产物 hash、checkpoint/metrics
字段、模型 ID 和固定 revision 全部校验通过后，任务才会跳过。任何缺失、变化或被篡改的
产物都会触发重新运行。

历史结果目录组件 `foundation_models` 为恢复兼容性而保留。它只是结果路径，不是 Python
包或实现目录。
