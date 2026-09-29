# 仅 TSFM 的功率预测

服务器直接运行请参阅：[服务器终端启动命令](docs/SERVER_RUN_COMMANDS_ZH.md)。

> 服务器无法访问 Hugging Face 时，请按 [浏览器下载权重与桶上离线加载](docs/OFFLINE_WEIGHTS_ZH.md) 配置 `TSFM_WEIGHTS_ROOT`。

[English](README.md) | [简体中文](README.zh-CN.md)

本仓库提供仅使用功率数据的预测，注册了六个权重：Sundial、TimeMoE-50M、
TimeMoE-200M、Chronos2、TiRex 和 TimesFM。每次运行都使用形状为 `[B, L, 1]` 的单一归一化功率通道，并返回
`[B, H, 1]`。实现位于 `models/` 下；可选模型包采用惰性导入，因此模型目录和任务列表
命令可以离线工作。
不存在 `foundation_models/` 实现包。

## 模型与能力

registry 是检查点身份、revision、模式和最后一层选择器的事实来源。

| 模型 | 检查点 ID | 固定 revision | 模式 | 最后一层选择器 |
| --- | --- | --- | --- | --- |
| `Sundial` | `thuml/sundial-base-128m` | `3212e42564493f520593e5414af4367fc4b49226` | `zero_shot`、`adapter`、`full`、`last_layer` | `flow_loss` |
| `TimeMoE` | `Maple728/TimeMoE-50M` | `446753ee48ff3726d0606a81d0092d54acee995e` | `zero_shot`、`adapter`、`full`、`last_layer` | `lm_heads` |
| `TimeMoE200M` | `Maple728/TimeMoE-200M` | `794591bfeb1225fdf742cec0f4c71f20c3f3b87e` | `zero_shot`、`adapter`、`full`、`last_layer` | `lm_heads` |
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

共有四种数据集/窗口设置。每种设置都有五个支持训练的权重，各支持四种模式，另加一个
TiRex zero-shot 模式：

`4 settings × (5 trainable checkpoints × 4 modes + 1 TiRex mode) = 84 tasks`。

PVMMoE 一键脚本排除 TimesFM，因此运行 68 个任务。

数据范围策略为：Luoyang 恢复经过验证的固定切分，训练/验证使用
`2026-04-05～2026-05-11`，测试使用 `2026-05-11～2026-06-12`；YLJ 保持
`configs/datasets/pvod_station00_ylj.yaml` 中的原始划分不变。
`scripts/run_pvmoe_5models_8gpu.sh` 会自动检查两份数据内容并生成运行配置。

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

### 修复 CUDA 13 PyTorch 与 CUDA 12.2 驱动不兼容

如果检查结果显示已安装的 PyTorch 使用 CUDA 13，而服务器驱动只支持 CUDA 12.2，请在
`py3_10` 环境中改装 `torch==2.4.1`。该版本使用 CUDA 12.1 构建，可以在支持 CUDA 12.2
的驱动上运行。以下命令显式使用华为内部 pip 源，不访问公共 PyPI：

```bash
cd /home/ma-user/work/why/huawei-ts-tsfm-main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py3_10

export PIP_INDEX_URL="http://repo.myhuaweicloud.com/repository/pypi/simple/"
export PIP_TRUSTED_HOST="repo.myhuaweicloud.com"
export PIP_TIMEOUT=600

python -m pip config list
python -m pip index versions torch

python -m pip uninstall -y torch triton

python -m pip freeze \
  | awk -F'==' 'tolower($1) ~ /^nvidia-/ {print $1}' \
  | xargs -r python -m pip uninstall -y

python -m pip install --no-cache-dir --force-reinstall "torch==2.4.1"

python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.version.cuda != "12.1":
    raise SystemExit(f"expected a CUDA 12.1 PyTorch build, got {torch.version.cuda!r}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch still cannot initialize CUDA")
if torch.cuda.device_count() != 8:
    raise SystemExit(f"expected 8 GPUs, found {torch.cuda.device_count()}")

for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
PY

python scripts/check_foundation_environment.py --offline --json
```

仓库已将现代环境的 PyTorch 固定为 `torch==2.4.1`，因此以后重新运行
`scripts/setup_foundation_runtime.sh` 时不会再次升级到 CUDA 13 版本。

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
cd /home/ma-user/work/why/huawei-ts-tsfm-main
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
cd /home/ma-user/work/why/huawei-ts-tsfm-main
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

SUNDIAL_CACHE="$HF_HOME/hub/models--thuml--sundial-base-128m"
find "$SUNDIAL_CACHE" \
  -type f \
  -name '*.incomplete' \
  -print \
  -delete

python scripts/download_foundation_weights.py --model Sundial

python scripts/download_foundation_weights.py

python scripts/check_foundation_environment.py --offline --json

unset HF_TOKEN
```

### 在本地下载权重并通过挂载桶导入服务器

如果华为服务器无法直接下载 Hugging Face 大文件，可以在任意另一台能访问 Hugging Face
的 Windows 电脑上克隆本仓库，然后运行仓库自带的跨机器打包程序。它会下载六个固定
revision、生成完整的 `checkpoints_huggingface` 缓存、打包为 `tar.gz` 并生成 SHA256。
不要直接拖拽未打包的缓存目录，因为 Hugging Face 缓存可能包含符号链接，而对象存储挂载
可能破坏链接关系。

如果另一台电脑没有 Git，在 PowerShell 中直接下载并运行独立脚本：

```powershell
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/2333-why/huawei-ts-tsfm/main/scripts/download_weights_without_git.ps1" `
  -OutFile "download_weights_without_git.ps1"

powershell -ExecutionPolicy Bypass -File ".\download_weights_without_git.ps1"
```

该脚本不依赖本仓库的其他文件，也不会创建或激活 Python 环境。请先激活需要使用的
Python 3.9+ 环境；脚本会直接使用当前终端中的 `python`，安装/更新 `huggingface_hub`，
创建 `huawei-ts-tsfm-weights` 目录，安全提示输入 Hugging Face 只读 Token，下载六个固定
revision，并自动生成缓存归档及 SHA256 文件。下载中断后重新运行同一命令即可继续。

在另一台 Windows 电脑的 PowerShell 中执行；仓库可以克隆到任意路径：

```powershell
git clone https://github.com/2333-why/huawei-ts-tsfm.git
Set-Location huawei-ts-tsfm

py -3 -m venv .venv-hf-download

$Python = Join-Path $PWD ".venv-hf-download\Scripts\python.exe"

& $Python -m pip install --upgrade pip huggingface_hub

& $Python scripts\build_offline_weight_bundle.py
```

程序会安全提示输入 Hugging Face 只读 Token，输入内容不会回显，也不会写入仓库。下载中断时
直接重新执行最后一条命令即可复用缓存并断点续传。成功后仓库根目录会生成：

```text
huawei-ts-tsfm-hf-cache.tar.gz
huawei-ts-tsfm-hf-cache.tar.gz.sha256
```

将下面两个文件上传到挂载桶目录 `/data/PVMMoE/why`：

```text
huawei-ts-tsfm-hf-cache.tar.gz
huawei-ts-tsfm-hf-cache.tar.gz.sha256
```

文件在服务器上的预期位置为：

```text
/data/PVMMoE/why/huawei-ts-tsfm-hf-cache.tar.gz
/data/PVMMoE/why/huawei-ts-tsfm-hf-cache.tar.gz.sha256
```

上传完成后，先在华为服务器校验归档：

```bash
cd /data/PVMMoE/why

sha256sum -c huawei-ts-tsfm-hf-cache.tar.gz.sha256
```

校验结果必须为 `huawei-ts-tsfm-hf-cache.tar.gz: OK`。然后直接将缓存解压到项目实际使用的
位置：

```bash
PROJECT_ROOT="/home/ma-user/work/why/huawei-ts-tsfm-main"
ARCHIVE="/data/PVMMoE/why/huawei-ts-tsfm-hf-cache.tar.gz"

mkdir -p "$PROJECT_ROOT"

tar --no-same-owner \
  -xzf "$ARCHIVE" \
  -C "$PROJECT_ROOT"

export HF_HOME="$PROJECT_ROOT/checkpoints_huggingface"
```

只清理服务器此前下载失败遗留的临时分片，并检查归档中的符号链接是否完整：

```bash
find "$HF_HOME" \
  -type f \
  -name '*.incomplete' \
  -print \
  -delete

BROKEN_LINK=$(find "$HF_HOME" -xtype l -print -quit)

if [ -n "$BROKEN_LINK" ]; then
  echo "发现损坏链接：$BROKEN_LINK"
  exit 1
else
  echo "Hugging Face 缓存链接检查通过"
fi
```

最后在完全离线模式下验证环境、数据集和六个模型权重：

```bash
cd /home/ma-user/work/why/huawei-ts-tsfm-main

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py3_10

export HF_HOME="$PWD/checkpoints_huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/check_foundation_environment.py --offline --json
```

六个模型的 `status` 都应为 `pass`，不能再出现 `download_required`。程序最终使用的权重
目录为 `/home/ma-user/work/why/huawei-ts-tsfm-main/checkpoints_huggingface`。

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
`scripts/run_all_foundation_models_2gpu.sh` 的有边界包装器；后者会执行 84 个注册任务。
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

`scripts/run_all_foundation_models_8gpu.sh` 是运行全部 84 个注册任务的固定八 GPU 批量入口。
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

## 缓存预热与离线预检

### 更新分支一键测试

`scripts/test_updated_tsfm.sh` 会依次执行代码编译检查、全量 pytest、离线环境与
固定权重预检，以及真实权重 GPU smoke 矩阵。默认按 `GPUS` 自动选择双卡或八卡入口：

```bash
PYTHON=.venv-tsfm-modern/bin/python \
GPUS="0 1 2 3 4 5 6 7" \
SKIPPD_PARQUET=/path/to/Luoyang.parquet \
PVOD_PARQUET=/path/to/YLJ.parquet \
HF_HOME="$PWD/checkpoints_huggingface" \
  bash scripts/test_updated_tsfm.sh
```

只进行不需要数据、权重或 GPU 的快速代码测试：

```bash
PYTHON=.venv-tsfm-modern/bin/python TEST_SCOPE=unit \
  bash scripts/test_updated_tsfm.sh
```

日志和测试产物默认写入 `results_updated_tsfm_test/`，可通过 `RESULTS_ROOT` 修改。

要预热精确的固定 revision，请使用一个一致的本地 Hub 缓存：

```bash
export HF_HOME="$PWD/.cache/huggingface"
hf download thuml/sundial-base-128m --revision 3212e42564493f520593e5414af4367fc4b49226
hf download Maple728/TimeMoE-50M --revision 446753ee48ff3726d0606a81d0092d54acee995e
hf download Maple728/TimeMoE-200M --revision 794591bfeb1225fdf742cec0f4c71f20c3f3b87e
hf download amazon/chronos-2 --revision 29ec3766d36d6f73f0696f85560a422f50e8498c
hf download NX-AI/TiRex --revision 63c740922493f5fbe60b277609ec62babfba2762
hf download google/timesfm-2.5-200m-transformers --revision 5a9806b9b291fad9233b5249d88263f1846304d3
```

然后在不联系 Hub 的情况下检查就绪状态：

```bash
/opt/data/private/penv/time/bin/python \
  scripts/check_foundation_environment.py --offline --json
```

离线预检不会暴露凭据，并且只使用本地缓存。它不会下载权重或安装包。失败的退出码会
报告结构化阻塞项，例如 Python 版本下限不符、缺少包、缺少缓存文件、缺少数据集或 GPU
不可用。

使用指定的基础解释器时，预检预期退出码为 `1`，并为 Chronos2、TiRex 和 TimesFM 报告
Python 版本下限/包阻塞项。在配置好的现代环境中安装现代 profile 并预热固定缓存后，
使用同一命令；真实检查点 smoke 仍是独立的验证步骤。

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

## 当前环境限制

指定的 `/opt/data/private/penv/time/bin/python` 仍是 Python 3.8.18 legacy profile，使用
Torch 2.3.1、Transformers 4.46.2 和 PEFT 0.13.2；它无法承载全部已注册的现代后端。

在 2026-08-28 使用 `.venv/tsfm-modern` 验证（Python 3.11.0、torch 2.4.1 with CUDA 12.1
runtime、Transformers 5.3.0、PEFT 0.18.1、chronos-forecasting 2.3.1 和 tirex-ts 1.4.2）
时，规范的真实检查点 CUDA smoke 矩阵在两张 NVIDIA RTX 4090 GPU 上运行，并完成
`68/68 PASS`。已验证的检查点固定值为
`thuml/sundial-base-128m@3212e42564493f520593e5414af4367fc4b49226`、
`Maple728/TimeMoE-50M@446753ee48ff3726d0606a81d0092d54acee995e`、
`amazon/chronos-2@29ec3766d36d6f73f0696f85560a422f50e8498c`、
`NX-AI/TiRex@63c740922493f5fbe60b277609ec62babfba2762` 和
`google/timesfm-2.5-200m-transformers@5a9806b9b291fad9233b5249d88263f1846304d3`。
