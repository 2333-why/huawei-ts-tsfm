# HuaWei-TS：纯时序光伏功率预测测试说明

本文档供拿到代码后负责复现、训练和测试的人员使用。项目只使用历史光伏功率
进行预测，不向神经网络提供图像、天气、未来目标或其他模态。经典方法可以额外
使用配置文件中的站点静态信息，以及由预测发布时间推导出的日历和太阳位置。

仓库包含两类方法：

- 8 个纯时序神经网络模型，需要训练，推荐使用 GPU。
- 10 个经典预测方法，不进行梯度训练，统一在 CPU 上拟合或评估。

正式批量训练入口为 `scripts/run_all_pure_time_series.sh`，快速验收入口为
`scripts/smoke_all_pure_time_series.sh`，单模型入口为 `run_time_series.py`。

## 1. 接收代码后的检查顺序

建议严格按以下顺序测试：

1. 创建 Python 环境并安装依赖。
2. 修改两个数据配置中的 Parquet 绝对路径，确认数据文件存在。
3. 检查模型目录是否能正常输出 8 个神经模型和 10 个经典方法。
4. 运行 Python 自动化测试。
5. 分别运行一个神经模型 smoke test 和一个经典方法 smoke test。
6. 有两张 GPU 时运行完整的 144 项 smoke test。
7. smoke 全部通过后，再启动正式训练或完整经典方法评估。

所有命令都应在仓库根目录执行：

```bash
cd /path/to/pure-ts
```

## 2. 环境准备

当前代码已在 Python 3.8 环境验证。建议使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest
```

主要依赖包括 PyTorch、NumPy、Pandas、PyArrow、PyYAML 和 pvlib。需要 GPU 训练
时，应先确认安装的 PyTorch 与测试机器的 CUDA 驱动匹配：

```bash
python - <<'PY'
import torch
import pvlib

print("torch:", torch.__version__)
print("pvlib:", pvlib.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda devices:", torch.cuda.device_count())
PY
```

批量脚本使用 Bash，并依赖常见 Linux 命令，如 `realpath`、`sha256sum`、`awk`、
`grep` 和 `mapfile`。Windows 测试人员应在 WSL 或 Linux 环境中运行脚本。

## 3. 数据准备

仓库预置两个数据集配置：

| 数据集参数 | 配置文件 | 采样间隔 | 时间列 | 功率列 |
| --- | --- | ---: | --- | --- |
| `skippd_luoyang` | `configs/datasets/skippd_luoyang.json` | 5 分钟 | `timestamp` | `final_power` |
| `pvod_station00_ylj` | `configs/datasets/pvod_station00_ylj.yaml` | 15 分钟 | `timestamp` | `observe_power` |

配置中的 `paths.parquet_file` 是原开发环境的绝对路径。接收代码后必须改成测试
机器上的真实路径：

```text
configs/datasets/skippd_luoyang.json
  paths.parquet_file: /your/path/skippd_luoyang.parquet

configs/datasets/pvod_station00_ylj.yaml
  paths.parquet_file: /your/path/station00_ylj.parquet
```

Parquet 数据必须满足：

- 时间列能够由 Pandas 解析，不能为空且不能重复。
- 功率列能够转换为数值；缺失的未来目标会在评估时通过 mask 排除。
- 时间戳表示数据源当地的 wall-clock time。
- 配置中的采样间隔、训练起点、测试区间、功率缩放和额定功率必须与数据一致。
- 使用 pvlib 的方法还需要正确的经纬度、IANA 时区、组件倾角和方位角。

当前站点参数如下：

| 站点 | latitude | longitude | timezone | surface_tilt | surface_azimuth |
| --- | ---: | ---: | --- | ---: | ---: |
| SKIPP'D | 37.427 | -122.174 | `America/Los_Angeles` | 37 | 195 |
| PVOD | 38.04778 | 114.95139 | `Asia/Shanghai` | 33 | 180 |

可以用下面的命令先确认配置中的文件存在：

```bash
python - <<'PY'
from pathlib import Path
from data_provider.power_only import load_power_only_config

for config_path in (
    Path("configs/datasets/skippd_luoyang.json"),
    Path("configs/datasets/pvod_station00_ylj.yaml"),
):
    config = load_power_only_config(config_path)
    parquet = Path(config["paths"]["parquet_file"])
    print(config_path, "->", parquet, "exists=", parquet.is_file())
PY
```

若要测试同结构的其他 Parquet，可以复制一份配置并通过 `--config` 显式传入；
`--dataset` 仍需选择现有的 `skippd_luoyang` 或 `pvod_station00_ylj`。

## 4. 模型目录

### 4.1 纯时序神经模型

```bash
python run_time_series.py --list-models
```

应严格输出以下 8 个名称：

```text
TSMixer
Pyraformer
SegRNN
Transformer
LightTS
Crossformer
FreTS
MICN
```

这些模型只接收形状为 `[batch, seq_len, 1]` 的历史归一化功率。正式运行会建立
训练、验证和测试数据，使用 Adam 优化器，并把验证损失最优的模型写入
`best.pt`。

### 4.2 经典预测方法

```bash
python run_time_series.py --list-baselines
```

应严格输出以下 10 个名称：

```text
Persistence
SmartPersistence
SeasonalPersistence
Climatology
MovingMedian
DriftPersistence
ClearSkyEWMA
ClearSkyAR
SimilarDay
PersistenceClimatologyBlend
```

前 4 个是最初的经典基线，后 6 个是新增方法：

| 方法 | 预测规则 | 是否拟合训练数据 |
| --- | --- | --- |
| `Persistence` | 将最后一个历史功率重复到全部 horizon | 否 |
| `SmartPersistence` | 保持当前晴空指数，并乘未来 pvlib clear-sky POA | 否 |
| `SeasonalPersistence` | 读取目标时刻前一天的因果可用功率，缺失时回退 Persistence | 否 |
| `Climatology` | 相同 clock time、日历圆周距离不超过 15 天的训练均值 | 是 |
| `MovingMedian` | 最近一个物理小时的有限功率中位数 | 否 |
| `DriftPersistence` | 最近一个物理小时首尾观测的线性斜率外推 | 否 |
| `ClearSkyEWMA` | 半衰期 30 分钟的晴空指数 EWMA，再乘未来 clear-sky POA | 否 |
| `ClearSkyAR` | 在训练晴空指数上拟合最高三阶岭正则 AR 并递推 | 是 |
| `SimilarDay` | 选择训练集中同 clock time 的 3 个最近历史轨迹 | 是 |
| `PersistenceClimatologyBlend` | 按 horizon 拟合 Smart Persistence 与 Climatology 的组合权重 | 是 |

所有经典方法都通过与神经模型相同的测试集、预测 CSV 和指标函数评估，但不建立
优化器、不运行 epoch，也不占用 GPU。成功结果应记录 `epochs=0`、
`train_steps=0`、`val_steps=0` 和正数 `test_steps`。

## 5. 预测设置

批量脚本固定测试以下 4 组设置：

| 设置目录 | 历史点数 | 预测目标 | SKIPP'D `pred_len` | PVOD `pred_len` |
| --- | ---: | --- | ---: | ---: |
| `seq24_pred1` | 24 | 未来 1 点 | 1 | 1 |
| `seq48_pred1` | 48 | 未来 1 点 | 1 | 1 |
| `seq48_h4` | 48 | 未来 4 小时 | 48 | 16 |
| `seq96_h4` | 96 | 未来 4 小时 | 48 | 16 |

注意：`pred_len` 是预测点数，不是小时数。SKIPP'D 每 5 分钟一个点，所以 4 小时
对应 48 点；PVOD 每 15 分钟一个点，所以 4 小时对应 16 点。

## 6. 先运行自动化测试

完整测试命令：

```bash
python -m pytest -q
```

命令必须以退出码 0 结束。只检查经典方法和统一 runner 时，可以先运行：

```bash
python -m pytest -q \
  tests/test_classical_baselines.py \
  tests/test_extended_classical_baselines.py \
  tests/test_baseline_runner.py
```

只检查 144 项训练脚本的任务展开、GPU/CPU 分配和断点恢复时，运行：

```bash
python -m pytest -q tests/test_pure_time_series_scripts.py
```

上述脚本测试使用受控 fake runner，不会真的训练 144 个模型。

## 7. 单个神经模型怎么测试和训练

### 7.1 最小 smoke test

下面的命令只运行最多 1 个训练 batch、1 个验证 batch 和 1 个测试 batch，用于
检查数据、CUDA、模型前向传播和结果写入：

```bash
CUDA_VISIBLE_DEVICES=0 python run_time_series.py \
  --dataset skippd_luoyang \
  --model TSMixer \
  --seq_len 24 \
  --pred_len 1 \
  --smoke \
  --device cuda:0 \
  --output_dir results_manual_smoke/seq24_pred1/skippd_luoyang/TSMixer
```

`--smoke` 会自动使用 `epochs=1`、`batch_size=2`，并将训练、验证、测试阶段分别
限制为最多一个 batch。它只能证明执行链路可运行，不能代表正式模型精度。

没有 GPU 时可以临时用 `--device cpu` 检查小型 smoke，但神经模型正式训练仍推荐
使用 GPU。

### 7.2 正式训练一个神经模型

```bash
CUDA_VISIBLE_DEVICES=0 python run_time_series.py \
  --dataset skippd_luoyang \
  --model TSMixer \
  --seq_len 48 \
  --pred_len 48 \
  --epochs 40 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --seed 2024 \
  --device cuda:0 \
  --output_dir results_manual/seq48_h4/skippd_luoyang/TSMixer
```

在命令中设置 `CUDA_VISIBLE_DEVICES=1` 时，进程内部仍使用 `--device cuda:0`；此时
`cuda:0` 表示该进程可见的第一张卡，也就是物理 GPU 1。

### 7.3 单张 GPU 顺序训练 8 个神经模型

下面是一份可以直接执行的 Bash 训练脚本。它在一个数据集和一组设置上顺序训练
全部 8 个神经模型，避免同一张 GPU 上并发抢占显存：

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
DATASET="${DATASET:-skippd_luoyang}"
SEQ_LEN="${SEQ_LEN:-24}"
PRED_LEN="${PRED_LEN:-1}"
EPOCHS="${EPOCHS:-40}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_single_gpu}"

while IFS= read -r MODEL; do
  OUTPUT_DIR="$OUTPUT_ROOT/seq${SEQ_LEN}_pred${PRED_LEN}/$DATASET/$MODEL"
  echo "[训练] $DATASET / $MODEL / ${SEQ_LEN}->${PRED_LEN}"
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" run_time_series.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --seq_len "$SEQ_LEN" \
    --pred_len "$PRED_LEN" \
    --epochs "$EPOCHS" \
    --device cuda:0 \
    --output_dir "$OUTPUT_DIR"
done < <("$PYTHON" run_time_series.py --list-models)
```

可以将这段内容保存为测试机器自己的脚本，也可以直接使用第 9 节按卡数选择的内置
多 GPU 批量脚本完成全部设置。

## 8. 经典方法怎么测试

经典方法不进行神经网络训练，但 `Climatology`、`ClearSkyAR`、`SimilarDay` 和
`PersistenceClimatologyBlend` 会仅使用训练区间拟合统计量。命令中的
`--epochs 0 --device cpu` 不可省略，便于清楚表达运行协议。

### 8.1 测试一个经典方法

快速 smoke test：

```bash
CUDA_VISIBLE_DEVICES="" python run_time_series.py \
  --dataset skippd_luoyang \
  --model ClearSkyAR \
  --seq_len 24 \
  --pred_len 1 \
  --epochs 0 \
  --smoke \
  --device cpu \
  --output_dir results_baseline_smoke/seq24_pred1/skippd_luoyang/ClearSkyAR
```

完整测试集评估：

```bash
CUDA_VISIBLE_DEVICES="" python run_time_series.py \
  --dataset skippd_luoyang \
  --model ClearSkyAR \
  --seq_len 48 \
  --pred_len 48 \
  --epochs 0 \
  --device cpu \
  --output_dir results_baseline/seq48_h4/skippd_luoyang/ClearSkyAR
```

### 8.2 在一个设置上测试全部 10 个经典方法

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-skippd_luoyang}"
SEQ_LEN="${SEQ_LEN:-24}"
PRED_LEN="${PRED_LEN:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results_all_baselines}"

while IFS= read -r BASELINE; do
  OUTPUT_DIR="$OUTPUT_ROOT/seq${SEQ_LEN}_pred${PRED_LEN}/$DATASET/$BASELINE"
  echo "[评估] $DATASET / $BASELINE / ${SEQ_LEN}->${PRED_LEN}"
  CUDA_VISIBLE_DEVICES="" "$PYTHON" run_time_series.py \
    --dataset "$DATASET" \
    --model "$BASELINE" \
    --seq_len "$SEQ_LEN" \
    --pred_len "$PRED_LEN" \
    --epochs 0 \
    --device cpu \
    --output_dir "$OUTPUT_DIR"
done < <("$PYTHON" run_time_series.py --list-baselines)
```

若只是检查 10 个方法能否运行，可在循环内增加 `--smoke`。正式比较指标时不要使用
`--smoke`，否则只会评估一个测试 batch。

## 9. 仓库内置的完整训练脚本

### 9.1 按 GPU 数量训练全部时序模型

三套多卡脚本只训练 8 个神经时序模型，不运行经典方法。它们覆盖完全相同的 64 项
正式训练：

```text
4 组长度设置 × 2 个数据集 × 8 个神经时序模型 = 64 项
```

这里的两个预测长度是“未来 1 个采样点”和“未来 4 小时”，每个预测长度各使用两个
历史输入长度：

| 设置目录 | 历史点数 | SKIPP'D 预测点数 | PVOD 预测点数 | 物理预测长度 |
| --- | ---: | ---: | ---: | --- |
| `seq24_pred1` | 24 | 1 | 1 | 1 个采样点 |
| `seq48_pred1` | 48 | 1 | 1 | 1 个采样点 |
| `seq48_h4` | 48 | 48 | 16 | 4 小时 |
| `seq96_h4` | 96 | 48 | 16 | 4 小时 |

根据机器上可用的 GPU 数量选择一个脚本：

| GPU 数量 | 训练脚本 | 默认 GPU 编号 | 每张卡的任务数 |
| ---: | --- | --- | ---: |
| 2 | `scripts/train_time_series_2gpu.sh` | `0 1` | 32 |
| 4 | `scripts/train_time_series_4gpu.sh` | `0 1 2 3` | 16 |
| 8 | `scripts/train_time_series_8gpu.sh` | `0 1 2 3 4 5 6 7` | 8 |

每个脚本顶部都有下面这段醒目的数据集地址配置。把两个占位地址改成测试机器上的
Parquet 绝对路径：

```bash
# ==================== 必须修改：数据集地址 ====================
SKIPPD_PARQUET="${SKIPPD_PARQUET:-/REPLACE_WITH_ABSOLUTE_PATH/skippd_luoyang.parquet}"
PVOD_PARQUET="${PVOD_PARQUET:-/REPLACE_WITH_ABSOLUTE_PATH/station00_ylj.parquet}"
# =============================================================
```

也可以不修改脚本，运行时通过环境变量提供地址。以四卡机器为例：

```bash
PYTHON="$(command -v python)" \
SKIPPD_PARQUET="/绝对路径/skippd_luoyang.parquet" \
PVOD_PARQUET="/绝对路径/station00_ylj.parquet" \
GPUS="0 1 2 3" \
EPOCHS=40 \
RESUME=0 \
OUTPUT_ROOT="results_time_series_4gpu" \
  bash scripts/train_time_series_4gpu.sh
```

双卡和八卡机器只需换成对应脚本。`GPUS` 中必须提供与脚本名称一致数量且互不重复的
物理 GPU 编号，例如一台机器只开放编号 `2 3 6 7` 时，四卡版设置
`GPUS="2 3 6 7"`。每张卡内部顺序训练，不会在同一张卡上并发启动两个模型。

成功时终端会显示“全部 64 个实验完成”，汇总文件默认为：

```text
<OUTPUT_ROOT>/run_summary.tsv
```

脚本会对覆盖配置和 Parquet 内容计算 `data_fingerprint`。使用 `RESUME=1` 时，地址或
内容发生变化的数据集会自动重新训练；完全相同的数据集才会复用已有结果。若要长期
保留两次独立实验，仍建议使用不同的 `OUTPUT_ROOT`。

### 9.2 144 项 smoke test（神经模型加经典方法）

完整矩阵为：

```text
4 组设置 × 2 个数据集 ×（8 个神经模型 + 10 个经典方法）= 144 项
```

其中 64 个神经任务被平均分到两条 GPU 队列，每张 GPU 32 项；80 个经典方法任务
进入一条 CPU 队列。三条队列同时运行，每条队列内部串行。

批量脚本要求恰好提供两个不同的 GPU 编号。先运行 smoke：

```bash
PYTHON="$(command -v python)" \
GPUS="0 1" \
OUTPUT_ROOT="results_pure_time_series_smoke" \
  bash scripts/smoke_all_pure_time_series.sh
```

成功时应输出“全部 144 个实验完成”，并生成：

```text
results_pure_time_series_smoke/smoke_summary.tsv
```

### 9.3 正式运行全部 144 项

```bash
PYTHON="$(command -v python)" \
GPUS="0 1" \
EPOCHS=40 \
RESUME=0 \
OUTPUT_ROOT="results_pure_time_series" \
  bash scripts/run_all_pure_time_series.sh
```

环境变量说明：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PYTHON` | `/opt/data/private/penv/time/bin/python` | Python 解释器；其他机器应显式覆盖 |
| `GPUS` | `0 1` | 两个不同的物理 GPU 编号 |
| `EPOCHS` | `40` | 神经模型正式训练 epoch；经典方法始终为 0 |
| `RESUME` | `0` | `1` 时复用通过完整校验的已有任务 |
| `OUTPUT_ROOT` | `results_pure_time_series` | 所有任务输出根目录 |
| `SUMMARY_PATH` | `<OUTPUT_ROOT>/run_summary.tsv` | 汇总文件路径 |
| `SKIPPD_PARQUET` | 空，使用仓库配置 | 临时覆盖 SKIPP'D Parquet 地址 |
| `PVOD_PARQUET` | 空，使用仓库配置 | 临时覆盖 PVOD Parquet 地址 |
| `EXPECTED_GPU_COUNT` | `2` | 公共编排器要求的 GPU 数；通常由多卡脚本设置 |
| `INCLUDE_BASELINES` | `1` | `0` 时只运行 64 个神经时序任务 |

脚本为每个 GPU 启动一条顺序队列，并为经典方法启动一条 CPU 顺序队列。CPU 子进程
显式设置 `CUDA_VISIBLE_DEVICES=""`，不会初始化或占用 GPU。

### 9.4 中断后继续训练

使用相同的输出目录和 epoch，设置 `RESUME=1`：

```bash
PYTHON="$(command -v python)" \
GPUS="0 1" \
EPOCHS=40 \
RESUME=1 \
OUTPUT_ROOT="results_pure_time_series" \
  bash scripts/run_all_pure_time_series.sh
```

只有 summary 身份、运行模式、epoch、阶段步数、规范输出目录、完成清单，以及
数据指纹、`best.pt`、`predictions.csv`、`metrics.json` 三个 SHA-256 都匹配时，
任务才会跳过。失败、缺失、截断、篡改、数据变化或参数不一致的任务会重新运行。

## 10. 输出文件和通过标准

单个任务的结果目录为：

```text
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/
```

包含以下文件：

| 文件 | 内容 |
| --- | --- |
| `best.pt` | 神经模型最优权重，或经典方法的可恢复拟合状态 |
| `predictions.csv` | 每个 issue time、target time 和 horizon 的真实值与预测值 |
| `metrics.json` | 原功率单位下的 MAE、RMSE、NMAE、NRMSE 等指标 |
| `completion.tsv` | 任务身份、运行限制、阶段步数和三个产物的 SHA-256 |
| `data_fingerprint.txt` | 批量脚本记录的配置与 Parquet 内容指纹，用于安全恢复 |
| `run.log` | 仅由批量脚本重定向生成的标准输出和错误日志 |

直接运行 `run_time_series.py` 时，日志默认显示在终端，不会自动创建 `run.log`。

查看单个任务指标：

```bash
python -m json.tool \
  results_pure_time_series/seq24_pred1/skippd_luoyang/TSMixer/metrics.json
```

查看预测前几行：

```bash
head -n 6 \
  results_pure_time_series/seq24_pred1/skippd_luoyang/TSMixer/predictions.csv
```

使用第 9.3 节统一脚本时，检查正式汇总是否恰好有 144 个任务，且全部为 `PASS`：

```bash
SUMMARY="results_pure_time_series/run_summary.tsv"
test "$(($(wc -l < "$SUMMARY") - 1))" -eq 144
awk -F '\t' 'NR > 1 && $5 != "PASS" {print; failed=1} END {exit failed}' "$SUMMARY"
```

使用第 9.1 节多卡神经训练脚本时，把上面的 `144` 改为 `64`。最终通过标准：

- 自动化测试退出码为 0。
- 两个数据配置都能找到真实 Parquet。
- 单模型 smoke 能生成 `best.pt`、`predictions.csv`、`metrics.json` 和
  `completion.tsv`。
- 多卡神经训练汇总包含 64 个唯一任务，或统一 smoke/正式汇总包含 144 个唯一任务，
  所有状态均为 `PASS`。
- 神经任务的训练、验证、测试步数为正数。
- 经典方法的训练和验证步数为 0，测试步数为正数。
- `metrics.json` 中的指标为有限数值，预测值已限制在有效归一化功率范围。

评估指标会先恢复到原功率单位，再计算：

```text
nmae = mae / rated_power
nrmse = rmse / rated_power
```

同时输出 `nmae_percent` 和 `nrmse_percent`，只统计 mask 标记为有效的目标点。

## 11. 常见问题

### 找不到 Parquet 文件

修改对应配置文件中的 `paths.parquet_file`，不要依赖原开发机器的 `/opt/data/...`
绝对路径。也可以通过 `--config /path/to/config.json` 覆盖默认配置。

### `GPUS` 参数报错

双卡、四卡、八卡脚本分别要求 2、4、8 个不同编号，具体选择见第 9.1 节。只有一张
GPU 时，请使用第 7.3 节的单 GPU 顺序训练脚本；经典方法可以完全在 CPU 上独立测试。

### CUDA 不可用或显存不足

先运行第 2 节的 PyTorch 检查。单模型运行时可以减小 `--batch_size`。不要在同一张
GPU 上同时启动多个正式神经训练任务。

### 经典方法为什么写 `epochs=0`

经典方法没有反向传播和优化器。`Climatology`、`ClearSkyAR`、`SimilarDay` 和 Blend
只是在训练区间拟合统计量，其产物仍通过统一 checkpoint 保存。

### smoke 指标为什么不能用于模型比较

smoke 只处理最多一个测试 batch，目标是验证程序链路。模型精度比较必须去掉
`--smoke`，运行完整训练和完整测试区间。

更详细的架构、数据边界和恢复协议见 [PURE_TIME_SERIES.md](PURE_TIME_SERIES.md)。
