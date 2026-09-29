# TSFM 服务器终端启动命令

以下命令直接使用服务器当前已经激活的 Python 环境，不创建或切换环境。当前完整实验
运行 Sundial、TimeMoE-50M、TimeMoE-200M、Chronos2 和 TiRex，暂不运行 TimesFM。

## 1. 更新代码并确认环境

```bash
cd /你的实际路径/huawei-ts-tsfm

git fetch origin
git checkout offline-5models-test
git pull origin offline-5models-test

which python
python --version
```

## 2. 数据与权重位置

脚本已经内置下列服务器路径：

```text
/data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/Luoyang-Unified_format-V1.parquet
/data/PVMMoE/DATA/01-Solar/YLJ/Benchmark/YLJ-Unified_format.parquet

/data/PVMMoE/PRETRAINED_MODELS/sundial-base-128m
/data/PVMMoE/PRETRAINED_MODELS/TimeMoE-50M
/data/PVMMoE/PRETRAINED_MODELS/TimeMoE-200M
/data/PVMMoE/PRETRAINED_MODELS/chronos2
/data/PVMMoE/PRETRAINED_MODELS/TiRex
```

如路径没有变化，不需要再设置环境变量。

## 3. 单独检查 Luoyang 数据范围

只有 Luoyang 数据进行了扩充。以下命令自动取最新四个自然月，前两个月训练、后两个月
测试；YLJ 继续使用原配置，不重新划分。

```bash
python scripts/prepare_recent_four_months.py \
  --luoyang-parquet /data/PVMMoE/DATA/01-Solar/Luoyang-XS/Benchmark_V1/Luoyang-Unified_format-V1.parquet \
  --output-dir "$PWD/generated_configs/recent_four_months"
```

范围报告位于：

```text
generated_configs/recent_four_months/data_range_report.json
```

若要求 Luoyang 四个月内一个时间点都不能缺少，添加 `--strict-grid`。

## 4. 一键运行完整 68 项实验

```bash
PYTHON="$(which python)" \
GPUS="0 1 2 3 4 5 6 7" \
RESUME=1 \
OUTPUT_ROOT="$PWD/results_foundation_5models_8gpu" \
bash scripts/run_pvmoe_5models_8gpu.sh
```

该入口会自动完成：

1. 读取 Luoyang 完整数据范围并生成“前两月训练、后两月测试”配置；
2. 保持 YLJ 原始时间划分；
3. 校验五个本地权重的文件、大小和 SHA-256；
4. 校验 Python、依赖、数据和 8 张 GPU；
5. 运行 68 项实验；
6. 汇总全部 `metrics.json` 到 Markdown。

## 5. 输出文件

```text
results_foundation_5models_8gpu/recent_four_months_configs/data_range_report.json
results_foundation_5models_8gpu/recent_four_months_configs/luoyang_recent_four_months.json
results_foundation_5models_8gpu/local_weights.json
results_foundation_5models_8gpu/preflight.json
results_foundation_5models_8gpu/run_summary.tsv
results_foundation_5models_8gpu/results_summary.md
```

`RESUME=1` 会验证模型身份、运行模式、数据指纹和结果文件 hash；只有全部一致的成功任务
才会跳过。若要强制全部重新运行，设置 `RESUME=0`。

## 6. 自定义路径（可选）

```bash
PRETRAINED_MODELS_ROOT=/新的权重根目录 \
SKIPPD_PARQUET=/新的Luoyang.parquet \
PVOD_PARQUET=/新的YLJ.parquet \
PYTHON="$(which python)" \
GPUS="0 1 2 3 4 5 6 7" \
RESUME=1 \
bash scripts/run_pvmoe_5models_8gpu.sh
```
