# FACTS 光伏功率预测

本仓库在保留 FACTS 核心模型、Teacher/Student、知识蒸馏、物理约束、路由与主要编码器逻辑的基础上，适配 Luoyang 和 YLJ 两套 Parquet 数据集，用于未来 4 小时光伏功率预测。

## 数据集适配

| 数据集 | 主配置 | 采样间隔 | 历史长度 | 预测长度 | 额定功率 |
|---|---|---:|---:|---:|---:|
| Luoyang | `configs/luoyang_parquet.json` | 5 分钟 | 16 点 | 48 点 | 48629.73 |
| Luoyang tuned | `configs/luoyang_tuned.json` | 5 分钟 | 16 点 | 48 点 | 48629.73 |
| YLJ | `configs/datasets/ylj.yaml` | 15 分钟 | 16 点 | 16 点 | 468.0 MW |

所有数据路径、字段、时间范围、输入输出维度、容量、缺失值规则和训练参数均由配置文件管理。

- Luoyang Student 使用历史时序和历史天空图像；Teacher 额外使用未来实测 GHI 与未来图像。
- YLJ Student 使用历史观测和气象预报；Teacher 额外使用未来实测气象，不使用图像。
- 未来真实功率只作为监督标签，不进入 Student 的训练、验证或推理输入。
- Luoyang 缺失图像采用因果前向保持，并通过 `image_mask` 显式标记；YLJ 使用全 False 图像掩码。

## 环境

推荐使用 Python 3.9、PyTorch 2.x 和 CUDA GPU：

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
```

默认训练配置使用 8 张 GPU、`batch_size=64`、`num_workers=16`、`prefetch_factor=2` 和 AMP。多卡训练基于 `nn.DataParallel`。

## 运行

只运行 Luoyang 调优流程：

```bash
bash scripts/run_luoyang_tuned.sh
```

依次运行 Luoyang 和 YLJ 全流程：

```bash
bash scripts/run_full_experiments.sh
```

只运行 YLJ：

```bash
bash scripts/run_full_experiments.sh --datasets ylj
```

启动器会依次完成数据预检、Teacher 训练与测试、Student/KD 训练、Student 测试和输出校验。完整的前台、后台及复用 checkpoint 命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。

Luoyang 和 YLJ 默认同时生成不含样本级数据的训练监控包。监控指标、隐私边界和现场回传要求见 [MONITORING.md](MONITORING.md)。

## 输出

正式测试会在配置指定的 `results_root` 下生成：

```text
official_test_predictions.csv
official_test_metrics.json
```

预测文件字段为：

```text
issue_time,target_time,horizon_minutes,y_true,y_pred
```

Luoyang 每个 `issue_time` 输出 48 个预测点，YLJ 输出 16 个预测点。指标在原始功率单位下计算，包含 15 分钟和 240 分钟的 overall、hard-delta、NRMSE 与 NMAE；归一化使用对应配置中的 `rated_power`。

有效 checkpoint 由权重和同名契约文件共同组成：

```text
checkpoint.pth
checkpoint.pth.metadata.json
```

契约记录数据集、输入维度、预测长度、额定功率和 Teacher 输入配置，避免加载不匹配的 checkpoint。
