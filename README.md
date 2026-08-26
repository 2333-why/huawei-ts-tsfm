# HuaWei-TS：纯时序光伏功率预测

`pure-ts` 分支是一个精简的纯时序实验入口。训练和评估只读取历史
`power`，目标变量为功率，不使用图像、天气或其他模态。

## 数据集

| 数据集 | 配置 | 采样间隔 | 目标列 |
| --- | --- | ---: | --- |
| `skippd_luoyang` | `configs/datasets/skippd_luoyang.json` | 5 分钟 | `final_power` |
| `pvod_station00_ylj` | `configs/datasets/pvod_station00_ylj.yaml` | 15 分钟 | `observe_power` |

## 保留模型

模型注册表严格保留综合前八：

`TSMixer`、`Pyraformer`、`SegRNN`、`Transformer`、`LightTS`、`Crossformer`、
`FreTS`、`MICN`。

## 训练设置

支持四组设置：`24→1`、`48→1`、`48→4小时`、`96→4小时`。四小时按采样间隔
换算为 SKIPP'D 的 48 点或 PVOD 的 16 点；一点预测始终输出 1 点。

单模型示例：

```bash
cd /opt/data/private/code/pure-ts
CUDA_VISIBLE_DEVICES=0 /opt/data/private/penv/time/bin/python run_time_series.py \
  --dataset skippd_luoyang --model TSMixer \
  --seq_len 24 --pred_len 1 --epochs 40 \
  --device cuda:0 \
  --output_dir results_pure_time_series/seq24_pred1/skippd_luoyang/TSMixer
```

完整实验共 64 个任务，由两个 GPU 队列并行运行、每张卡内部串行：

```bash
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" EPOCHS=40 RESUME=0 \
  bash scripts/run_all_pure_time_series.sh
```

只做有界 smoke test（每任务训练、验证、测试最多一个 batch，不跑完整 epoch）：

```bash
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" \
  bash scripts/smoke_all_pure_time_series.sh
```

详细说明、输出目录和恢复规则见 [PURE_TIME_SERIES.md](PURE_TIME_SERIES.md)。
