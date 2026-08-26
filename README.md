# HuaWei-TS：纯时序光伏功率预测

`pure-ts` 分支是一个精简的纯时序实验入口。神经模型训练和评估只读取历史
`power`，目标变量为功率，不使用图像、天气或其他模态。经典基线只额外使用
站点静态元数据和由 issue time 推导的日历/时间输入；这些输入不进入神经模型。

## 数据集与站点元数据

| 数据集 | 配置 | 采样间隔 | 目标列 |
| --- | --- | ---: | --- |
| `skippd_luoyang` | `configs/datasets/skippd_luoyang.json` | 5 分钟 | `final_power` |
| `pvod_station00_ylj` | `configs/datasets/pvod_station00_ylj.yaml` | 15 分钟 | `observe_power` |

基线的 clear-sky 计算由 `pvlib>=0.10.5,<0.12` 提供。配置中的站点元数据为：

| 站点 | latitude | longitude | timezone | surface_tilt | surface_azimuth |
| --- | ---: | ---: | --- | ---: | ---: |
| SKIPP'D | 37.427 | -122.174 | `America/Los_Angeles` | 37 | 195 |
| PVOD | 38.04778 | 114.95139 | `Asia/Shanghai` | 33 | 180 |

## 两个方法目录

`run_time_series.py --list-models` 严格返回八个神经模型：

`TSMixer`、`Pyraformer`、`SegRNN`、`Transformer`、`LightTS`、`Crossformer`、
`FreTS`、`MICN`。

`run_time_series.py --list-baselines` 严格返回四个经典基线：

`Persistence`、`SmartPersistence`、`SeasonalPersistence`、`Climatology`。

四个基线的公式（`H` 为预测步数，`x` 为归一化功率）如下：

1. `Persistence`：`ŷ(t+h) = x(t)`，`h = 1..H`。
2. `SmartPersistence`：`ŷ(t+h) = x(t) * POA_clear(t+h) / POA_clear(t)`；当当前 clear-sky POA 不大于 1 W/m² 时回退为 0。
3. `SeasonalPersistence`：`ŷ(t+h) = x_obs(t+h-1 day)`，读取因果可用的观测过去功率；缺少对应历史点时回退为 `x(t)`。
4. `Climatology`：仅对训练区间内相同 clock time 且日历圆周距离不超过 15 天的观测取均值；无样本时回退为训练均值。

只有 `Climatology` 的拟合限制在训练区间；`SeasonalPersistence` 在预测时读取因果可用的观测过去功率。

## 训练设置与单次运行

支持四组设置：`24→1`、`48→1`、`48→4小时`、`96→4小时`。四小时按采样间隔
换算为 SKIPP'D 的 48 点或 PVOD 的 16 点；一点预测始终输出 1 点。

神经模型单次运行示例：

```bash
cd /opt/data/private/code/pure-ts
CUDA_VISIBLE_DEVICES=0 /opt/data/private/penv/time/bin/python run_time_series.py \
  --dataset skippd_luoyang --model TSMixer \
  --seq_len 24 --pred_len 1 --epochs 40 \
  --device cuda:0 \
  --output_dir results_pure_time_series/seq24_pred1/skippd_luoyang/TSMixer
```

基线单次运行示例（不训练、不需要 GPU）：

```bash
cd /opt/data/private/code/pure-ts
/opt/data/private/penv/time/bin/python run_time_series.py \
  --dataset skippd_luoyang --model Persistence \
  --seq_len 24 --pred_len 1 --epochs 0 --device cpu \
  --output_dir results_pure_time_series/seq24_pred1/skippd_luoyang/Persistence
```

## 统一 96 任务实验

四组设置 × 两个数据集 ×（八个神经模型 + 四个基线）共 96 个唯一任务：64
个神经任务由两条 GPU 队列并行运行，每张卡内部串行并各承担 32 个；32 个
基线任务在一条 CPU 队列串行运行，使用 `launch_gpu=cpu` 和 `--device cpu`。

```bash
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" EPOCHS=40 RESUME=0 \
  bash scripts/run_all_pure_time_series.sh
```

只做有界 smoke test（每个神经任务训练、验证、测试最多一个 batch；基线训练和
验证步数保持为 0，测试最多一个 batch）：

```bash
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" \
  bash scripts/smoke_all_pure_time_series.sh
```

每个结果目录的身份由 `<setting>/<dataset>/<method>` 唯一确定，目录中包含：

```text
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/best.pt
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/metrics.json
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/predictions.csv
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/run.log
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/completion.tsv
```

评估指标在恢复原功率单位后计算，`nmae = mae / rated_power`、
`nrmse = rmse / rated_power`，并同时输出 `nmae_percent`/`nrmse_percent`；只统计
有效目标点。成功基线的 `completion.tsv` 记录 `epochs=0`、`train_steps=0`、
`val_steps=0` 和正数 `test_steps`。

`RESUME=1` 仅复用 summary 身份、规范输出目录、运行限制、完成清单和
`best.pt`、`predictions.csv`、`metrics.json` 三个 SHA-256 均完全匹配的成功产物。
神经任务必须有正的训练/验证/测试步数；只有四个已注册基线允许训练/验证步数
为 0。损坏、截断、篡改、重复、失败或方法身份不匹配的任务都会重跑。

详细说明见 [PURE_TIME_SERIES.md](PURE_TIME_SERIES.md)。
