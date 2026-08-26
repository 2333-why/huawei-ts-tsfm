# 纯时序光伏功率预测

本分支的神经模型只使用历史 `power` 数值，输入不包含图像、天气、未来信息或
其他模态。经典基线额外使用站点静态元数据和由 issue time 推导的日历/时间输入；
这些额外输入只用于基线，不进入神经模型。

## 数据集与站点元数据

| 数据集 | 配置 | 采样间隔 | 目标列 |
| --- | --- | ---: | --- |
| `skippd_luoyang` | `configs/datasets/skippd_luoyang.json` | 5 分钟 | `final_power` |
| `pvod_station00_ylj` | `configs/datasets/pvod_station00_ylj.yaml` | 15 分钟 | `observe_power` |

基线的 clear-sky POA 使用 pvlib 的 Ineichen 模型和以下配置。依赖约束为
`pvlib>=0.10.5,<0.12`。

| 站点 | latitude | longitude | timezone | surface_tilt | surface_azimuth |
| --- | ---: | ---: | --- | ---: | ---: |
| SKIPP'D | 37.427 | -122.174 | `America/Los_Angeles` | 37 | 195 |
| PVOD | 38.04778 | 114.95139 | `Asia/Shanghai` | 33 | 180 |

## 方法目录

`run_time_series.py --list-models` 返回严格有序的八个神经模型：

`TSMixer`、`Pyraformer`、`SegRNN`、`Transformer`、`LightTS`、`Crossformer`、
`FreTS`、`MICN`。

`run_time_series.py --list-baselines` 返回严格有序的四个经典基线：

`Persistence`、`SmartPersistence`、`SeasonalPersistence`、`Climatology`。

基线预测都在归一化功率空间工作，`H` 为预测步数，`x` 为当前归一化功率：

1. `Persistence`：`ŷ(t+h) = x(t)`，`h = 1..H`。
2. `SmartPersistence`：`ŷ(t+h) = x(t) * POA_clear(t+h) / POA_clear(t)`；当前 clear-sky POA 不大于 1 W/m² 时回退为 0。
3. `SeasonalPersistence`：`ŷ(t+h) = x_train(t+h - 1 day)`；找不到对应历史点时回退为 `x(t)`。
4. `Climatology`：对训练集内相同 clock time 且日历圆周距离不超过 15 天的观测取均值；无样本时回退为训练均值。

预测值会限制在归一化功率 `[0, 1]`。基线只拟合/读取训练数据；验证集不会被
用于基线构造。

## 四组设置

| 输出标签 | 历史点数 | 预测目标 |
| --- | ---: | --- |
| `seq24_pred1` | 24 | 未来 1 点 |
| `seq48_pred1` | 48 | 未来 1 点 |
| `seq48_h4` | 48 | 未来 4 小时 |
| `seq96_h4` | 96 | 未来 4 小时 |

四小时按物理时间换算：`skippd_luoyang` 为 48 点，`pvod_station00_ylj` 为 16
点。一点预测始终为 1 点，与采样间隔无关。

## 单次运行

神经模型进程内统一使用 `cuda:0`；环境变量中的物理卡号决定实际 GPU：

```bash
cd /opt/data/private/code/pure-ts
CUDA_VISIBLE_DEVICES=0 /opt/data/private/penv/time/bin/python run_time_series.py \
  --dataset skippd_luoyang --model TSMixer \
  --seq_len 24 --pred_len 1 --epochs 40 \
  --device cuda:0 \
  --output_dir results_pure_time_series/seq24_pred1/skippd_luoyang/TSMixer
```

基线单次运行使用 CPU，不需要 `CUDA_VISIBLE_DEVICES`：

```bash
cd /opt/data/private/code/pure-ts
/opt/data/private/penv/time/bin/python run_time_series.py \
  --dataset skippd_luoyang --model Persistence \
  --seq_len 24 --pred_len 1 --device cpu \
  --output_dir results_pure_time_series/seq24_pred1/skippd_luoyang/Persistence
```

## 统一 96 任务实验

四组设置 × 两个数据集 ×（八个神经模型 + 四个基线）共 96 个唯一
`(setting, dataset, method)` 任务。其中 64 个神经任务由两条 GPU 队列并行、
每张卡内部串行，使用配置的两张卡各 32 个任务；32 个基线任务由一条 CPU 队列
串行执行，summary 的 `launch_gpu` 固定为 `cpu`，进程参数为 `--device cpu`。

```bash
cd /opt/data/private/code/pure-ts
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" EPOCHS=40 RESUME=0 \
  bash scripts/run_all_pure_time_series.sh
```

Smoke 脚本使用同一 96 任务矩阵。神经任务的训练、验证、测试阶段各最多一个
batch；基线不训练且不建立验证阶段，`train_steps=val_steps=0`，测试最多一个
batch：

```bash
cd /opt/data/private/code/pure-ts
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" \
  bash scripts/smoke_all_pure_time_series.sh
```

完整运行写入 `results_pure_time_series/run_summary.tsv`，Smoke 运行写入
`results_pure_time_series_smoke/smoke_summary.tsv`；两者都必须包含 96 行且全部为
`PASS` 才表示对应运行成功。

## 输出与恢复

任务身份由 `<setting>/<dataset>/<method>` 唯一确定，结果位于：

```text
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/best.pt
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/metrics.json
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/predictions.csv
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/run.log
<OUTPUT_ROOT>/<setting>/<dataset>/<method>/completion.tsv
```

`run_summary.tsv` 保持八列：
`seq_len`、`pred_len`、`dataset`、`model`、`status`、`output_dir`、`exit_code`、
`launch_gpu`。每个成功任务的 `completion.tsv` 还记录运行模式、限制、阶段步数和
三个产物的 SHA-256。神经任务必须有正的训练/验证/测试步数；基线的有效完成清单
必须记录 `epochs=0`、`train_steps=0`、`val_steps=0` 和正数 `test_steps`。

设置 `RESUME=1` 后，只有 summary 行身份、规范输出目录、运行模式、epoch 和
阶段限制、完成清单身份/步数，以及 `best.pt`、`predictions.csv`、`metrics.json`
三者哈希都精确匹配的 `PASS` 任务才会跳过。GPU 只记录调度来源，不参与任务身份，
因此更换 GPU 分配仍可复用神经产物；基线始终保留 `launch_gpu=cpu`。损坏、截断、
篡改、重复、失败或方法身份不匹配的任务都会重跑。

预测 CSV 的时间戳保持数据源的无时区 clock time，不附加虚假的时区偏移。

## 指标

指标在恢复原功率单位后计算，只统计有效目标点：

`nmae = mae / rated_power`

`nrmse = rmse / rated_power`

同时输出 `nmae_percent = 100 * nmae` 和 `nrmse_percent = 100 * nrmse`。若某个
训练、验证或测试阶段没有有效目标点，运行失败且不会写入可恢复的成功清单。
