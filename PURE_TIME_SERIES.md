# 纯时序光伏功率预测

本分支只使用历史 `power` 数值，输入不包含图像、天气、未来信息或其他
模态。数据集配置和目标列如下：

| 数据集 | 配置 | 采样间隔 | 目标列 |
| --- | --- | ---: | --- |
| `skippd_luoyang` | `configs/datasets/skippd_luoyang.json` | 5 分钟 | `final_power` |
| `pvod_station00_ylj` | `configs/datasets/pvod_station00_ylj.yaml` | 15 分钟 | `observe_power` |

## 模型

`run_time_series.py --list-models` 按综合排名返回且只返回以下八个模型：

`TSMixer`、`Pyraformer`、`SegRNN`、`Transformer`、`LightTS`、`Crossformer`、
`FreTS`、`MICN`。

## 四组设置

| 输出标签 | 历史点数 | 预测目标 |
| --- | ---: | --- |
| `seq24_pred1` | 24 | 未来 1 点 |
| `seq48_pred1` | 48 | 未来 1 点 |
| `seq48_h4` | 48 | 未来 4 小时 |
| `seq96_h4` | 96 | 未来 4 小时 |

四小时按物理时间换算：`skippd_luoyang` 为 48 点，
`pvod_station00_ylj` 为 16 点。一点预测始终为 1 点，与采样间隔无关。

## 单模型运行

进程内统一使用 `cuda:0`；环境变量中的物理卡号决定实际使用 GPU：

```bash
cd /opt/data/private/code/pure-ts
CUDA_VISIBLE_DEVICES=0 /opt/data/private/penv/time/bin/python run_time_series.py \
  --dataset skippd_luoyang --model TSMixer \
  --seq_len 24 --pred_len 1 --epochs 40 \
  --device cuda:0 \
  --output_dir results_pure_time_series/seq24_pred1/skippd_luoyang/TSMixer
```
## 完整 64 任务实验

四组设置 × 两个数据集（四小时按数据集换算）× 八个模型，共 64 个任务。
下面的脚本将任务平均分给 GPU 0 和 GPU 1；每张卡内部串行执行，卡间并行，
并保留每任务日志和 `run_summary.tsv`：

```bash
cd /opt/data/private/code/pure-ts
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" EPOCHS=40 RESUME=0 \
  bash scripts/run_all_pure_time_series.sh
```

设置 `RESUME=1` 会先读取上一次的 summary，只复用对应行状态为 `PASS` 且
`best.pt`、`metrics.json` 和 `predictions.csv` 均非空的任务；失败、缺少
summary 记录或产物不完整的任务都会重试。summary 的最后一列 `gpu` 持久化
实际 `CUDA_VISIBLE_DEVICES` 分配；旧版本没有该列的 summary 会被视为不兼容
并重跑。GPU 不参与任务身份，因此更换 GPU 分配仍可复用成功结果。

## 有界 Smoke Test

Smoke 使用完全相同的 64 任务矩阵，并将每个任务的训练、验证、测试阶段限制
为最多一个 batch，不运行完整 epoch：

```bash
cd /opt/data/private/code/pure-ts
PYTHON=/opt/data/private/penv/time/bin/python \
GPUS="0 1" \
  bash scripts/smoke_all_pure_time_series.sh
```

结果写入 `results_pure_time_series_smoke/smoke_summary.tsv`；只有 64 行均为
`PASS` 才表示 smoke 成功。每个任务的产物位于：

```text
<OUTPUT_ROOT>/<setting>/<dataset>/<model>/best.pt
<OUTPUT_ROOT>/<setting>/<dataset>/<model>/metrics.json
<OUTPUT_ROOT>/<setting>/<dataset>/<model>/predictions.csv
<OUTPUT_ROOT>/<setting>/<dataset>/<model>/run.log
```
