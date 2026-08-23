# Luoyang / YLJ 训练监控说明

本监控用于在不导出涉密样本的前提下，定位 Luoyang 和 YLJ 上的数据质量、训练稳定性、Teacher 能力和 Student 蒸馏效果问题。默认配置已启用监控，不改变模型结构、损失权重或训练超参数。

## 运行方式

继续使用原有流水线命令即可：

```bash
python scripts/luoyang_pipeline.py teacher --config configs/luoyang_tuned.json
python scripts/luoyang_pipeline.py student --config configs/luoyang_tuned.json
python scripts/luoyang_pipeline.py test --config configs/luoyang_tuned.json --prefer-latest-checkpoint

python scripts/ylj_pipeline.py teacher --config configs/datasets/ylj.yaml
python scripts/ylj_pipeline.py student --config configs/datasets/ylj.yaml
python scripts/ylj_pipeline.py test --config configs/datasets/ylj.yaml --prefer-latest-checkpoint
```

监控开关位于各数据集配置的 `monitoring` 段。推荐保持：

```text
enabled: true
include_test_during_training: false
save_raw_predictions: false
min_slice_count: 100
min_slice_days: 5
```

`100` 和 `5` 是代码强制的最低隐私门槛，不能通过命令行降低。模型指标要求至少 100 个有效点并覆盖 5 个自然日；数据画像还要求至少 100 个独立贡献样本并覆盖 5 个自然日，不能用单个样本的多个历史步或预测步凑数。日期缺失、样本不足或天数不足时，均值、分位数、比例、误差和分布漂移等数值会留空并标记 `privacy_suppressed=true`。稀有缺失、来源码、功率区间和图像状态按各自贡献样本再次检查，不能从另一项总体均值反推。

训练阶段只监控 train/val。checkpoint 选择完成后，正式测试阶段才会写入 test 聚合诊断，避免测试集参与模型选择。

## 应回传的内容

每个 Teacher 或 Student 运行目录下会生成独立的 `monitor/` 目录。请完整回传该目录，其中只包含以下聚合文件：

| 文件 | 内容 |
|---|---|
| `run_manifest.json` | 模型规模、安全参数、split 样本数、候选窗口接受率、月/小时/星期覆盖和采样间隔 |
| `data_quality.json` | 输入/标签画像、逐位置缺失、逐预测步难度、来源与模态覆盖、train-val/test 漂移 |
| `epochs.jsonl` | 每 epoch 损失、学习率、梯度、参数更新、AMP、耗时与显存 |
| `horizon_metrics.csv` | 每个预测步的 RMSE/MAE/Bias/相关性、持久性基线、裁剪前后指标 |
| `slice_metrics.csv` | 功率区间、爬坡方向/幅度、昼夜、缺失率和图像覆盖等分组指标 |
| `distillation.jsonl` | Teacher/Student 差异、Teacher 胜率、模态消融与蒸馏诊断 |
| `summary.json` | 最优 checkpoint、最终测试摘要、自动告警和人工复核提示 |

监控目录是专用目录：若其中出现上述七个文件之外的 NPZ、NPY、CSV、权重、日志或子目录，程序会拒绝启动监控。七个同名项若是符号链接或非普通文件也会被拒绝。仅测试模式复用目录时，还会校验 schema、`aggregate_only` 标记以及 CSV/JSONL 结构。自定义 `--monitor_dir` 不能指向训练运行目录或其父目录。

只需打包 `monitor/`。不要回传下列文件：

- `official_test_predictions.csv` 或其他逐样本预测文件
- 训练日志、Parquet 数据、图片、checkpoint 权重
- `test_results/` 下的图片、NPY、NPZ 或 CSV

`official_test_metrics.json` 本身是聚合指标，但不属于监控包；如需回传，应由数据管理方单独确认。

## 隐私边界

监控器不会持久化预测数组、标签数组、时间戳、图片路径、特征向量或 embedding。日期只在内存中转换成不可回传的自然日编号以计算支持度；路径不保留正文、文件名或可关联摘要；数值分布不输出精确最小值和最大值。缺少可验证日期时，数据画像和场景切片均 fail closed。

## 数据集画像字段

`data_quality.json` 的 `train`、`val`、`test` 均为独立聚合画像：

- `privacy_support`：有效日期样本数、自然日数和本 split 是否达到回传门槛。
- `features`：每个模型输入特征的有效/缺失支持度、均值、标准差和分位数。输入值处于数据加载器送入模型后的归一化空间。
- `target`、`current_power`、`ramp`：总体标签、当前功率和未来变化量，同时给出归一化值与 `original_units`。
- `missingness_by_position`：主时间序列每个 history step x feature、每个 target horizon、历史图像帧、YLJ 原始预报产品、当前功率，以及 Teacher 历史/未来特权输入的可用率。
- `sample_completeness`：每个样本的输入和标签完整度分布；任一位置存在低支持度缺失时会联动抑制。
- `horizon_label_profile`：每个预测步的标签分布、ramp 分布及无需模型即可计算的持久性 MAE/RMSE。Luoyang 生成 48 个 `lead_NNN`，YLJ 生成 16 个。
- `target_operating_ranges`、`ramp_ranges`：固定功率区间及上升/下降/稳定/大爬坡构成，用于判断训练样本是否覆盖困难工况。
- `images`：仅在图像模态启用时记录历史图像逐帧可用率、每样本 none/partial/full 状态和相对目标槽位的填充滞后。
- `source_codes`、`source_code_rates`：原始值、前向填充/替代产品、issue-time 持久性和训练均值回退的固定码统计；未知码统一归入 `other`，不会写任意类别文本。
- `comparisons_to_train`：val/test 相对 train 的均值、p05/p50/p95、IQR、标准差、缺失率、逐 horizon 和来源码比例变化；只有两侧均未被抑制时才出现。

来源码仅用于统计输入回退比例：

```text
0 = 原始值 / 本产品预报
1 = 因果前向填充 / 替代产品
2 = issue-time 持久性值
3 = 训练集均值
```

## 优先查看顺序

1. 查看 `summary.json` 的 `alerts` 和 `result.warnings`。
2. 查看 `data_quality.json` 的 `privacy_support`、逐位置缺失和来源码，先判断数据是否可用及回退是否过多。
3. 比较 `horizon_label_profile` 和 `comparisons_to_train`，判断远期标签更难、train-val/test 域漂移或困难工况覆盖不足。
4. 在 `horizon_metrics.csv` 将 Student、Teacher 与数据画像中的持久性基线按远近预测步比较。
5. 在 `slice_metrics.csv` 定位高功率、强爬坡、缺图或高缺失场景。
6. 用 `epochs.jsonl` 判断学习率衰减、梯度消失/爆炸、AMP 跳步和参数是否实际更新。
7. 用 `distillation.jsonl` 结合 Teacher 特权输入覆盖率，判断 Teacher 是否确实优于 Student，以及特权模态是否带来有效增益。

若切片没有输出，先检查数据是否达到两个隐私门槛；不要通过降低门槛来导出小样本分组，除非数据管理方明确批准。
