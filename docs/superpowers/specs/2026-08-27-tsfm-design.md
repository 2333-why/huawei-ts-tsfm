# 时序基础模型迁移设计

## 目标与范围

在 `tsfm` 分支和 `/opt/data/private/code/tsfm-ts` 工作区中，为现有纯功率时序框架增加可审计的时序基础模型实验链路。链路覆盖三个模型、两个数据集、两个预测任务和四种运行模式，同时复用现有数据边界、预测 CSV、原功率单位指标与完成清单。

本轮主模型目录固定为：

- `Chronos2`：`amazon/chronos-2`；
- `Sundial`：`thuml/sundial-base-128m`；
- `TimeMoE`：`Maple728/TimeMoE-50M`。

选择这三个模型，是因为它们既存在于 `/opt/data/private/code/Time-Series-Library` 的基础模型目录中，又能暴露真实 PyTorch 参数和训练损失。`TiRex` 官方只提供 zero-shot 推理、不开放用户微调；`TimesFM 2.5` 当前可训练 Transformers 端要求的版本与指定环境中 TimeMoE/Sundial 的远程代码不兼容；旧 `Chronos` 与 Chronos-2 属于同一模型族。因此它们不进入本轮四模式主矩阵，代码也不得把冻结黑盒或外部校准器称为“全量微调”。

## 明确解释

- “两个模拟数据集”解释为当前框架已配置且本机文件存在的 `skippd_luoyang` 与 `pvod_station00_ylj`。不新增或合成第三份数据。
- `48-1` 表示 `seq_len=48`、未来 1 个采样点。
- `96-4小时` 表示 `seq_len=96`、未来 4 个物理小时：SKIPP'D 的 `pred_len=48`，PVOD 的 `pred_len=16`。
- 模型输入仍只有历史归一化功率 `[batch, seq_len, 1]`；不得向模型传入天气、图像、未来目标、测试统计量或其他特权信息。
- 统一运行模式为 `zero_shot`、`adapter`、`full`、`last_layer`。

## 训练模式语义

`zero_shot` 不建立优化器、不访问训练或验证 loader，直接用预训练权重预测测试集。产物必须记录 `epochs=0`、`train_steps=0`、`val_steps=0` 和正数 `test_steps`。

`adapter` 使用 PEFT LoRA 注入后端声明的线性层。基础参数全部冻结，只有 LoRA 参数可训练。若未注入参数、基础参数仍可训练或可训练比例异常，运行必须失败。

`full` 解冻基础模型全部可学习参数，按模型原生训练损失更新。可训练参数必须等于模型全部参数；否则运行失败。

`last_layer` 冻结基础模型，只解冻后端明确声明的预测输出模块：Chronos-2 的输出 patch/projection head、Sundial 的 `flow_loss` 预测头、TimeMoE 的输出层。必须至少有一个而非全部参数可训练；模块名匹配为空或过宽时运行失败。

每个 checkpoint 和 `metrics.json` 都记录：运行模式、模型 ID、模型 revision、总参数数、可训练参数数、可训练参数名摘要、训练/验证/测试步数。训练模式不能通过仅训练一个新建的通用回归头来冒充基础模型微调。

## 架构

新增独立入口 `run_foundation_model.py`，不把基础模型的特殊生命周期塞入现有 `run_time_series.py`。数据读取、反归一化、指标和预测文件格式复用现有纯时序代码中的稳定行为。

`foundation_models/registry.py` 保存不可变模型规格和懒加载工厂。列目录和解析 CLI 时不得导入可选依赖或下载权重。

`foundation_models/backends.py` 定义统一后端协议：

```python
class FoundationBackend(Protocol):
    model_name: str
    model_id: str

    def predict(self, history: torch.Tensor, pred_len: int) -> torch.Tensor: ...
    def training_loss(
        self,
        history: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor: ...
    def configure_trainable(self, mode: str, lora: LoraSettings) -> TrainabilityReport: ...
    def save(self, path: Path) -> None: ...
```

所有 `predict` 返回 `[B, pred_len, 1]` 的有限浮点 Tensor。后端负责调用各模型的官方 API、取点预测（样本均值或中位数）并保持设备一致。后端不得在构造时硬编码 `cuda`；设备由 CLI 传入。

`foundation_models/trainability.py` 集中实现冻结、LoRA、最后一层选择和参数审计。runner 只消费 `TrainabilityReport`，不按模型名散落条件分支。

## 数据流与防泄漏

```text
Parquet + dataset config
  -> PowerOnlyParquetDataset
  -> history / target / target_mask / issue_time
  -> backend (history only for predict)
  -> normalized forecast
  -> inverse power scaling and clipping
  -> predictions.csv + metrics.json + completion.tsv
```

训练模式使用 train loader 更新参数，用 val loader选择最佳 checkpoint，最终只在 test loader 评估。zero-shot 只构造 test loader。缺失目标由 mask 排除；训练损失必须拒绝整个阶段零有效目标。

## 实验矩阵与双 GPU 编排

固定任务为：

| setting | dataset | seq_len | pred_len |
| --- | --- | ---: | ---: |
| `seq48_pred1` | `skippd_luoyang` | 48 | 1 |
| `seq48_pred1` | `pvod_station00_ylj` | 48 | 1 |
| `seq96_h4` | `skippd_luoyang` | 96 | 48 |
| `seq96_h4` | `pvod_station00_ylj` | 96 | 16 |

三个模型 × 四个任务 × 四种模式，共 48 个唯一实验。`scripts/run_all_foundation_models_2gpu.sh` 启动两条 GPU 队列，每张卡顺序执行 24 个任务；同一物理 GPU 上不得并发两个模型。子进程设置 `CUDA_VISIBLE_DEVICES=<physical>`，内部统一传 `--device cuda:0`。

脚本支持 `SMOKE=1`、`RESUME=1`、`GPUS="0 1"`、`OUTPUT_ROOT`、两个 Parquet 覆盖变量。恢复只接受身份、模式、模型 revision、数据指纹、阶段步数和产物哈希全部匹配的任务。

## 产物与失败语义

单任务目录为 `<OUTPUT_ROOT>/<setting>/<dataset>/<model>/<mode>/`，包含：

- `best.pt` 或后端可恢复的 `save_pretrained` 目录；
- `predictions.csv`；
- `metrics.json`；
- `completion.tsv`；
- `data_fingerprint.txt`；
- 批处理时的 `run.log`。

依赖缺失、模型权重下载失败、模型 revision 不可解析、预测 shape 错误、输出非有限、训练模式无真实可训练参数、CUDA 不可用或 loader 为空都必须返回非零状态，且不得留下可被恢复逻辑误认的 `completion.tsv`。

## 验证标准

- 原仓库测试保持通过。
- 新单元测试不下载权重，通过 fake backend 验证目录、模式、参数审计、shape、checkpoint 与指标契约。
- 脚本测试使用 fake runner，证明恰好展开 48 个唯一任务、两张 GPU 各 24 个、任务长度映射正确、同卡无重叠且失败能够传播。
- 指定环境能够列出三个模型和四种模式而不下载权重。
- 安装可选依赖与缓存权重后，每个后端至少完成一次真实 zero-shot 单 batch GPU smoke；三个训练模式至少用一个 batch 完成反向传播并证明预期参数发生变化、冻结参数保持不变。
- 完整 48 项训练不是代码交付的默认验证步骤；它由批量脚本显式启动，避免在未确认训练预算时自动占用两张 GPU 数小时。
