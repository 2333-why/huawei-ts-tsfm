# 五模型 TSFM-only 重构设计

## 目标

把当前仓库收敛为只服务时序基础模型（TSFM）的实验库：保留 Sundial、TimeMoE，新增 Chronos2、TiRex、TimesFM；基础模型代码全部放在 `models/`，删除独立 `foundation_models/` 包和纯时序/经典基线实现。入口统一为 `run.py`，继续支持两个功率数据集、`48→1` 与 `96→4小时` 两类任务，并提供一个接近 Time-Series-Library 示例风格的简单训练脚本。

本设计参考 `/opt/data/private/code/Time-Series-Library` 的模型 ID、预测长度、batch 和入口参数，但不复制其中的 eager import、硬编码 CUDA、不可微训练路径、多变量 reshape 错误或失效参数。

## 不可协商的边界

- 指定解释器仍是 `/opt/data/private/penv/time/bin/python`，本机两张 GPU 由两个独立进程使用；单个模型实例不套 `DataParallel`。
- 模型输入为有限浮点 `history [B, L, 1]`，输出为有限浮点 `forecast [B, H, 1]`；本仓库不新增 ETT/CSV 数据管线。
- 模型目录只保留 TSFM 代码；不得留下 `foundation_models/` 兼容包，也不得留下纯时序模型、经典 baseline、其 layers/runner/scripts/tests/docs。
- catalog、CLI 列表和测试收集不得导入可选模型包、下载权重或访问网络。
- 训练能力必须来自可微的基础模型路径。不能用通用外接回归头冒充 full/last-layer 微调，也不能为 TiRex 编造训练支持。
- 保留用户文件 `task.md` 与 `时序基础模型.md`，以及用户对后者的未提交修改。

## 模型目录与固定身份

`models/` 的最终职责划分：

```text
models/
  __init__.py       # 轻量公开接口，不 eager-import 可选依赖
  base.py           # backend protocol 与 shape/loss 校验
  common.py         # 五个 backend 共用的输入、设备、归一化辅助
  factory.py        # registry 字符串 entrypoint 的懒加载
  registry.py       # 固定模型身份、能力与参数选择器
  tasks.py          # 按 capability 生成实验矩阵
  trainability.py   # freeze/full/last-layer/LoRA 与参数审计
  Sundial.py
  TimeMoE.py
  Chronos2.py
  TiRex.py
  TimesFM.py
```

模型规格固定如下：

| name | checkpoint | revision | modes | last layer |
| --- | --- | --- | --- | --- |
| Sundial | `thuml/sundial-base-128m` | `3212e42564493f520593e5414af4367fc4b49226` | 4 modes | `flow_loss` |
| TimeMoE | `Maple728/TimeMoE-50M` | `446753ee48ff3726d0606a81d0092d54acee995e` | 4 modes | `lm_heads` |
| Chronos2 | `amazon/chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` | 4 modes | `output_patch_embedding` |
| TiRex | `NX-AI/TiRex` | `63c740922493f5fbe60b277609ec62babfba2762` | `zero_shot` only | none |
| TimesFM | `google/timesfm-2.5-200m-transformers` | `5a9806b9b291fad9233b5249d88263f1846304d3` | 4 modes | `output_projection_point` |

四个全局模式仍为 `zero_shot`、`adapter`、`full`、`last_layer`。`FoundationModelSpec.supported_modes` 是模型能力的唯一来源；runner 必须在加载包或权重前拒绝不支持的组合，并列出该模型的有效模式。

TimesFM 有意使用官方可微的 Transformers checkpoint，而不是参考仓库的 `google/timesfm-2.5-200m-pytorch` NumPy forecast 包装器。后者只能推理，无法满足 full/last-layer/LoRA 的真实梯度语义。

## Backend 契约

runner 只依赖下列真实契约：

```python
class FoundationBackend(Protocol):
    model_name: str
    model_id: str
    revision: str
    model: torch.nn.Module

    def predict(self, history: Tensor, pred_len: int) -> Tensor: ...
    def training_loss(self, history: Tensor, target: Tensor, target_mask: Tensor) -> Tensor: ...
    def configure_trainable(self, mode: str, lora: LoraSettings) -> TrainabilityReport: ...
```

checkpoint 保存/恢复由 runner 对 `backend.model.state_dict()` 负责；删除未被调用的 `save()` 假协议。所有 backend 在构造时接收显式 `device` 和 pinned `revision`，不得自行选择 `cuda:0`。

### 各模型调用

- Sundial 与 TimeMoE 沿用已验证的 generate 预测和现有可微训练目标；共同验证/归一化代码移到 `models/common.py`，模型特有逻辑分别留在模型文件中。
- Chronos2 懒加载 `Chronos2Pipeline.from_pretrained`，将其原生 `nn.Module` 注册为 `backend.model`。预测使用 pipeline 的点预测结果；训练直接调用原生 model 的 `context`、`future_target`、`future_target_mask` 与足够的 `num_output_patches`，验证返回的 scalar loss。
- TiRex 懒加载 `tirex.load_model`，调用 `forecast(context=..., prediction_length=...)`，只取 point forecast。它只有 zero-shot；`training_loss` 和任何训练模式均给出定向错误。
- TimesFM 懒加载 `transformers.TimesFm2_5ModelForPrediction`。预测从 `mean_predictions` 裁到 `pred_len`；训练用同一可微 forward 的 point forecast 和仓库的二值 `target_mask` 计算 masked MSE。官方 `future_values` loss 没有 mask 参数，因此不能直接用于含缺失目标的功率数据。

Chronos2、TiRex、TimesFM 负责自己的缩放，不能再叠加参考包装器的手工标准化；Sundial/TimeMoE 保持现有且已有测试覆盖的标准化所有权。

## 参数训练与审计

`trainability.py` 接受 registry 中的 model-specific adapter targets：Sundial/TimeMoE 使用现有 transformer linear suffix，Chronos2 与 TimesFM 使用其实际线性层集合（允许 PEFT `all-linear`）。每次配置都返回不可变 `TrainabilityReport`，包含总参数、可训练参数、比例、至多 50 个名字和完整名字摘要。

- `zero_shot`：全部冻结，eval，无 optimizer。
- `adapter`：基础参数冻结，只有 `lora_` 参数可训练；未匹配到模块或发生基础参数泄漏即失败。
- `full`：基础模型全部参数可训练。
- `last_layer`：只允许 registry 的精确模块前缀，必须命中严格子集。

TiRex 的 unsupported mode 在上述策略执行前失败，所以不会出现“冻结黑盒但记录为 full”的结果。

## Runner、参数兼容和数据流

主入口为 `run.py`。保留现有可靠行为：数据 split、防泄漏、训练/验证/测试生命周期、最佳 checkpoint、预测 CSV、指标 JSON、完成清单和失败时移除完成标记。`PowerBatch/power_only_batch` 迁入 `data_provider/power_only.py`；`json_safe/sha256/write_predictions` 迁入 `utils/artifacts.py`，从而可以删除 `models/tslib_adapter.py` 和 `run_time_series.py`。

canonical 参数仍为 `--dataset --model --mode --seq_len --pred_len --epochs --smoke`。为使简单脚本接近参考仓库，增加并实际归一化这些别名：

- `--data` → `--dataset`
- `--train_epochs` → `--epochs`
- `--debug True|False` → `--smoke`
- `--model_id` 进入实验身份与产物 metadata
- `--task_name` 与 `--is_training` 只做一致性校验：zero-shot 必须是 `zero_shot_forecast/0`，训练模式必须是 `long_term_forecast/1`

别名和 canonical 参数同时提供且冲突时必须失败。不要用 `parse_known_args` 静默吞掉参考仓库的无效参数。

数据流保持：

```text
config + Parquet
  -> PowerOnlyParquetDataset
  -> history / target / target_mask / issue_time
  -> backend
  -> forecast [B,H,1]
  -> power-scale inverse + metrics
  -> best.pt / predictions.csv / metrics.json / completion.tsv
```

## 任务矩阵与脚本

固定四个数据窗口不变：

| setting | dataset | seq_len | pred_len |
| --- | --- | ---: | ---: |
| `seq48_pred1` | `skippd_luoyang` | 48 | 1 |
| `seq48_pred1` | `pvod_station00_ylj` | 48 | 1 |
| `seq96_h4` | `skippd_luoyang` | 96 | 48 |
| `seq96_h4` | `pvod_station00_ylj` | 96 | 16 |

每个窗口有 17 个有效 model/mode 组合：4 个可训练模型 × 4 modes，加 TiRex zero-shot，共 68 个实验。矩阵只生成支持项；计数、summary 行数和 GPU 分配均从实际任务迭代器推导，不再硬编码 32。

保留双进程/双 GPU 批处理与 smoke wrapper，并新增 `scripts/train_model.sh`。新脚本使用 `model_name`、`seq_len`、`debug_mode`、shell 数组和嵌套循环，调用 `python -u run.py`；默认示例为 TimesFM adapter，遍历仓库真实的两个数据集/两类任务，不伪装支持 ETTh1.csv。

## 清理结果

删除 `foundation_models/`、`baselines/`、`layers/`、八个纯时序模型、`models/tslib_*`、`run_time_series.py`、纯时序脚本和测试、`PURE_TIME_SERIES.md`、旧 classical/pure 设计文档，以及与本设计冲突的两模型旧 spec/plan。`README.md` 整体重写为 TSFM-only；`FOUNDATION_MODELS.md` 删除，避免两份运行说明漂移。

结果目录中的字面量 `foundation_models` 是历史产物路径，不是 Python 包；本轮保留它以免无必要破坏恢复契约。

## 环境事实与依赖策略

指定解释器当前为 Python 3.8.18，已有 Torch 2.3.1、Transformers 4.46.2、PEFT 0.13.2，但没有 `chronos-forecasting`、`tirex-ts` 或 TimesFM 2.5 Transformers class。三项现代上游均要求 Python 3.10+；因此不能在该解释器中诚实完成真实权重 smoke。

基础 `requirements.txt` 保持当前 Python 3.8 可安装并删除纯时序遗留依赖。另提供带 Python 版本标记和精确说明的 modern optional requirements/README 命令；预检必须将版本不兼容、缺包和 cache 缺失分别报告，且非零退出。实现过程中不自动升级解释器、不修改外部环境、不把 fake 测试称为真实模型验证。

## 验证标准

- 指定解释器运行完整 pytest；基线现状是 411 passed、1 failed，唯一失败来自测试复制仓库时遇到工作区 `codex` 特殊文件。新脚本测试不得复制整个仓库，最终 suite 必须为零失败。
- 单测使用完整接口 fake，覆盖五模型 loader 参数、预测 shape、三套新训练路径、capability 拒绝、参数审计和 missing-dependency 错误；测试不下载权重。
- `run.py --list-models` 输出五个模型，未安装 optional packages 时仍成功；任务迭代器产生 68 个唯一任务。
- fake-runner 双 GPU smoke 产生 68 个 PASS，且两个物理 GPU 队列各自串行、整体存在并发、失败可传播。
- `scripts/check_foundation_environment.py` 在当前解释器诚实报告三模型的 Python/package 阻塞；Sundial/TimeMoE 的已安装路径继续可检查。
- 只有依赖和 checkpoint 真正可用时才运行每个模型的单 batch GPU smoke。未运行的真实 smoke 在交付报告中列为外部环境限制，不能由 fake 结果替代。
