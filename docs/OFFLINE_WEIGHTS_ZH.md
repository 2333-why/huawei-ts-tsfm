# 浏览器下载权重与桶上离线加载

服务器不需要访问 Hugging Face。在本地浏览器下载以下固定 revision 的文件，上传到对象存储后，将桶挂载目录交给代码即可。

## 如何搜索

在浏览器搜索框中输入 `site:huggingface.co <model ID>`，只进入 `huggingface.co` 官方模型页。为防止上游更新导致实验漂移，使用下面已经锁定 commit 的链接，不要下载 `main` 的未锁定版本。

| 本地目录 | 搜索词 / 模型 ID | 固定文件页 | 必需大文件 |
|---|---|---|---|
| `Sundial` | `thuml/sundial-base-128m` | <https://huggingface.co/thuml/sundial-base-128m/tree/3212e42564493f520593e5414af4367fc4b49226> | `model.safetensors` (513,341,448 bytes) |
| `TimeMoE` | `Maple728/TimeMoE-50M` | <https://huggingface.co/Maple728/TimeMoE-50M/tree/446753ee48ff3726d0606a81d0092d54acee995e> | `model.safetensors` (226,760,264 bytes) |
| `TimeMoE200M` | `Maple728/TimeMoE-200M` | <https://huggingface.co/Maple728/TimeMoE-200M/tree/794591bfeb1225fdf742cec0f4c71f20c3f3b87e> | `model.safetensors` (906,450,104 bytes) |
| `Chronos2` | `amazon/chronos-2` | <https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c> | `model.safetensors` (477,930,472 bytes) |
| `TiRex` | `NX-AI/TiRex` | <https://huggingface.co/NX-AI/TiRex/tree/63c740922493f5fbe60b277609ec62babfba2762> | `model.ckpt` (141,230,262 bytes) |
| `TimesFM` | `google/timesfm-2.5-200m-transformers` | <https://huggingface.co/google/timesfm-2.5-200m-transformers/tree/5a9806b9b291fad9233b5249d88263f1846304d3> | `model.safetensors` (925,187,448 bytes) |

进入文件页后，逐个点击文件，再点右上角 **download** 图标。大文件必须下载到表中字节数；如果只有几百字节，下载到的是 Git LFS 指针，不是权重。

## 必需目录结构

```text
tsfm_weights/
├── Sundial/
│   ├── config.json
│   ├── generation_config.json
│   ├── configuration_sundial.py
│   ├── modeling_sundial.py
│   ├── flow_loss.py
│   ├── ts_generation_mixin.py
│   └── model.safetensors
├── TimeMoE-50M/
│   ├── config.json
│   ├── generation_config.json
│   ├── configuration_time_moe.py
│   ├── modeling_time_moe.py
│   ├── ts_generation_mixin.py
│   └── model.safetensors
├── TimeMoE-200M/
│   ├── config.json
│   ├── generation_config.json
│   ├── configuration_time_moe.py
│   ├── modeling_time_moe.py
│   ├── ts_generation_mixin.py
│   └── model.safetensors
├── Chronos2/
│   ├── config.json
│   └── model.safetensors
├── TiRex/
│   └── model.ckpt
└── TimesFM/
    ├── config.json
    └── model.safetensors
```

六个大权重合计 3,190,899,998 bytes，约 3.19 GB（2.97 GiB）。其他 README、图片和 TiRex ONNX 文件不是当前 PyTorch 实验所必需。

## 服务器上指定桶挂载目录

若六个子目录都在同一根目录：

```bash
export TSFM_WEIGHTS_ROOT=/mnt/your-bucket/tsfm_weights
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/check_local_weights.py --verify-sha256
```

如果六个权重在不同位置，分别设置（优先级高于 `TSFM_WEIGHTS_ROOT`）：

```bash
export SUNDIAL_WEIGHT_DIR=/mnt/bucket-a/Sundial
export TIMEMOE_WEIGHT_DIR=/mnt/bucket-a/TimeMoE-50M
export TIMEMOE_200M_WEIGHT_DIR=/mnt/bucket-a/TimeMoE-200M
export CHRONOS2_WEIGHT_DIR=/mnt/bucket-b/Chronos2
export TIREX_WEIGHT_DIR=/mnt/bucket-b/TiRex
export TIMESFM_WEIGHT_DIR=/mnt/bucket-c/TimesFM
```

配置后，原有训练、推理和 `scripts/test_updated_tsfm.sh` 命令无需增加模型路径参数。代码会优先从这些目录加载，且仍将实验身份记录为仓库中锁定的 model ID 和 revision。

## PVMMoE 服务器固定路径

仓库已提供路径脚本，对应目录为：

```text
/data/PVMMoE/PRETRAINED_MODELS/sundial-base-128m
/data/PVMMoE/PRETRAINED_MODELS/chronos2
/data/PVMMoE/PRETRAINED_MODELS/TimeMoE-50M
/data/PVMMoE/PRETRAINED_MODELS/TimeMoE-200M
/data/PVMMoE/PRETRAINED_MODELS/TiRex
/data/PVMMoE/PRETRAINED_MODELS/TimesFM
```

在服务器上执行：

```bash
source scripts/use_pvmoe_local_weights.sh
python scripts/check_local_weights.py --verify-sha256
```

当前 PVMMoE 完整实验使用 Sundial、TimeMoE-50M、TimeMoE-200M、Chronos2 和 TiRex，暂时不加载或测试 TimesFM。

在已激活的 Python 环境中执行一条命令，将使用 8 张 GPU 运行全部 68 个任务：

```bash
PYTHON="$(which python)" \
GPUS="0 1 2 3 4 5 6 7" \
RESUME=1 \
bash scripts/run_pvmoe_5models_8gpu.sh
```

该入口会先读取 Luoyang 数据实际范围，并恢复原固定切分：训练/验证为
`2026-04-05～2026-05-11`，测试为 `2026-05-11～2026-06-12`。YLJ 继续严格使用
`configs/datasets/pvod_station00_ylj.yaml` 中原来的时间范围，不做重新划分。

脚本会实际读取两个 Parquet，检查列、时间戳、有效功率和 train/val/test 样本数，
再生成 Luoyang 一个月训练/一个月测试及 YLJ 原切分配置。随后校验五个权重、环境和 GPU。
任何预检失败都不会开始实验。全部通过后会生成：

```text
results_foundation_5models_8gpu/dataset_preflight/dataset_preflight.json
results_foundation_5models_8gpu/dataset_preflight/luoyang_one_month_train_one_month_test.json
results_foundation_5models_8gpu/dataset_preflight/ylj_original_split.json
results_foundation_5models_8gpu/run_summary.tsv
results_foundation_5models_8gpu/results_summary.md
```

`RESUME=1` 会严格校验已有产物，只跳过身份、数据指纹和文件 hash 全部一致的成功任务。
