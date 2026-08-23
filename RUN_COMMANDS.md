# 核心运行命令

以下命令均在仓库根目录执行：

```bash
cd /home/ma-user/work/why/FACTS-Luoyang-why
```

## Luoyang 调优流程

依次执行数据预检、Teacher 训练与测试、Student/KD 训练、Student 正式测试和结果校验：

```bash
bash scripts/run_luoyang_tuned.sh
```

后台运行：

```bash
mkdir -p logs_luoyang_tuned
nohup bash scripts/run_luoyang_tuned.sh \
  > logs_luoyang_tuned/nohup_full_run.log 2>&1 < /dev/null &
echo $! > logs_luoyang_tuned/nohup_full_run.pid
tail -f logs_luoyang_tuned/nohup_full_run.log
```

退出 `tail -f` 使用 `Ctrl+C`，后台训练不会停止。

使用已有调优 checkpoint 只重新测试 Student：

```bash
python scripts/luoyang_pipeline.py test \
  --config configs/luoyang_tuned.json \
  --prefer-latest-checkpoint
python scripts/verify_official_outputs.py luoyang \
  --config configs/luoyang_tuned.json
```

## Luoyang 与 YLJ 全流程

依次运行两套数据集：

```bash
bash scripts/run_full_experiments.sh
```

复用已有 Luoyang Teacher，从 Luoyang Student 开始，再运行 YLJ：

```bash
bash scripts/run_full_experiments.sh --after-luoyang-teacher
```

只运行 YLJ：

```bash
bash scripts/run_full_experiments.sh --datasets ylj
```

## 复用已有 Student

只测试、不重新训练：

```bash
# Luoyang
python scripts/run_all_datasets.py \
  --datasets luoyang --skip-preflight --skip-train

# YLJ
python scripts/run_all_datasets.py \
  --datasets ylj --skip-preflight --skip-train
```

## 运行状态

标准双数据集流程：

```bash
bash scripts/check_full_experiment_status.sh
```

Luoyang 调优流程：

```bash
tail -f logs_luoyang_tuned/nohup_full_run.log
ps -fp "$(cat logs_luoyang_tuned/nohup_full_run.pid)"
nvidia-smi
```

## 结果位置

- Luoyang 标准流程：`results_luoyang/`
- Luoyang 调优流程：`results_luoyang_tuned/`
- YLJ：`results_ylj/`

每次正式测试生成 `official_test_predictions.csv` 和 `official_test_metrics.json`。
