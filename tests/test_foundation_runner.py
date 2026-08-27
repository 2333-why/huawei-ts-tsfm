from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


class TinyDataset:
    instances = []

    def __init__(self, config_path, flag, history_points, forecast_steps):
        self.config_path = Path(config_path)
        self.flag = flag
        self.seq_len = int(history_points)
        self.pred_len = int(forecast_steps)
        self.power_scale = 2.0
        self.rated_power = 4.0
        self.target_floor = 0.0
        self.forecast_step = timedelta(minutes=5)
        TinyDataset.instances.append(self)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "history": np.full((self.seq_len, 1), float(index + 1), dtype=np.float32),
            "target": np.full((self.pred_len, 1), 1.0, dtype=np.float32),
            "target_mask": np.ones((self.pred_len,), dtype=bool),
            "issue_time_ns": np.int64(1735689600000000000 + index * 300000000000),
        }


class LeakageDataset(TinyDataset):
    def __getitem__(self, index):
        return {
            "history": np.array([[11.0], [12.0]], dtype=np.float32),
            "target": np.array([[101.0], [102.0], [103.0]], dtype=np.float32),
            "target_mask": np.ones((3,), dtype=bool),
            "issue_time_ns": np.int64(1735689600000000000),
        }


class ExtremeDataset(TinyDataset):
    def __init__(self, config_path, flag, history_points, forecast_steps):
        super().__init__(config_path, flag, history_points, forecast_steps)
        self.power_scale = 2.0
        self.rated_power = 4.0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "history": np.full((2, 1), 7.0, dtype=np.float32),
            "target": np.array([[1.0], [1.0]], dtype=np.float32),
            "target_mask": np.array([True, False], dtype=bool),
            "issue_time_ns": np.int64(1735689600000000000),
        }


class RestoreDataset(TinyDataset):
    def __getitem__(self, index):
        return {
            "history": np.zeros((self.seq_len, 1), dtype=np.float32),
            "target": np.zeros((self.pred_len, 1), dtype=np.float32),
            "target_mask": np.ones((self.pred_len,), dtype=bool),
            "issue_time_ns": np.int64(1735689600000000000),
        }


def _batch(value=1.0, target=1.0, mask=None, issue=1735689600000000000):
    if mask is None:
        mask = torch.ones(1, 1, dtype=torch.bool)
    return {
        "history": torch.full((1, 2, 1), value),
        "target": torch.full((1, 1, 1), target),
        "target_mask": mask,
        "issue_time_ns": torch.tensor([issue], dtype=torch.int64),
    }


class RecordingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.strict_loads = []

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        self.strict_loads.append((dict(state_dict), strict))
        return super().load_state_dict(state_dict, strict=strict, *args, **kwargs)


class FakeBackend:
    instances = []
    optimizer_created = False

    def __init__(self, *args, **kwargs):
        self.model_name = "Sundial"
        self.model_id = "thuml/sundial-base-128m"
        self.revision = "3212e42564493f520593e5414af4367fc4b49226"
        self.model = RecordingModule()
        self.configure_calls = []
        self.predict_histories = []
        self.predict_pred_lens = []
        self.loss_calls = []
        self.load_calls = self.model.strict_loads
        FakeBackend.instances.append(self)

    def configure_trainable(self, mode, lora):
        self.configure_calls.append((mode, lora))
        for parameter in self.model.parameters():
            parameter.requires_grad_(mode != "zero_shot")
        return SimpleNamespace(
            mode=mode,
            total_parameters=1,
            trainable_parameters=0 if mode == "zero_shot" else 1,
            trainable_ratio=0.0 if mode == "zero_shot" else 1.0,
            trainable_names=() if mode == "zero_shot" else ("weight",),
            trainable_names_sha256=hashlib.sha256(
                b"" if mode == "zero_shot" else b"weight"
            ).hexdigest(),
        )

    def training_loss(self, history, target, target_mask):
        self.loss_calls.append((history.detach().clone(), target.detach().clone(), target_mask.detach().clone()))
        return (self.model.weight - target[target_mask].mean()).square()

    def predict(self, history, pred_len):
        self.predict_histories.append(history.detach().clone())
        self.predict_pred_lens.append(pred_len)
        return self.model.weight.detach().expand(history.shape[0], pred_len, 1)


class NestedRestoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1, 1, bias=False)
        self.lora = nn.Module()
        self.lora.A = nn.Parameter(torch.tensor(0.0))
        self.lora.B = nn.Parameter(torch.tensor(0.0))
        self.head = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.base.weight.zero_()
            self.head.weight.zero_()
        self.strict_loads = []

    def forecast_value(self):
        return self.base.weight[0, 0] + self.lora.A + self.lora.B + self.head.weight[0, 0]

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        self.strict_loads.append((dict(state_dict), strict))
        return super().load_state_dict(state_dict, strict=strict, *args, **kwargs)


class RestoreBackend:
    def __init__(self, *args, **kwargs):
        self.model_name = "Sundial"
        self.model_id = "thuml/sundial-base-128m"
        self.revision = "3212e42564493f520593e5414af4367fc4b49226"
        self.model = NestedRestoreModel()
        self.loss_calls = 0

    def configure_trainable(self, mode, lora):
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if mode == "full":
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
        elif mode == "adapter":
            self.model.lora.A.requires_grad_(True)
            self.model.lora.B.requires_grad_(True)
        elif mode == "last_layer":
            self.model.head.weight.requires_grad_(True)
        names = tuple(sorted(name for name, parameter in self.model.named_parameters() if parameter.requires_grad))
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        return SimpleNamespace(
            mode=mode,
            total_parameters=total,
            trainable_parameters=trainable,
            trainable_ratio=float(trainable) / total,
            trainable_names=names,
            trainable_names_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest(),
        )

    def training_loss(self, history, target, target_mask):
        self.loss_calls += 1
        desired = 0.0 if self.loss_calls == 1 else 10.0
        return (self.model.forecast_value() - desired).square()

    def predict(self, history, pred_len):
        return self.model.forecast_value().detach().expand(history.shape[0], pred_len, 1)


def _args(module, tmp_path, mode="zero_shot", **overrides):
    values = dict(
        dataset="skippd_luoyang",
        model="Sundial",
        mode=mode,
        config=tmp_path / "config.json",
        seq_len=2,
        pred_len=1,
        epochs=1,
        max_train_steps=0,
        max_eval_steps=0,
        max_test_steps=0,
        batch_size=1,
        num_workers=0,
        prefetch_factor=2,
        learning_rate=0.1,
        seed=1,
        device="cpu",
        output_dir=tmp_path / "output",
        smoke=False,
        lora_r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_bias="none",
        lora_task_type=None,
        lora_fan_in_fan_out=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_fakes(monkeypatch, tmp_path):
    import run

    TinyDataset.instances = []
    FakeBackend.instances = []
    monkeypatch.setattr(run, "PowerOnlyParquetDataset", TinyDataset)
    monkeypatch.setattr(run, "build_backend", lambda *args, **kwargs: FakeBackend())
    return run


def test_list_commands_are_lazy_and_exact(capsys, monkeypatch):
    import run
    from models import MODEL_NAMES, RUN_MODES

    monkeypatch.setattr(
        run,
        "PowerOnlyParquetDataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dataset constructed")),
    )
    monkeypatch.setattr(
        run,
        "build_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backend constructed")),
    )

    assert run.main(["--list-models"]) == 0
    assert tuple(capsys.readouterr().out.splitlines()) == MODEL_NAMES
    assert run.main(["--list-modes"]) == 0
    assert tuple(capsys.readouterr().out.splitlines()) == RUN_MODES


def test_list_commands_are_isolated_in_a_fresh_process():
    repository = Path(__file__).resolve().parents[1]
    runner = repository / "run.py"
    code = """
import builtins
import runpy
import sys

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'transformers', 'peft'}:
        raise AssertionError('optional model dependency imported during listing')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.argv = ['run.py', '--list-models']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    raise SystemExit(exc.code)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repository),
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["Sundial", "TimeMoE", "Chronos2", "TiRex", "TimesFM"]
    assert completed.stderr == ""


def test_parser_requires_runtime_identity_and_positive_lengths():
    import run

    with pytest.raises(SystemExit):
        run.parse_args([])
    with pytest.raises(SystemExit):
        run.parse_args(["--dataset", "skippd_luoyang", "--model", "Sundial"])
    with pytest.raises(SystemExit):
        run.parse_args([
            "--dataset", "skippd_luoyang", "--model", "Sundial", "--mode", "zero_shot",
            "--seq_len", "0", "--pred_len", "1",
        ])


def test_parser_defaults_config_and_mode_specific_smoke():
    import run

    args = run.parse_args([
        "--dataset", "pvod_station00_ylj", "--model", "TimeMoE", "--mode", "zero_shot",
        "--seq_len", "48", "--pred_len", "1", "--epochs", "8", "--smoke",
    ])
    assert args.config.name == "pvod_station00_ylj.yaml"
    assert (args.epochs, args.batch_size, args.max_train_steps, args.max_eval_steps, args.max_test_steps) == (0, 2, 1, 1, 1)

    trained = run.parse_args([
        "--dataset", "skippd_luoyang", "--model", "Sundial", "--mode", "full",
        "--seq_len", "48", "--pred_len", "1", "--smoke",
    ])
    assert trained.epochs == 1


def test_runtime_rejects_cuda_when_unavailable(monkeypatch):
    import run

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises((RuntimeError, ValueError), match="CUDA|cuda"):
        run._device_from_arg("cuda:0")


def test_seed_is_applied_before_dataset_and_backend_and_recorded(monkeypatch, tmp_path):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    observations = []

    class SeedDataset(TinyDataset):
        def __init__(self, config_path, flag, history_points, forecast_steps):
            observations.append(
                ("dataset", random.random(), float(np.random.random()), float(torch.rand(1)))
            )
            super().__init__(config_path, flag, history_points, forecast_steps)

    class SeedBackend(FakeBackend):
        def __init__(self, *args, **kwargs):
            observations.append(
                ("backend", random.random(), float(np.random.random()), float(torch.rand(1)))
            )
            super().__init__(*args, **kwargs)

    run.PowerOnlyParquetDataset = SeedDataset
    monkeypatch.setattr(run, "build_backend", lambda *args, **kwargs: SeedBackend())
    args = _args(run, tmp_path, mode="zero_shot", seed=12345)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    expected = [
        ("dataset", random.random(), float(np.random.random()), float(torch.rand(1))),
        ("backend", random.random(), float(np.random.random()), float(torch.rand(1))),
    ]
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)

    result = run.run(args)

    assert observations == expected
    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    assert metrics["seed"] == 12345
    assert checkpoint["seed"] == 12345


def test_backend_identity_must_match_pinned_registry(monkeypatch, tmp_path):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    backend = FakeBackend()
    backend.model_id = "other/model"
    backend.revision = "other-revision"
    monkeypatch.setattr(run, "build_backend", lambda *args, **kwargs: backend)
    args = _args(run, tmp_path, mode="zero_shot")

    with pytest.raises(ValueError, match="pinned|identity"):
        run.run(args)
    assert not (args.output_dir / "completion.tsv").exists()


def test_zero_shot_constructs_only_test_without_optimizer_and_writes_artifacts(monkeypatch, tmp_path):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    run.PowerOnlyParquetDataset = LeakageDataset
    monkeypatch.setattr(torch.optim, "Adam", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("optimizer")))
    args = _args(run, tmp_path, mode="zero_shot", smoke=True, pred_len=3)
    result = run.run(args)

    assert [dataset.flag for dataset in TinyDataset.instances] == ["test"]
    backend = FakeBackend.instances[0]
    assert [call[0] for call in backend.configure_calls] == ["zero_shot"]
    assert len(backend.loss_calls) == 0
    assert len(backend.predict_histories) == 1
    assert backend.predict_histories[0].shape == (2, 2, 1)
    assert torch.equal(
        backend.predict_histories[0],
        torch.tensor([[[11.0], [12.0]], [[11.0], [12.0]]]),
    )
    assert backend.predict_pred_lens == [3]
    assert set(result) == {"checkpoint", "predictions", "metrics", "completion"}
    assert all(path.is_file() for path in result.values())
    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    assert metrics["epochs"] == 0
    assert metrics["phase_steps"] == {"train": 0, "val": 0, "test": 1}
    assert metrics["limits"] == {"train": 1, "val": 1, "test": 1}
    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    assert checkpoint["mode"] == "zero_shot"
    assert "state_dict" not in checkpoint
    assert checkpoint["epochs"] == 0
    assert checkpoint["phase_steps"] == {"train": 0, "val": 0, "test": 1}


@pytest.mark.parametrize("mode", ["adapter", "full", "last_layer"])
def test_training_modes_restore_best_state_and_use_trainable_parameters(monkeypatch, tmp_path, mode):
    run = _install_fakes(monkeypatch, tmp_path)
    args = _args(run, tmp_path, mode=mode, smoke=True)
    result = run.run(args)

    assert [dataset.flag for dataset in TinyDataset.instances] == ["train", "val", "test"]
    backend = FakeBackend.instances[0]
    assert [call[0] for call in backend.configure_calls] == [mode]
    assert backend.loss_calls
    assert backend.load_calls or torch.load(result["checkpoint"], map_location="cpu")["state_dict"]
    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    assert metrics["mode"] == mode
    assert metrics["trainability"]["mode"] == mode
    assert metrics["phase_steps"] == {"train": 1, "val": 1, "test": 1}
    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    assert checkpoint["epochs"] == 1
    assert checkpoint["phase_steps"] == {"train": 1, "val": 1, "test": 1}


@pytest.mark.parametrize("mode", ["adapter", "full", "last_layer"])
def test_training_restores_epoch_zero_best_state_before_test(monkeypatch, tmp_path, mode):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    run.PowerOnlyParquetDataset = RestoreDataset
    backend = RestoreBackend()
    monkeypatch.setattr(run, "build_backend", lambda *args, **kwargs: backend)
    args = _args(
        run,
        tmp_path,
        mode=mode,
        epochs=2,
        max_train_steps=1,
        max_eval_steps=1,
        max_test_steps=1,
    )

    result = run.run(args)

    assert backend.model.strict_loads
    restored_state, strict = backend.model.strict_loads[-1]
    assert strict is True
    assert any(name.startswith("lora.") for name in restored_state)
    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    assert metrics["best_epoch"] == 0
    assert metrics["history"][1]["val_loss"] > metrics["history"][0]["val_loss"]
    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    assert checkpoint["epochs"] == 2
    assert checkpoint["phase_steps"] == {"train": 2, "val": 2, "test": 1}
    with result["predictions"].open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["y_pred"]) == 0.0


def test_validation_loss_is_weighted_by_valid_points(monkeypatch, tmp_path):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    val_batches = [
        {
            "history": torch.ones(1, 2, 1),
            "target": torch.tensor([[[2.0], [100.0], [100.0]]]),
            "target_mask": torch.tensor([[True, False, False]]),
            "issue_time_ns": torch.tensor([1735689600000000000]),
        },
        {
            "history": torch.ones(1, 2, 1),
            "target": torch.ones(1, 3, 1),
            "target_mask": torch.tensor([[True, True, True]]),
            "issue_time_ns": torch.tensor([1735689600300000000]),
        },
    ]
    backend = FakeBackend()
    result = run._collect_predictions(
        backend, val_batches, torch.device("cpu"), 2, 3, 0, "val"
    )
    # Fake predictions are 0.5; literal SSE is 2.25 + .25 + .25 + .25 over 4 points.
    assert result["loss"] == pytest.approx(3.0 / 4.0)


def test_predictions_are_clipped_before_csv_and_metrics(monkeypatch, tmp_path):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    backend = FakeBackend()
    backend.model.weight.requires_grad_(False)
    backend.predict = lambda history, pred_len: torch.tensor([[[-1.0]]]).expand(history.shape[0], pred_len, 1)
    monkeypatch.setattr(run, "build_backend", lambda *args, **kwargs: backend)
    args = _args(run, tmp_path, mode="zero_shot")
    run.run(args)
    with (args.output_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["y_pred"]) == 0.0
    metrics = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_loss_normalized"] == pytest.approx(1.0)


def test_float32_extreme_and_bounds_are_clipped_before_test_loss_and_artifacts(monkeypatch, tmp_path):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    run.PowerOnlyParquetDataset = ExtremeDataset
    backend = FakeBackend()
    extreme = torch.finfo(torch.float32).max
    backend.predict = lambda history, pred_len: torch.tensor(
        [[[extreme], [-1.0]]], dtype=torch.float32
    ).expand(history.shape[0], pred_len, 1)
    monkeypatch.setattr(run, "build_backend", lambda *args, **kwargs: backend)
    args = _args(run, tmp_path, mode="zero_shot", pred_len=2)

    result = run.run(args)

    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    assert metrics["test_loss_normalized"] == pytest.approx(1.0)
    assert metrics["metrics_original_power_units"] == {
        "count": 1,
        "mae": 2.0,
        "rmse": 2.0,
        "nmae": 0.5,
        "nrmse": 0.5,
        "mape_percent": 100.0,
        "nmae_percent": 50.0,
        "nrmse_percent": 50.0,
    }
    with result["predictions"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["horizon_minutes"] == "5"
    assert rows[0]["issue_time"] == "2025-01-01T00:00:00"
    assert rows[0]["target_time"] == "2025-01-01T00:05:00"
    assert float(rows[0]["y_true"]) == 2.0
    assert float(rows[0]["y_pred"]) == 4.0


class CountingLoader:
    def __init__(self, batches):
        self.batches = list(batches)
        self.next_calls = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        self.next_calls += 1
        if self.index >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.index]
        self.index += 1
        return batch


def test_max_step_limit_does_not_request_an_extra_batch(monkeypatch, tmp_path):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    loader = CountingLoader([_batch(), _batch(value=2.0)])
    monkeypatch.setattr(run, "_dataset_and_loader", lambda args, flag: (TinyDataset(args.config, flag, 2, 1), loader))
    args = _args(run, tmp_path, mode="zero_shot", max_test_steps=1)
    run.run(args)
    assert loader.next_calls == 1


@pytest.mark.parametrize("bad_prediction", [torch.tensor([[[float("nan")]]]), torch.ones(1, 2, 1, 1), torch.ones(1, 1, 2)])
def test_bad_prediction_leaves_completion_absent(monkeypatch, tmp_path, bad_prediction):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    backend = FakeBackend()
    backend.predict = lambda history, pred_len: bad_prediction
    monkeypatch.setattr(run, "build_backend", lambda *args, **kwargs: backend)
    args = _args(run, tmp_path, mode="zero_shot")
    with pytest.raises(ValueError):
        run.run(args)
    assert not (args.output_dir / "completion.tsv").exists()


@pytest.mark.parametrize("failure", ["empty", "masked"])
@pytest.mark.parametrize("phase", ["train", "val", "test"])
def test_empty_and_all_masked_phases_leave_completion_absent(monkeypatch, tmp_path, phase, failure):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    valid = [_batch()]
    failing = [] if failure == "empty" else [_batch(mask=torch.tensor([[False]]))]
    batches = {name: (failing if name == phase else valid) for name in ("train", "val", "test")}
    monkeypatch.setattr(
        run,
        "_dataset_and_loader",
        lambda args, flag: (TinyDataset(args.config, flag, 2, 1), batches[flag]),
    )
    args = _args(run, tmp_path, mode="full")
    expected = "empty" if failure == "empty" else "zero valid targets"
    with pytest.raises(RuntimeError, match=expected):
        run.run(args)
    assert not (args.output_dir / "completion.tsv").exists()


def test_completion_contains_absolute_identity_and_exact_hashes(monkeypatch, tmp_path):
    import run

    run = _install_fakes(monkeypatch, tmp_path)
    args = _args(run, tmp_path, mode="zero_shot")
    result = run.run(args)
    rows = result["completion"].read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    header = rows[0].split("\t")
    assert header == list(run.FOUNDATION_COMPLETION_HEADER)
    assert all(len(row.split("\t")) == len(header) for row in rows)
    values = dict(zip(header, rows[1].split("\t")))
    assert values["schema_version"] == "2"
    assert values["dataset"] == "skippd_luoyang"
    assert values["model"] == "Sundial"
    assert values["seq_len"] == "2"
    assert values["pred_len"] == "1"
    assert values["mode"] == "zero_shot"
    assert values["run_mode"] == "full"
    assert values["epochs"] == "0"
    assert values["max_train_steps"] == "0"
    assert values["max_eval_steps"] == "0"
    assert values["max_test_steps"] == "0"
    assert values["train_steps"] == "0"
    assert values["val_steps"] == "0"
    assert values["test_steps"] == "2"
    assert values["model_id"] == "thuml/sundial-base-128m"
    assert values["model_revision"] == "3212e42564493f520593e5414af4367fc4b49226"
    assert values["output_dir"] == str(args.output_dir.resolve())
    assert values["best_sha256"] == hashlib.sha256(result["checkpoint"].read_bytes()).hexdigest()
    assert values["predictions_sha256"] == hashlib.sha256(result["predictions"].read_bytes()).hexdigest()
    assert values["metrics_sha256"] == hashlib.sha256(result["metrics"].read_bytes()).hexdigest()


def test_all_masked_training_batch_is_counted_but_skips_loss_and_optimizer(monkeypatch, tmp_path):
    run = _install_fakes(monkeypatch, tmp_path)
    train_loader = [_batch(mask=torch.tensor([[False]])), _batch(mask=torch.tensor([[True]]))]
    batches = {"train": train_loader, "val": [_batch()], "test": [_batch()]}
    monkeypatch.setattr(
        run,
        "_dataset_and_loader",
        lambda args, flag: (TinyDataset(args.config, flag, 2, 1), batches[flag]),
    )
    args = _args(run, tmp_path, mode="full")
    result = run.run(args)
    backend = FakeBackend.instances[0]
    assert len(backend.loss_calls) == 1
    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    assert metrics["phase_steps"]["train"] == 2


@pytest.mark.parametrize("bad_loss", [torch.tensor(1.0), torch.tensor(float("nan")), torch.ones(1)])
def test_invalid_native_loss_leaves_completion_absent(monkeypatch, tmp_path, bad_loss):
    run = _install_fakes(monkeypatch, tmp_path)
    backend = FakeBackend()
    backend.training_loss = lambda history, target, target_mask: bad_loss
    monkeypatch.setattr(run, "build_backend", lambda *args, **kwargs: backend)
    args = _args(run, tmp_path, mode="full")
    with pytest.raises(ValueError, match="loss"):
        run.run(args)
    assert not (args.output_dir / "completion.tsv").exists()


def test_strict_checkpoint_restore_failure_leaves_completion_absent(monkeypatch, tmp_path):
    run = _install_fakes(monkeypatch, tmp_path)
    backend = FakeBackend()

    def fail_restore(state, strict=False):
        assert strict is True
        raise RuntimeError("bad state")

    backend.model.load_state_dict = fail_restore
    monkeypatch.setattr(run, "build_backend", lambda *args, **kwargs: backend)
    args = _args(run, tmp_path, mode="full")
    with pytest.raises(RuntimeError, match="restoration"):
        run.run(args)
    assert not (args.output_dir / "completion.tsv").exists()


def test_optimizer_receives_exact_requires_grad_parameters(monkeypatch, tmp_path):
    run = _install_fakes(monkeypatch, tmp_path)
    original_adam = torch.optim.Adam
    captured = []

    def recording_adam(parameters, *args, **kwargs):
        captured.append(list(parameters))
        return original_adam(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", recording_adam)
    args = _args(run, tmp_path, mode="full", smoke=True)
    run.run(args)
    backend = FakeBackend.instances[0]
    assert captured == [[parameter for parameter in backend.model.parameters() if parameter.requires_grad]]


def test_stale_completion_is_removed_before_backend_failure(monkeypatch, tmp_path):
    run = _install_fakes(monkeypatch, tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "completion.tsv").write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(
        run,
        "build_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("construction failed")),
    )
    args = _args(run, tmp_path, mode="zero_shot")
    with pytest.raises(RuntimeError, match="construction"):
        run.run(args)
    assert not (output_dir / "completion.tsv").exists()


def test_run_module_normalizes_aliases_and_rejects_unsupported_mode_before_backend(monkeypatch):
    import run

    observed = {}
    monkeypatch.setattr(
        run,
        "build_backend",
        lambda *args, **kwargs: observed.setdefault("built", True),
    )
    with pytest.raises(SystemExit):
        run.parse_args(
            [
                "--data", "skippd_luoyang",
                "--model", "TiRex",
                "--mode", "adapter",
                "--seq_len", "48",
                "--pred_len", "1",
                "--train_epochs", "1",
                "--debug", "False",
            ]
        )
    assert observed == {}


@pytest.mark.parametrize(
    ("canonical", "alias", "value"),
    [
        ("--dataset", "--data", "skippd_luoyang"),
        ("--epochs", "--train_epochs", "1"),
    ],
)
def test_parser_accepts_equal_canonical_and_alias_values(canonical, alias, value):
    import run

    args = run.parse_args(
        [
            "--dataset", "skippd_luoyang",
            "--model", "Sundial",
            "--mode", "full",
            "--seq_len", "2",
            "--pred_len", "1",
            canonical, value,
            alias, value,
        ]
    )
    assert args.dataset == "skippd_luoyang"
    assert args.epochs == 1


def test_parser_accepts_alias_only_values_and_removes_alias_attributes():
    import run

    args = run.parse_args(
        [
            "--data", "skippd_luoyang",
            "--model", "Sundial",
            "--mode", "full",
            "--seq_len", "2",
            "--pred_len", "1",
            "--train_epochs", "1",
            "--debug", "False",
        ]
    )
    assert args.dataset == "skippd_luoyang"
    assert args.epochs == 1
    assert args.smoke is False
    assert not hasattr(args, "data")
    assert not hasattr(args, "train_epochs")
    assert not hasattr(args, "debug")


def test_parser_accepts_equal_smoke_and_debug_values():
    import run

    args = run.parse_args(
        [
            "--dataset", "skippd_luoyang",
            "--model", "Sundial",
            "--mode", "full",
            "--seq_len", "2",
            "--pred_len", "1",
            "--smoke",
            "--debug", "True",
        ]
    )
    assert args.smoke is True
    assert args.epochs == 1


@pytest.mark.parametrize(
    "argv",
    [
        ["--dataset", "skippd_luoyang", "--data", "pvod_station00_ylj"],
        ["--epochs", "1", "--train_epochs", "2"],
        ["--smoke", "--debug", "False"],
    ],
)
def test_parser_rejects_genuine_alias_conflicts(argv):
    import run

    base = [
        "--dataset", "skippd_luoyang",
        "--model", "Sundial",
        "--mode", "full",
        "--seq_len", "2",
        "--pred_len", "1",
    ]
    with pytest.raises(SystemExit):
        run.parse_args(base + argv)


@pytest.mark.parametrize("value", ["maybe", "", "1", "2", "yes", "truthy"])
def test_parser_rejects_malformed_debug_boolean(value):
    import run

    with pytest.raises(SystemExit):
        run.parse_args(
            [
                "--dataset", "skippd_luoyang",
                "--model", "Sundial",
                "--mode", "full",
                "--seq_len", "2",
                "--pred_len", "1",
                "--debug", value,
            ]
        )


def test_parser_rejects_unsupported_model_mode_before_backend_import():
    import run

    with pytest.raises(SystemExit):
        run.parse_args(
            [
                "--dataset", "skippd_luoyang",
                "--model", "TiRex",
                "--mode", "adapter",
                "--seq_len", "48",
                "--pred_len", "1",
            ]
        )


def test_parser_rejects_invalid_dataset_alias_without_traceback():
    import run

    with pytest.raises(SystemExit):
        run.parse_args(
            [
                "--data", "not-a-dataset",
                "--model", "Sundial",
                "--mode", "zero_shot",
                "--seq_len", "48",
                "--pred_len", "1",
            ]
        )


def test_direct_namespace_rejects_unsupported_model_mode_before_backend(monkeypatch, tmp_path):
    import run

    module = _install_fakes(monkeypatch, tmp_path)
    observed = []
    monkeypatch.setattr(module, "build_backend", lambda *args, **kwargs: observed.append(True))
    args = _args(module, tmp_path, model="TiRex", mode="adapter")
    with pytest.raises(ValueError, match="does not support mode"):
        module.run(args)
    assert observed == []


def test_parser_rejects_unknown_arguments():
    import run

    with pytest.raises(SystemExit):
        run.parse_args(
            [
                "--dataset", "skippd_luoyang",
                "--model", "Sundial",
                "--mode", "full",
                "--seq_len", "2",
                "--pred_len", "1",
                "--not-a-runner-option",
            ]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("task_name", "long_term_forecast"), ("is_training", True)],
)
def test_direct_namespace_consistency_assertions_precede_backend(monkeypatch, tmp_path, field, value):
    import run

    module = _install_fakes(monkeypatch, tmp_path)
    observed = []
    monkeypatch.setattr(module, "build_backend", lambda *args, **kwargs: observed.append(True))
    args = _args(module, tmp_path, mode="zero_shot", **{field: value})
    with pytest.raises(ValueError, match="inconsistent"):
        module.run(args)
    assert observed == []


def test_direct_namespace_aliases_are_normalized_before_lifecycle(monkeypatch, tmp_path):
    import run

    module = _install_fakes(monkeypatch, tmp_path)
    args = _args(module, tmp_path, mode="zero_shot")
    delattr(args, "dataset")
    args.data = "skippd_luoyang"
    delattr(args, "epochs")
    args.train_epochs = 1
    delattr(args, "smoke")
    args.debug = "False"
    result = module.run(args)
    assert result["completion"].is_file()


def test_default_output_cleanup_happens_before_alias_conflict(monkeypatch, tmp_path):
    import run

    module = _install_fakes(monkeypatch, tmp_path)
    config = tmp_path / "config.json"
    results_root = tmp_path / "default-results"
    config.write_text(json.dumps({"paths": {"results_root": str(results_root)}}), encoding="utf-8")
    output = results_root / "foundation_models" / "skippd_luoyang" / "Sundial" / "zero_shot"
    output.mkdir(parents=True)
    completion = output / "completion.tsv"
    completion.write_text("stale\n", encoding="utf-8")
    args = _args(
        module,
        tmp_path,
        mode="zero_shot",
        config=config,
        output_dir=None,
    )
    args.data = "pvod_station00_ylj"

    with pytest.raises(ValueError, match="conflicting"):
        module.run(args)
    assert not completion.exists()


@pytest.mark.parametrize(
    ("canonical_dataset", "alias_dataset"),
    [
        ("skippd_luoyang", "pvod_station00_ylj"),
        ("pvod_station00_ylj", "skippd_luoyang"),
    ],
)
def test_alias_conflict_cleans_every_default_output_candidate(
    monkeypatch, tmp_path, canonical_dataset, alias_dataset
):
    import run

    module = _install_fakes(monkeypatch, tmp_path)
    config = tmp_path / "config.json"
    results_root = tmp_path / "default-results"
    config.write_text(json.dumps({"paths": {"results_root": str(results_root)}}), encoding="utf-8")
    outputs = []
    for dataset in (canonical_dataset, alias_dataset):
        output = results_root / "foundation_models" / dataset / "Sundial" / "zero_shot"
        output.mkdir(parents=True)
        completion = output / "completion.tsv"
        completion.write_text("stale\n", encoding="utf-8")
        outputs.append(completion)
    args = _args(
        module,
        tmp_path,
        mode="zero_shot",
        config=config,
        output_dir=None,
        dataset=canonical_dataset,
        data=alias_dataset,
    )

    with pytest.raises(ValueError, match="conflicting"):
        module.run(args)
    assert all(not completion.exists() for completion in outputs)


def test_cli_parse_failure_cleans_explicit_output_completion(monkeypatch, tmp_path):
    import run

    output = tmp_path / "cli-output"
    output.mkdir()
    completion = output / "completion.tsv"
    completion.write_text("stale\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        run.main(
            [
                "--dataset",
                "skippd_luoyang",
                "--model",
                "Sundial",
                "--mode",
                "zero_shot",
                "--seq_len",
                "2",
                "--pred_len",
                "1",
                "--output_dir",
                str(output),
                "--debug",
                "yes",
            ]
        )
    assert not completion.exists()


def test_requested_model_id_is_metadata_only_and_pinned_loader_identity_survives(monkeypatch, tmp_path):
    import run

    module = _install_fakes(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(module, "build_backend", lambda *args, **kwargs: (calls.append((args, kwargs)) or FakeBackend()))
    args = _args(module, tmp_path, mode="zero_shot", model_id="caller/model")
    result = module.run(args)

    assert calls == [(("Sundial", torch.device("cpu")), {})]
    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    assert metrics["model_id"] == "thuml/sundial-base-128m"
    assert metrics["requested_model_id"] == "caller/model"
    assert checkpoint["model_id"] == "thuml/sundial-base-128m"
    assert checkpoint["requested_model_id"] == "caller/model"
