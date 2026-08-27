from __future__ import annotations

import csv
import hashlib
import json
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


def _batch(value=1.0, target=1.0, mask=None, issue=1735689600000000000):
    if mask is None:
        mask = torch.ones(1, 1, dtype=torch.bool)
    return {
        "history": torch.full((1, 2, 1), value),
        "target": torch.full((1, 1, 1), target),
        "target_mask": mask,
        "issue_time_ns": torch.tensor([issue], dtype=torch.int64),
    }


class FakeBackend:
    instances = []
    optimizer_created = False

    def __init__(self, *args, **kwargs):
        self.model_name = "Sundial"
        self.model_id = "tiny/model"
        self.revision = "tiny-revision"
        self.model = nn.Module()
        self.model.weight = nn.Parameter(torch.tensor(0.5))
        self.configure_calls = []
        self.predict_histories = []
        self.loss_calls = []
        self.load_calls = []
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
        return self.model.weight.detach().expand(history.shape[0], pred_len, 1)


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
    import run_foundation_model

    TinyDataset.instances = []
    FakeBackend.instances = []
    monkeypatch.setattr(run_foundation_model, "PowerOnlyParquetDataset", TinyDataset)
    monkeypatch.setattr(run_foundation_model, "build_backend", lambda *args, **kwargs: FakeBackend())
    return run_foundation_model


def test_list_commands_are_lazy_and_exact(capsys, monkeypatch):
    import run_foundation_model
    from foundation_models import MODEL_NAMES, RUN_MODES

    monkeypatch.setattr(
        run_foundation_model,
        "PowerOnlyParquetDataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dataset constructed")),
    )
    monkeypatch.setattr(
        run_foundation_model,
        "build_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backend constructed")),
    )

    assert run_foundation_model.main(["--list-models"]) == 0
    assert tuple(capsys.readouterr().out.splitlines()) == MODEL_NAMES
    assert run_foundation_model.main(["--list-modes"]) == 0
    assert tuple(capsys.readouterr().out.splitlines()) == RUN_MODES


def test_parser_requires_runtime_identity_and_positive_lengths():
    import run_foundation_model

    with pytest.raises(SystemExit):
        run_foundation_model.parse_args([])
    with pytest.raises(SystemExit):
        run_foundation_model.parse_args(["--dataset", "skippd_luoyang", "--model", "Sundial"])
    with pytest.raises(SystemExit):
        run_foundation_model.parse_args([
            "--dataset", "skippd_luoyang", "--model", "Sundial", "--mode", "zero_shot",
            "--seq_len", "0", "--pred_len", "1",
        ])


def test_parser_defaults_config_and_mode_specific_smoke():
    import run_foundation_model

    args = run_foundation_model.parse_args([
        "--dataset", "pvod_station00_ylj", "--model", "TimeMoE", "--mode", "zero_shot",
        "--seq_len", "48", "--pred_len", "1", "--epochs", "8", "--smoke",
    ])
    assert args.config.name == "pvod_station00_ylj.yaml"
    assert (args.epochs, args.batch_size, args.max_train_steps, args.max_eval_steps, args.max_test_steps) == (0, 2, 1, 1, 1)

    trained = run_foundation_model.parse_args([
        "--dataset", "skippd_luoyang", "--model", "Sundial", "--mode", "full",
        "--seq_len", "48", "--pred_len", "1", "--smoke",
    ])
    assert trained.epochs == 1


def test_runtime_rejects_cuda_when_unavailable(monkeypatch):
    import run_foundation_model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises((RuntimeError, ValueError), match="CUDA|cuda"):
        run_foundation_model._device_from_arg("cuda:0")


def test_zero_shot_constructs_only_test_without_optimizer_and_writes_artifacts(monkeypatch, tmp_path):
    import run_foundation_model

    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    monkeypatch.setattr(torch.optim, "Adam", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("optimizer")))
    args = _args(run_foundation_model, tmp_path, mode="zero_shot", smoke=True)
    result = run_foundation_model.run(args)

    assert [dataset.flag for dataset in TinyDataset.instances] == ["test"]
    backend = FakeBackend.instances[0]
    assert [call[0] for call in backend.configure_calls] == ["zero_shot"]
    assert len(backend.loss_calls) == 0
    assert len(backend.predict_histories) == 1
    assert set(result) == {"checkpoint", "predictions", "metrics", "completion"}
    assert all(path.is_file() for path in result.values())
    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    assert metrics["epochs"] == 0
    assert metrics["phase_steps"] == {"train": 0, "val": 0, "test": 1}
    assert metrics["limits"] == {"train": 1, "val": 1, "test": 1}
    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    assert checkpoint["mode"] == "zero_shot"
    assert "state_dict" not in checkpoint


@pytest.mark.parametrize("mode", ["adapter", "full", "last_layer"])
def test_training_modes_restore_best_state_and_use_trainable_parameters(monkeypatch, tmp_path, mode):
    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    args = _args(run_foundation_model, tmp_path, mode=mode, smoke=True)
    result = run_foundation_model.run(args)

    assert [dataset.flag for dataset in TinyDataset.instances] == ["train", "val", "test"]
    backend = FakeBackend.instances[0]
    assert [call[0] for call in backend.configure_calls] == [mode]
    assert backend.loss_calls
    assert backend.load_calls or torch.load(result["checkpoint"], map_location="cpu")["state_dict"]
    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    assert metrics["mode"] == mode
    assert metrics["trainability"]["mode"] == mode
    assert metrics["phase_steps"] == {"train": 1, "val": 1, "test": 1}


def test_validation_loss_is_weighted_by_valid_points(monkeypatch, tmp_path):
    import run_foundation_model

    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
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
    result = run_foundation_model._collect_predictions(
        backend, val_batches, torch.device("cpu"), 2, 3, 0, "val"
    )
    # Fake predictions are 0.5; literal SSE is 2.25 + .25 + .25 + .25 over 4 points.
    assert result["loss"] == pytest.approx(3.0 / 4.0)


def test_predictions_are_clipped_before_csv_and_metrics(monkeypatch, tmp_path):
    import run_foundation_model

    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    backend = FakeBackend()
    backend.model.weight.requires_grad_(False)
    backend.predict = lambda history, pred_len: torch.tensor([[[-1.0]]]).expand(history.shape[0], pred_len, 1)
    monkeypatch.setattr(run_foundation_model, "build_backend", lambda *args, **kwargs: backend)
    args = _args(run_foundation_model, tmp_path, mode="zero_shot")
    run_foundation_model.run(args)
    with (args.output_dir / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["y_pred"]) == 0.0
    metrics = json.loads((args.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["test_loss_normalized"] == pytest.approx(1.0)


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
    import run_foundation_model

    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    loader = CountingLoader([_batch(), _batch(value=2.0)])
    monkeypatch.setattr(run_foundation_model, "_dataset_and_loader", lambda args, flag: (TinyDataset(args.config, flag, 2, 1), loader))
    args = _args(run_foundation_model, tmp_path, mode="zero_shot", max_test_steps=1)
    run_foundation_model.run(args)
    assert loader.next_calls == 1


@pytest.mark.parametrize("bad_prediction", [torch.tensor([[[float("nan")]]]), torch.ones(1, 2, 1, 1), torch.ones(1, 1, 2)])
def test_bad_prediction_leaves_completion_absent(monkeypatch, tmp_path, bad_prediction):
    import run_foundation_model

    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    backend = FakeBackend()
    backend.predict = lambda history, pred_len: bad_prediction
    monkeypatch.setattr(run_foundation_model, "build_backend", lambda *args, **kwargs: backend)
    args = _args(run_foundation_model, tmp_path, mode="zero_shot")
    with pytest.raises(ValueError):
        run_foundation_model.run(args)
    assert not (args.output_dir / "completion.tsv").exists()


def test_empty_and_all_masked_phases_leave_completion_absent(monkeypatch, tmp_path):
    import run_foundation_model

    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    args = _args(run_foundation_model, tmp_path, mode="full")
    monkeypatch.setattr(run_foundation_model, "_dataset_and_loader", lambda args, flag: (TinyDataset(args.config, flag, 2, 1), [] if flag == "train" else [_batch()]))
    with pytest.raises(RuntimeError, match="empty"):
        run_foundation_model.run(args)
    assert not (args.output_dir / "completion.tsv").exists()

    args = _args(run_foundation_model, tmp_path / "masked", mode="full")
    monkeypatch.setattr(run_foundation_model, "_dataset_and_loader", lambda args, flag: (TinyDataset(args.config, flag, 2, 1), [_batch(mask=torch.tensor([[False]]))]))
    with pytest.raises(RuntimeError, match="zero valid targets"):
        run_foundation_model.run(args)
    assert not (args.output_dir / "completion.tsv").exists()


def test_completion_contains_absolute_identity_and_exact_hashes(monkeypatch, tmp_path):
    import run_foundation_model

    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    args = _args(run_foundation_model, tmp_path, mode="zero_shot")
    result = run_foundation_model.run(args)
    rows = result["completion"].read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    header = rows[0].split("\t")
    values = dict(zip(header, rows[1].split("\t")))
    assert values["schema_version"] == "2"
    assert values["mode"] == "zero_shot"
    assert values["model_id"] == "tiny/model"
    assert values["model_revision"] == "tiny-revision"
    assert values["output_dir"] == str(args.output_dir.resolve())
    assert values["best_sha256"] == hashlib.sha256(result["checkpoint"].read_bytes()).hexdigest()
    assert values["predictions_sha256"] == hashlib.sha256(result["predictions"].read_bytes()).hexdigest()
    assert values["metrics_sha256"] == hashlib.sha256(result["metrics"].read_bytes()).hexdigest()


def test_all_masked_training_batch_is_counted_but_skips_loss_and_optimizer(monkeypatch, tmp_path):
    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    train_loader = [_batch(mask=torch.tensor([[False]])), _batch(mask=torch.tensor([[True]]))]
    batches = {"train": train_loader, "val": [_batch()], "test": [_batch()]}
    monkeypatch.setattr(
        run_foundation_model,
        "_dataset_and_loader",
        lambda args, flag: (TinyDataset(args.config, flag, 2, 1), batches[flag]),
    )
    args = _args(run_foundation_model, tmp_path, mode="full")
    result = run_foundation_model.run(args)
    backend = FakeBackend.instances[0]
    assert len(backend.loss_calls) == 1
    metrics = json.loads(result["metrics"].read_text(encoding="utf-8"))
    assert metrics["phase_steps"]["train"] == 2


@pytest.mark.parametrize("bad_loss", [torch.tensor(1.0), torch.tensor(float("nan")), torch.ones(1)])
def test_invalid_native_loss_leaves_completion_absent(monkeypatch, tmp_path, bad_loss):
    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    backend = FakeBackend()
    backend.training_loss = lambda history, target, target_mask: bad_loss
    monkeypatch.setattr(run_foundation_model, "build_backend", lambda *args, **kwargs: backend)
    args = _args(run_foundation_model, tmp_path, mode="full")
    with pytest.raises(ValueError, match="loss"):
        run_foundation_model.run(args)
    assert not (args.output_dir / "completion.tsv").exists()


def test_strict_checkpoint_restore_failure_leaves_completion_absent(monkeypatch, tmp_path):
    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    backend = FakeBackend()

    def fail_restore(state, strict=False):
        assert strict is True
        raise RuntimeError("bad state")

    backend.model.load_state_dict = fail_restore
    monkeypatch.setattr(run_foundation_model, "build_backend", lambda *args, **kwargs: backend)
    args = _args(run_foundation_model, tmp_path, mode="full")
    with pytest.raises(RuntimeError, match="restoration"):
        run_foundation_model.run(args)
    assert not (args.output_dir / "completion.tsv").exists()


def test_optimizer_receives_exact_requires_grad_parameters(monkeypatch, tmp_path):
    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    original_adam = torch.optim.Adam
    captured = []

    def recording_adam(parameters, *args, **kwargs):
        captured.append(list(parameters))
        return original_adam(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", recording_adam)
    args = _args(run_foundation_model, tmp_path, mode="full", smoke=True)
    run_foundation_model.run(args)
    backend = FakeBackend.instances[0]
    assert captured == [[parameter for parameter in backend.model.parameters() if parameter.requires_grad]]


def test_stale_completion_is_removed_before_backend_failure(monkeypatch, tmp_path):
    run_foundation_model = _install_fakes(monkeypatch, tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "completion.tsv").write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(
        run_foundation_model,
        "build_backend",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("construction failed")),
    )
    args = _args(run_foundation_model, tmp_path, mode="zero_shot")
    with pytest.raises(RuntimeError, match="construction"):
        run_foundation_model.run(args)
    assert not (output_dir / "completion.tsv").exists()
