import copy
import csv
import hashlib
from types import SimpleNamespace
from datetime import timedelta

import numpy as np
import pytest
import torch
from torch import nn

from baselines import BASELINE_NAMES
from models.TSMixer import Model as TSMixer
from run_time_series import (
    _collect_predictions,
    _dataset_and_loader,
    _metrics,
    _run_epoch,
    _write_completion_manifest,
    _write_predictions,
    main,
    parse_args,
    run,
)


EXPECTED_MODELS = (
    "TSMixer",
    "Pyraformer",
    "SegRNN",
    "Transformer",
    "LightTS",
    "Crossformer",
    "FreTS",
    "MICN",
)


def _tsmixer_model():
    return TSMixer(SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=2,
        pred_len=1,
        enc_in=1,
        d_model=4,
        e_layers=1,
        dropout=0.0,
    ))


def _three_batch_loader():
    batches = []
    for offset in range(3):
        history = torch.tensor([[[1.0 + offset], [2.0 + offset]]])
        target = torch.tensor([[[3.0 + offset]]])
        batches.append({
            "history": history,
            "target": target,
            "target_mask": torch.tensor([[True]]),
            "issue_time_ns": torch.tensor([offset], dtype=torch.int64),
        })
    return batches


class _CountingLoader:
    def __init__(self, batches):
        self._batches = list(batches)
        self.next_calls = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        self.next_calls += 1
        if self._index >= len(self._batches):
            raise StopIteration
        batch = self._batches[self._index]
        self._index += 1
        return batch


def test_smoke_arguments_limit_every_phase():
    args = parse_args([
        "--dataset", "skippd_luoyang", "--model", "TSMixer", "--smoke",
    ])

    assert (
        args.epochs,
        args.max_train_steps,
        args.max_eval_steps,
        args.max_test_steps,
        args.batch_size,
    ) == (1, 1, 1, 1, 2)


def test_smoke_overrides_unrestricted_limits():
    args = parse_args([
        "--dataset", "skippd_luoyang", "--model", "TSMixer", "--smoke",
        "--epochs", "7", "--max_train_steps", "0", "--max_eval_steps", "9",
        "--max_test_steps", "11", "--batch_size", "64",
    ])

    assert (
        args.epochs,
        args.max_train_steps,
        args.max_eval_steps,
        args.max_test_steps,
        args.batch_size,
    ) == (1, 1, 1, 1, 2)


@pytest.mark.parametrize("seq_len,pred_len", [(24, 1), (48, 12)])
def test_parser_accepts_point_based_history_and_prediction_lengths(
    seq_len, pred_len,
):
    args = parse_args([
        "--dataset", "skippd_luoyang", "--model", "TSMixer",
        "--seq_len", str(seq_len), "--pred_len", str(pred_len),
    ])

    assert (args.seq_len, args.pred_len) == (seq_len, pred_len)


def test_length_overrides_are_forwarded_to_the_dataset(monkeypatch):
    import run_time_series

    received = {}

    class RecordingDataset:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            raise AssertionError("loader construction must not read a sample")

    monkeypatch.setitem(
        run_time_series.DATASET_LOADERS, "skippd_luoyang", RecordingDataset
    )
    args = parse_args([
        "--dataset", "skippd_luoyang", "--model", "TSMixer",
        "--seq_len", "24", "--pred_len", "1",
    ])
    args.device = torch.device("cpu")

    _dataset_and_loader(args, "train")

    assert received["history_points"] == 24
    assert received["forecast_steps"] == 1


def test_pure_runner_uses_generic_power_only_dataset_contract(monkeypatch):
    import run_time_series

    received = {}

    class RecordingDataset:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def __len__(self):
            return 1

    monkeypatch.setitem(
        run_time_series.DATASET_LOADERS, "pvod_station00_ylj", RecordingDataset
    )
    args = parse_args([
        "--dataset", "pvod_station00_ylj", "--model", "TSMixer",
        "--seq_len", "48", "--pred_len", "12",
    ])
    args.device = torch.device("cpu")

    _dataset_and_loader(args, "train")

    assert received == {
        "config_path": args.config,
        "flag": "train",
        "history_points": 48,
        "forecast_steps": 12,
    }


@pytest.mark.parametrize("option", ["--seq_len", "--pred_len"])
@pytest.mark.parametrize("value", [0, -1])
def test_parser_rejects_nonpositive_sequence_lengths(option, value):
    with pytest.raises(SystemExit):
        parse_args([
            "--dataset", "skippd_luoyang", "--model", "TSMixer",
            option, str(value),
        ])


def test_one_step_limit_is_exact():
    model = _tsmixer_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss, steps = _run_epoch(
        model,
        _three_batch_loader(),
        optimizer,
        torch.device("cpu"),
        max_steps=1,
        model_name="TSMixer",
        label_len=2,
        pred_len=1,
    )

    assert steps == 1
    assert torch.isfinite(torch.tensor(loss))


def test_one_step_limit_does_not_request_a_second_batch():
    model = _tsmixer_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _CountingLoader(_three_batch_loader())

    _, steps = _run_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        max_steps=1,
        model_name="TSMixer",
        label_len=2,
        pred_len=1,
    )

    assert steps == 1
    assert loader.next_calls == 1


def test_empty_train_loader_is_rejected_when_a_step_is_required():
    model = _tsmixer_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with pytest.raises(RuntimeError, match="train loader is empty"):
        _run_epoch(
            model,
            [],
            optimizer,
            torch.device("cpu"),
            max_steps=1,
            model_name="TSMixer",
            label_len=2,
            pred_len=1,
        )


@pytest.mark.parametrize("phase", ["val", "test"])
def test_empty_eval_loader_is_rejected_when_a_step_is_required(phase):
    model = _tsmixer_model()

    with pytest.raises(RuntimeError, match=f"{phase} loader is empty"):
        _collect_predictions(
            model,
            [],
            torch.device("cpu"),
            max_steps=1,
            model_name="TSMixer",
            label_len=2,
            pred_len=1,
            phase=phase,
        )


@pytest.mark.parametrize("phase", ["train", "val", "test"])
@pytest.mark.parametrize("max_steps", [None, 0])
def test_empty_required_loader_is_rejected_for_unlimited_limits(phase, max_steps):
    model = _tsmixer_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) if phase == "train" else None

    with pytest.raises(RuntimeError, match=f"{phase} loader is empty"):
        if phase == "train":
            _run_epoch(
                model,
                [],
                optimizer,
                torch.device("cpu"),
                max_steps=max_steps,
                model_name="TSMixer",
                label_len=2,
                pred_len=1,
                phase=phase,
            )
        else:
            _collect_predictions(
                model,
                [],
                torch.device("cpu"),
                max_steps=max_steps,
                model_name="TSMixer",
                label_len=2,
                pred_len=1,
                phase=phase,
            )


def test_zero_test_limit_is_unlimited_instead_of_using_eval_limit():
    model = _tsmixer_model()
    loader = _three_batch_loader()

    result = _collect_predictions(
        model,
        loader,
        torch.device("cpu"),
        max_steps=0,
        model_name="TSMixer",
        label_len=2,
        pred_len=1,
        phase="test",
    )

    assert result["steps"] == 3


@pytest.mark.parametrize(
    "option",
    ["--max_train_steps", "--max_eval_steps", "--max_test_steps"],
)
def test_parser_rejects_negative_step_limits(option):
    with pytest.raises(SystemExit):
        parse_args([
            "--dataset", "skippd_luoyang", "--model", "TSMixer",
            option, "-1",
        ])


@pytest.mark.parametrize("batch_size", [0, -1])
def test_parser_rejects_nonpositive_batch_size(batch_size):
    with pytest.raises(SystemExit):
        parse_args([
            "--dataset", "skippd_luoyang", "--model", "TSMixer",
            "--batch_size", str(batch_size),
        ])


@pytest.mark.parametrize(
    "field,value,needle",
    [
        ("max_train_steps", -1, "max_train_steps"),
        ("max_eval_steps", -1, "max_eval_steps"),
        ("max_test_steps", -1, "max_test_steps"),
        ("batch_size", 0, "batch_size"),
    ],
)
def test_runtime_rejects_invalid_limits(monkeypatch, field, value, needle):
    import run_time_series

    args = parse_args([
        "--dataset", "skippd_luoyang", "--model", "TSMixer",
    ])
    setattr(args, field, value)
    monkeypatch.setattr(
        run_time_series,
        "_dataset_and_loader",
        lambda *unused: (_ for _ in ()).throw(
            AssertionError("runtime validation did not run")
        ),
    )

    with pytest.raises(ValueError, match=needle):
        run(args)


def test_test_phase_receives_its_own_zero_limit(monkeypatch, tmp_path):
    import run_time_series

    calls = []
    dataset = SimpleNamespace(
        seq_len=2,
        pred_len=1,
        power_scale=1.0,
        rated_power=10.0,
        forecast_step=timedelta(minutes=1),
    )
    batches = _three_batch_loader()

    def fake_collect(model, loader, device, max_steps, **kwargs):
        calls.append(max_steps)
        return {
            "loss": 0.0,
            "steps": 1,
            "prediction": torch.zeros(1, 1, 1).numpy(),
            "target": torch.zeros(1, 1, 1).numpy(),
            "mask": torch.ones(1, 1, dtype=torch.bool).numpy(),
            "issue_time_ns": torch.full((1,), float("nan")).numpy(),
        }

    monkeypatch.setattr(run_time_series, "_read_config", lambda path: {
        "paths": {"results_root": str(tmp_path)},
    })
    monkeypatch.setattr(
        run_time_series,
        "_dataset_and_loader",
        lambda args, flag: (dataset, batches),
    )
    monkeypatch.setattr(
        run_time_series,
        "_build_model",
        lambda args, train_dataset: (_tsmixer_model(), SimpleNamespace(
            label_len=2, pred_len=1,
        )),
    )
    monkeypatch.setattr(run_time_series, "_collect_predictions", fake_collect)

    def fake_write_predictions(path, *unused):
        path.write_text("issue_time,target_time,horizon_minutes,y_true,y_pred\n", encoding="utf-8")
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "nmae": None,
            "nrmse": None,
            "mape_percent": None,
            "nmae_percent": None,
            "nrmse_percent": None,
        }

    monkeypatch.setattr(run_time_series, "_write_predictions", fake_write_predictions)

    args = parse_args([
        "--dataset", "skippd_luoyang", "--model", "TSMixer",
        "--max_eval_steps", "1", "--max_test_steps", "0",
        "--output_dir", str(tmp_path / "output"),
    ])
    result = run(args)

    assert calls == [1, 0]
    assert result["payload"]["seq_len"] == 2
    assert result["payload"]["pred_len"] == 1


@pytest.mark.parametrize("empty_phase", ["train", "val", "test"])
def test_run_rejects_empty_unlimited_phase_before_success_artifacts(
    monkeypatch, tmp_path, empty_phase,
):
    import run_time_series

    dataset = SimpleNamespace(
        seq_len=2,
        pred_len=1,
        power_scale=1.0,
        forecast_step=timedelta(minutes=1),
    )
    output_dir = tmp_path / "output"

    monkeypatch.setattr(run_time_series, "_read_config", lambda path: {
        "paths": {"results_root": str(tmp_path)},
    })
    monkeypatch.setattr(
        run_time_series,
        "_dataset_and_loader",
        lambda args, flag: (
            dataset,
            [] if flag == empty_phase else _three_batch_loader(),
        ),
    )
    monkeypatch.setattr(
        run_time_series,
        "_build_model",
        lambda args, train_dataset: (_tsmixer_model(), SimpleNamespace(
            label_len=2, pred_len=1,
        )),
    )

    args = parse_args([
        "--dataset", "skippd_luoyang", "--model", "TSMixer",
        "--max_train_steps", "0", "--max_eval_steps", "0",
        "--max_test_steps", "0", "--output_dir", str(output_dir),
    ])
    with pytest.raises(RuntimeError, match=f"{empty_phase} loader is empty"):
        run(args)

    assert not (output_dir / "predictions.csv").exists()
    assert not (output_dir / "metrics.json").exists()
    if empty_phase == "test":
        assert (output_dir / "best.pt").is_file()
    else:
        assert not (output_dir / "best.pt").exists()


def test_runner_accepts_a_model_with_standard_forecast_inputs():
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.calls = []

        def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
            self.calls.append((x_enc, x_mark_enc, x_dec, x_mark_dec))
            return x_enc[:, -1:, :] * self.scale

    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _, steps = _run_epoch(
        model,
        _three_batch_loader(),
        optimizer,
        torch.device("cpu"),
        max_steps=1,
        model_name="TSMixer",
        label_len=2,
        pred_len=1,
    )

    assert steps == 1
    assert len(model.calls) == 1
    x_enc, x_mark_enc, x_dec, x_mark_dec = model.calls[0]
    assert x_enc.shape == (1, 2, 1)
    assert x_mark_enc is None
    assert x_dec.tolist() == [[[1.0], [2.0], [0.0]]]
    assert x_mark_dec is None


def test_list_models_does_not_require_dataset_or_model():
    args = parse_args(["--list-models"])
    assert args.list_models is True


def test_list_models_prints_exact_selected_catalog(capsys):
    assert main(["--list-models"]) == 0
    assert tuple(capsys.readouterr().out.splitlines()) == EXPECTED_MODELS


def test_list_baselines_does_not_require_dataset_or_model():
    args = parse_args(["--list-baselines"])
    assert args.list_baselines is True


def test_list_baselines_prints_exact_baseline_catalog(capsys):
    assert main(["--list-baselines"]) == 0
    assert tuple(capsys.readouterr().out.splitlines()) == BASELINE_NAMES


def test_unknown_method_is_rejected_by_cli():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "skippd_luoyang", "--model", "UnknownMethod"])


def test_metrics_uses_rated_power_and_respects_target_mask():
    prediction = np.array([[[12.0], [100.0], [8.0]]])
    target = np.array([[[10.0], [0.0], [10.0]]])
    mask = np.array([[True, False, True]])

    metrics = _metrics(prediction, target, mask, rated_power=20.0)

    assert metrics == {
        "count": 2,
        "mae": 2.0,
        "rmse": 2.0,
        "nmae": 0.1,
        "nrmse": 0.1,
        "mape_percent": 20.0,
        "nmae_percent": 10.0,
        "nrmse_percent": 10.0,
    }


def test_metrics_changes_normalized_errors_when_rated_power_changes():
    prediction = np.array([[[15.0], [5.0]]])
    target = np.array([[[10.0], [10.0]]])
    mask = np.array([[True, True]])

    metrics = _metrics(prediction, target, mask, rated_power=10.0)
    larger_rating = _metrics(prediction, target, mask, rated_power=20.0)

    assert metrics["nmae"] == 0.5
    assert metrics["nrmse"] == 0.5
    assert metrics["nmae_percent"] == 50.0
    assert metrics["nrmse_percent"] == 50.0
    assert larger_rating["nmae"] == 0.25
    assert larger_rating["nrmse"] == 0.25
    assert larger_rating["nmae_percent"] == 25.0
    assert larger_rating["nrmse_percent"] == 25.0


def test_metrics_returns_none_for_all_error_fields_when_no_target_is_valid():
    metrics = _metrics(
        np.array([[[1.0], [2.0]]]),
        np.array([[[3.0], [4.0]]]),
        np.array([[False, False]]),
        rated_power=10.0,
    )

    assert metrics == {
        "count": 0,
        "mae": None,
        "rmse": None,
        "nmae": None,
        "nrmse": None,
        "mape_percent": None,
        "nmae_percent": None,
        "nrmse_percent": None,
    }


@pytest.mark.parametrize("rated_power", [0.0, -1.0, float("nan")])
def test_metrics_rejects_nonpositive_or_nonfinite_rated_power(rated_power):
    with pytest.raises(ValueError, match="rated_power"):
        _metrics(
            np.zeros((1, 1, 1)),
            np.zeros((1, 1, 1)),
            np.ones((1, 1), dtype=bool),
            rated_power=rated_power,
        )


class _KnownForecast(nn.Module):
    def __init__(self, output_value=0.0):
        super().__init__()
        self.output_value = float(output_value)
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.bias + x_enc.new_full(
            (x_enc.shape[0], x_dec.shape[1] - x_enc.shape[1] + 1, 1),
            self.output_value,
        )


def _uneven_valid_batches():
    return [
        {
            "history": torch.zeros(1, 2, 1),
            "target": torch.tensor([[[2.0], [100.0], [100.0]]]),
            "target_mask": torch.tensor([[True, False, False]]),
            "issue_time_ns": torch.tensor([0], dtype=torch.int64),
        },
        {
            "history": torch.zeros(1, 2, 1),
            "target": torch.tensor([[[1.0], [1.0], [1.0]]]),
            "target_mask": torch.tensor([[True, True, True]]),
            "issue_time_ns": torch.tensor([1], dtype=torch.int64),
        },
    ]


def test_collect_predictions_weights_loss_by_valid_target_count():
    result = _collect_predictions(
        _KnownForecast(),
        _uneven_valid_batches(),
        torch.device("cpu"),
        max_steps=0,
        model_name="TSMixer",
        label_len=2,
        pred_len=3,
        phase="val",
    )

    # Literal SSE is 4 + 1 + 1 + 1 over four valid targets.
    assert result["loss"] == 7.0 / 4.0


def test_run_epoch_reports_loss_weighted_by_valid_target_count():
    model = _KnownForecast()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    loss, steps = _run_epoch(
        model,
        _uneven_valid_batches(),
        optimizer,
        torch.device("cpu"),
        max_steps=0,
        model_name="TSMixer",
        label_len=2,
        pred_len=3,
        phase="train",
    )

    assert steps == 2
    assert loss == 7.0 / 4.0


def test_run_epoch_zero_valid_batch_does_not_advance_adam_state():
    valid_batch = {
        "history": torch.zeros(1, 2, 1),
        "target": torch.tensor([[[2.0]]]),
        "target_mask": torch.tensor([[True]]),
        "issue_time_ns": torch.tensor([0], dtype=torch.int64),
    }
    zero_valid_batch = {
        "history": torch.zeros(1, 2, 1),
        "target": torch.tensor([[[100.0]]]),
        "target_mask": torch.tensor([[False]]),
        "issue_time_ns": torch.tensor([1], dtype=torch.int64),
    }
    single_batch_model = _KnownForecast()
    extra_zero_batch_model = copy.deepcopy(single_batch_model)
    single_batch_optimizer = torch.optim.Adam(single_batch_model.parameters(), lr=0.01)
    extra_zero_batch_optimizer = torch.optim.Adam(extra_zero_batch_model.parameters(), lr=0.01)

    _, single_steps = _run_epoch(
        single_batch_model,
        [valid_batch],
        single_batch_optimizer,
        torch.device("cpu"),
        max_steps=0,
        model_name="TSMixer",
        label_len=2,
        pred_len=1,
        phase="train",
    )
    _, extra_zero_steps = _run_epoch(
        extra_zero_batch_model,
        [valid_batch, zero_valid_batch],
        extra_zero_batch_optimizer,
        torch.device("cpu"),
        max_steps=0,
        model_name="TSMixer",
        label_len=2,
        pred_len=1,
        phase="train",
    )

    assert single_steps == 1
    assert extra_zero_steps == 2
    for expected, observed in zip(
        single_batch_model.parameters(), extra_zero_batch_model.parameters()
    ):
        assert torch.equal(expected, observed)


@pytest.mark.parametrize("phase", ["train", "val"])
def test_zero_valid_targets_fail_the_whole_phase(phase):
    batch = {
        "history": torch.zeros(1, 2, 1),
        "target": torch.ones(1, 1, 1),
        "target_mask": torch.tensor([[False]]),
        "issue_time_ns": torch.tensor([0], dtype=torch.int64),
    }
    model = _KnownForecast()

    with pytest.raises(RuntimeError, match="zero valid targets"):
        if phase == "train":
            _run_epoch(
                model,
                [batch],
                torch.optim.SGD(model.parameters(), lr=0.0),
                torch.device("cpu"),
                max_steps=0,
                model_name="TSMixer",
                label_len=2,
                pred_len=1,
                phase=phase,
            )
        else:
            _collect_predictions(
                model,
                [batch],
                torch.device("cpu"),
                max_steps=0,
                model_name="TSMixer",
                label_len=2,
                pred_len=1,
                phase=phase,
            )


def test_write_predictions_keeps_source_clock_timestamp_naive(tmp_path):
    dataset = SimpleNamespace(power_scale=1.0, rated_power=10.0)
    result = {
        "prediction": np.array([[[12.0]]]),
        "target": np.array([[[10.0]]]),
        "mask": np.array([[True]]),
        "issue_time_ns": np.array([np.datetime64("2025-01-01T00:00:00").astype("datetime64[ns]").astype(np.int64)]),
    }

    metrics = _write_predictions(tmp_path / "predictions.csv", result, dataset, 5)

    assert metrics["nmae"] == 0.2
    with (tmp_path / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["issue_time"] == "2025-01-01T00:00:00"
    assert row["target_time"] == "2025-01-01T00:05:00"
    assert "+00:00" not in row["issue_time"]
    assert "+00:00" not in row["target_time"]


def test_completion_manifest_records_identity_steps_and_artifact_hashes(tmp_path):
    output_dir = tmp_path / "seq24_pred1" / "skippd_luoyang" / "TSMixer"
    output_dir.mkdir(parents=True)
    checkpoint = output_dir / "best.pt"
    predictions = output_dir / "predictions.csv"
    metrics = output_dir / "metrics.json"
    checkpoint.write_bytes(b"checkpoint")
    predictions.write_text("predictions\n", encoding="utf-8")
    metrics.write_text("{\"count\": 1}\n", encoding="utf-8")
    args = SimpleNamespace(
        dataset="skippd_luoyang",
        model="TSMixer",
        smoke=True,
        epochs=1,
        max_train_steps=1,
        max_eval_steps=1,
        max_test_steps=1,
    )
    dataset = SimpleNamespace(seq_len=24, pred_len=1)
    payload = {"phase_steps": {"train": 1, "val": 1, "test": 1}}

    manifest = _write_completion_manifest(
        output_dir, args, dataset, payload, checkpoint, predictions, metrics
    )

    rows = manifest.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    header = rows[0].split("\t")
    values = rows[1].split("\t")
    fields = dict(zip(header, values))
    assert fields["dataset"] == "skippd_luoyang"
    assert fields["model"] == "TSMixer"
    assert fields["seq_len"] == "24"
    assert fields["pred_len"] == "1"
    assert fields["output_dir"] == str(output_dir.resolve())
    assert fields["run_mode"] == "smoke"
    assert fields["train_steps"] == fields["val_steps"] == fields["test_steps"] == "1"
    assert fields["best_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert fields["predictions_sha256"] == hashlib.sha256(predictions.read_bytes()).hexdigest()
    assert fields["metrics_sha256"] == hashlib.sha256(metrics.read_bytes()).hexdigest()
