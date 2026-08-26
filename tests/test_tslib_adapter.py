import pytest
import torch

from models.tslib_adapter import (
    PowerBatch,
    decoder_from_history,
    forward_power_model,
    normalize_forecast,
    power_only_batch,
    prepare_model_inputs,
)


def test_power_batch_accepts_single_channel_mapping():
    x = torch.tensor([[[1.], [2.], [3.]]])
    y = torch.tensor([[[4.], [5.]]])
    batch = {
        "history": x,
        "target": y,
        "target_mask": torch.tensor([[True, False]]),
        "issue_time_ns": torch.tensor([0], dtype=torch.int64),
    }

    result = power_only_batch(batch, torch.device("cpu"))

    assert isinstance(result, PowerBatch)
    assert result.x.tolist() == [[[1.], [2.], [3.]]]
    assert result.y.tolist() == [[[4.], [5.]]]
    assert result.mask.tolist() == [[True, False]]


def test_decoder_uses_history_then_future_zeros():
    x = torch.tensor([[[1.], [2.], [3.], [4.]]])

    decoder = decoder_from_history(x, label_len=2, pred_len=3)

    assert decoder.tolist() == [[[3.], [4.], [0.], [0.], [0.]]]


def test_prepare_model_inputs_has_no_future_target_parameter():
    x = torch.tensor([[[1.], [2.], [3.], [4.]]])

    x_model, x_mark, x_dec, x_mark_dec = prepare_model_inputs(
        "Transformer", x, label_len=2, pred_len=3
    )

    assert x_model is x
    assert x_mark is None
    assert x_dec.tolist() == [[[3.], [4.], [0.], [0.], [0.]]]
    assert x_mark_dec is None


@pytest.mark.parametrize("model_name", ["Transformer", "TemporalFusionTransformer", "MultiPatchFormer"])
def test_all_model_names_use_the_same_power_only_inputs(model_name):
    x = torch.randn(2, 5, 1)

    inputs = prepare_model_inputs(
        model_name, x, label_len=2, pred_len=3
    )

    assert inputs[0] is x
    assert inputs[1] is None
    assert inputs[2].shape == (2, 5, 1)
    assert inputs[3] is None


def test_normalize_forecast_slices_history_prefix():
    output = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)

    result = normalize_forecast("SCINet", output, batch_size=1, pred_len=3)

    assert result.tolist() == [[[4.], [5.], [6.]]]


def test_normalize_forecast_accepts_first_tensor_from_tuple():
    output = (torch.arange(6, dtype=torch.float32).reshape(1, 6, 1), "attention")

    result = normalize_forecast("Autoformer", output, batch_size=1, pred_len=2)

    assert result.tolist() == [[[4.], [5.]]]


@pytest.mark.parametrize(
    "output,batch_size,match",
    [
        (torch.zeros(2, 2), 2, "3-D"),
        (torch.zeros(1, 2, 2), 1, "one output channel"),
        (torch.zeros(1, 3, 1), 2, "batch size"),
        (torch.zeros(1, 2, 1), 1, "at least"),
        (("not a tensor",), 2, "tensor"),
    ],
)
def test_normalize_forecast_rejects_invalid_shapes(output, batch_size, match):
    with pytest.raises(ValueError, match=match):
        normalize_forecast("TestModel", output, batch_size=batch_size, pred_len=3)


def test_forward_power_model_calls_model_and_normalizes_output():
    calls = []

    class RecordingModel:
        def __call__(self, *args):
            calls.append(args)
            return torch.arange(10, dtype=torch.float32).reshape(2, 5, 1)

    x = torch.tensor(
        [[[1.], [2.], [3.], [4.]],
         [[5.], [6.], [7.], [8.]]]
    )

    result = forward_power_model(
        "Transformer", RecordingModel(), x, label_len=2, pred_len=2
    )

    assert result.tolist() == [[[3.], [4.]], [[8.], [9.]]]
    assert len(calls) == 1
    assert torch.equal(calls[0][0], x)
    assert calls[0][1] is None
    assert calls[0][2].tolist() == [[[3.], [4.], [0.], [0.]], [[7.], [8.], [0.], [0.]]]
    assert calls[0][3] is None


def test_model_facing_boundary_accepts_only_power_channel():
    calls = []

    class RecordingModel:
        def __call__(self, *args):
            calls.append(args)
            return torch.zeros(1, 2, 1)

    x = torch.tensor([[[1.], [2.], [3.]]])

    prepared = prepare_model_inputs("Transformer", x, label_len=2, pred_len=2)
    forward_power_model("Transformer", RecordingModel(), x, label_len=2, pred_len=2)

    assert prepared[0].tolist() == [[[1.], [2.], [3.]]]
    assert prepared[2].tolist() == [[[2.], [3.], [0.], [0.]]]
    assert calls[0][0].tolist() == [[[1.], [2.], [3.]]]
    assert calls[0][2].tolist() == [[[2.], [3.], [0.], [0.]]]


def test_power_only_batch_rejects_missing_or_malformed_input():
    with pytest.raises(ValueError, match="mapping"):
        power_only_batch((torch.zeros(1, 2, 1),), torch.device("cpu"))
    with pytest.raises(ValueError, match="one power channel"):
        power_only_batch(
            {
                "history": torch.zeros(1, 2, 2),
                "target": torch.zeros(1, 1, 1),
            },
            torch.device("cpu"),
        )
