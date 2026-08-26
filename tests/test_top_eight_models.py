from types import SimpleNamespace

import pytest
import torch

from models.tslib_adapter import forward_power_model
from models.tslib_factory import build_power_model
from models.tslib_registry import SELECTED_MODEL_NAMES
from run_time_series import parse_args


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


def tiny_args():
    return SimpleNamespace(
        d_model=16,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        factor=3,
        dropout=0.0,
        moving_avg=3,
        top_k=2,
        num_kernels=2,
        patch_len=8,
        patch_stride=4,
        batch_size=2,
        device="cpu",
    )


def test_registry_exposes_only_ranked_top_eight():
    assert SELECTED_MODEL_NAMES == EXPECTED_MODELS


@pytest.mark.parametrize("name", EXPECTED_MODELS)
@pytest.mark.parametrize(
    "seq_len,pred_len",
    [(24, 1), (48, 1), (48, 16), (48, 48), (96, 16), (96, 48)],
)
def test_retained_model_forecasts_supported_shapes(name, seq_len, pred_len):
    args = tiny_args()
    dataset = SimpleNamespace(seq_len=seq_len, pred_len=pred_len)
    model, config = build_power_model(name, args, dataset)
    prediction = forward_power_model(
        name,
        model.eval(),
        torch.randn(2, seq_len, 1),
        config.label_len,
        pred_len,
    )
    assert prediction.shape == (2, pred_len, 1)
    assert torch.isfinite(prediction).all()


def test_removed_model_is_rejected_by_cli():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "skippd_luoyang", "--model", "DLinear"])
