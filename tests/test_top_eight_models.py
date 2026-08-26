from types import SimpleNamespace

import pytest
import torch

from models.tslib_adapter import forward_power_model
from models.tslib_factory import build_power_model
from models.tslib_registry import SELECTED_MODEL_NAMES
from layers.tslib.Embed import TimeFeatureEmbedding
from layers.tslib.Pyraformer_EncDec import Encoder
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


def test_pyraformer_forwards_embedding_configuration_by_name():
    configs = SimpleNamespace(
        seq_len=24,
        d_model=8,
        d_ff=16,
        n_heads=2,
        dropout=0.37,
        e_layers=1,
        enc_in=1,
        embed="timeF",
        freq="t",
    )

    encoder = Encoder(configs, window_size=[4, 4], inner_size=5)

    assert isinstance(encoder.enc_embedding.temporal_embedding, TimeFeatureEmbedding)
    assert encoder.enc_embedding.temporal_embedding.embed.in_features == 5
    assert encoder.enc_embedding.dropout.p == 0.37
