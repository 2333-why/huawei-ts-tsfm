from types import SimpleNamespace

import pytest
import torch

from models import encoder_img


def _config(seq_len=16):
    return SimpleNamespace(
        enc_in=1, pred_len=1, seq_len=seq_len, task_name="long_term_forecast",
        d_model=4, e_layers=1, n_heads=1, d_ff=8, dropout=0.0, factor=1)


def test_encoder_is_registered_and_eval_is_deterministic():
    torch.manual_seed(7)
    model = encoder_img.Model(_config()).eval()
    assert any(name.startswith("encoder.") for name, _ in model.named_parameters())
    assert any(name.startswith("encoder.") for name in model.state_dict())
    inputs = torch.randn(2, 16, 1)
    first = model(inputs)
    second = model(inputs)
    torch.testing.assert_close(first, second)


def test_encoder_rejects_an_unregistered_runtime_length():
    model = encoder_img.Model(_config(seq_len=16))
    with pytest.raises(ValueError, match="configured segment count"):
        model(torch.randn(2, 25, 1))
