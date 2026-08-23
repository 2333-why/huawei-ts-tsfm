from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from models import MTS_31F


class ConfigAwareFakeSpatialEncoder(nn.Module):
    def __init__(self, configs, image_encoder_type):
        super().__init__()
        self.history_order = getattr(configs, "history_order", "current_first")
        self.frame_encoder = nn.ModuleList([nn.Conv2d(1, 1, kernel_size=1)])
        self.encoded_sequence = None

    def _ordered(self, x):
        if self.history_order == "current_first":
            return torch.flip(x, dims=[1])
        return x

    def encode_spatial_sequence(self, x, all_steps=False):
        assert all_steps
        self.encoded_sequence = self._ordered(x)
        return self.encoded_sequence

    def forward(self, x):
        ordered = self._ordered(x)
        return ordered[:, -1].reshape(x.shape[0], 1, 1)


@pytest.fixture
def lightweight_teacher_encoders(monkeypatch):
    monkeypatch.setattr(MTS_31F, "encoder", lambda args: nn.Identity())
    monkeypatch.setattr(MTS_31F, "encoder_img", lambda args: nn.Identity())
    monkeypatch.setattr(
        MTS_31F,
        "StanfordImageSequenceEncoder",
        ConfigAwareFakeSpatialEncoder,
    )


def teacher_model(history_order=None):
    args = SimpleNamespace(
        seq_len=3,
        pred_len=2,
        enc_in=1,
        c_out=1,
        data="Stanford",
        image_encoder_type="cnn",
    )
    args_img = SimpleNamespace(c_out=1)
    if history_order is not None:
        args_img.history_order = history_order
    args_weather = SimpleNamespace(c_out=1)
    return MTS_31F.Model(args, args_img, args_weather)


def run_teacher_with_current_first_input(model):
    model.forward(
        x_ts=torch.zeros(1, 1, 1),
        x_img_h=torch.tensor([3.0, 2.0, 1.0]).view(1, 3, 1, 1),
        x_img_f=torch.tensor([4.0, 5.0]).view(1, 2, 1, 1),
        x_weather_h=torch.empty(1, 0, 1),
        x_weather_f=torch.empty(1, 0, 1),
        ab="no_weather",
    )
    return model.branch_img.encoded_sequence.flatten().tolist()


def test_teacher_defaults_missing_image_history_order_to_current_first(
    lightweight_teacher_encoders,
):
    model = teacher_model()

    assert model.image_history_order == "current_first"


@pytest.mark.parametrize(
    ("history_order", "history"),
    [
        ("current_first", [3.0, 2.0, 1.0]),
        ("past_first", [1.0, 2.0, 3.0]),
    ],
)
def test_teacher_orders_history_before_appending_future(history_order, history):
    model = object.__new__(MTS_31F.Model)
    model.image_history_order = history_order
    x_h = torch.tensor(history).view(1, 3, 1, 1)
    x_f = torch.tensor([4.0, 5.0]).view(1, 2, 1, 1)

    ordered = model._ordered_teacher_images(x_h, x_f)

    assert ordered.flatten().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_teacher_rejects_unknown_image_history_order():
    model = object.__new__(MTS_31F.Model)
    model.image_history_order = "unknown"

    with pytest.raises(ValueError, match="unsupported image history_order: unknown"):
        model._ordered_teacher_images(torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))


def test_teacher_spatial_target_is_the_last_future_frame(
    lightweight_teacher_encoders,
):
    model = teacher_model(history_order="current_first")

    ordered = run_teacher_with_current_first_input(model)

    assert model.branch_img.history_order == "past_first"
    assert ordered == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert model.last_image_spatial_target.flatten().tolist() == [5.0]


def test_explicit_legacy_mode_preserves_raw_current_first_history(
    lightweight_teacher_encoders,
):
    corrected = teacher_model(history_order="current_first")
    legacy = teacher_model(history_order="past_first")

    corrected_sequence = run_teacher_with_current_first_input(corrected)
    legacy_sequence = run_teacher_with_current_first_input(legacy)

    assert corrected_sequence == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert legacy_sequence == [3.0, 2.0, 1.0, 4.0, 5.0]
    assert corrected_sequence != legacy_sequence
