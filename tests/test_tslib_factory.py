from types import SimpleNamespace

import pytest
import torch

from models.tslib_factory import (
    build_model_config,
    build_power_model,
    compute_koopa_mask,
)
from models.tslib_adapter import forward_power_model, prepare_model_inputs


def _base_args():
    return SimpleNamespace(
        d_model=8,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=16,
        factor=2,
        dropout=0.0,
        moving_avg=4,
        top_k=2,
        num_kernels=1,
        patch_len=16,
        patch_stride=8,
        batch_size=2,
        device="cpu",
    )


def test_factory_sets_univariate_dimensions():
    args = _base_args()
    dataset = SimpleNamespace(seq_len=16, pred_len=4)

    config = build_model_config(args, dataset)

    assert (config.enc_in, config.dec_in, config.c_out) == (1, 1, 1)
    assert config.features == "M"
    assert config.data == "power_only"


@pytest.mark.parametrize("seq_len,pred_len", [(24, 1), (48, 12)])
def test_segrnn_factory_supports_planned_point_count_settings(seq_len, pred_len):
    args = _base_args()
    dataset = SimpleNamespace(seq_len=seq_len, pred_len=pred_len)
    model, config = build_power_model("SegRNN", args, dataset)
    model.eval()

    prediction = forward_power_model(
        "SegRNN", model, torch.randn(2, seq_len, 1), label_len=seq_len,
        pred_len=pred_len,
    )

    assert seq_len % config.seg_len == 0
    assert pred_len % config.seg_len == 0
    assert prediction.shape == (2, pred_len, 1)


def test_multipatchformer_factory_uses_32_point_internal_history():
    args = _base_args()
    dataset = SimpleNamespace(seq_len=16, pred_len=8)

    _, config = build_power_model("MultiPatchFormer", args, dataset)

    assert config.seq_len == 32


def test_pattn_factory_caps_patch_and_stride_to_history():
    args = _base_args()
    dataset = SimpleNamespace(seq_len=4, pred_len=8)

    model, config = build_power_model("PAttn", args, dataset)

    assert model.patch_size == config.patch_len == 4
    assert model.stride == 4


@pytest.mark.parametrize("history_len", [16, 32, 40])
def test_multipatchformer_factory_forward_normalizes_history_and_channel(
    history_len,
):
    args = _base_args()
    dataset = SimpleNamespace(seq_len=16, pred_len=8)
    model, config = build_power_model("MultiPatchFormer", args, dataset)
    model.eval()
    x = torch.randn(2, history_len, 2)

    prepared = prepare_model_inputs(
        "MultiPatchFormer", x, label_len=history_len, pred_len=dataset.pred_len
    )
    assert prepared[0].shape == (2, 32, 1)
    prediction = forward_power_model(
        "MultiPatchFormer", model, x, label_len=history_len, pred_len=dataset.pred_len
    )
    power_prediction = forward_power_model(
        "MultiPatchFormer", model, x[..., :1], label_len=history_len,
        pred_len=dataset.pred_len,
    )

    assert config.seq_len == 32
    assert prediction.shape == (2, dataset.pred_len, 1)
    assert torch.allclose(prediction, power_prediction)


@pytest.mark.parametrize("pred_len", [1, 4, 8, 16])
def test_multipatchformer_factory_supports_positive_prediction_lengths(pred_len):
    args = _base_args()
    dataset = SimpleNamespace(seq_len=16, pred_len=pred_len)
    model, config = build_power_model("MultiPatchFormer", args, dataset)
    model.eval()

    prediction = forward_power_model(
        "MultiPatchFormer", model, torch.randn(2, 16, 2), label_len=16,
        pred_len=pred_len,
    )

    assert config.seq_len == 32
    assert prediction.shape == (2, pred_len, 1)


class _FiniteKoopaDataset:
    seq_len = 16

    def __init__(self, pred_len):
        self.pred_len = pred_len
        self.accesses = []

    def __len__(self):
        return 2

    def __getitem__(self, index):
        self.accesses.append(index)
        if index >= len(self):
            raise IndexError
        return (
            torch.arange(32, dtype=torch.float32).reshape(16, 2),
            torch.zeros(self.pred_len, 1),
        )


@pytest.mark.parametrize("pred_len", [16, 48])
def test_koopa_factory_supports_planned_prediction_lengths(pred_len):
    args = _base_args()
    dataset = _FiniteKoopaDataset(pred_len)
    model, config = build_power_model("Koopa", args, dataset)
    model.eval()

    prediction = forward_power_model(
        "Koopa", model, torch.randn(2, 16, 1), label_len=16,
        pred_len=pred_len,
    )

    assert model.seg_len == config.seg_len == 8
    assert prediction.shape == (2, pred_len, 1)


def test_koopa_smoke_factory_limits_mask_scan_to_one_sample():
    args = _base_args()
    args.smoke = True
    dataset = _FiniteKoopaDataset(pred_len=16)

    build_power_model("Koopa", args, dataset)

    assert dataset.accesses == [0]


def test_koopa_non_smoke_factory_scans_full_dataset():
    args = _base_args()
    dataset = _FiniteKoopaDataset(pred_len=16)

    build_power_model("Koopa", args, dataset)

    assert dataset.accesses == [0, 1, 2]


@pytest.mark.parametrize("model_name", ["Mamba", "MambaSimple"])
@pytest.mark.parametrize("pred_len", [8, 48])
def test_mamba_variants_project_history_to_requested_horizon(model_name, pred_len):
    if not torch.cuda.is_available():
        pytest.fail("Mamba real-horizon regression requires CUDA")

    args = _base_args()
    args.device = "cuda:0"
    dataset = SimpleNamespace(seq_len=16, pred_len=pred_len)
    model, config = build_power_model(model_name, args, dataset)
    model.to(torch.device("cuda:0"))
    model.eval()

    with torch.no_grad():
        prediction = forward_power_model(
            model_name,
            model,
            torch.randn(2, 16, 1, device="cuda:0"),
            label_len=16,
            pred_len=pred_len,
        )

    assert config.seq_len == 16
    assert prediction.shape == (2, pred_len, 1)


def test_timemixer_factory_single_scale_forward():
    args = _base_args()
    dataset = SimpleNamespace(seq_len=16, pred_len=8)
    model, config = build_power_model("TimeMixer", args, dataset)
    model.eval()

    prediction = forward_power_model(
        "TimeMixer", model, torch.randn(2, 16, 1), label_len=16, pred_len=8
    )

    assert config.down_sampling_layers == 0
    assert prediction.shape == (2, 8, 1)


def test_wpmixer_factory_normalizes_cpu_device():
    args = _base_args()
    args.device = "cpu"
    dataset = SimpleNamespace(seq_len=16, pred_len=8)

    _, config = build_power_model("WPMixer", args, dataset)

    assert config.device == torch.device("cpu")


class _PowerOnlyDataset:
    seq_len = 16
    pred_len = 4

    def __init__(self):
        self._items = [
            (torch.arange(16, dtype=torch.float32).reshape(16, 1),),
            (torch.arange(16, dtype=torch.float32).reshape(16, 1) + 1,),
        ]
        self.non_power_channel_read = False

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def fail_if_non_power_channel_was_read(self):
        assert not self.non_power_channel_read


class _FakeMultichannelDataset(_PowerOnlyDataset):
    def __init__(self):
        self._items = []
        self.non_power_channel_read = False
        for offset in range(2):
            power = torch.arange(16, dtype=torch.float32).reshape(16, 1) + offset
            non_power = torch.full_like(power, 1000 + offset)
            self._items.append((torch.cat((power, non_power), dim=-1),))

    def __getitem__(self, index):
        item = self._items[index]
        class GuardedTensor:
            def __init__(self, tensor, dataset):
                self.tensor = tensor
                self.dataset = dataset

            def __getitem__(self, key):
                if isinstance(key, tuple) and any(
                    isinstance(part, int) and part != 0 for part in key
                ):
                    self.dataset.non_power_channel_read = True
                return self.tensor[key]

            @property
            def shape(self):
                return self.tensor.shape

        return (GuardedTensor(item[0], self),)


def test_koopa_mask_uses_only_power_channel():
    dataset = _FakeMultichannelDataset()

    mask = compute_koopa_mask(dataset)

    dataset.fail_if_non_power_channel_was_read()
    assert mask.ndim == 1


def test_koopa_requires_factory_supplied_mask():
    from models.Koopa import Model

    config = build_model_config(_base_args(), SimpleNamespace(seq_len=16, pred_len=8))

    with pytest.raises(ValueError, match="mask_spectrum"):
        Model(config)


def test_temporal_fusion_transformer_uses_power_only_definition_and_broadcast_mask():
    from models.TemporalFusionTransformer import (
        InterpretableMultiHeadAttention,
        datatype_dict,
    )

    config = build_model_config(_base_args(), SimpleNamespace(seq_len=16, pred_len=8))
    attention = InterpretableMultiHeadAttention(config)

    assert datatype_dict["power_only"].observed == [0]
    assert tuple(attention.mask.shape) == (1, 1, 24, 24)


def test_micn_does_not_hardcode_cuda_constructor_device():
    from models.MICN import Model

    config = build_model_config(_base_args(), SimpleNamespace(seq_len=16, pred_len=8))
    model = Model(config)

    assert model.conv_trans.mic[0].device != torch.device("cuda:0")
