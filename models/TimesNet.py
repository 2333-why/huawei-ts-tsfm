import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from layers.Conv_Blocks import Inception_Block_V1
from layers.Embed import DataEmbedding


def FFT_for_Period(x, k=2):
    """Return dominant periods and their per-sample amplitudes."""
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        batch, length, channels = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        outputs = []
        total_length = self.seq_len + self.pred_len
        for period in period_list:
            if total_length % period != 0:
                padded_length = (total_length // period + 1) * period
                padding = torch.zeros(
                    batch, padded_length - total_length, channels,
                    dtype=x.dtype, device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                padded_length = total_length
                out = x
            out = out.reshape(batch, padded_length // period, period, channels)
            out = self.conv(out.permute(0, 3, 1, 2))
            out = out.permute(0, 2, 3, 1).reshape(batch, -1, channels)
            outputs.append(out[:, :total_length, :])
        outputs = torch.stack(outputs, dim=-1)
        weights = F.softmax(period_weight, dim=1)
        weights = weights.unsqueeze(1).unsqueeze(1).repeat(1, length, channels, 1)
        return torch.sum(outputs * weights, dim=-1) + x


class Model(nn.Module):
    """TimesNet forecasting model from the Time-Series-Library."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name in {"imputation", "anomaly_detection"}:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        elif self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for block in self.model:
            enc_out = self.layer_norm(block(enc_out))
        dec_out = self.projection(enc_out)
        scale = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        mean = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        return dec_out * scale + mean

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for block in self.model:
            enc_out = self.layer_norm(block(enc_out))
        dec_out = self.projection(enc_out)
        return dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

    def anomaly_detection(self, x_enc):
        return self.imputation(x_enc, None, None, None, None)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        for block in self.model:
            enc_out = self.layer_norm(block(enc_out))
        output = self.dropout(self.act(enc_out)).reshape(enc_out.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)[:, -self.pred_len:, :]
        if self.task_name == "imputation":
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        return None
