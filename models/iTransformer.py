import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    """iTransformer forecasting model from the Time-Series-Library."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.task_name in {"imputation", "anomaly_detection"}:
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        elif self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, _, n_vars = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        _, length, n_vars = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, length, 1)
        return dec_out + means[:, 0, :].unsqueeze(1).repeat(1, length, 1)

    def anomaly_detection(self, x_enc):
        return self.imputation(x_enc, None, None, None, None)

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projection(self.dropout(self.act(enc_out)).reshape(enc_out.shape[0], -1))

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
