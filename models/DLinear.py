import torch
import torch.nn as nn

from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """DLinear forecasting model from the Time-Series-Library."""

    def __init__(self, configs, individual=False):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = (
            configs.seq_len
            if self.task_name in {"classification", "anomaly_detection", "imputation"}
            else configs.pred_len
        )
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in

        if individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                seasonal = nn.Linear(self.seq_len, self.pred_len)
                trend = nn.Linear(self.seq_len, self.pred_len)
                seasonal.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))
                trend.weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))
                self.Linear_Seasonal.append(seasonal)
                self.Linear_Trend.append(trend)
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))

        if self.task_name == "classification":
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros(
                seasonal_init.size(0), seasonal_init.size(1), self.pred_len,
                dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros(
                trend_init.size(0), trend_init.size(1), self.pred_len,
                dtype=trend_init.dtype, device=trend_init.device)
            for index in range(self.channels):
                seasonal_output[:, index, :] = self.Linear_Seasonal[index](
                    seasonal_init[:, index, :])
                trend_output[:, index, :] = self.Linear_Trend[index](
                    trend_init[:, index, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        return (seasonal_output + trend_output).permute(0, 2, 1)

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        return self.projection(enc_out.reshape(enc_out.shape[0], -1))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc)[:, -self.pred_len:, :]
        if self.task_name == "imputation":
            return self.imputation(x_enc)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        if self.task_name == "classification":
            return self.classification(x_enc)
        return None
