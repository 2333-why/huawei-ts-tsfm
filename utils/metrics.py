import numpy as np
import torch.nn as nn

STANFORD_SUNNY_DATES = np.array([
    "2017-09-15", "2017-10-06", "2017-10-22", "2018-02-16", "2018-06-12",
    "2018-06-23", "2019-01-25", "2019-06-23", "2019-07-14", "2019-10-14",
], dtype="datetime64[D]")

STANFORD_CLOUDY_DATES = np.array([
    "2017-06-24", "2017-09-20", "2017-10-11", "2018-01-25", "2018-03-09",
    "2018-10-04", "2019-05-27", "2019-06-28", "2019-08-10", "2019-10-19",
], dtype="datetime64[D]")


class Weighted_MSE_MAE(nn.Module):
    def __init__(self, alpha_weight=0.5):
        super(Weighted_MSE_MAE, self).__init__()
        self.alpha_weight = alpha_weight
        self.loss_mse = nn.MSELoss()
        self.loss_mae = nn.L1Loss()

    def forward(self, pred, true):
        loss_mse = self.loss_mse(pred, true)    
        loss_mae = self.loss_mae(pred, true)
        return loss_mse * self.alpha_weight + loss_mae * (1 - self.alpha_weight)

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)



def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))

def NMAE(pred, true):
    return np.mean(np.abs(true - pred) / np.abs(true))


def normalized_metric(pred, true, normalizer):
    normalizer = float(normalizer)
    if normalizer <= 0:
        raise ValueError("normalizer must be positive")

    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    nmae = mae / normalizer
    nrmse = rmse / normalizer
    return nrmse, nmae


def _single_split_metric(pred, true, normalizer):
    if len(pred) == 0:
        return {
            "count": 0,
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "nrmse": np.nan,
            "nmae": np.nan,
        }

    mae, mse, rmse, _, _ = metric(pred, true)
    nrmse, nmae = normalized_metric(pred, true, normalizer)
    return {
        "count": len(pred),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "nmae": nmae,
    }


def stanford_sunny_cloudy_metrics(pred, true, sample_times, normalizer):
    sample_times = np.asarray(sample_times, dtype="datetime64[s]")
    if len(sample_times) != len(pred):
        raise ValueError(
            f"sample_times length {len(sample_times)} does not match prediction length {len(pred)}"
        )

    dates = sample_times.astype("datetime64[D]")
    sunny_mask = np.isin(dates, STANFORD_SUNNY_DATES)
    # Original Stanford notebooks define cloudy test samples as the complement of sunny samples.
    cloudy_mask = ~sunny_mask

    return {
        "Sunny": _single_split_metric(pred[sunny_mask], true[sunny_mask], normalizer),
        "Cloudy": _single_split_metric(pred[cloudy_mask], true[cloudy_mask], normalizer),
        "Overall": _single_split_metric(pred, true, normalizer),
    }


def format_stanford_split_metrics(split_metrics):
    lines = [
        "split,count,nrmse,nrmse_percent,nmae,nmae_percent,rmse,mae",
    ]
    for split in ["Sunny", "Cloudy", "Overall"]:
        values = split_metrics[split]
        lines.append(
            "{},{},{:.6f},{:.3f},{:.6f},{:.3f},{:.6f},{:.6f}".format(
                split,
                values["count"],
                values["nrmse"],
                values["nrmse"] * 100,
                values["nmae"],
                values["nmae"] * 100,
                values["rmse"],
                values["mae"],
            )
        )
    return "\n".join(lines)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

# import torch
# import numpy as np
# def RSE_torch(pred, true):
#     return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - torch.mean(true)) ** 2))
#
# def CORR_torch(pred, true):
#     u = ((true - torch.mean(true, dim=0)) * (pred - torch.mean(pred, dim=0))).sum(dim=0)
#     d = torch.sqrt(((true - torch.mean(true, dim=0)) ** 2 * (pred - torch.mean(pred, dim=0)) ** 2).sum(dim=0))
#     return torch.mean(u / (d + 1e-8))  # 加1e-8防止除0
#
# def MAE_torch(pred, true):
#     return torch.mean(torch.abs(true - pred))
#
# def MSE_torch(pred, true):
#     return torch.mean((true - pred) ** 2)
#
# def RMSE_torch(pred, true):
#     return torch.sqrt(MSE_torch(pred, true))
#
# def MAPE_torch(pred, true):
#     return torch.mean(torch.abs((true - pred) / (true + 1e-5)))  # 避免除0
#
# def MSPE_torch(pred, true):
#     return torch.mean(torch.square((true - pred) / (true + 1e-5)))  # 避免除0
#
# def metric(pred, true):
#     # 如果是 numpy，先转换为 Tensor
#     if isinstance(pred, np.ndarray):
#         pred = torch.tensor(pred, dtype=torch.float32).cuda()
#     if isinstance(true, np.ndarray):
#         true = torch.tensor(true, dtype=torch.float32).cuda()
#
#     mae = MAE_torch(pred, true)
#     mse = MSE_torch(pred, true)
#     rmse = RMSE_torch(pred, true)
#     mape = MAPE_torch(pred, true)
#     mspe = MSPE_torch(pred, true)
#
#     return mae.item(), mse.item(), rmse.item(), mape.item(), mspe.item()
