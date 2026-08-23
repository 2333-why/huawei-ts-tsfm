import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Model as encoder
from models.encoder_img import Model as encoder_img
from models.image_sequence_encoder import StanfordImageSequenceEncoder
from models.utils import BilinearOrthogonalProjector as bop

def cal_similarity(x, y, delay=3):
    if x.shape != y.shape:
        raise ValueError("Input tensors x and y must have the same shape.")

    _, seq_len, _ = x.shape
    
    # Store cross-correlation scores for each delay
    correlations_per_delay = []

    # Iterate through all possible time delays p from 0 to delay
    for p in range(delay + 1):
        # Simulate time shift X(t+p) through slicing operation
        # This is equivalent to shifting x left by p units
        shifted_x = x[:, p:, :]
        
        # Slice y accordingly to match the length of shifted_x
        aligned_y = y[:, :seq_len - p, :]
        
        # Calculate element-wise product
        product = shifted_x * aligned_y
        
        # Sum along the sequence length dimension to get cross-correlation score at delay p
        # This is a discrete implementation of the integral formula in the paper
        # Output shape: (batch_size, feature_dim)
        correlation_at_p = torch.sum(product, dim=1)
        
        correlations_per_delay.append(correlation_at_p)

    # Stack all delay score tensors
    # New tensor shape: (delay + 1, batch_size, feature_dim)
    stacked_correlations = torch.stack(correlations_per_delay, dim=0)

    # Take maximum along delay dimension (dim=0) to find best matching score for each feature
    # torch.max returns (values, indices), we only need values
    max_correlation, _ = torch.max(stacked_correlations, dim=0)
    
    # Add sequence dimension to match (batch_size, seq_len, feature_dim) shape
    # Preserve the correlation value and only parameterize its forecast axis.
    return max_correlation.unsqueeze(1).expand(-1, seq_len, -1)

class Model(nn.Module):
    def __init__(self, args, args_img, args_weather):
        super(Model, self).__init__()
        self.args = args
        self.args_img = args_img
        self.image_history_order = getattr(args_img, 'history_order', 'current_first')
        teacher_seq_len = int(args.seq_len) + int(args.pred_len)
        teacher_img_args = copy.copy(args_img)
        teacher_img_args.seq_len = teacher_seq_len
        teacher_weather_args = copy.copy(args_weather)
        teacher_weather_args.seq_len = teacher_seq_len
        self.branch_ts = encoder(args)
        self.ts_projection = (
            nn.Linear(args.enc_in, args.c_out)
            if getattr(args, 'data', None) in ['LuoyangParquet', 'YLJParquet']
            and args.enc_in != args.c_out
            else nn.Identity()
        )
        self.image_encoder_type = getattr(args, 'image_encoder_type', 'bop')
        if getattr(args, 'data', None) in ['Stanford', 'LuoyangParquet', 'YLJParquet'] and self.image_encoder_type != 'bop':
            teacher_img_args.history_order = 'past_first'
            self.branch_img = StanfordImageSequenceEncoder(teacher_img_args, self.image_encoder_type)
            self.bop_img = None
        else:
            self.branch_img = encoder_img(teacher_img_args)
            self.bop_img = bop(teacher_img_args.H, teacher_img_args.W, teacher_img_args.r_h, teacher_img_args.r_w)
        self.branch_weather = encoder_img(teacher_weather_args)
        
        # Project image and weather features to time series feature dimension separately
        self.img_projection = nn.Linear(args_img.c_out, args.c_out)  # 512 -> 42
        self.weather_projection = nn.Linear(args_weather.c_out, args.c_out)  # 6 -> 42
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))  # [time_series, image, weather]
        self.last_image_spatial_target = None

    def _ordered_teacher_images(self, x_img_h, x_img_f):
        if self.image_history_order == 'current_first':
            x_img_h = torch.flip(x_img_h, dims=[1])
        elif self.image_history_order != 'past_first':
            raise ValueError(
                f'unsupported image history_order: {self.image_history_order}')
        return torch.cat([x_img_h, x_img_f], dim=1)
        
    def forward(self, x_ts, x_img_h, x_img_f, x_weather_h, x_weather_f, ab=None,
                image_mask=None):
        # Process time series features
        x_ts_f_pred = self.ts_projection(self.branch_ts(x_ts))
        self.last_image_spatial_target = None

        x_img = self._ordered_teacher_images(x_img_h, x_img_f)
        
        # Process image sequences of different lengths separately with complete parameters
        if ab in ['no_img', 'ts_only', 'ts']:
            x_img_f_pred = torch.zeros_like(x_ts_f_pred)
            sim_img_ts = torch.zeros_like(x_ts_f_pred[:, :1, :])
        elif self.image_encoder_type != 'bop':
            expected_channels = self.branch_img.frame_encoder[0].in_channels
            if x_img.dim() == 5 and x_img.shape[2] == 3 and expected_channels == 1:
                x_img = x_img.mean(dim=2)
            spatial_sequence = self.branch_img.encode_spatial_sequence(
                x_img, all_steps=True)
            self.last_image_spatial_target = spatial_sequence[:, -1]
            x_img_f_pred = self.branch_img(x_img)
            x_img_f_pred = self.img_projection(x_img_f_pred)
            sim_img_ts = cal_similarity(x_img_f_pred, x_ts_f_pred)
        elif x_img.dim() >= 4:
            self.last_image_spatial_target = None
            x_img = self.bop_img(x_img)
            x_img_f_pred = self.branch_img(x_img)
            x_img_f_pred = self.img_projection(x_img_f_pred)
            sim_img_ts = cal_similarity(x_img_f_pred, x_ts_f_pred)
        else:
            self.last_image_spatial_target = None
            x_img_f_pred = self.branch_img(x_img)
            x_img_f_pred = self.img_projection(x_img_f_pred)
            sim_img_ts = cal_similarity(x_img_f_pred, x_ts_f_pred)
    
        if image_mask is not None and ab not in ['no_img', 'ts_only', 'ts']:
            image_available = image_mask.to(x_img_f_pred.device).bool().any(dim=1)
            x_img_f_pred = x_img_f_pred * image_available[:, None, None].to(x_img_f_pred.dtype)
            sim_img_ts = cal_similarity(x_img_f_pred, x_ts_f_pred)

        if ab in ['no_weather', 'ts_only', 'ts']:
            x_weather_f_pred = torch.zeros_like(x_ts_f_pred)
            sim_weather_ts = torch.zeros_like(x_ts_f_pred[:, :1, :])
        else:
            x_weather = torch.cat([x_weather_h, x_weather_f], dim=1)
            x_weather_f_pred = self.branch_weather(x_weather)
            x_weather_f_pred = self.weather_projection(x_weather_f_pred)
            sim_weather_ts = cal_similarity(x_weather_f_pred, x_ts_f_pred)
        
        x_fused = x_ts_f_pred + x_img_f_pred * sim_img_ts + x_weather_f_pred * sim_weather_ts
                
        # Feature processing after fusion
        feat = x_fused.clone()
        return x_fused, sim_img_ts, sim_weather_ts, feat
