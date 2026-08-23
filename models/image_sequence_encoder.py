import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class StanfordImageSequenceEncoder(nn.Module):
    """CNN + temporal GRU encoder for Stanford sky-image sequences.

    Input shape:
      gray: [B, T, H, W]
      rgb:  [B, T, C, H, W]

    The cnn_motion variant concatenates frame differences as extra channels,
    giving the image branch a cheap cloud-motion signal without optical-flow
    preprocessing.
    """

    def __init__(self, configs, encoder_type="cnn_motion"):
        super().__init__()
        self.output_dim = int(configs.c_out)
        self.pred_len = int(configs.pred_len)
        self.encoder_type = encoder_type
        self.history_order = getattr(configs, "history_order", "current_first")
        image_mode = getattr(configs, "stanford_image_mode", "gray")
        base_channels = 3 if image_mode == "rgb" else 1
        self.use_motion = encoder_type in ["cnn_motion", "cnn_motion_only"]
        self.motion_only = encoder_type == "cnn_motion_only"
        self.latest_only = encoder_type == "cnn_latest"
        in_channels = base_channels * (2 if encoder_type == "cnn_motion" else 1)
        conv_dim = int(getattr(configs, "image_cnn_dim", 64))
        hidden_dim = int(getattr(configs, "image_temporal_hidden_dim", max(self.output_dim, conv_dim)))
        dropout = float(getattr(configs, "dropout", 0.1))

        self.frame_encoder = nn.Sequential(
            nn.Conv2d(in_channels, conv_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim // 2),
            nn.GELU(),
            nn.Conv2d(conv_dim // 2, conv_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.GELU(),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.temporal_encoder = nn.GRU(
            input_size=conv_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def _normalize_shape(self, x):
        if x.dim() == 4:
            return x.unsqueeze(2)
        if x.dim() == 5:
            return x
        raise ValueError(f"Unsupported Stanford image tensor shape: {tuple(x.shape)}")

    def _add_motion_channels(self, x):
        if not self.use_motion:
            return x
        diff = torch.zeros_like(x)
        diff[:, 1:] = x[:, 1:] - x[:, :-1]
        if self.motion_only:
            return diff
        return torch.cat([x, diff], dim=2)

    def encode_spatial_sequence(self, x, all_steps=True):
        """Return the legacy CNN map before global spatial pooling.

        The returned sequence is chronological (oldest to current) regardless
        of the loader's configured history order. For a 64x64 Stanford input,
        the shape is [B, T, image_cnn_dim, 8, 8]. ``all_steps=True`` exposes
        every history frame even for the ``cnn_latest`` encoder; passing False
        follows that encoder's legacy latest-frame behavior.

        This method deliberately reuses ``frame_encoder`` in place. It adds no
        parameters or state-dict keys, so existing checkpoints remain strict-
        load compatible.
        """
        x = self._normalize_shape(x)
        if self.history_order == "current_first":
            x = torch.flip(x, dims=[1])
        x = self._add_motion_channels(x)
        if self.latest_only and not all_steps:
            x = x[:, -1:]

        batch, steps, channels, height, width = x.shape
        spatial = x.reshape(batch * steps, channels, height, width)
        for layer in self.frame_encoder:
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                break
            spatial = layer(spatial)
        return spatial.reshape(batch, steps, spatial.shape[1], spatial.shape[2], spatial.shape[3])

    def forward(self, x):
        x = self._normalize_shape(x)
        if self.history_order == "current_first":
            x = torch.flip(x, dims=[1])
        x = self._add_motion_channels(x)
        if self.latest_only:
            x = x[:, -1:]
        batch, steps, channels, height, width = x.shape
        frames = x.reshape(batch * steps, channels, height, width)
        frame_features = self.frame_encoder(frames).flatten(1)
        frame_features = frame_features.reshape(batch, steps, -1)
        temporal_features, _ = self.temporal_encoder(frame_features)
        last = temporal_features[:, -1]
        output = self.head(last).unsqueeze(1)
        if self.pred_len > 1:
            output = output.repeat(1, self.pred_len, 1)
        return output


class StanfordDualAppearanceMotionEncoder(nn.Module):
    """Causal dual-path encoder using the latest frame and ordered frame differences."""

    def __init__(self, configs):
        super().__init__()
        self.appearance = StanfordImageSequenceEncoder(configs, "cnn_latest")
        self.motion = StanfordImageSequenceEncoder(configs, "cnn_motion_only")
        self.history_order = getattr(configs, "history_order", "current_first")

    def forward(self, x):
        appearance = self.appearance(x)
        motion = self.motion(x)
        normalized = self.appearance._normalize_shape(x)
        if self.history_order == "current_first":
            normalized = torch.flip(normalized, dims=[1])
        motion_energy = normalized[:, 1:].sub(normalized[:, :-1]).abs().mean(
            dim=(1, 2, 3, 4), keepdim=False).view(-1, 1, 1)
        return appearance, motion, motion_energy


class StanfordSolarAdvectionEncoder(nn.Module):
    """Causal, sun-conditioned feature advection for Stanford sky histories.

    ``spatial_sequence`` must be the chronological output of
    ``StanfordImageSequenceEncoder.encode_spatial_sequence`` with shape
    [B, T, C, H, W]. ``x_mark_h`` and ``pv_history`` retain the loader order;
    they are reversed internally when ``history_order=current_first``.
    ``y_mark`` is the known forecast timestamp, not a future observation.

    The module estimates local feature displacement between adjacent history
    maps, extrapolates the latest map to the forecast horizon, and pools the
    result with a time-conditioned solar query. It never accepts or reads a
    future image.
    """

    uses_future_images = False
    time_feature_dim = 6

    def __init__(self, configs):
        super().__init__()
        self.input_dim = int(getattr(configs, "image_cnn_dim", 64))
        self.output_dim = int(getattr(configs, "advection_output_dim", 64))
        self.motion_dim = int(getattr(
            configs, "advection_motion_dim", min(32, self.input_dim)))
        self.history_order = getattr(configs, "history_order", "current_first")
        self.forecast_horizon_minutes = float(getattr(
            configs, "forecast_horizon_minutes", 15.0))
        self.sample_interval_minutes = float(getattr(
            configs, "sample_interval_minutes", 1.0))
        self.capacity_kw = float(getattr(configs, "stanford_capacity_kw", 30.1))
        self.correlation_radius = 2
        self.correlation_temperature = float(getattr(
            configs, "advection_correlation_temperature", 0.1))
        self.recent_flow_steps = int(getattr(configs, "advection_recent_flow_steps", 5))

        if self.input_dim <= 0 or self.output_dim <= 0 or self.motion_dim <= 0:
            raise ValueError("advection feature dimensions must be positive")
        if self.history_order not in ["current_first", "past_first"]:
            raise ValueError("history_order must be current_first or past_first")
        if self.sample_interval_minutes <= 0:
            raise ValueError("sample_interval_minutes must be positive")
        if self.capacity_kw <= 0:
            raise ValueError("stanford_capacity_kw must be positive")
        if self.correlation_temperature <= 0:
            raise ValueError("advection_correlation_temperature must be positive")
        if self.recent_flow_steps <= 0:
            raise ValueError("advection_recent_flow_steps must be positive")
        if self.forecast_horizon_minutes < 0:
            raise ValueError("forecast_horizon_minutes must be non-negative")

        self.motion_projection = nn.Conv2d(
            self.input_dim, self.motion_dim, kernel_size=1, bias=False)
        self.spatial_key = nn.Conv2d(
            self.input_dim, self.output_dim, kernel_size=1, bias=False)
        self.coordinate_projection = nn.Linear(5, self.output_dim, bias=False)
        self.time_projection = nn.Sequential(
            nn.Linear(self.time_feature_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        self.pv_encoder = nn.GRU(
            input_size=2,
            hidden_size=self.output_dim,
            num_layers=1,
            batch_first=True,
        )
        self.image_context_projection = nn.Linear(
            self.input_dim * 2, self.output_dim)
        self.motion_context_projection = nn.Linear(3, self.output_dim)
        self.event_norm = nn.LayerNorm(self.output_dim)
        self.event_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )

        offsets = [
            (dx, dy)
            for dy in range(-self.correlation_radius, self.correlation_radius + 1)
            for dx in range(-self.correlation_radius, self.correlation_radius + 1)
        ]
        self.register_buffer(
            "_correlation_offsets",
            torch.tensor(offsets, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _coordinate_grid(height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)

    @staticmethod
    def _coordinate_features(grid):
        x = grid[..., 0]
        y = grid[..., 1]
        return torch.stack([x, y, x.square(), y.square(), x * y], dim=-1)

    @staticmethod
    def _time_features(marks):
        if marks.dim() != 3 or marks.shape[-1] < 5:
            raise ValueError(
                "Stanford time marks must have shape [B,T,>=5] with month/day/weekday/hour/minute")
        month = marks[..., 0]
        day = marks[..., 1]
        hour = marks[..., 3]
        minute = marks[..., 4]

        time_angle = 2.0 * math.pi * (hour + minute / 60.0) / 24.0
        year_phase = ((month - 1.0) + (day - 1.0) / 31.0) / 12.0
        year_angle = 2.0 * math.pi * year_phase
        sin_time = torch.sin(time_angle)
        cos_time = torch.cos(time_angle)
        sin_year = torch.sin(year_angle)
        cos_year = torch.cos(year_angle)
        return torch.stack([
            sin_time,
            cos_time,
            sin_year,
            cos_year,
            sin_time * sin_year,
            cos_time * cos_year,
        ], dim=-1)

    def _ordered_auxiliary(self, values):
        if self.history_order == "current_first":
            return torch.flip(values, dims=[1])
        return values

    def _validate_inputs(self, spatial_sequence, x_mark_h, y_mark, pv_history):
        if spatial_sequence.dim() != 5:
            raise ValueError("spatial_sequence must have shape [B,T,C,H,W]")
        batch, steps, channels, _, _ = spatial_sequence.shape
        if steps < 2:
            raise ValueError("solar advection requires at least two historical frames")
        if channels != self.input_dim:
            raise ValueError(
                f"spatial_sequence has {channels} channels, expected {self.input_dim}")
        if x_mark_h.dim() != 3 or x_mark_h.shape[:2] != (batch, steps):
            raise ValueError("x_mark_h must align with spatial_sequence as [B,T,M]")
        if y_mark.dim() == 2:
            y_mark = y_mark.unsqueeze(1)
        if y_mark.dim() != 3 or y_mark.shape[0] != batch or y_mark.shape[1] < 1:
            raise ValueError("y_mark must have shape [B,Ty,M] with Ty >= 1")
        if pv_history.dim() == 2:
            pv_history = pv_history.unsqueeze(-1)
        if pv_history.dim() != 3 or pv_history.shape[:2] != (batch, steps):
            raise ValueError("pv_history must align with spatial_sequence as [B,T,Cp]")
        return y_mark, pv_history

    def _solar_attention(self, spatial_sequence, queries):
        batch, steps, channels, height, width = spatial_sequence.shape
        flat = spatial_sequence.reshape(batch * steps, channels, height, width)
        keys = self.spatial_key(flat).reshape(
            batch, steps, self.output_dim, height, width)
        coordinate_grid = self._coordinate_grid(
            height, width, spatial_sequence.device, spatial_sequence.dtype)
        position = self.coordinate_projection(
            self._coordinate_features(coordinate_grid)).permute(2, 0, 1)
        keys = keys + position.view(1, 1, self.output_dim, height, width)
        scores = (keys * queries.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)
        scores = scores * (self.output_dim ** -0.5)
        attention = torch.softmax(scores.flatten(start_dim=2), dim=-1).reshape(
            batch, steps, 1, height, width)
        context = (spatial_sequence * attention).sum(dim=(-1, -2))
        return context, attention

    def _local_correlation_flow(self, spatial_sequence):
        batch, steps, channels, height, width = spatial_sequence.shape
        projected = self.motion_projection(
            spatial_sequence.reshape(batch * steps, channels, height, width))
        projected = projected.reshape(batch, steps, self.motion_dim, height, width)
        projected = F.normalize(projected, dim=2, eps=1e-6)

        previous = projected[:, :-1].reshape(-1, self.motion_dim, height, width)
        current = projected[:, 1:].reshape(-1, self.motion_dim, height, width)
        kernel = 2 * self.correlation_radius + 1
        patches = F.unfold(
            current,
            kernel_size=kernel,
            padding=self.correlation_radius,
        ).reshape(-1, self.motion_dim, kernel * kernel, height, width)
        correlation = (previous.unsqueeze(2) * patches).sum(dim=1)
        probabilities = torch.softmax(
            correlation / self.correlation_temperature, dim=1)

        offsets = self._correlation_offsets.to(
            device=probabilities.device, dtype=probabilities.dtype)
        offsets = offsets.transpose(0, 1).view(1, 2, kernel * kernel, 1, 1)
        flow = (probabilities.unsqueeze(1) * offsets).sum(dim=2)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(
            dim=1, keepdim=True)
        confidence = 1.0 - entropy / math.log(float(kernel * kernel))
        confidence = confidence.clamp(min=0.0, max=1.0)
        return (
            flow.reshape(batch, steps - 1, 2, height, width),
            confidence.reshape(batch, steps - 1, 1, height, width),
        )

    def _aggregate_velocity(self, flow, confidence, history_attention):
        recent_steps = min(self.recent_flow_steps, flow.shape[1])
        flow = flow[:, -recent_steps:]
        confidence = confidence[:, -recent_steps:]
        endpoint_attention = history_attention[:, 1:][:, -recent_steps:]
        spatial_size = flow.shape[-2] * flow.shape[-1]
        weights = confidence * (0.25 + endpoint_attention * float(spatial_size))
        denominator = weights.sum(dim=(1, 3, 4)).clamp_min(1e-6)
        velocity = (flow * weights).sum(dim=(1, 3, 4)) / denominator
        attention_denominator = (
            0.25 + endpoint_attention * float(spatial_size)
        ).sum(dim=(1, 3, 4)).clamp_min(1e-6)
        aggregate_confidence = (
            confidence * (0.25 + endpoint_attention * float(spatial_size))
        ).sum(dim=(1, 3, 4)) / attention_denominator
        return velocity, aggregate_confidence

    def _advect(self, latest_spatial, velocity):
        batch, _, height, width = latest_spatial.shape
        horizon_steps = self.forecast_horizon_minutes / self.sample_interval_minutes
        displacement = velocity * horizon_steps
        base_grid = self._coordinate_grid(
            height, width, latest_spatial.device, latest_spatial.dtype)
        base_grid = base_grid.unsqueeze(0).expand(batch, -1, -1, -1)
        scale = latest_spatial.new_tensor([
            2.0 / max(width - 1, 1),
            2.0 / max(height - 1, 1),
        ])
        sampling_grid = base_grid - displacement[:, None, None, :] * scale
        future_spatial = F.grid_sample(
            latest_spatial,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        validity = sampling_grid.abs().le(1.0).all(dim=-1, keepdim=True)
        validity = validity.permute(0, 3, 1, 2).to(latest_spatial.dtype)
        return future_spatial, displacement, sampling_grid, validity

    def _pv_context(self, pv_history):
        pv = pv_history[..., :1] / self.capacity_kw
        delta = torch.zeros_like(pv)
        delta[:, 1:] = pv[:, 1:] - pv[:, :-1]
        _, hidden = self.pv_encoder(torch.cat([pv, delta], dim=-1))
        return hidden[-1]

    def forward(self, spatial_sequence, x_mark_h, y_mark, pv_history):
        y_mark, pv_history = self._validate_inputs(
            spatial_sequence, x_mark_h, y_mark, pv_history)
        dtype = spatial_sequence.dtype
        x_mark_h = self._ordered_auxiliary(x_mark_h).to(
            device=spatial_sequence.device, dtype=dtype)
        pv_history = self._ordered_auxiliary(pv_history).to(
            device=spatial_sequence.device, dtype=dtype)
        y_mark = y_mark[:, -1:].to(device=spatial_sequence.device, dtype=dtype)

        history_queries = self.time_projection(self._time_features(x_mark_h))
        target_query = self.time_projection(self._time_features(y_mark))
        history_context, history_attention = self._solar_attention(
            spatial_sequence, history_queries)

        pairwise_flow, pairwise_confidence = self._local_correlation_flow(
            spatial_sequence)
        velocity, advection_confidence = self._aggregate_velocity(
            pairwise_flow, pairwise_confidence, history_attention)
        future_spatial, displacement, sampling_grid, validity = self._advect(
            spatial_sequence[:, -1], velocity)

        future_context, future_attention = self._solar_attention(
            future_spatial.unsqueeze(1), target_query)
        image_context = self.image_context_projection(torch.cat([
            history_context[:, -1], future_context[:, 0]
        ], dim=-1))
        normalized_velocity = torch.stack([
            velocity[:, 0] / max(spatial_sequence.shape[-1] - 1, 1),
            velocity[:, 1] / max(spatial_sequence.shape[-2] - 1, 1),
            advection_confidence[:, 0],
        ], dim=-1)
        motion_context = self.motion_context_projection(normalized_velocity)
        pv_context = self._pv_context(pv_history)
        fused = self.event_norm(
            image_context + motion_context + pv_context + target_query[:, 0])
        event_feature = self.event_head(fused).unsqueeze(1)

        return {
            "event_feature": event_feature,
            "future_spatial_map": future_spatial,
            "pairwise_flow": pairwise_flow,
            "pairwise_confidence": pairwise_confidence,
            "advection_velocity": velocity,
            "advection_displacement": displacement,
            "advection_confidence": advection_confidence,
            "advection_grid": sampling_grid,
            "advection_validity": validity,
            "history_attention": history_attention,
            "future_attention": future_attention[:, 0],
        }


_STANFORD_SOLAR_CALIBRATION_SHA256 = (
    "fffe056483a3a42594af1b0897789a16207f4add6f7697204775c06205b2a211"
)


def _stanford_solar_calibration_path(configs):
    path = getattr(configs, "solar_calibration_path", "")
    if not path:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "stanford_sun_path_calibration_fold0.json"
        )
    return str(Path(path).resolve())


def _load_stanford_solar_calibration(module, configs):
    """Install the audited train-only sun-path calibration on ``module``.

    Keeping checksum validation and buffer registration here prevents causal
    image encoders from silently drifting to different calibration artifacts.
    Buffers deliberately retain their legacy names so SolarToken checkpoints
    remain load-compatible.
    """
    path = Path(module.calibration_path)
    if not path.is_file():
        raise FileNotFoundError(f"Stanford solar calibration not found: {path}")
    payload_bytes = path.read_bytes()
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if digest != _STANFORD_SOLAR_CALIBRATION_SHA256:
        raise ValueError(
            f"Stanford solar calibration checksum mismatch: {digest}")
    payload = json.loads(payload_bytes)
    causal = payload.get("causal_scope", {})
    validation_images_used = causal.get(
        "uses_validation_images_for_fit_or_detection",
        causal.get("uses_validation_or_test_images", True),
    )
    test_images_used = causal.get(
        "uses_test_images",
        causal.get("uses_validation_or_test_images", True),
    )
    if (
        bool(causal.get("uses_future_images", True))
        or bool(causal.get("uses_pv_targets", True))
        or bool(validation_images_used)
        or bool(test_images_used)
    ):
        raise ValueError("solar calibration is not train-only causal")
    source = payload.get("source", {})
    fold_index = int(getattr(configs, "fold_index", 0))
    if int(source.get("fold_index", -1)) != fold_index:
        raise ValueError(
            f"solar calibration fold {source.get('fold_index')} "
            f"does not match {fold_index}")
    coefficients = torch.tensor(
        payload["regression"]["coefficients"], dtype=torch.float32)
    if coefficients.shape != (2, 28):
        raise ValueError("solar calibration coefficients must have shape [2,28]")
    attention = payload["attention_8x8"]
    module.register_buffer("solar_calibration_coefficients", coefficients)
    module.register_buffer(
        "solar_calibration_month_offsets",
        torch.tensor(
            [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334],
            dtype=torch.float32,
        ),
    )
    module.solar_attention_sigma = float(attention["sigma_normalized_grid"])
    module.solar_attention_truncate_cells = float(
        attention["truncate_radius_feature_cells"])
    module.solar_calibration_name = payload["name"]
    module.solar_calibration_payload_sha256 = payload["payload_sha256"]


def _stanford_solar_location_sequence(module, marks, grid_size=(8, 8)):
    """Return calibrated sun logits/coordinates for every supplied timestamp."""
    if marks.dim() != 3 or marks.shape[-1] < 5:
        raise ValueError("solar calibration requires month/day/hour/minute marks")
    height, width = (int(grid_size[0]), int(grid_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("solar calibration grid dimensions must be positive")

    batch, steps = marks.shape[:2]
    flat_marks = marks.reshape(batch * steps, marks.shape[-1])
    month_index = flat_marks[:, 0].round().long().clamp(1, 12) - 1
    ordinal = (
        module.solar_calibration_month_offsets[month_index].to(flat_marks)
        + flat_marks[:, 1]
    )
    minute = flat_marks[:, 3] * 60.0 + flat_marks[:, 4]
    clock = 2.0 * math.pi * minute / 1440.0
    year = 2.0 * math.pi * (ordinal - 1.0) / 365.25
    base = torch.stack([
        torch.sin(clock),
        torch.cos(clock),
        torch.sin(2.0 * clock),
        torch.cos(2.0 * clock),
        torch.sin(year),
        torch.cos(year),
    ], dim=-1)
    features = [torch.ones_like(base[:, :1]), base]
    features.append(torch.cat([
        base[:, left:left + 1] * base[:, right:right + 1]
        for left in range(base.shape[1])
        for right in range(left, base.shape[1])
    ], dim=-1))
    features = torch.cat(features, dim=-1)
    pixels = (
        features
        @ module.solar_calibration_coefficients.to(flat_marks).transpose(0, 1)
    )
    pixels = pixels.clamp(0.0, 63.0)
    normalized = 2.0 * pixels / 63.0 - 1.0

    axis_y = torch.linspace(
        -1.0, 1.0, height, device=marks.device, dtype=marks.dtype)
    axis_x = torch.linspace(
        -1.0, 1.0, width, device=marks.device, dtype=marks.dtype)
    grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
    distance_squared = (
        (grid_x[None] - normalized[:, 0, None, None]).square()
        + (grid_y[None] - normalized[:, 1, None, None]).square()
    )
    logits = -0.5 * distance_squared / (module.solar_attention_sigma ** 2)
    # The audited calibration defines its truncation radius on an 8x8 grid.
    truncate_radius = module.solar_attention_truncate_cells * (2.0 / 7.0)
    logits = logits.masked_fill(
        distance_squared > truncate_radius ** 2,
        torch.finfo(logits.dtype).min,
    )
    return (
        logits.reshape(batch, steps, height * width),
        normalized.reshape(batch, steps, 2),
    )


class StanfordSolarTokenEncoder(nn.Module):
    """Forecast a target-sun token from causal RGB space-time tokens.

    The advection encoder above reduces motion to one global velocity.  This
    encoder keeps local RGB/chromaticity changes as space-time tokens and lets
    several target-time queries attend to different plausible cloud paths.
    Its solar-location distribution depends only on the known forecast time;
    a future image may supervise that distribution in the training loss, but
    is deliberately absent from this forward interface.
    """

    uses_future_images = False
    time_feature_dim = StanfordSolarAdvectionEncoder.time_feature_dim
    calibration_sha256 = _STANFORD_SOLAR_CALIBRATION_SHA256

    def __init__(self, configs):
        super().__init__()
        self.output_dim = int(getattr(configs, "solar_token_dim", 64))
        self.conv_dim = int(getattr(configs, "solar_token_conv_dim", 32))
        self.hypothesis_count = int(getattr(configs, "solar_token_hypotheses", 4))
        self.attention_heads = int(getattr(configs, "solar_token_heads", 4))
        self.history_order = getattr(configs, "history_order", "current_first")
        self.capacity_kw = float(getattr(configs, "stanford_capacity_kw", 30.1))
        self.calibration_path = _stanford_solar_calibration_path(configs)

        if self.output_dim <= 0 or self.conv_dim <= 0:
            raise ValueError("solar token dimensions must be positive")
        if self.hypothesis_count <= 0:
            raise ValueError("solar_token_hypotheses must be positive")
        if self.output_dim % self.attention_heads:
            raise ValueError("solar_token_dim must be divisible by solar_token_heads")
        if self.history_order not in ["current_first", "past_first"]:
            raise ValueError("history_order must be current_first or past_first")

        # RGB, chromaticity, chronological RGB delta and luminance.
        self.frame_stem = nn.Sequential(
            nn.Conv2d(10, self.conv_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(self._group_count(self.conv_dim), self.conv_dim),
            nn.GELU(),
            nn.Conv2d(
                self.conv_dim, self.output_dim,
                kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(self._group_count(self.output_dim), self.output_dim),
            nn.GELU(),
        )
        self.local_space_time = nn.Sequential(
            nn.Conv3d(
                self.output_dim, self.output_dim, kernel_size=3,
                padding=1, groups=self.output_dim, bias=False),
            nn.Conv3d(self.output_dim, self.output_dim, kernel_size=1, bias=False),
            nn.GroupNorm(self._group_count(self.output_dim), self.output_dim),
            nn.GELU(),
        )
        self.position_projection = nn.Linear(7, self.output_dim, bias=False)
        self.time_projection = nn.Sequential(
            nn.Linear(self.time_feature_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        self.hypothesis_queries = nn.Parameter(
            torch.empty(self.hypothesis_count, self.output_dim))
        nn.init.normal_(self.hypothesis_queries, std=0.02)
        self.trajectory_attention = nn.MultiheadAttention(
            self.output_dim,
            self.attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.solar_coordinate_projection = nn.Linear(2, self.output_dim, bias=False)
        self.solar_roi_encoder = nn.GRU(
            input_size=self.output_dim,
            hidden_size=self.output_dim,
            num_layers=1,
            batch_first=True,
        )
        self.pv_encoder = nn.GRU(
            input_size=2,
            hidden_size=self.output_dim,
            num_layers=1,
            batch_first=True,
        )
        self.pv_summary_projection = nn.Linear(7, self.output_dim)
        self.image_transition_projection = nn.Linear(10, self.output_dim)
        self.transition_projection = nn.Sequential(
            nn.Linear(self.output_dim * 4, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )
        self.event_projection = nn.Sequential(
            nn.Linear(self.output_dim * 5, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.future_token_projection = nn.Sequential(
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.clear_sky_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.GELU(),
            nn.Linear(self.output_dim // 2, 1),
        )
        self._load_solar_calibration(configs)

    def _load_solar_calibration(self, configs):
        _load_stanford_solar_calibration(self, configs)

    @staticmethod
    def _group_count(channels):
        for groups in [8, 4, 2]:
            if channels % groups == 0:
                return groups
        return 1

    def _ordered_rgb(self, images):
        if images.dim() == 4:
            images = images.unsqueeze(2)
        if images.dim() != 5:
            raise ValueError(
                f"solar token images must have shape [B,T,C,H,W], got {tuple(images.shape)}")
        if images.shape[2] == 1:
            images = images.repeat(1, 1, 3, 1, 1)
        if images.shape[2] != 3:
            raise ValueError("solar token encoder requires one or three image channels")
        if self.history_order == "current_first":
            images = torch.flip(images, dims=[1])
        return images

    def _ordered_history(self, values):
        if self.history_order == "current_first":
            return torch.flip(values, dims=[1])
        return values

    @staticmethod
    def _position_features(steps, height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        lag = torch.linspace(-1.0, 0.0, steps, device=device, dtype=dtype)
        grid_t, grid_y, grid_x = torch.meshgrid(lag, y, x, indexing="ij")
        return torch.stack([
            grid_x,
            grid_y,
            grid_t,
            grid_x * grid_t,
            grid_y * grid_t,
            grid_x.square(),
            grid_y.square(),
        ], dim=-1)

    def _calibrated_solar_location(self, marks):
        logits, normalized = _stanford_solar_location_sequence(
            self, marks[:, -1:], grid_size=(8, 8))
        return logits[:, 0], normalized[:, 0]

    def _pv_context(self, pv_history):
        pv = pv_history[..., :1] / self.capacity_kw
        delta = torch.zeros_like(pv)
        delta[:, 1:] = pv[:, 1:] - pv[:, :-1]
        _, hidden = self.pv_encoder(torch.cat([pv, delta], dim=-1))

        current = pv[:, -1, 0]
        lag_indices = [
            max(pv.shape[1] - 2, 0),
            max(pv.shape[1] - 4, 0),
            max(pv.shape[1] - 6, 0),
            0,
        ]
        changes = [current - pv[:, index, 0] for index in lag_indices]
        adjacent = delta[:, 1:, 0]
        summary = torch.stack([
            current,
            *changes,
            adjacent.std(dim=1, unbiased=False),
            adjacent.abs().amax(dim=1),
        ], dim=-1)
        return hidden[-1], self.pv_summary_projection(summary)

    def _image_transition_context(self, images, delta):
        luminance = images.mean(dim=2)
        frame_mean = luminance.mean(dim=(-1, -2))
        current = frame_mean[:, -1]
        lag_indices = [
            max(images.shape[1] - 2, 0),
            max(images.shape[1] - 6, 0),
            0,
        ]
        adjacent_mean = frame_mean[:, 1:] - frame_mean[:, :-1]
        absolute_delta = delta[:, 1:].abs()
        summary = torch.stack([
            current,
            luminance[:, -1].std(dim=(-1, -2), unbiased=False),
            current - frame_mean[:, lag_indices[0]],
            current - frame_mean[:, lag_indices[1]],
            current - frame_mean[:, lag_indices[2]],
            absolute_delta[:, -1].mean(dim=(1, 2, 3)),
            absolute_delta.mean(dim=(1, 2, 3, 4)),
            absolute_delta.std(dim=(1, 2, 3, 4), unbiased=False),
            frame_mean.std(dim=1, unbiased=False),
            adjacent_mean.abs().amax(dim=1),
        ], dim=-1)
        return self.image_transition_projection(summary)

    def forward(self, images, x_mark_h, y_mark, pv_history):
        images = self._ordered_rgb(images)
        if x_mark_h.dim() != 3 or x_mark_h.shape[:2] != images.shape[:2]:
            raise ValueError("x_mark_h must align with the historical image sequence")
        if y_mark.dim() == 2:
            y_mark = y_mark.unsqueeze(1)
        if y_mark.dim() != 3 or y_mark.shape[0] != images.shape[0]:
            raise ValueError("y_mark must have shape [B,Ty,M]")
        if pv_history.dim() == 2:
            pv_history = pv_history.unsqueeze(-1)
        if pv_history.dim() != 3 or pv_history.shape[:2] != images.shape[:2]:
            raise ValueError("pv_history must align with the historical image sequence")

        x_mark_h = self._ordered_history(x_mark_h).to(images)
        pv_history = self._ordered_history(pv_history).to(images)
        y_mark = y_mark[:, -1:].to(images)

        chromaticity = images / images.sum(dim=2, keepdim=True).clamp_min(0.03)
        delta = torch.zeros_like(images)
        delta[:, 1:] = images[:, 1:] - images[:, :-1]
        image_transition = self._image_transition_context(images, delta)
        luminance = images.mean(dim=2, keepdim=True)
        frame_input = torch.cat([images, chromaticity, delta, luminance], dim=2)

        batch, steps, channels, image_h, image_w = frame_input.shape
        encoded = self.frame_stem(
            frame_input.reshape(batch * steps, channels, image_h, image_w))
        height, width = encoded.shape[-2:]
        encoded = encoded.reshape(
            batch, steps, self.output_dim, height, width)
        local = self.local_space_time(encoded.permute(0, 2, 1, 3, 4))
        encoded = encoded + local.permute(0, 2, 1, 3, 4)

        position = self.position_projection(self._position_features(
            steps, height, width, encoded.device, encoded.dtype))
        history_time = self.time_projection(
            StanfordSolarAdvectionEncoder._time_features(x_mark_h))
        tokens = encoded.permute(0, 1, 3, 4, 2)
        tokens = tokens + position.unsqueeze(0)
        tokens = tokens + history_time[:, :, None, None, :]
        tokens = tokens.reshape(batch, steps * height * width, self.output_dim)

        target_time = self.time_projection(
            StanfordSolarAdvectionEncoder._time_features(y_mark))[:, 0]
        solar_location_logits, solar_coordinates = self._calibrated_solar_location(
            y_mark)
        target_time = target_time + self.solar_coordinate_projection(solar_coordinates)
        queries = target_time[:, None] + self.hypothesis_queries[None]
        hypotheses, trajectory_attention = self.trajectory_attention(
            queries, tokens, tokens, need_weights=True)

        solar_location = torch.softmax(solar_location_logits, dim=-1)
        roi_maps = F.adaptive_avg_pool2d(
            encoded.reshape(batch * steps, self.output_dim, height, width),
            (8, 8),
        ).reshape(batch, steps, self.output_dim, 8 * 8)
        roi_sequence = (
            roi_maps * solar_location[:, None, None, :]
        ).sum(dim=-1)
        _, roi_hidden = self.solar_roi_encoder(roi_sequence)

        pv_hidden, pv_summary = self._pv_context(pv_history)
        hypothesis_mean = hypotheses.mean(dim=1)
        transition_feature = self.transition_projection(torch.cat([
            pv_hidden,
            pv_summary,
            image_transition,
            target_time,
        ], dim=-1))
        event_feature = self.event_projection(torch.cat([
            hypothesis_mean,
            roi_hidden[-1],
            pv_hidden,
            pv_summary,
            target_time,
        ], dim=-1))
        future_token = self.future_token_projection(hypothesis_mean)
        clear_sky_kw = self.capacity_kw * torch.sigmoid(
            self.clear_sky_head(target_time))

        return {
            "event_feature": event_feature.unsqueeze(1),
            "transition_feature": transition_feature.unsqueeze(1),
            "future_solar_token": future_token,
            "solar_location_logits": solar_location_logits,
            "solar_location": solar_location,
            "solar_coordinates": solar_coordinates,
            "clear_sky_kw": clear_sky_kw.unsqueeze(1),
            "trajectory_hypotheses": hypotheses,
            "trajectory_attention": trajectory_attention,
            "roi_sequence": roi_sequence,
        }


def hierarchical_phase_probabilities(factor_logits):
    """Factor endpoint phase into event, active-at-lead and direction states."""
    if factor_logits.shape[-1] != 3:
        raise ValueError('phase factor logits must end with three states')
    event, active, up = torch.sigmoid(factor_logits).unbind(dim=-1)
    down = 1.0 - up
    inside = 1.0 - active
    probabilities = torch.stack([
        1.0 - event,
        event * active * down,
        event * active * up,
        event * inside * down,
        event * inside * up,
    ], dim=-1)
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class StanfordCausalPhaseFieldEncoder(nn.Module):
    """Causal local cloud-transport features at fixed Stanford forecast leads.

    Unlike a global optical-flow summary, this encoder retains a categorical
    displacement distribution at every feature-map pixel.  It uses each
    historical timestamp's calibrated sun position for the historical stream,
    then transports the latest local feature/phase fields to the calibrated
    target-sun positions at 1, 3, 5, 10 and 15 minutes.

    The forward boundary intentionally contains only historical images, their
    known timestamps and PV values, plus known forecast timestamps.  Future
    images and future PV observations cannot enter this module.
    """

    uses_future_images = False
    forecast_leads_minutes = (1, 3, 5, 10, 15)
    time_feature_dim = StanfordSolarAdvectionEncoder.time_feature_dim
    calibration_sha256 = _STANFORD_SOLAR_CALIBRATION_SHA256

    def __init__(self, configs):
        super().__init__()
        self.output_dim = int(getattr(configs, "phase_field_dim", 64))
        self.stem_dim = int(getattr(
            configs, "phase_field_stem_dim", max(8, self.output_dim // 2)))
        self.motion_dim = int(getattr(
            configs, "phase_field_motion_dim", max(4, self.output_dim // 2)))
        self.correlation_radius = int(getattr(
            configs, "phase_field_correlation_radius", 2))
        self.correlation_temperature = float(getattr(
            configs, "phase_field_correlation_temperature", 0.07))
        self.temporal_decay = float(getattr(
            configs, "phase_field_temporal_decay", 3.0))
        self.transport_mode = str(getattr(
            configs, "phase_transport_mode", "mean_flow"))
        self.transport_hypotheses = int(getattr(
            configs, "phase_transport_hypotheses", 3))
        self.sample_interval_minutes = float(getattr(
            configs, "sample_interval_minutes", 1.0))
        self.capacity_kw = float(getattr(configs, "stanford_capacity_kw", 30.1))
        self.history_order = getattr(configs, "history_order", "current_first")
        self.calibration_path = _stanford_solar_calibration_path(configs)

        if self.output_dim <= 0 or self.stem_dim <= 0 or self.motion_dim <= 0:
            raise ValueError("phase-field feature dimensions must be positive")
        if self.correlation_radius < 1:
            raise ValueError("phase_field_correlation_radius must be at least one")
        if self.correlation_temperature <= 0:
            raise ValueError("phase_field_correlation_temperature must be positive")
        if self.temporal_decay <= 0:
            raise ValueError("phase_field_temporal_decay must be positive")
        if self.transport_mode not in [
            "mean_flow", "multi_hypothesis", "hypothesis_tokens",
            "set_attention"
        ]:
            raise ValueError(
                f"unsupported phase transport mode: {self.transport_mode}")
        if self.transport_hypotheses <= 0:
            raise ValueError("phase_transport_hypotheses must be positive")
        if self.sample_interval_minutes <= 0:
            raise ValueError("sample_interval_minutes must be positive")
        if self.capacity_kw <= 0:
            raise ValueError("stanford_capacity_kw must be positive")
        if self.history_order not in ["current_first", "past_first"]:
            raise ValueError("history_order must be current_first or past_first")

        # Raw RGB preserves brightness, chromaticity separates cloud colour,
        # and per-frame exposure-normalized RGB reduces camera auto-exposure
        # shift. Motion remains explicit in the cost volume.
        self.frame_stem = nn.Sequential(
            nn.Conv2d(10, self.stem_dim, kernel_size=5, stride=2, padding=2,
                      bias=False),
            nn.GroupNorm(
                StanfordSolarTokenEncoder._group_count(self.stem_dim),
                self.stem_dim),
            nn.GELU(),
            nn.Conv2d(
                self.stem_dim, self.output_dim, kernel_size=3, stride=2,
                padding=1, bias=False),
            nn.GroupNorm(
                StanfordSolarTokenEncoder._group_count(self.output_dim),
                self.output_dim),
            nn.GELU(),
        )
        self.motion_projection = nn.Conv2d(
            self.output_dim, self.motion_dim, kernel_size=1, bias=False)

        kernel = 2 * self.correlation_radius + 1
        offsets = [
            (dx, dy)
            for dy in range(-self.correlation_radius,
                            self.correlation_radius + 1)
            for dx in range(-self.correlation_radius,
                            self.correlation_radius + 1)
        ]
        self._correlation_offset_pairs = tuple(offsets)
        self.register_buffer(
            "_correlation_offsets",
            torch.tensor(offsets, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_lead_minutes",
            torch.tensor(self.forecast_leads_minutes, dtype=torch.float32),
            persistent=False,
        )
        self.phase_map_projection = nn.Sequential(
            nn.Conv2d(kernel * kernel + 3, self.output_dim, kernel_size=1,
                      bias=False),
            nn.GroupNorm(
                StanfordSolarTokenEncoder._group_count(self.output_dim),
                self.output_dim),
            nn.GELU(),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(self.time_feature_dim, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        self.solar_coordinate_projection = nn.Linear(
            2, self.output_dim, bias=False)
        self.lead_projection = nn.Sequential(
            nn.Linear(1, self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.historical_sun_encoder = nn.GRU(
            input_size=self.output_dim,
            hidden_size=self.output_dim,
            num_layers=1,
            batch_first=True,
        )
        self.pv_encoder = nn.GRU(
            input_size=2,
            hidden_size=self.output_dim,
            num_layers=1,
            batch_first=True,
        )
        self.pv_statistics_projection = nn.Sequential(
            nn.Linear(16, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )
        self.path_geometry_projection = nn.Linear(7, self.output_dim)
        if self.transport_mode == "set_attention":
            # Keep every motion mode's source, target and relative arrival
            # geometry.  A source coordinate alone is ambiguous because the
            # calibrated sun moves across leads and the same cloud track can
            # approach one target while departing another.
            self.hypothesis_token_projection = nn.Sequential(
                nn.Linear(self.output_dim * 5 + 2, self.output_dim),
                nn.GELU(),
                nn.LayerNorm(self.output_dim),
            )
            attention_heads = 4 if self.output_dim % 4 == 0 else 1
            self.hypothesis_set_attention = nn.MultiheadAttention(
                self.output_dim, attention_heads, batch_first=True)
            self.hypothesis_set_queries = nn.Parameter(
                torch.zeros(1, 2, self.output_dim))
            nn.init.normal_(
                self.hypothesis_set_queries, mean=0.0,
                std=self.output_dim ** -0.5)
        else:
            self.hypothesis_token_projection = None
            self.hypothesis_set_attention = None
            self.hypothesis_set_queries = None
        if self.transport_mode == "hypothesis_tokens":
            phase_context_fields = self.transport_hypotheses + 4
            path_context_fields = 2 * self.transport_hypotheses + 5
        elif self.transport_mode == "set_attention":
            phase_context_fields = 5
            path_context_fields = 7
        elif self.transport_mode == "multi_hypothesis":
            phase_context_fields = 6
            path_context_fields = 9
        else:
            phase_context_fields = 5
            path_context_fields = 7
        self.phase_feature_head = nn.Sequential(
            nn.Linear(self.output_dim * phase_context_fields, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )
        self.path_feature_head = nn.Sequential(
            nn.Linear(self.output_dim * path_context_fields, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )
        self.hazard_feature_head = nn.Sequential(
            nn.Linear(self.output_dim * 5, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )
        self.hazard_head = nn.Linear(self.output_dim, 1)
        # The five endpoint phases are not unrelated classes.  Factoring them
        # into event occurrence, active-vs-return and direction makes the
        # scarce active/return labels share the correct statistical structure.
        self.phase_factor_head = nn.Linear(self.output_dim, 3)
        # A single path regressor averages active and returned paths toward the
        # event boundary.  Keep one signed hypothesis per phase instead.
        self.class_path_delta_heads = nn.ModuleList([
            nn.Linear(self.output_dim, 1) for _ in range(5)
        ])

        _load_stanford_solar_calibration(self, configs)

    def _ordered_rgb(self, images):
        if images.dim() == 4:
            images = images.unsqueeze(2)
        if images.dim() != 5:
            raise ValueError(
                "phase-field images must have shape [B,T,C,H,W], got "
                f"{tuple(images.shape)}")
        if images.shape[2] == 1:
            images = images.repeat(1, 1, 3, 1, 1)
        if images.shape[2] != 3:
            raise ValueError("phase-field encoder requires one or three image channels")
        if self.history_order == "current_first":
            images = torch.flip(images, dims=[1])
        return images

    def _ordered_history(self, values):
        if self.history_order == "current_first":
            return torch.flip(values, dims=[1])
        return values

    def _validate_inputs(self, images, x_mark_h, lead_marks, pv_history):
        if images.shape[1] < 2:
            raise ValueError("phase-field encoder requires at least two history frames")
        batch, steps = images.shape[:2]
        if x_mark_h.dim() != 3 or x_mark_h.shape[:2] != (batch, steps):
            raise ValueError("x_mark_h must align with historical images as [B,T,M]")
        if lead_marks.dim() == 2:
            lead_marks = lead_marks.unsqueeze(1)
        if (
            lead_marks.dim() != 3
            or lead_marks.shape[0] != batch
            or lead_marks.shape[1] != len(self.forecast_leads_minutes)
        ):
            raise ValueError(
                "lead_marks must have shape [B,5,M] in fixed "
                "[1,3,5,10,15]-minute order")
        if pv_history.dim() == 2:
            pv_history = pv_history.unsqueeze(-1)
        if pv_history.dim() != 3 or pv_history.shape[:2] != (batch, steps):
            raise ValueError("pv_history must align with historical images as [B,T,Cp]")
        return lead_marks, pv_history

    def _local_cost_volume(self, projected_features):
        """Dense endpoint-aligned correspondence without spatial reduction."""
        if projected_features.dim() != 5:
            raise ValueError("projected_features must have shape [B,T,C,H,W]")
        batch, steps, channels, height, width = projected_features.shape
        if steps < 2:
            raise ValueError("local correspondence requires at least two frames")

        features = F.normalize(projected_features, dim=2, eps=1e-6)
        previous = features[:, :-1].reshape(
            batch * (steps - 1), channels, height, width)
        current = features[:, 1:].reshape(
            batch * (steps - 1), channels, height, width)
        kernel = 2 * self.correlation_radius + 1
        candidates = kernel * kernel
        # Building the full unfolded [B*(T-1),C,K^2,H,W] tensor is several
        # GiB at the production batch size.  Compute one displacement at a
        # time and retain only the channel-reduced cost volume.
        radius = self.correlation_radius
        padded_previous = F.pad(previous, (radius, radius, radius, radius))
        padded_valid = F.pad(
            torch.ones(
                previous.shape[0], 1, height, width,
                device=previous.device, dtype=previous.dtype),
            (radius, radius, radius, radius),
        )
        cost_planes = []
        valid_planes = []
        for dx, dy in self._correlation_offset_pairs:
            start_y = radius + int(dy)
            start_x = radius + int(dx)
            shifted = padded_previous[
                :, :, start_y:start_y + height, start_x:start_x + width]
            cost_planes.append((current * shifted).sum(dim=1))
            valid_planes.append(padded_valid[
                :, 0, start_y:start_y + height, start_x:start_x + width])
        cost = torch.stack(cost_planes, dim=1)
        valid = torch.stack(valid_planes, dim=1).gt(0.5)
        masked_cost = cost.masked_fill(
            ~valid, torch.finfo(cost.dtype).min)
        probabilities = torch.softmax(
            masked_cost / self.correlation_temperature, dim=1)

        # A candidate offset points from the endpoint into the previous frame;
        # negating it yields forward motion at the endpoint pixel.
        offsets = self._correlation_offsets.to(probabilities)
        forward_offsets = -offsets.transpose(0, 1).reshape(
            1, 2, candidates, 1, 1)
        flow = (probabilities.unsqueeze(1) * forward_offsets).sum(dim=2)
        # Compute entropy in FP32.  In FP16, 1e-8 rounds to zero and the old
        # expression could evaluate exact-zero probabilities as 0 * -inf.
        probability_fp32 = probabilities.float()
        entropy = -(
            probability_fp32
            * probability_fp32.clamp_min(1e-12).log()
        ).sum(dim=1, keepdim=True).to(probabilities.dtype)
        valid_count = valid.sum(dim=1, keepdim=True).to(entropy.dtype)
        maximum_entropy = valid_count.clamp_min(2.0).log()
        confidence = (1.0 - entropy / maximum_entropy).clamp(0.0, 1.0)
        confidence = torch.where(
            valid_count > 1.0, confidence, torch.ones_like(confidence))

        pair_shape = (batch, steps - 1)
        return (
            masked_cost.reshape(*pair_shape, candidates, height, width),
            probabilities.reshape(*pair_shape, candidates, height, width),
            flow.reshape(*pair_shape, 2, height, width),
            confidence.reshape(*pair_shape, 1, height, width),
        )

    def _pv_statistics(self, pv):
        """Explicit causal level, volatility and change-point features."""
        current = pv[:, -1, 0]
        changes = []
        for lag in (1, 3, 5, 10, 15):
            index = max(pv.shape[1] - 1 - lag, 0)
            changes.append(current - pv[:, index, 0])
        changes = torch.stack(changes, dim=-1)
        recent = pv[:, -min(6, pv.shape[1]):, 0]
        full = pv[..., 0]
        recent_jumps = recent[:, 1:] - recent[:, :-1]
        full_jumps = full[:, 1:] - full[:, :-1]
        statistics = torch.cat([
            current.unsqueeze(-1),
            (1.0 - current).unsqueeze(-1),
            changes,
            changes.abs(),
            recent.std(dim=1, unbiased=False).unsqueeze(-1),
            recent_jumps.abs().amax(dim=1).unsqueeze(-1),
            full.std(dim=1, unbiased=False).unsqueeze(-1),
            full_jumps.abs().amax(dim=1).unsqueeze(-1),
        ], dim=-1)
        if statistics.shape[-1] != 16:
            raise RuntimeError('Stanford PV statistics must contain 16 features')
        return statistics

    def _recent_local_field(self, probabilities, flow, confidence):
        """Temporally smooth each pixel while preserving its spatial motion."""
        pairs = flow.shape[1]
        age = torch.arange(
            pairs - 1, -1, -1, device=flow.device, dtype=flow.dtype)
        recency = torch.exp(-age / self.temporal_decay).view(1, pairs, 1, 1, 1)
        weights = recency * confidence.clamp_min(0.05)
        denominator = weights.sum(dim=1).clamp_min(1e-6)
        local_flow = (flow * weights).sum(dim=1) / denominator
        local_probabilities = (
            probabilities * weights
        ).sum(dim=1) / denominator
        recency_denominator = recency.sum(dim=1).clamp_min(1e-6)
        local_confidence = (
            confidence * recency
        ).sum(dim=1) / recency_denominator
        return local_probabilities, local_flow, local_confidence

    def _motion_photometric_consistency(
            self, photometric_sequence, pairwise_flow, pairwise_confidence,
            output_size):
        """Reconstruct each historical frame from its causal predecessor."""
        batch, steps, channels = photometric_sequence.shape[:3]
        height, width = output_size
        photometric = F.interpolate(
            photometric_sequence.reshape(
                batch * steps, channels,
                photometric_sequence.shape[-2], photometric_sequence.shape[-1]),
            size=(height, width), mode="bilinear", align_corners=True,
        ).reshape(batch, steps, channels, height, width)
        previous = photometric[:, :-1].reshape(
            batch * (steps - 1), channels, height, width)
        current = photometric[:, 1:].reshape_as(previous)
        flow = pairwise_flow.reshape(
            batch * (steps - 1), 2, height, width)
        confidence = pairwise_confidence.reshape(
            batch * (steps - 1), 1, height, width)
        base_grid = self._coordinate_grid(
            height, width, flow.device, flow.dtype)
        base_grid = base_grid.unsqueeze(0).expand(flow.shape[0], -1, -1, -1)
        scale = flow.new_tensor([
            2.0 / max(width - 1, 1),
            2.0 / max(height - 1, 1),
        ])
        sampling_grid = base_grid - flow.permute(0, 2, 3, 1) * scale
        reconstructed = F.grid_sample(
            previous, sampling_grid, mode="bilinear",
            padding_mode="border", align_corners=True)
        valid = sampling_grid.abs().le(1.0).all(dim=-1)
        # Detaching confidence prevents the encoder from reducing the loss by
        # declaring every difficult correspondence uncertain.
        weight = valid.to(flow.dtype) * (
            0.25 + confidence.detach().squeeze(1))
        robust_error = torch.sqrt(
            (reconstructed - current).square() + 1e-4).mean(dim=1)
        return (robust_error * weight).sum() / weight.sum().clamp_min(1.0)

    def _local_motion_hypotheses(self, probabilities, confidence):
        """Preserve dominant local correspondence modes before extrapolation."""
        count = min(self.transport_hypotheses, probabilities.shape[1])
        mode_probability, mode_index = probabilities.topk(count, dim=1)
        forward_offsets = -self._correlation_offsets.to(probabilities)
        velocity = forward_offsets[mode_index]
        velocity = velocity.permute(0, 1, 4, 2, 3).contiguous()
        represented_mass = mode_probability.sum(dim=1, keepdim=True)
        mode_weight = mode_probability / represented_mass.clamp_min(1e-6)
        mode_confidence = (
            confidence.unsqueeze(1) * mode_weight.unsqueeze(2)
        )
        return velocity, mode_confidence, represented_mass

    @staticmethod
    def _coordinate_grid(height, width, device, dtype):
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1)

    def _transport_local_fields(
            self, latest_features, phase_map, local_flow, local_confidence):
        """Semi-Lagrangian transport driven by the full local flow field."""
        batch, _, height, width = latest_features.shape
        base_grid = self._coordinate_grid(
            height, width, latest_features.device, latest_features.dtype)
        base_grid = base_grid.unsqueeze(0).expand(batch, -1, -1, -1)
        source_grid = base_grid.permute(0, 3, 1, 2)
        scale = latest_features.new_tensor([
            2.0 / max(width - 1, 1),
            2.0 / max(height - 1, 1),
        ]).view(1, 2, 1, 1)
        scale = scale / self.sample_interval_minutes

        feature_field = latest_features
        phase_field = phase_map
        confidence_field = local_confidence
        velocity_field = local_flow
        requested = set(self.forecast_leads_minutes)
        feature_leads = []
        phase_leads = []
        confidence_leads = []
        source_grid_leads = []

        for minute in range(1, self.forecast_leads_minutes[-1] + 1):
            sampling_grid = (
                base_grid
                - (velocity_field * scale).permute(0, 2, 3, 1)
            )
            feature_field = F.grid_sample(
                feature_field, sampling_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True)
            phase_field = F.grid_sample(
                phase_field, sampling_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True)
            confidence_field = F.grid_sample(
                confidence_field, sampling_grid, mode="bilinear",
                padding_mode="zeros", align_corners=True)
            source_grid = F.grid_sample(
                source_grid, sampling_grid, mode="bilinear",
                padding_mode="border", align_corners=True)
            velocity_field = F.grid_sample(
                velocity_field, sampling_grid, mode="bilinear",
                padding_mode="border", align_corners=True)

            if minute in requested:
                feature_leads.append(feature_field)
                phase_leads.append(phase_field)
                confidence_leads.append(confidence_field)
                source_grid_leads.append(source_grid.permute(0, 2, 3, 1))

        return (
            torch.stack(feature_leads, dim=1),
            torch.stack(phase_leads, dim=1),
            torch.stack(confidence_leads, dim=1),
            torch.stack(source_grid_leads, dim=1),
        )

    @staticmethod
    def _attention_pool(fields, attention):
        batch, steps, channels, height, width = fields.shape
        flattened = fields.reshape(batch, steps, channels, height * width)
        return (flattened * attention.unsqueeze(2)).sum(dim=-1)

    def forward(self, images, x_mark_h, lead_marks, pv_history):
        images = self._ordered_rgb(images)
        lead_marks, pv_history = self._validate_inputs(
            images, x_mark_h, lead_marks, pv_history)
        x_mark_h = self._ordered_history(x_mark_h).to(images)
        pv_history = self._ordered_history(pv_history).to(images)
        lead_marks = lead_marks.to(images)

        chromaticity = images / images.sum(dim=2, keepdim=True).clamp_min(0.03)
        luminance = images.mean(dim=2, keepdim=True)
        exposure = luminance.mean(dim=(-2, -1), keepdim=True).clamp_min(0.03)
        exposure_normalized = (images / exposure).clamp(0.0, 4.0)
        normalized_luminance = (luminance / exposure).clamp(0.0, 4.0)
        photometric_sequence = torch.cat([
            chromaticity, normalized_luminance], dim=2)
        frame_input = torch.cat([
            images, chromaticity, exposure_normalized, luminance], dim=2)
        batch, steps, channels, image_h, image_w = frame_input.shape
        encoded = self.frame_stem(
            frame_input.reshape(batch * steps, channels, image_h, image_w))
        height, width = encoded.shape[-2:]
        encoded = encoded.reshape(
            batch, steps, self.output_dim, height, width)

        motion = self.motion_projection(encoded.reshape(
            batch * steps, self.output_dim, height, width))
        motion = motion.reshape(
            batch, steps, self.motion_dim, height, width)
        cost_volume, correspondence, pairwise_flow, pairwise_confidence = (
            self._local_cost_volume(motion)
        )
        motion_consistency_loss = self._motion_photometric_consistency(
            photometric_sequence, pairwise_flow, pairwise_confidence,
            (height, width))
        local_probability, local_flow, local_confidence = (
            self._recent_local_field(
                correspondence, pairwise_flow, pairwise_confidence)
        )
        phase_input = torch.cat([
            local_probability,
            local_flow,
            local_confidence,
        ], dim=1)
        phase_map = self.phase_map_projection(phase_input)

        historical_logits, historical_coordinates = (
            _stanford_solar_location_sequence(
                self, x_mark_h, grid_size=(height, width))
        )
        historical_attention = torch.softmax(historical_logits, dim=-1)
        historical_sun_stream = self._attention_pool(
            encoded, historical_attention)
        historical_time = self.time_projection(
            StanfordSolarAdvectionEncoder._time_features(x_mark_h))
        historical_sun_input = (
            historical_sun_stream
            + historical_time
            + self.solar_coordinate_projection(historical_coordinates)
        )
        _, historical_hidden = self.historical_sun_encoder(
            historical_sun_input)

        pv = pv_history[..., :1] / self.capacity_kw
        pv_delta = torch.zeros_like(pv)
        pv_delta[:, 1:] = pv[:, 1:] - pv[:, :-1]
        _, pv_hidden = self.pv_encoder(torch.cat([pv, pv_delta], dim=-1))
        pv_statistics = self._pv_statistics(pv)
        pv_summary = pv_hidden[-1] + self.pv_statistics_projection(pv_statistics)

        target_logits, target_coordinates = _stanford_solar_location_sequence(
            self, lead_marks, grid_size=(height, width))
        target_attention = torch.softmax(target_logits, dim=-1)
        hypothesis_weights = None
        hypothesis_source_coordinates = None
        if self.transport_mode in [
            "multi_hypothesis", "hypothesis_tokens", "set_attention"
        ]:
            velocities, mode_confidences, represented_mass = (
                self._local_motion_hypotheses(
                    local_probability, local_confidence)
            )
            transported_hypotheses = []
            phase_hypotheses = []
            confidence_hypotheses = []
            grid_hypotheses = []
            for hypothesis in range(velocities.shape[1]):
                values = self._transport_local_fields(
                    encoded[:, -1], phase_map,
                    velocities[:, hypothesis],
                    mode_confidences[:, hypothesis])
                transported_hypotheses.append(values[0])
                phase_hypotheses.append(values[1])
                confidence_hypotheses.append(values[2])
                grid_hypotheses.append(values[3])
            transported_stack = torch.stack(transported_hypotheses, dim=1)
            phase_stack = torch.stack(phase_hypotheses, dim=1)
            confidence_stack = torch.stack(confidence_hypotheses, dim=1)
            grid_stack = torch.stack(grid_hypotheses, dim=1)

            field_streams = torch.stack([
                self._attention_pool(value, target_attention)
                for value in transported_hypotheses
            ], dim=1)
            phase_streams = torch.stack([
                self._attention_pool(value, target_attention)
                for value in phase_hypotheses
            ], dim=1)
            confidence_streams = torch.stack([
                self._attention_pool(value, target_attention)
                for value in confidence_hypotheses
            ], dim=1)
            source_streams = []
            for grid in grid_hypotheses:
                flat = grid.reshape(
                    batch, len(self.forecast_leads_minutes),
                    height * width, 2)
                source_streams.append(
                    (flat * target_attention.unsqueeze(-1)).sum(dim=2))
            hypothesis_source_coordinates = torch.stack(
                source_streams, dim=1)

            raw_weight = confidence_streams[..., 0].clamp_min(0.0)
            hypothesis_weights = raw_weight / raw_weight.sum(
                dim=1, keepdim=True).clamp_min(1e-6)
            mixture_weight = hypothesis_weights.unsqueeze(-1)
            target_sun_field_stream = (
                mixture_weight * field_streams).sum(dim=1)
            target_phase_stream = (
                mixture_weight * phase_streams).sum(dim=1)
            target_sun_field_std = torch.sqrt((
                mixture_weight
                * (field_streams - target_sun_field_stream.unsqueeze(1)).square()
            ).sum(dim=1).clamp_min(1e-8))
            target_phase_std = torch.sqrt((
                mixture_weight
                * (phase_streams - target_phase_stream.unsqueeze(1)).square()
            ).sum(dim=1).clamp_min(1e-8))
            source_coordinates = (
                mixture_weight * hypothesis_source_coordinates).sum(dim=1)
            source_dispersion = torch.sqrt((
                hypothesis_weights
                * (hypothesis_source_coordinates
                   - source_coordinates.unsqueeze(1)).square().sum(dim=-1)
            ).sum(dim=1, keepdim=False).clamp_min(1e-8)).unsqueeze(-1)
            mean_confidence = (
                mixture_weight * confidence_streams).sum(dim=1)
            target_confidence = mean_confidence * (
                1.0 - source_dispersion / math.sqrt(8.0)
            ).clamp(0.0, 1.0)
            grid_weight = (
                hypothesis_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            path_grids = (grid_weight * grid_stack).sum(dim=1)
            field_weight = hypothesis_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            transported = (field_weight * transported_stack).sum(dim=1)
            transported_phase = (field_weight * phase_stack).sum(dim=1)
            transported_confidence = (
                field_weight * confidence_stack).sum(dim=1)
        else:
            transported, transported_phase, transported_confidence, path_grids = (
                self._transport_local_fields(
                    encoded[:, -1], phase_map, local_flow, local_confidence)
            )
            target_sun_field_stream = self._attention_pool(
                transported, target_attention)
            target_phase_stream = self._attention_pool(
                transported_phase, target_attention)
            target_confidence = self._attention_pool(
                transported_confidence, target_attention)
            path_flat = path_grids.reshape(
                batch, len(self.forecast_leads_minutes), height * width, 2)
            source_coordinates = (
                path_flat * target_attention.unsqueeze(-1)
            ).sum(dim=2)
            target_sun_field_std = torch.zeros_like(target_sun_field_stream)
            target_phase_std = torch.zeros_like(target_phase_stream)
            hypothesis_weights = torch.ones(
                batch, 1, len(self.forecast_leads_minutes),
                device=images.device, dtype=images.dtype)
            hypothesis_source_coordinates = source_coordinates.unsqueeze(1)
            source_dispersion = torch.zeros_like(target_confidence)
            represented_mass = torch.ones(
                batch, 1, height, width,
                device=images.device, dtype=images.dtype)
        path_displacement = target_coordinates - source_coordinates
        path_geometry = torch.cat([
            target_coordinates,
            source_coordinates,
            path_displacement,
            target_confidence,
        ], dim=-1)
        path_geometry_feature = self.path_geometry_projection(path_geometry)

        target_time = self.time_projection(
            StanfordSolarAdvectionEncoder._time_features(lead_marks))
        lead_feature = self.lead_projection(
            (self._lead_minutes.to(images) / self._lead_minutes[-1])
            .view(1, -1, 1)
            .expand(batch, -1, -1)
        )
        history_context = historical_hidden[-1].unsqueeze(1).expand(
            -1, len(self.forecast_leads_minutes), -1)
        pv_context = pv_summary.unsqueeze(1).expand_as(history_context)
        if self.transport_mode == "set_attention":
            source_coordinate_tokens = self.solar_coordinate_projection(
                hypothesis_source_coordinates)
            target_coordinate_tokens = self.solar_coordinate_projection(
                target_coordinates).unsqueeze(1).expand_as(
                    source_coordinate_tokens)
            hypothesis_displacements = (
                target_coordinates.unsqueeze(1)
                - hypothesis_source_coordinates)
            displacement_tokens = self.solar_coordinate_projection(
                hypothesis_displacements)
            set_inputs = torch.cat([
                field_streams,
                phase_streams,
                source_coordinate_tokens,
                target_coordinate_tokens,
                displacement_tokens,
                hypothesis_weights.unsqueeze(-1),
                confidence_streams[..., :1],
            ], dim=-1)
            set_tokens = self.hypothesis_token_projection(set_inputs)
            set_tokens = set_tokens.permute(0, 2, 1, 3).reshape(
                batch * len(self.forecast_leads_minutes),
                hypothesis_weights.shape[1], self.output_dim)
            set_queries = self.hypothesis_set_queries.expand(
                set_tokens.shape[0], -1, -1)
            set_summary, _ = self.hypothesis_set_attention(
                set_queries, set_tokens, set_tokens,
                need_weights=False)
            set_summary = set_summary.reshape(
                batch, len(self.forecast_leads_minutes), 2, self.output_dim)
            phase_set_summary = set_summary[:, :, 0]
            path_set_summary = set_summary[:, :, 1]
            phase_inputs = [phase_set_summary]
            path_inputs = [path_set_summary, phase_set_summary]
        elif self.transport_mode == "hypothesis_tokens":
            coordinate_tokens = self.solar_coordinate_projection(
                hypothesis_source_coordinates)
            probability_scale = torch.sqrt(
                hypothesis_weights * hypothesis_weights.shape[1]
            ).unsqueeze(-1)
            phase_mode_tokens = (
                phase_streams + coordinate_tokens) * probability_scale
            field_mode_tokens = (
                field_streams + coordinate_tokens) * probability_scale
            phase_inputs = [
                phase_mode_tokens.permute(0, 2, 1, 3).reshape(
                    batch, len(self.forecast_leads_minutes), -1)
            ]
            path_inputs = [
                field_mode_tokens.permute(0, 2, 1, 3).reshape(
                    batch, len(self.forecast_leads_minutes), -1),
                phase_mode_tokens.permute(0, 2, 1, 3).reshape(
                    batch, len(self.forecast_leads_minutes), -1),
            ]
        else:
            phase_inputs = [target_phase_stream]
            path_inputs = [target_sun_field_stream, target_phase_stream]
        if self.transport_mode == "multi_hypothesis":
            phase_inputs.append(target_phase_std)
            path_inputs.extend([target_sun_field_std, target_phase_std])
        phase_features = self.phase_feature_head(torch.cat(phase_inputs + [
            path_geometry_feature,
            target_time,
            history_context,
            pv_context,
        ], dim=-1))
        path_features = self.path_feature_head(torch.cat(path_inputs + [
            path_geometry_feature,
            target_time,
            lead_feature,
            history_context,
            pv_context,
        ], dim=-1))
        hazard_features = self.hazard_feature_head(torch.cat([
            path_features,
            phase_features,
            history_context,
            pv_context,
            target_time,
        ], dim=-1))

        phase_factor_logits = self.phase_factor_head(phase_features)
        phase_probability = hierarchical_phase_probabilities(
            phase_factor_logits)
        phase_logits = phase_probability.clamp_min(1e-7).log()
        class_path_delta_kw = self.capacity_kw * torch.stack([
            head(path_features) for head in self.class_path_delta_heads
        ], dim=2)
        predicted_phase = phase_probability.argmax(
            dim=-1, keepdim=True).unsqueeze(-1)
        selected_path_delta_kw = class_path_delta_kw.gather(
            dim=2, index=predicted_phase).squeeze(2)

        return {
            "hazard_features": hazard_features,
            "phase_features": phase_features,
            "path_features": path_features,
            "hazard_logits": self.hazard_head(hazard_features).squeeze(-1),
            "phase_factor_logits": phase_factor_logits,
            "phase_logits": phase_logits,
            "phase_probability": phase_probability,
            "class_path_delta_kw": class_path_delta_kw,
            "path_delta_kw": selected_path_delta_kw,
            "event_feature": hazard_features[:, -1:],
            "lead_minutes": self._lead_minutes.to(images),
            "historical_sun_stream": historical_sun_stream,
            "historical_sun_coordinates": historical_coordinates,
            "historical_sun_attention": historical_attention,
            "target_sun_field_stream": target_sun_field_stream,
            "target_sun_coordinates": target_coordinates,
            "target_sun_attention": target_attention,
            "target_phase_stream": target_phase_stream,
            "target_path_source_coordinates": source_coordinates,
            "target_path_displacement": path_displacement,
            "target_path_confidence": target_confidence,
            "transport_hypothesis_weights": hypothesis_weights,
            "transport_hypothesis_source_coordinates": (
                hypothesis_source_coordinates),
            "transport_hypothesis_displacements": (
                None if hypothesis_source_coordinates is None
                else target_coordinates.unsqueeze(1)
                - hypothesis_source_coordinates),
            "transport_source_dispersion": source_dispersion,
            "transport_represented_probability_mass": represented_mass,
            "target_sun_field_std": target_sun_field_std,
            "target_phase_std": target_phase_std,
            "pv_statistics": pv_statistics,
            "pv_summary": pv_summary,
            "correspondence_cost_volume": cost_volume,
            "correspondence_probabilities": correspondence,
            "pairwise_flow_field": pairwise_flow,
            "pairwise_flow_confidence": pairwise_confidence,
            "motion_consistency_loss": motion_consistency_loss,
            "local_flow_field": local_flow,
            "local_flow_confidence": local_confidence,
            "transported_feature_fields": transported,
            "transported_phase_fields": transported_phase,
            "path_sampling_grids": path_grids,
        }
