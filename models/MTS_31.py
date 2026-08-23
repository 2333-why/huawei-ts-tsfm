import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import Model as encoder
from models.encoder_img import Model as encoder_img
from models.image_sequence_encoder import (
    StanfordDualAppearanceMotionEncoder,
    StanfordImageSequenceEncoder,
    StanfordSolarAdvectionEncoder,
    StanfordSolarTokenEncoder,
    StanfordCausalPhaseFieldEncoder,
)
import random
from models.utils import BilinearOrthogonalProjector as bop


def phase_path_correction(current_pv, parent_pred, class_path_delta_kw):
    """Convert each class-specific endpoint path into a parent residual."""
    if current_pv.shape != parent_pred.shape:
        raise ValueError('current PV and parent prediction must align')
    if class_path_delta_kw.shape[:-1] != current_pv.shape[:-1]:
        raise ValueError('class path deltas must align with current PV')
    if class_path_delta_kw.shape[-1] != 5:
        raise ValueError('class path deltas must contain five phase hypotheses')
    return current_pv + class_path_delta_kw - parent_pred


def conjunctive_event_support(path_support, phase_event_support):
    """Require both event detectors to be confident before routing a correction."""
    return (path_support + phase_event_support - 1.0).clamp(0.0, 1.0)


def winner_conjunctive_support(path_support, active_support, winner_probability):
    """Route only when path, endpoint phase and candidate benefit all agree."""
    return (
        path_support + active_support + winner_probability - 2.0
    ).clamp(0.0, 1.0)


def hard_phase_winner_support(
        path_support, endpoint_phase_probability, winner_probability):
    """Select the endpoint-active expert with calibrated class decisions."""
    endpoint_phase = endpoint_phase_probability.argmax(dim=-1, keepdim=True)
    endpoint_active = (endpoint_phase == 1) | (endpoint_phase == 2)
    return (
        (path_support >= 0.5)
        & endpoint_active
        & (winner_probability >= 0.5)
    ).to(path_support.dtype)


def confidence_margin_support(
        path_support, endpoint_phase_probability, winner_probability,
        winner_prior=None):
    """Use only confidence above each detector's natural decision boundary."""
    path_margin = (2.0 * path_support - 1.0).clamp(0.0, 1.0)
    active_confidence = endpoint_phase_probability[..., 1:3].amax(
        dim=-1, keepdim=True)
    inactive_confidence = torch.cat([
        endpoint_phase_probability[..., :1],
        endpoint_phase_probability[..., 3:5],
    ], dim=-1).amax(dim=-1, keepdim=True)
    phase_margin = (active_confidence - inactive_confidence).clamp(0.0, 1.0)
    if winner_prior is None:
        winner_prior = torch.full_like(winner_probability, 0.5)
    winner_prior = torch.as_tensor(
        winner_prior, device=winner_probability.device,
        dtype=winner_probability.dtype)
    winner_margin = (
        (winner_probability - winner_prior)
        / (1.0 - winner_prior).clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    return path_margin * phase_margin * winner_margin


def endpoint_magnitude_support(endpoint_delta_kw, boundary_kw=5.0):
    """Measure whether the predicted endpoint remains outside the event band."""
    if boundary_kw <= 0:
        raise ValueError('endpoint event boundary must be positive')
    return (endpoint_delta_kw.abs() / boundary_kw).clamp(0.0, 1.0)


def phase_regime_support(
        path_support, endpoint_phase_probability, endpoint_support,
        winner_probability, winner_prior):
    """Combine phase confidence with the physics of each endpoint regime."""
    top_values, top_indices = endpoint_phase_probability.topk(2, dim=-1)
    class_margin = (top_values[..., :1] - top_values[..., 1:2]).clamp(0.0, 1.0)
    predicted_phase = top_indices[..., :1]
    inside_support = 1.0 - endpoint_support
    no_event_support = (1.0 - path_support) * inside_support
    active_support = path_support * endpoint_support
    return_support = path_support * inside_support
    physical_support = torch.where(
        predicted_phase == 0,
        no_event_support,
        torch.where(
            (predicted_phase == 1) | (predicted_phase == 2),
            active_support,
            return_support,
        ),
    )
    winner_prior = torch.as_tensor(
        winner_prior, device=winner_probability.device,
        dtype=winner_probability.dtype)
    winner_evidence = (
        (winner_probability - winner_prior)
        / (1.0 - winner_prior).clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    return class_margin * physical_support * winner_evidence, physical_support


def conditional_phase_physical_support(
        path_support, class_path_delta_kw, boundary_kw=5.0):
    """Signed physical support for no-event, active and return hypotheses."""
    if boundary_kw <= 0:
        raise ValueError('endpoint event boundary must be positive')
    if class_path_delta_kw.shape[-1] != 5:
        raise ValueError('class path deltas must end with five phase hypotheses')
    if path_support.shape != class_path_delta_kw.shape[:-1] + (1,):
        raise ValueError('path support must align with class path hypotheses')
    inside = (
        1.0 - class_path_delta_kw.abs() / boundary_kw
    ).clamp(0.0, 1.0)
    active_down = (-class_path_delta_kw[..., 1] / boundary_kw).clamp(0.0, 1.0)
    active_up = (class_path_delta_kw[..., 2] / boundary_kw).clamp(0.0, 1.0)
    event = path_support[..., 0]
    survival = 1.0 - event
    return torch.stack([
        survival * inside[..., 0],
        event * active_down,
        event * active_up,
        event * inside[..., 3],
        event * inside[..., 4],
    ], dim=-1)


def conditional_phase_route_support(
        path_support, phase_probability, class_path_delta_kw,
        class_winner_probability, winner_prior, class_mask=None,
        boundary_kw=5.0):
    """Soft mixture route with class-specific physics and train-only priors."""
    expected_shape = phase_probability.shape
    for name, value in [
        ('class_path_delta_kw', class_path_delta_kw),
        ('class_winner_probability', class_winner_probability),
    ]:
        if value.shape != expected_shape:
            raise ValueError(f'{name} must match phase probability')
    physical = conditional_phase_physical_support(
        path_support, class_path_delta_kw, boundary_kw)
    # Winner BCE is balanced across winner/non-winner and endpoint regimes.
    # Its calibrated decision boundary is therefore 0.5; the separately
    # recorded natural prior remains an audit statistic, not a route cutoff.
    winner_evidence = (
        2.0 * class_winner_probability - 1.0
    ).clamp(0.0, 1.0)
    route = phase_probability * physical * winner_evidence
    if class_mask is not None:
        route = route * torch.as_tensor(
            class_mask, device=route.device, dtype=route.dtype)
    return route, physical


def conditional_phase_risk_route_support(
        path_support, phase_probability, class_path_delta_kw,
        class_excess_risk, normalized_risk_scale, class_mask=None,
        boundary_kw=5.0):
    """Route only where the candidate's predicted risk is below the parent.

    Excess risk and its scale are normalized by plant capacity squared.  The
    signed magnitude therefore retains the asymmetric cost that a binary
    winner target discards.
    """
    expected_shape = phase_probability.shape
    for name, value in [
        ('class_path_delta_kw', class_path_delta_kw),
        ('class_excess_risk', class_excess_risk),
    ]:
        if value.shape != expected_shape:
            raise ValueError(f'{name} must match phase probability')
    if normalized_risk_scale <= 0:
        raise ValueError('normalized phase risk scale must be positive')
    physical = conditional_phase_physical_support(
        path_support, class_path_delta_kw, boundary_kw)
    risk_evidence = (
        -class_excess_risk / float(normalized_risk_scale)
    ).clamp(0.0, 1.0)
    route = phase_probability * physical * risk_evidence
    if class_mask is not None:
        route = route * torch.as_tensor(
            class_mask, device=route.device, dtype=route.dtype)
    return route, physical


def cal_similarity(x, y, delay=3):

    if x.shape != y.shape:
            raise ValueError("输入张量x和y的形状必须相同。")

    batch_size, seq_len, feature_dim = x.shape
    
    correlations_per_delay = []

    for p in range(delay + 1):
  
        shifted_x = x[:, p:, :]
        
        aligned_y = y[:, :seq_len - p, :]
        
        product = shifted_x * aligned_y
        
       
        correlation_at_p = torch.sum(product, dim=1)
        
        correlations_per_delay.append(correlation_at_p)

    
    stacked_correlations = torch.stack(correlations_per_delay, dim=0)

    
    max_correlation, _ = torch.max(stacked_correlations, dim=0)
    
    # The legacy one-step model used [B,1,D].  Repeat the same correlation
    # score across configured forecast steps without changing its definition.
    return max_correlation.unsqueeze(1).expand(-1, seq_len, -1)

class Model(nn.Module):
    def __init__(self, args, args_img, args_weather):
        super(Model, self).__init__()
        self.args = args
        self.args_img = args_img
        student_img_args = copy.copy(args_img)
        student_img_args.seq_len = int(args.seq_len)
        student_weather_args = copy.copy(args_weather)
        student_weather_args.seq_len = int(args.seq_len)
        self.branch_ts = encoder(args)
        self.ts_projection = (
            nn.Linear(args.enc_in, args.c_out)
            if getattr(args, 'data', None) in ['LuoyangParquet', 'YLJParquet']
            and args.enc_in != args.c_out
            else nn.Identity()
        )
        self.image_encoder_type = getattr(args, 'image_encoder_type', 'bop')
        self.solar_advection_encoder = self.image_encoder_type == 'cnn_solar_advection'
        self.event_routing_mode = getattr(args, 'event_routing_mode', 'soft')
        if self.event_routing_mode not in ['soft', 'hard', 'confidence']:
            raise ValueError(f'unsupported event routing mode: {self.event_routing_mode}')
        self.tail_routing_mode = getattr(args, 'tail_routing_mode', 'confidence')
        if self.tail_routing_mode not in ['soft', 'hard', 'confidence', 'posterior']:
            raise ValueError(f'unsupported tail routing mode: {self.tail_routing_mode}')
        self.tail_confidence_threshold = float(
            getattr(args, 'tail_confidence_threshold', 0.5))
        if not 0.0 <= self.tail_confidence_threshold < 1.0:
            raise ValueError('tail_confidence_threshold must be in [0, 1)')
        self.dual_image_encoder = self.image_encoder_type in [
            'cnn_dual_motion', 'cnn_solar_advection']
        if getattr(args, 'data', None) in ['Stanford', 'LuoyangParquet', 'YLJParquet'] and self.dual_image_encoder:
            legacy_img_args = copy.copy(student_img_args)
            if self.solar_advection_encoder:
                legacy_img_args.stanford_image_mode = 'gray'
            self.branch_img = StanfordDualAppearanceMotionEncoder(legacy_img_args)
            self.bop_img = None
        elif getattr(args, 'data', None) in ['Stanford', 'LuoyangParquet', 'YLJParquet'] and self.image_encoder_type != 'bop':
            self.branch_img = StanfordImageSequenceEncoder(student_img_args, self.image_encoder_type)
            self.bop_img = None
        else:
            self.branch_img = encoder_img(student_img_args)
            self.bop_img = bop(student_img_args.H, student_img_args.W, student_img_args.r_h, student_img_args.r_w)
        self.branch_weather = encoder_img(student_weather_args)
        self.img_projection = nn.Linear(args_img.c_out, args.c_out)  # 512 -> 42
        self.weather_projection = nn.Linear(args_weather.c_out, args.c_out)  # 6 -> 42
        self.tors = args.tors
        self.modality_dropout_rate = args.modality_dropout_rate
        self.ramp_head = nn.Linear(args.c_out * 3, 1)
        self.ramp_direction_head = nn.Linear(args.c_out * 3, 1)
        self.expert_gate = nn.Linear(args.c_out * 3 + 1, 1)
        nn.init.zeros_(self.expert_gate.weight)
        nn.init.constant_(self.expert_gate.bias, float(getattr(args, 'expert_gate_bias', -1.0)))
        self.last_ramp_pred = None
        self.last_ramp_direction_logit = None
        self.last_expert_gate = None
        self.last_expert_gate_logit = None
        self.last_stable_pred = None
        self.last_ramp_expert_pred = None
        self.last_residual_pred = None
        self.last_applied_correction = None
        self.last_raw_applied_correction = None
        self.last_motion_energy = None
        self.last_event_state_logits = None
        self.last_event_state_probabilities = None
        self.last_event_state_routing = None
        self.last_event_correction = None
        self.last_raw_event_correction = None
        self.last_spatial_forecast = None
        self.last_spatial_forecast_projected = None
        self.last_solar_advection = None
        self.last_solar_token = None
        self.last_solar_attention_logits = None
        self.last_clear_sky_kw = None
        self.last_tail_parent_pred = None
        self.last_tail_scale = None
        self.last_tail_event_logit = None
        self.last_tail_direction_logit = None
        self.last_tail_event_probability = None
        self.last_tail_direction_probability = None
        self.last_tail_routing_strength = None
        self.last_tail_up_raw_correction = None
        self.last_tail_down_raw_correction = None
        self.last_tail_expert_correction = None
        self.last_phase_field = None
        self.last_phase_hazard_logits = None
        self.last_phase_state_logits = None
        self.last_phase_path_delta_kw = None
        self.last_phase_class_path_delta_kw = None
        self.last_phase_parent_pred = None
        self.last_phase_class_raw_correction = None
        self.last_phase_candidate_correction = None
        self.last_phase_benefit_logit = None
        self.last_phase_scale_logit = None
        self.last_phase_class_benefit_logits = None
        self.last_phase_class_scale_logits = None
        self.last_phase_benefit_strength = None
        self.last_phase_route_support = None
        self.last_phase_class_route_support = None
        self.last_phase_active_support = None
        self.last_phase_safety_strength = None
        self.last_phase_safety_correction = None
        self.last_phase_endpoint_magnitude_support = None
        self.residual_correction_clip_kw = float(
            getattr(args, 'residual_correction_clip_kw', 10.0))
        self.appearance_correction_head = nn.Linear(args_img.c_out, args.c_out)
        self.motion_correction_head = nn.Linear(args_img.c_out * 2, args.c_out)
        nn.init.zeros_(self.appearance_correction_head.weight)
        nn.init.zeros_(self.appearance_correction_head.bias)
        nn.init.zeros_(self.motion_correction_head.weight)
        nn.init.zeros_(self.motion_correction_head.bias)
        gate_hidden = int(getattr(args, 'expert_gate_hidden_dim', 32))
        self.soft_expert_gate = nn.Sequential(
            nn.Linear(args_img.c_out * 2 + 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.soft_expert_gate[-1].weight)
        nn.init.constant_(
            self.soft_expert_gate[-1].bias,
            float(getattr(args, 'expert_gate_bias', -2.0)),
        )
        if self.solar_advection_encoder:
            event_dim = int(getattr(args_img, 'advection_output_dim', args_img.c_out))
            spatial_dim = int(getattr(args_img, 'image_cnn_dim', 64))
            self.solar_event_encoder = StanfordSolarAdvectionEncoder(args_img)
            self.solar_color_projection = nn.Sequential(
                nn.Linear(12 * 4 * 4, event_dim),
                nn.GELU(),
                nn.LayerNorm(event_dim),
            )
            self.event_state_head = nn.Linear(event_dim, 3)
            self.event_up_correction_head = nn.Linear(event_dim, args.c_out)
            self.event_down_correction_head = nn.Linear(event_dim, args.c_out)
            self.spatial_kd_projection = nn.Conv2d(
                spatial_dim, spatial_dim, kernel_size=1, bias=False)
            self.solar_token_encoder = StanfordSolarTokenEncoder(args_img)
            tail_context_dim = event_dim + 3
            self.tail_event_head = nn.Linear(tail_context_dim, 1)
            self.tail_direction_head = nn.Linear(tail_context_dim, 1)
            self.tail_up_head = nn.Sequential(
                nn.Linear(tail_context_dim, event_dim),
                nn.GELU(),
                nn.Linear(event_dim, args.c_out),
            )
            self.tail_down_head = nn.Sequential(
                nn.Linear(tail_context_dim, event_dim),
                nn.GELU(),
                nn.Linear(event_dim, args.c_out),
            )
            self.register_buffer(
                'tail_prior_event_positive', torch.zeros((), dtype=torch.float64))
            self.register_buffer(
                'tail_prior_event_total', torch.zeros((), dtype=torch.float64))
            self.register_buffer(
                'tail_prior_up_positive', torch.zeros((), dtype=torch.float64))
            self.register_buffer(
                'tail_prior_direction_total', torch.zeros((), dtype=torch.float64))
            self.register_buffer(
                'tail_prior_frozen', torch.zeros((), dtype=torch.bool))
            self.phase_field_encoder = StanfordCausalPhaseFieldEncoder(args_img)
            if int(args.c_out) != 1:
                raise ValueError(
                    'Stanford causal phase-field routing requires c_out=1')
            phase_dim = self.phase_field_encoder.output_dim
            phase_route_state_dim = 5 * 5 + 5 + 5 * 5
            phase_context_dim = phase_dim + 3 + phase_route_state_dim
            self.phase_class_correction_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(phase_context_dim, phase_dim),
                    nn.GELU(),
                    nn.Linear(phase_dim, args.c_out),
                )
                for _ in range(5)
            ])
            self.phase_class_benefit_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(
                        phase_dim * 2 + 8 + phase_route_state_dim,
                        phase_dim),
                    nn.GELU(),
                    nn.Linear(phase_dim, 2),
                )
                for _ in range(5)
            ])
            self.phase_family_stack_head = nn.Sequential(
                nn.Linear(
                    phase_dim * 2 + 9 + phase_route_state_dim,
                    phase_dim),
                nn.GELU(),
                nn.Linear(phase_dim, 2),
            )
            self.phase_family_stack_head.register_buffer(
                'family_calibration', torch.ones(2))
            self.phase_correction_mode = str(getattr(
                args, 'phase_correction_mode', 'learned_residual'))
            if self.phase_correction_mode not in [
                'learned_residual', 'physics_path'
            ]:
                raise ValueError(
                    f'unsupported phase correction mode: '
                    f'{self.phase_correction_mode}')
            self.phase_benefit_mode = str(getattr(
                args, 'phase_benefit_mode', 'winner'))
            if self.phase_benefit_mode not in [
                'winner', 'net_risk', 'conditional_mean', 'family_stack',
                'constrained_stack'
            ]:
                raise ValueError(
                    f'unsupported phase benefit mode: {self.phase_benefit_mode}')
            self.phase_risk_scale_kw = float(getattr(
                args, 'phase_risk_scale_kw', 3.0))
            if self.phase_risk_scale_kw <= 0:
                raise ValueError('phase_risk_scale_kw must be positive')
            route_names = (
                'no_event', 'active_down', 'active_up',
                'return_from_down', 'return_from_up')
            requested_routes = str(getattr(
                args, 'phase_route_classes', 'active_down,active_up'))
            requested_routes = {
                value.strip() for value in requested_routes.split(',')
                if value.strip()
            }
            if requested_routes == {'all'}:
                requested_routes = set(route_names)
            unknown_routes = requested_routes.difference(route_names)
            if unknown_routes:
                raise ValueError(
                    'unsupported phase route classes: '
                    + ','.join(sorted(unknown_routes)))
            self.register_buffer(
                'phase_route_class_mask',
                torch.tensor([
                    name in requested_routes for name in route_names
                ], dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                'phase_route_enabled', torch.zeros((), dtype=torch.bool))
            self.register_buffer(
                'phase_winner_positive', torch.zeros(5, dtype=torch.float64))
            self.register_buffer(
                'phase_winner_total', torch.zeros(5, dtype=torch.float64))
            self.register_buffer(
                'phase_winner_prior_frozen', torch.zeros((), dtype=torch.bool))
            with torch.no_grad():
                self.spatial_kd_projection.weight.zero_()
                diagonal = min(
                    self.spatial_kd_projection.weight.shape[0],
                    self.spatial_kd_projection.weight.shape[1])
                indices = torch.arange(diagonal)
                self.spatial_kd_projection.weight[indices, indices, 0, 0] = 1.0
                self.event_state_head.weight.zero_()
                self.event_state_head.bias.copy_(torch.tensor([2.0, -1.0, -1.0]))
                self.event_up_correction_head.weight.zero_()
                self.event_up_correction_head.bias.zero_()
                self.event_down_correction_head.weight.zero_()
                self.event_down_correction_head.bias.zero_()
                self.tail_event_head.weight.zero_()
                self.tail_event_head.bias.zero_()
                self.tail_direction_head.weight.zero_()
                self.tail_direction_head.bias.zero_()
                self.tail_up_head[-1].weight.zero_()
                self.tail_up_head[-1].bias.zero_()
                self.tail_down_head[-1].weight.zero_()
                self.tail_down_head[-1].bias.zero_()
                for head in self.phase_class_correction_heads:
                    head[-1].weight.zero_()
                    head[-1].bias.zero_()
                for head in self.phase_class_benefit_heads:
                    head[-1].weight.zero_()
                    head[-1].bias.zero_()
                self.phase_family_stack_head[-1].weight.zero_()
                self.phase_family_stack_head[-1].bias.zero_()
        else:
            self.solar_event_encoder = None
            self.solar_color_projection = None
            self.event_state_head = None
            self.event_up_correction_head = None
            self.event_down_correction_head = None
            self.spatial_kd_projection = None
            self.solar_token_encoder = None
            self.tail_event_head = None
            self.tail_direction_head = None
            self.tail_up_head = None
            self.tail_down_head = None
            self.tail_scale = None
            self.phase_field_encoder = None
            self.phase_class_correction_heads = None
            self.phase_class_benefit_heads = None
            self.phase_family_stack_head = None

    def set_phase_route_enabled(self, enabled):
        if not self.solar_advection_encoder:
            raise RuntimeError('phase routing requires the Stanford image encoder')
        self.phase_route_enabled.fill_(bool(enabled))

    def update_phase_winner_prior(self, winner, eligible):
        if (
            not self.training
            or bool(self.phase_winner_prior_frozen.item())
        ):
            return
        winner = winner.detach().to(dtype=torch.bool).reshape(-1, 5)
        eligible = eligible.detach().to(dtype=torch.bool).reshape(-1, 5)
        with torch.no_grad():
            self.phase_winner_positive.add_(
                (winner & eligible).sum(dim=0).to(self.phase_winner_positive))
            self.phase_winner_total.add_(
                eligible.sum(dim=0).to(self.phase_winner_total))

    def freeze_phase_winner_prior(self):
        if bool((self.phase_winner_total <= 0).any().item()):
            raise RuntimeError(
                'cannot freeze phase winner prior before train observations')
        self.phase_winner_prior_frozen.fill_(True)

    def phase_winner_prior_probability(self):
        fallback = torch.full_like(self.phase_winner_total, 0.5)
        return torch.where(
            self.phase_winner_total > 0,
            self.phase_winner_positive
            / self.phase_winner_total.clamp_min(1.0),
            fallback,
        ).clamp(1e-6, 1.0 - 1e-6)

    def phase_winner_prior_statistics(self):
        return {
            'winner_prior': self.phase_winner_prior_probability().detach()
            .cpu().tolist(),
            'winner_positive_count': self.phase_winner_positive.detach()
            .cpu().tolist(),
            'winner_total_count': self.phase_winner_total.detach()
            .cpu().tolist(),
            'frozen': bool(self.phase_winner_prior_frozen.detach().cpu()),
        }

    @staticmethod
    def _empirical_prior(positive, total, fallback):
        fallback_value = torch.as_tensor(
            fallback, device=total.device, dtype=total.dtype)
        return torch.where(
            total > 0,
            positive / total.clamp_min(1.0),
            fallback_value,
        ).clamp(1e-6, 1.0 - 1e-6)

    def update_tail_priors(self, event_mask, positive_direction):
        """Accumulate train-only priors for Bayes-corrected tail routing."""
        if (
            not self.training
            or not self.solar_advection_encoder
            or bool(self.tail_prior_frozen.item())
        ):
            return
        event_mask = event_mask.detach().to(dtype=torch.bool).view(-1)
        positive_direction = positive_direction.detach().to(
            dtype=torch.bool).view(-1)
        with torch.no_grad():
            self.tail_prior_event_positive.add_(
                event_mask.sum().to(self.tail_prior_event_positive))
            self.tail_prior_event_total.add_(
                torch.as_tensor(
                    event_mask.numel(), device=event_mask.device,
                    dtype=self.tail_prior_event_total.dtype))
            self.tail_prior_up_positive.add_(
                (event_mask & positive_direction).sum().to(
                    self.tail_prior_up_positive))
            self.tail_prior_direction_total.add_(
                event_mask.sum().to(self.tail_prior_direction_total))

    def freeze_tail_priors(self):
        if not self.solar_advection_encoder:
            return
        if self.tail_prior_event_total.item() <= 0:
            raise RuntimeError('cannot freeze tail priors before train observations')
        self.tail_prior_frozen.fill_(True)

    def tail_prior_probabilities(self):
        if not self.solar_advection_encoder:
            return None, None
        event_prior = self._empirical_prior(
            self.tail_prior_event_positive,
            self.tail_prior_event_total,
            fallback=0.05,
        )
        up_prior = self._empirical_prior(
            self.tail_prior_up_positive,
            self.tail_prior_direction_total,
            fallback=0.5,
        )
        return event_prior, up_prior

    def tail_prior_statistics(self):
        event_prior, up_prior = self.tail_prior_probabilities()
        if event_prior is None:
            return {}
        return {
            'event_prior': float(event_prior.detach().cpu()),
            'up_given_event_prior': float(up_prior.detach().cpu()),
            'event_positive_count': float(
                self.tail_prior_event_positive.detach().cpu()),
            'event_total_count': float(
                self.tail_prior_event_total.detach().cpu()),
            'up_positive_count': float(
                self.tail_prior_up_positive.detach().cpu()),
            'direction_total_count': float(
                self.tail_prior_direction_total.detach().cpu()),
            'frozen': bool(self.tail_prior_frozen.detach().cpu()),
        }

    @staticmethod
    def _legacy_gray_images(images):
        if images.dim() == 5 and images.shape[2] == 3:
            return images.mean(dim=2)
        return images

    def _solar_color_features(self, images):
        normalized = images
        if normalized.dim() == 4:
            normalized = normalized.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        elif normalized.dim() != 5:
            raise ValueError(f'unsupported solar image shape: {tuple(images.shape)}')
        if normalized.shape[2] == 1:
            normalized = normalized.repeat(1, 1, 3, 1, 1)
        if getattr(self.args, 'history_order', 'current_first') == 'current_first':
            normalized = torch.flip(normalized, dims=[1])

        current = normalized[:, -1]
        previous_5 = normalized[:, max(normalized.shape[1] - 6, 0)]
        previous_15 = normalized[:, 0]
        chromaticity = current / current.sum(dim=1, keepdim=True).clamp_min(1e-4)
        maps = torch.cat([
            current,
            current - previous_5,
            current - previous_15,
            chromaticity,
        ], dim=1)
        pooled = F.adaptive_avg_pool2d(maps, (4, 4)).flatten(1)
        return self.solar_color_projection(pooled).unsqueeze(1)

  
        
    def forward(
            self, x_ts, x_img_h, x_img_f, x_weather_h, x_weather_f,
            used_data=None, x_mark_h=None, x_mark_f=None,
            phase_lead_marks=None, image_mask=None):
        x_ts_f_pred = self.ts_projection(self.branch_ts(x_ts))
        self.last_stable_pred = x_ts_f_pred
        self.last_phase_field = None
        self.last_phase_hazard_logits = None
        self.last_phase_state_logits = None
        self.last_phase_path_delta_kw = None
        self.last_phase_class_path_delta_kw = None
        self.last_phase_parent_pred = None
        self.last_phase_class_raw_correction = None
        self.last_phase_candidate_correction = None
        self.last_phase_benefit_logit = None
        self.last_phase_scale_logit = None
        self.last_phase_class_benefit_logits = None
        self.last_phase_class_scale_logits = None
        self.last_phase_benefit_strength = None
        self.last_phase_route_support = None
        self.last_phase_class_route_support = None
        self.last_phase_active_support = None
        self.last_phase_safety_strength = None
        self.last_phase_safety_correction = None
        self.last_phase_stable_family_correction = None
        self.last_phase_active_family_correction = None
        self.last_phase_motion_consistency_loss = None
        self.last_phase_endpoint_magnitude_support = None
        dual_strategy = used_data in [
            'causal_residual_appearance', 'causal_residual_motion',
            'causal_soft_gate', 'causal_solar_event',
            'causal_solar_token_tail', 'causal_phase_field_tail']
        appearance_features = None
        motion_features = None
        motion_energy = torch.zeros_like(x_ts_f_pred[:, :1, :1])

        if used_data in ['no_img', 'ts_only', 'ts']:
            x_img_f_pred = torch.zeros_like(x_ts_f_pred)
            drop_img = torch.tensor(0.0, device=x_ts_f_pred.device)
        elif self.dual_image_encoder:
            legacy_images = self._legacy_gray_images(x_img_h)
            appearance_features, motion_features, motion_energy = self.branch_img(
                legacy_images)
            appearance_correction = self.appearance_correction_head(appearance_features)
            motion_delta = self.motion_correction_head(
                torch.cat([appearance_features, motion_features], dim=-1))
            combined_correction = appearance_correction + motion_delta
            x_img_f_pred = (
                appearance_correction
                if used_data == 'causal_residual_appearance'
                else combined_correction
            )
            if used_data in [
                'causal_solar_event', 'causal_solar_token_tail',
                'causal_phase_field_tail'
            ]:
                if x_mark_h is None or x_mark_f is None:
                    raise ValueError(
                        'causal_solar_event requires historical and target time marks')
                spatial_sequence = self.branch_img.appearance.encode_spatial_sequence(
                    legacy_images, all_steps=True)
                solar = self.solar_event_encoder(
                    spatial_sequence, x_mark_h, x_mark_f, x_ts)
                event_feature = (
                    solar['event_feature'] + self._solar_color_features(x_img_h))
                self.last_spatial_forecast = solar['future_spatial_map']
                self.last_spatial_forecast_projected = self.spatial_kd_projection(
                    self.last_spatial_forecast)
                self.last_solar_advection = solar
                self.last_event_state_logits = self.event_state_head(event_feature)
                self.last_event_state_probabilities = torch.softmax(
                    self.last_event_state_logits, dim=-1)
                if self.training or self.event_routing_mode == 'soft':
                    event_routing = self.last_event_state_probabilities
                else:
                    hard_routing = F.one_hot(
                        self.last_event_state_probabilities.argmax(dim=-1),
                        num_classes=3,
                    ).to(self.last_event_state_probabilities.dtype)
                    if self.event_routing_mode == 'confidence':
                        event_confidence = self.last_event_state_probabilities[
                            ..., 1:].max(dim=-1, keepdim=True).values
                        confident_event = event_confidence >= 0.5
                        stable_routing = torch.zeros_like(hard_routing)
                        stable_routing[..., 0] = 1.0
                        event_routing = torch.where(
                            confident_event, hard_routing, stable_routing)
                    else:
                        event_routing = hard_routing
                self.last_event_state_routing = event_routing
                up_correction = self.event_up_correction_head(event_feature)
                down_correction = self.event_down_correction_head(event_feature)
                raw_event_correction = (
                    event_routing[..., 2:3] * up_correction
                    + event_routing[..., 1:2] * down_correction
                )
                self.last_raw_event_correction = raw_event_correction
            else:
                self.last_event_state_logits = None
                self.last_event_state_probabilities = None
                self.last_event_state_routing = None
                self.last_spatial_forecast = None
                self.last_spatial_forecast_projected = None
                self.last_solar_advection = None
                self.last_raw_event_correction = None
                self.last_solar_token = None
                self.last_solar_attention_logits = None
                self.last_clear_sky_kw = None
                self.last_tail_parent_pred = None
                self.last_tail_scale = None
                self.last_tail_event_logit = None
                self.last_tail_direction_logit = None
                self.last_tail_event_probability = None
                self.last_tail_direction_probability = None
                self.last_tail_routing_strength = None
                self.last_tail_up_raw_correction = None
                self.last_tail_down_raw_correction = None
                self.last_tail_expert_correction = None
                self.last_phase_field = None
                self.last_phase_hazard_logits = None
                self.last_phase_state_logits = None
                self.last_phase_path_delta_kw = None
                self.last_phase_class_path_delta_kw = None
                self.last_phase_parent_pred = None
                self.last_phase_class_raw_correction = None
                self.last_phase_candidate_correction = None
                self.last_phase_benefit_logit = None
                self.last_phase_scale_logit = None
                self.last_phase_class_benefit_logits = None
                self.last_phase_class_scale_logits = None
                self.last_phase_benefit_strength = None
                self.last_phase_route_support = None
                self.last_phase_class_route_support = None
                self.last_phase_active_support = None
                self.last_phase_safety_strength = None
                self.last_phase_safety_correction = None
                self.last_phase_endpoint_magnitude_support = None
            if self.training and random.random() < self.modality_dropout_rate:
                x_img_f_pred = torch.zeros_like(x_img_f_pred)
                appearance_correction = torch.zeros_like(appearance_correction)
                motion_delta = torch.zeros_like(motion_delta)
                combined_correction = torch.zeros_like(combined_correction)
                drop_img = torch.tensor(0.0, device=x_ts_f_pred.device)
            else:
                drop_img = torch.tensor(1.0, device=x_ts_f_pred.device)
        else:
            if self.image_encoder_type != 'bop':
                x_img = x_img_h
            else:
                x_img = self.bop_img(x_img_h) if x_img_h.dim() >= 4 else x_img_h
            x_img_f_pred = self.img_projection(self.branch_img(x_img))
            if self.training and random.random() < self.modality_dropout_rate:
                x_img_f_pred = torch.zeros_like(x_img_f_pred)
                drop_img = torch.tensor(0.0, device=x_ts_f_pred.device)
            else:
                drop_img = torch.tensor(1.0, device=x_ts_f_pred.device)
        if image_mask is not None and used_data not in ['no_img', 'ts_only', 'ts']:
            image_available = image_mask.to(x_img_f_pred.device).bool().any(dim=1)
            x_img_f_pred = x_img_f_pred * image_available[:, None, None].to(x_img_f_pred.dtype)
        sim_img_ts = cal_similarity(x_img_f_pred, x_ts_f_pred)

        causal_strategies = [
            'causal_img', 'causal_gated', 'causal_residual_appearance',
            'causal_residual_motion', 'causal_soft_gate', 'causal_solar_event',
            'causal_solar_token_tail', 'causal_phase_field_tail',
        ]
        if used_data in ['no_weather', 'ts_only', 'ts'] + causal_strategies:
            x_weather_f_pred = torch.zeros_like(x_ts_f_pred)
            drop_weather = torch.tensor(0.0, device=x_ts_f_pred.device)
        else:
            x_weather_f_pred = self.weather_projection(self.branch_weather(x_weather_h))
            if self.training and random.random() < self.modality_dropout_rate:
                x_weather_f_pred = torch.zeros_like(x_weather_f_pred)
                drop_weather = torch.tensor(0.0, device=x_ts_f_pred.device)
            else:
                drop_weather = torch.tensor(1.0, device=x_ts_f_pred.device)
        sim_weather_ts = cal_similarity(x_weather_f_pred, x_ts_f_pred)

        auxiliary_features = torch.cat([x_ts_f_pred, x_img_f_pred, sim_img_ts], dim=-1)
        self.last_ramp_pred = self.ramp_head(auxiliary_features)
        self.last_ramp_direction_logit = self.ramp_direction_head(auxiliary_features)
        self.last_motion_energy = motion_energy

        if dual_strategy:
            clip = self.residual_correction_clip_kw
            appearance_residual = (
                torch.clamp(appearance_correction, min=-clip, max=clip)
                if clip > 0 else appearance_correction
            )
            combined_residual = (
                torch.clamp(combined_correction, min=-clip, max=clip)
                if clip > 0 else combined_correction
            )
            if used_data == 'causal_residual_appearance':
                stable_pred = x_ts_f_pred
                ramp_expert_pred = x_ts_f_pred + appearance_residual
                expert_residual = appearance_residual
                applied_correction = appearance_residual
                raw_applied_correction = appearance_correction
            elif used_data in ['causal_residual_motion', 'causal_soft_gate']:
                # Motion and gate candidates are exact Appearance children at
                # initialization because motion_correction_head starts at zero.
                stable_pred = x_ts_f_pred + appearance_residual
                ramp_expert_pred = x_ts_f_pred + combined_residual
                expert_residual = combined_residual - appearance_residual
                applied_correction = combined_residual
                raw_applied_correction = combined_correction
            elif used_data in [
                'causal_solar_event', 'causal_solar_token_tail',
                'causal_phase_field_tail'
            ]:
                event_residual = (
                    torch.clamp(raw_event_correction, min=-clip, max=clip)
                    if clip > 0 else raw_event_correction
                )
                stable_pred = x_ts_f_pred + appearance_residual
                ramp_expert_pred = stable_pred + event_residual
                expert_residual = event_residual
                applied_correction = appearance_residual + event_residual
                raw_applied_correction = appearance_correction + raw_event_correction

            self.last_stable_pred = stable_pred
            self.last_residual_pred = expert_residual
            self.last_ramp_expert_pred = ramp_expert_pred
            if used_data == 'causal_soft_gate':
                historical_volatility = x_ts[:, 1:, :].sub(x_ts[:, :-1, :]).abs().mean(dim=1, keepdim=True)
                historical_volatility = historical_volatility / float(
                    getattr(self.args, 'stanford_capacity_kw', 30.1))
                gate_features = torch.cat([
                    appearance_features, motion_features, historical_volatility, motion_energy], dim=-1)
                self.last_expert_gate_logit = self.soft_expert_gate(gate_features)
                self.last_expert_gate = torch.sigmoid(self.last_expert_gate_logit)
                applied_correction = appearance_residual + self.last_expert_gate * expert_residual
                raw_applied_correction = appearance_correction + self.last_expert_gate * motion_delta
                x = x_ts_f_pred + applied_correction
            elif used_data in [
                'causal_solar_event', 'causal_solar_token_tail',
                'causal_phase_field_tail'
            ]:
                event_residual = (
                    torch.clamp(raw_event_correction, min=-clip, max=clip)
                    if clip > 0 else raw_event_correction
                )
                stable_pred = x_ts_f_pred + appearance_residual
                ramp_expert_pred = stable_pred + event_residual
                expert_residual = event_residual
                applied_correction = appearance_residual + event_residual
                raw_applied_correction = appearance_correction + raw_event_correction
                self.last_stable_pred = stable_pred
                self.last_ramp_expert_pred = ramp_expert_pred
                self.last_residual_pred = expert_residual
                self.last_event_correction = event_residual
                self.last_expert_gate = (
                    1.0 - self.last_event_state_probabilities[..., :1])
                self.last_expert_gate_logit = None
                x = ramp_expert_pred

                if used_data == 'causal_solar_token_tail':
                    parent_pred = ramp_expert_pred
                    tail = self.solar_token_encoder(
                        x_img_h, x_mark_h, x_mark_f, x_ts)
                    current_pv = (
                        x_ts[:, :1, :]
                        if getattr(self.args, 'history_order', 'current_first') == 'current_first'
                        else x_ts[:, -1:, :]
                    )
                    clear_sky_kw = tail['clear_sky_kw']
                    capacity = float(getattr(self.args, 'stanford_capacity_kw', 30.1))
                    if capacity <= 0:
                        raise ValueError('stanford_capacity_kw must be positive')
                    shared_scalars = torch.cat([
                        current_pv / capacity,
                        parent_pred.detach() / capacity,
                        clear_sky_kw / capacity,
                    ], dim=-1)
                    magnitude_context = torch.cat([
                        tail['event_feature'],
                        shared_scalars,
                    ], dim=-1)
                    transition_context = torch.cat([
                        tail['transition_feature'],
                        shared_scalars,
                    ], dim=-1)
                    tail_event_logit = self.tail_event_head(transition_context)
                    tail_direction_logit = self.tail_direction_head(magnitude_context)
                    event_probability = torch.sigmoid(tail_event_logit)
                    up_probability = torch.sigmoid(tail_direction_logit)
                    calibrated_event_probability = event_probability
                    if self.tail_routing_mode == 'soft':
                        routing_strength = event_probability
                        routed_up_probability = up_probability
                    elif self.tail_routing_mode == 'hard':
                        routing_strength = (
                            event_probability >= self.tail_confidence_threshold
                        ).to(event_probability.dtype)
                        routed_up_probability = (up_probability >= 0.5).to(
                            up_probability.dtype)
                    else:
                        routing_strength = (
                            (event_probability - self.tail_confidence_threshold)
                            / (1.0 - self.tail_confidence_threshold)
                        ).clamp(0.0, 1.0)
                        routed_up_probability = up_probability

                    if self.tail_routing_mode == 'posterior':
                        event_prior, up_prior = self.tail_prior_probabilities()
                        event_prior = event_prior.to(event_probability)
                        up_prior = up_prior.to(up_probability)
                        event_log_prior_odds = torch.logit(
                            event_prior)
                        up_log_prior_odds = torch.logit(
                            up_prior)
                        calibrated_event_probability = torch.sigmoid(
                            tail_event_logit + event_log_prior_odds)
                        # The frozen parent already represents the unconditional
                        # training prior.  Route only evidence above that prior,
                        # scaled to retain a unit-strength certain event.
                        routing_strength = (
                            (calibrated_event_probability - event_prior)
                            / (1.0 - event_prior)
                        ).clamp(0.0, 1.0)
                        routed_up_probability = torch.sigmoid(
                            tail_direction_logit + up_log_prior_odds)

                    state_probabilities = torch.cat([
                        1.0 - routing_strength,
                        routing_strength * (1.0 - routed_up_probability),
                        routing_strength * routed_up_probability,
                    ], dim=-1)

                    tail_routing = torch.cat([
                        1.0 - routing_strength,
                        routing_strength * (1.0 - routed_up_probability),
                        routing_strength * routed_up_probability,
                    ], dim=-1)

                    # Occurrence and direction are supervised classifiers.  Do
                    # not let regression inflate their probabilities merely to
                    # reduce an event-magnitude loss.
                    if self.training:
                        routing_strength = routing_strength.detach()
                        routed_up_probability = routed_up_probability.detach()

                    # The two heads predict normalized corrections and are
                    # supervised only on their own direction.  Zero
                    # initialization keeps this an exact Stage-1 child while
                    # avoiding a global scale that can reverse both signs.
                    tail_up_raw = capacity * self.tail_up_head(magnitude_context)
                    tail_down_raw = capacity * self.tail_down_head(magnitude_context)
                    tail_up_correction = F.relu(tail_up_raw)
                    tail_down_correction = -F.relu(-tail_down_raw)
                    tail_expert_correction = (
                        routed_up_probability * tail_up_correction
                        + (1.0 - routed_up_probability) * tail_down_correction
                    )
                    tail_correction = routing_strength * tail_expert_correction
                    if clip > 0:
                        tail_expert_correction = torch.clamp(
                            tail_expert_correction, min=-clip, max=clip)
                        tail_correction = torch.clamp(
                            tail_correction, min=-clip, max=clip)

                    self.last_tail_event_logit = tail_event_logit
                    self.last_tail_direction_logit = tail_direction_logit
                    self.last_tail_event_probability = calibrated_event_probability
                    self.last_tail_direction_probability = routed_up_probability
                    self.last_event_state_logits = state_probabilities.clamp_min(1e-8).log()
                    self.last_event_state_probabilities = state_probabilities
                    self.last_event_state_routing = tail_routing
                    self.last_solar_token = tail['future_solar_token']
                    self.last_solar_attention_logits = tail['solar_location_logits']
                    self.last_clear_sky_kw = clear_sky_kw
                    self.last_tail_parent_pred = parent_pred
                    self.last_tail_scale = None
                    self.last_tail_routing_strength = routing_strength
                    self.last_tail_up_raw_correction = tail_up_raw
                    self.last_tail_down_raw_correction = tail_down_raw
                    self.last_tail_expert_correction = tail_expert_correction
                    self.last_raw_event_correction = tail_correction
                    self.last_event_correction = tail_correction
                    self.last_stable_pred = parent_pred
                    self.last_residual_pred = tail_correction
                    self.last_ramp_expert_pred = parent_pred + tail_correction
                    self.last_expert_gate = routing_strength
                    applied_correction = applied_correction + tail_correction
                    raw_applied_correction = (
                        raw_applied_correction + tail_correction)
                    # Keep the rare-event mixture calibrated: its occurrence,
                    # direction and magnitude are trained by separate targets.
                    # Otherwise all-sample task MSE inflates the conditional
                    # magnitude to compensate for a small gate probability.
                    output_tail_correction = (
                        tail_correction.detach()
                        if self.training else tail_correction
                    )
                    x = parent_pred + output_tail_correction
                elif used_data == 'causal_phase_field_tail':
                    if phase_lead_marks is None:
                        raise ValueError(
                            'causal_phase_field_tail requires fixed-lead time marks')
                    parent_pred = ramp_expert_pred
                    phase = self.phase_field_encoder(
                        x_img_h, x_mark_h, phase_lead_marks, x_ts)
                    capacity = float(getattr(
                        self.args, 'stanford_capacity_kw', 30.1))
                    if capacity <= 0:
                        raise ValueError('stanford_capacity_kw must be positive')
                    current_pv = (
                        x_ts[:, :1, :]
                        if getattr(
                            self.args, 'history_order', 'current_first'
                        ) == 'current_first'
                        else x_ts[:, -1:, :]
                    )
                    hazard_probability = torch.sigmoid(
                        phase['hazard_logits'])
                    phase_probability = torch.softmax(
                        phase['phase_logits'], dim=-1)
                    class_path_delta = phase[
                        'class_path_delta_kw'].squeeze(-1)
                    predicted_phase_by_lead = phase_probability.argmax(
                        dim=-1, keepdim=True)
                    selected_path_delta = class_path_delta.gather(
                        dim=-1, index=predicted_phase_by_lead)
                    endpoint_class_path_delta = class_path_delta[:, -1:]
                    endpoint_path_delta = selected_path_delta[:, -1:]
                    route_state = torch.cat([
                        phase_probability.reshape(
                            phase_probability.shape[0], 1, -1),
                        hazard_probability.unsqueeze(1),
                        (class_path_delta / capacity).reshape(
                            class_path_delta.shape[0], 1, -1),
                    ], dim=-1)
                    phase_route_tensors = {
                        'path_features': phase['path_features'][:, -1:],
                        'class_path_delta_kw': class_path_delta,
                        'hazard_logits': phase['hazard_logits'],
                        'phase_logits': phase['phase_logits'],
                        'current_pv': current_pv,
                        'parent_pred': parent_pred,
                        'route_state': route_state,
                    }
                    non_finite_route_inputs = [
                        name for name, value in phase_route_tensors.items()
                        if not bool(torch.isfinite(value).all().item())
                    ]
                    if non_finite_route_inputs:
                        raise FloatingPointError(
                            'non-finite phase route inputs: '
                            + ','.join(non_finite_route_inputs))
                    # These heads are tiny, while their route state mixes
                    # regression values, probabilities and learned features.
                    # Keep them in FP32 under global AMP to avoid a single
                    # rare path overflowing a half-precision matrix multiply.
                    magnitude_contexts = []
                    for class_index in range(5):
                        class_scalars = torch.cat([
                            current_pv / capacity,
                            parent_pred.detach() / capacity,
                            endpoint_class_path_delta[
                                ..., class_index:class_index + 1] / capacity,
                        ], dim=-1)
                        magnitude_contexts.append(torch.cat([
                            phase['path_features'][:, -1:],
                            class_scalars,
                            route_state,
                        ], dim=-1))
                    if self.phase_correction_mode == 'physics_path':
                        phase_class_raw = phase_path_correction(
                            current_pv,
                            parent_pred.detach(),
                            endpoint_class_path_delta)
                    else:
                        with torch.autocast(
                            device_type=magnitude_contexts[0].device.type,
                            enabled=False,
                        ):
                            phase_class_raw = capacity * torch.cat([
                                head(context.float())
                                for head, context in zip(
                                    self.phase_class_correction_heads,
                                    magnitude_contexts)
                            ], dim=-1)
                    non_finite_magnitude_outputs = [
                        name for name, value in {
                            'phase_class_raw': phase_class_raw,
                        }.items()
                        if not bool(torch.isfinite(value).all().item())
                    ]
                    if non_finite_magnitude_outputs:
                        bad_parameters = [
                            name for name, parameter in self.named_parameters()
                            if name.startswith('phase_class_correction_heads.')
                            and not bool(torch.isfinite(parameter).all().item())
                        ]
                        context_max = float(
                            max(
                                context.float().abs().amax()
                                for context in magnitude_contexts
                            ).detach().cpu())
                        raise FloatingPointError(
                            'non-finite phase magnitude outputs: '
                            + ','.join(non_finite_magnitude_outputs)
                            + f'; context_abs_max={context_max}; '
                            + 'non_finite_parameters=' + ','.join(bad_parameters))
                    if clip > 0:
                        phase_class_raw = torch.clamp(
                            phase_class_raw, min=-clip, max=clip)

                    endpoint_phase_probability = phase_probability[:, -1:]
                    endpoint_active_probability = endpoint_phase_probability[
                        ..., 1:3]
                    active_probability = endpoint_active_probability.sum(
                        dim=-1, keepdim=True)
                    predicted_phase = endpoint_phase_probability.argmax(
                        dim=-1, keepdim=True)
                    candidate_correction = phase_class_raw.gather(
                        dim=-1, index=predicted_phase)

                    path_support = (
                        1.0 - (1.0 - hazard_probability).prod(
                            dim=1, keepdim=True)
                    ).unsqueeze(-1)
                    endpoint_no_event = endpoint_phase_probability[..., :1]
                    endpoint_active_support = active_probability
                    track_confidence = phase[
                        'target_path_confidence'][:, -1:]
                    benefit_common_context = torch.cat([
                        phase['hazard_features'][:, -1:],
                        phase['phase_features'][:, -1:],
                        current_pv / capacity,
                        parent_pred.detach() / capacity,
                        path_support.detach(),
                        endpoint_no_event.detach(),
                        track_confidence.detach(),
                        route_state.detach(),
                    ], dim=-1)
                    benefit_contexts = []
                    for class_index in range(5):
                        benefit_contexts.append(torch.cat([
                            benefit_common_context,
                            endpoint_class_path_delta[
                                ..., class_index:class_index + 1].detach()
                            / capacity,
                            phase_class_raw[
                                ..., class_index:class_index + 1].detach()
                            / capacity,
                            endpoint_phase_probability[
                                ..., class_index:class_index + 1].detach(),
                        ], dim=-1))
                    with torch.autocast(
                        device_type=benefit_contexts[0].device.type,
                        enabled=False,
                    ):
                        class_benefit_outputs = [
                            head(context.float())
                            for head, context in zip(
                                self.phase_class_benefit_heads,
                                benefit_contexts)
                        ]
                    class_benefit_logits = torch.cat([
                        output[..., :1] for output in class_benefit_outputs
                    ], dim=-1)
                    class_scale_logits = torch.cat([
                        output[..., 1:] for output in class_benefit_outputs
                    ], dim=-1)
                    if self.phase_benefit_mode in [
                        'family_stack', 'constrained_stack'
                    ]:
                        stable_indices = [0, 3, 4]
                        active_indices = [1, 2]
                        stable_candidate = (
                            endpoint_phase_probability[
                                ..., stable_indices].detach()
                            * phase_class_raw[..., stable_indices].detach()
                        ).sum(dim=-1, keepdim=True)
                        active_candidate = (
                            endpoint_phase_probability[
                                ..., active_indices].detach()
                            * phase_class_raw[..., active_indices].detach()
                        ).sum(dim=-1, keepdim=True)
                        stable_probability = endpoint_phase_probability[
                            ..., stable_indices].detach().sum(
                                dim=-1, keepdim=True)
                        active_family_probability = endpoint_phase_probability[
                            ..., active_indices].detach().sum(
                                dim=-1, keepdim=True)
                        family_context = torch.cat([
                            benefit_common_context,
                            stable_candidate / capacity,
                            active_candidate / capacity,
                            stable_probability,
                            active_family_probability,
                        ], dim=-1)
                        with torch.autocast(
                            device_type=family_context.device.type,
                            enabled=False,
                        ):
                            family_logits = self.phase_family_stack_head(
                                family_context.float())
                        stable_logit = family_logits[..., :1]
                        active_logit = family_logits[..., 1:]
                        class_scale_logits = torch.cat([
                            stable_logit,
                            active_logit,
                            active_logit,
                            stable_logit,
                            stable_logit,
                        ], dim=-1)
                        class_benefit_logits = torch.zeros_like(
                            class_scale_logits)
                        family_calibration = self.phase_family_stack_head.get_buffer(
                            'family_calibration').to(stable_logit)
                        stable_family_correction = (
                            family_calibration[0]
                            * torch.sigmoid(stable_logit) * stable_candidate)
                        active_family_correction = (
                            family_calibration[1]
                            * torch.sigmoid(active_logit) * active_candidate)
                    else:
                        stable_family_correction = None
                        active_family_correction = None
                    benefit_logit = class_benefit_logits.gather(
                        dim=-1, index=predicted_phase)
                    scale_logit = class_scale_logits.gather(
                        dim=-1, index=predicted_phase)
                    if self.phase_benefit_mode in [
                        'conditional_mean', 'family_stack',
                        'constrained_stack'
                    ]:
                        class_winner_probability = torch.ones_like(
                            class_benefit_logits)
                        class_physical_support = torch.ones_like(
                            endpoint_phase_probability)
                        route_class_support = (
                            endpoint_phase_probability.detach()
                            * self.phase_route_class_mask.to(
                                endpoint_phase_probability)
                        )
                    elif self.phase_benefit_mode == 'net_risk':
                        normalized_risk_scale = (
                            self.phase_risk_scale_kw / capacity) ** 2
                        route_class_support, class_physical_support = (
                            conditional_phase_risk_route_support(
                                path_support.detach(),
                                endpoint_phase_probability.detach(),
                                endpoint_class_path_delta.detach(),
                                class_benefit_logits,
                                normalized_risk_scale,
                                self.phase_route_class_mask,
                                float(getattr(
                                    self.args, 'expert_gate_threshold_kw', 5.0)),
                            )
                        )
                        # Keep the existing diagnostic field bounded while
                        # preserving zero risk as the abstention boundary.
                        class_winner_probability = torch.sigmoid(
                            -class_benefit_logits / normalized_risk_scale)
                    else:
                        class_winner_probability = torch.sigmoid(
                            class_benefit_logits)
                    winner_probability = class_winner_probability.gather(
                        dim=-1, index=predicted_phase)
                    class_winner_prior = self.phase_winner_prior_probability().to(
                        winner_probability).view(1, 1, 5)
                    if self.phase_benefit_mode == 'winner':
                        route_class_support, class_physical_support = (
                            conditional_phase_route_support(
                                path_support.detach(),
                                endpoint_phase_probability.detach(),
                                endpoint_class_path_delta.detach(),
                                class_winner_probability,
                                class_winner_prior,
                                self.phase_route_class_mask,
                                float(getattr(
                                    self.args, 'expert_gate_threshold_kw', 5.0)),
                            )
                        )
                    class_scale_probability = torch.sigmoid(class_scale_logits)
                    if self.phase_benefit_mode in [
                        'family_stack', 'constrained_stack'
                    ]:
                        family_calibration = self.phase_family_stack_head.get_buffer(
                            'family_calibration').to(class_scale_probability)
                        class_calibration = torch.stack([
                            family_calibration[0], family_calibration[1],
                            family_calibration[1], family_calibration[0],
                            family_calibration[0],
                        ]).view(1, 1, 5)
                        class_scale_probability = (
                            class_scale_probability * class_calibration)
                    scale_probability = class_scale_probability.gather(
                        dim=-1, index=predicted_phase)
                    class_benefit_strength = (
                        class_scale_probability * route_class_support)
                    benefit_strength = class_benefit_strength.sum(
                        dim=-1, keepdim=True)
                    route_support = route_class_support.sum(
                        dim=-1, keepdim=True)
                    if self.phase_benefit_mode in [
                        'net_risk', 'conditional_mean', 'family_stack',
                        'constrained_stack'
                    ]:
                        safety_class_strength = (
                            class_scale_probability * route_class_support)
                    else:
                        safety_class_strength = (
                            class_scale_probability
                            * class_winner_probability
                            * endpoint_phase_probability.detach()
                            * class_physical_support.detach()
                            * self.phase_route_class_mask.to(
                                class_scale_probability)
                        )
                    safety_strength = safety_class_strength.sum(
                        dim=-1, keepdim=True)
                    safety_correction = (
                        safety_class_strength * phase_class_raw.detach()
                    ).sum(dim=-1, keepdim=True)
                    endpoint_support = class_physical_support.gather(
                        dim=-1, index=predicted_phase)
                    route_enabled = bool(self.phase_route_enabled.item())
                    if not route_enabled:
                        benefit_strength = torch.zeros_like(benefit_strength)
                        # This branch is deliberately independent of every
                        # phase-field value, including any non-finite value in
                        # an untrained auxiliary head.  Epoch 0 and all
                        # pretraining stages therefore reproduce the frozen
                        # parent bit-for-bit.
                        phase_correction = torch.zeros_like(parent_pred)
                    else:
                        routed_class_strength = (
                            class_benefit_strength.detach()
                            if self.training else class_benefit_strength)
                        phase_correction = (
                            routed_class_strength * phase_class_raw
                        ).sum(dim=-1, keepdim=True)
                    self.last_phase_field = phase
                    self.last_phase_motion_consistency_loss = phase.get(
                        'motion_consistency_loss')
                    self.last_phase_hazard_logits = phase['hazard_logits']
                    self.last_phase_state_logits = phase['phase_logits']
                    self.last_phase_path_delta_kw = selected_path_delta
                    self.last_phase_class_path_delta_kw = phase[
                        'class_path_delta_kw']
                    self.last_phase_parent_pred = parent_pred
                    self.last_phase_class_raw_correction = phase_class_raw
                    self.last_phase_candidate_correction = candidate_correction
                    self.last_phase_benefit_logit = benefit_logit
                    self.last_phase_scale_logit = scale_logit
                    self.last_phase_class_benefit_logits = class_benefit_logits
                    self.last_phase_class_scale_logits = class_scale_logits
                    self.last_phase_benefit_strength = benefit_strength
                    self.last_phase_route_support = route_support
                    self.last_phase_class_route_support = route_class_support
                    self.last_phase_active_support = endpoint_active_support
                    self.last_phase_safety_strength = safety_strength
                    self.last_phase_safety_correction = safety_correction
                    self.last_phase_stable_family_correction = (
                        stable_family_correction)
                    self.last_phase_active_family_correction = (
                        active_family_correction)
                    self.last_phase_endpoint_magnitude_support = endpoint_support
                    self.last_clear_sky_kw = None
                    self.last_raw_event_correction = phase_correction
                    self.last_event_correction = phase_correction
                    self.last_stable_pred = parent_pred
                    self.last_residual_pred = phase_correction
                    self.last_ramp_expert_pred = parent_pred + phase_correction
                    self.last_expert_gate = benefit_strength
                    applied_correction = applied_correction + phase_correction
                    raw_applied_correction = (
                        raw_applied_correction + phase_correction)
                    output_phase_correction = (
                        phase_correction.detach()
                        if self.training else phase_correction
                    )
                    x = parent_pred + output_phase_correction
            else:
                self.last_expert_gate = None
                self.last_expert_gate_logit = None
                x = ramp_expert_pred
            self.last_applied_correction = applied_correction
            self.last_raw_applied_correction = raw_applied_correction
        else:
            self.last_residual_pred = None
            self.last_ramp_expert_pred = None
            self.last_applied_correction = None
            self.last_raw_applied_correction = None
            image_contribution = x_img_f_pred * sim_img_ts
            if used_data == 'causal_gated':
                historical_volatility = x_ts[:, 1:, :].sub(x_ts[:, :-1, :]).abs().mean(dim=1, keepdim=True)
                historical_volatility = historical_volatility / float(
                    getattr(self.args, 'stanford_capacity_kw', 30.1))
                gate_features = torch.cat([auxiliary_features, historical_volatility], dim=-1)
                self.last_expert_gate_logit = self.expert_gate(gate_features)
                self.last_expert_gate = torch.sigmoid(self.last_expert_gate_logit)
                x = x_ts_f_pred + self.last_expert_gate * image_contribution
            elif used_data in ['ts', 'ts_only']:
                x = x_ts_f_pred
            elif used_data == 'no_img':
                x = x_ts_f_pred + x_weather_f_pred * sim_weather_ts
            elif used_data in ['no_weather', 'causal_img', 'img']:
                x = x_ts_f_pred + image_contribution
            elif used_data == 'weather':
                x = x_ts_f_pred + x_weather_f_pred * sim_weather_ts
            else:
                x = x_ts_f_pred + image_contribution + x_weather_f_pred * sim_weather_ts
            if used_data != 'causal_gated':
                self.last_expert_gate = None
                self.last_expert_gate_logit = None

        feat = x.clone()
        if self.training:
            return x, sim_img_ts, sim_weather_ts, drop_img, drop_weather, feat
        return x, sim_img_ts, sim_weather_ts, feat
