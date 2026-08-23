from data_provider.data_factory import data_provider
from exp.exp_basic_kd import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, append_metrics_row
from utils.metrics import metric, stanford_sunny_cloudy_metrics, format_stanford_split_metrics
from utils.stanford_outputs import save_stanford_sample_predictions
from utils.luoyang_outputs import build_official_outputs
from utils.ylj_outputs import build_ylj_outputs
from utils.checkpoint_contract import write_checkpoint_contract, validate_checkpoint_contract
from utils.ylj_checkpoint_contract import write_ylj_checkpoint_contract, validate_ylj_checkpoint_contract
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import copy
import hashlib
import json
import shutil
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.trend_tag import patch_analyze_trends_gpu, calculate_accuracy_gpu
from utils.kd import relation_kd, response_kd, attention_kd, contrastive_kd, causal_kd, inter_causal_kd
from utils.img import temporal_contrastive_loss, enhanced_temporal_contrastive_loss
from utils.training_monitor import TrainingMonitor
from utils.stanford_phase import (
    ACTIVE_DOWN,
    ACTIVE_UP,
    build_stanford_path_targets,
    capacity_normalized_path_huber,
    discrete_hazard_survival_nll,
    macro_balanced_phase_loss,
)
warnings.filterwarnings('ignore')


def solve_phase_family_calibration(
        diagnostics, min_tail_improvement=0.0, grid_size=401):
    """Project two family scales onto validation non-degradation guards."""
    if grid_size < 2:
        raise ValueError('phase calibration grid must contain at least two points')
    target = diagnostics['target_kw'] - diagnostics['parent_kw']
    route = diagnostics['class_route_support']
    scale = diagnostics['class_scale_probability']
    raw = diagnostics['class_raw_correction_kw']
    stable_indices = [0, 3, 4]
    active_indices = [1, 2]
    stable = (route[:, stable_indices] * scale[:, stable_indices]
              * raw[:, stable_indices]).sum(axis=1)
    active = (route[:, active_indices] * scale[:, active_indices]
              * raw[:, active_indices]).sum(axis=1)
    ramp = np.abs(diagnostics['signed_ramp_kw'])
    cloud = diagnostics['cloud_event_day'].astype(bool)
    groups = {
        'ordinary': ramp < 5.0,
        'no_path': diagnostics['no_path'].astype(bool),
        'round_trip': diagnostics['round_trip'].astype(bool),
        'ge8': ramp >= 8.0,
        'cloud_ordinary': cloud & (ramp < 5.0),
        'cloud_no_path': cloud & diagnostics['no_path'].astype(bool),
        'cloud_tail5': cloud & (ramp >= 5.0),
        'cloud_ge8': cloud & (ramp >= 8.0),
    }
    empty = [name for name, mask in groups.items() if not bool(mask.any())]
    if empty:
        raise ValueError(
            'phase validation calibration has empty groups: ' + ','.join(empty))

    coefficients = {}
    for name, mask in groups.items():
        coefficients[name] = (
            float(np.mean(target[mask] ** 2)),
            float(np.mean(target[mask] * stable[mask])),
            float(np.mean(target[mask] * active[mask])),
            float(np.mean(stable[mask] ** 2)),
            float(np.mean(active[mask] ** 2)),
            float(np.mean(stable[mask] * active[mask])),
        )
    values = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
    stable_grid, active_grid = np.meshgrid(values, values, indexing='ij')
    rmses = {}
    baselines = {}
    for name, values_ in coefficients.items():
        base, ts, ta, ss, aa, sa = values_
        mse = (
            base - 2.0 * stable_grid * ts - 2.0 * active_grid * ta
            + stable_grid ** 2 * ss + active_grid ** 2 * aa
            + 2.0 * stable_grid * active_grid * sa)
        rmses[name] = np.sqrt(np.maximum(mse, 0.0))
        baselines[name] = float(np.sqrt(base))

    feasible = np.ones(stable_grid.shape, dtype=bool)
    for name in [
        'ordinary', 'no_path', 'round_trip', 'ge8',
        'cloud_ordinary', 'cloud_no_path', 'cloud_ge8',
    ]:
        feasible &= rmses[name] <= baselines[name] + 1e-10
    required_improvement = max(float(min_tail_improvement), 1e-7)
    feasible &= (
        rmses['cloud_tail5']
        <= baselines['cloud_tail5'] - required_improvement)
    if not bool(feasible.any()):
        return {
            'feasible': False,
            'stable_scale': 0.0,
            'active_scale': 0.0,
            'rmse_deltas': {name: 0.0 for name in groups},
        }
    tail_objective = np.where(feasible, rmses['cloud_tail5'], np.inf)
    best_tail = float(np.min(tail_objective))
    tail_optimal = feasible & (rmses['cloud_tail5'] <= best_tail + 1e-12)
    guard_objective = sum(
        rmses[name] / max(baselines[name], 1e-8)
        for name in [
            'ordinary', 'no_path', 'round_trip', 'ge8',
            'cloud_ordinary', 'cloud_no_path', 'cloud_ge8',
        ])
    objective = np.where(tail_optimal, guard_objective, np.inf)
    index = np.unravel_index(int(np.argmin(objective)), objective.shape)
    return {
        'feasible': True,
        'stable_scale': float(stable_grid[index]),
        'active_scale': float(active_grid[index]),
        'rmse_deltas': {
            name: float(rmses[name][index] - baselines[name])
            for name in groups
        },
    }


class ConservativeCheckpointGuard:
    """Track validation improvements relative to an immutable epoch-0 baseline."""

    def __init__(self, input_score, input_ordinary_rmse, patience,
                 min_improvement=0.0, ordinary_guard=False,
                 ordinary_relative_tolerance=0.0, input_no_path_rmse=None,
                 no_path_guard=False, no_path_relative_tolerance=0.0,
                 metric_guards=None):
        if not np.isfinite(input_score):
            raise ValueError(f'epoch-0 validation score is not finite: {input_score}')
        self.input_score = float(input_score)
        self.input_ordinary_rmse = float(input_ordinary_rmse)
        self.best_score = float(input_score)
        self.best_epoch = 0
        self.patience = int(patience)
        self.min_improvement = max(float(min_improvement), 0.0)
        self.ordinary_guard = bool(ordinary_guard)
        self.ordinary_relative_tolerance = max(float(ordinary_relative_tolerance), 0.0)
        if self.ordinary_guard and not np.isfinite(self.input_ordinary_rmse):
            raise ValueError(
                'ordinary validation guard requires a finite epoch-0 ordinary RMSE')
        self.input_no_path_rmse = (
            float(input_no_path_rmse)
            if input_no_path_rmse is not None else float('nan')
        )
        self.no_path_guard = bool(no_path_guard)
        self.no_path_relative_tolerance = max(
            float(no_path_relative_tolerance), 0.0)
        if self.no_path_guard and not np.isfinite(self.input_no_path_rmse):
            raise ValueError(
                'no-path validation guard requires a finite epoch-0 no-path RMSE')
        self.metric_guards = {}
        for name, raw_spec in (metric_guards or {}).items():
            spec = dict(raw_spec)
            baseline = float(spec['baseline'])
            mode = spec.get('mode', 'non_degrade')
            if not np.isfinite(baseline):
                raise ValueError(
                    f'{name} validation guard requires a finite epoch-0 metric')
            if mode not in ['non_degrade', 'improve']:
                raise ValueError(
                    f'{name} validation guard mode must be non_degrade or improve')
            self.metric_guards[name] = {
                'baseline': baseline,
                'mode': mode,
                'relative_tolerance': max(float(
                    spec.get('relative_tolerance', 0.0)), 0.0),
                'min_improvement': max(float(
                    spec.get('min_improvement', 0.0)), 0.0),
            }
        self.counter = 0
        self.early_stop = False

    @property
    def ordinary_limit(self):
        if not self.ordinary_guard:
            return None
        return self.input_ordinary_rmse * (1.0 + self.ordinary_relative_tolerance)

    @property
    def no_path_limit(self):
        if not self.no_path_guard:
            return None
        return self.input_no_path_rmse * (1.0 + self.no_path_relative_tolerance)

    def metric_guard_results(self, metrics):
        results = {}
        metrics = metrics or {}
        for name, spec in self.metric_guards.items():
            value = float(metrics.get(name, float('nan')))
            if spec['mode'] == 'improve':
                limit = spec['baseline'] - spec['min_improvement']
                passed = np.isfinite(value) and value < limit
            else:
                limit = spec['baseline'] * (
                    1.0 + spec['relative_tolerance'])
                passed = np.isfinite(value) and value <= limit
            results[name] = {
                'value': value,
                'limit': limit,
                'passed': bool(passed),
                'mode': spec['mode'],
            }
        return results

    def consider(self, epoch, score, ordinary_rmse, no_path_rmse=None,
                 guard_metrics=None):
        score = float(score)
        ordinary_rmse = float(ordinary_rmse)
        score_finite = np.isfinite(score)
        ordinary_ok = (
            not self.ordinary_guard
            or (
                np.isfinite(ordinary_rmse)
                and np.isfinite(self.input_ordinary_rmse)
                and ordinary_rmse <= self.ordinary_limit
            )
        )
        no_path_rmse = (
            float(no_path_rmse)
            if no_path_rmse is not None else float('nan')
        )
        no_path_ok = (
            not self.no_path_guard
            or (
                np.isfinite(no_path_rmse)
                and no_path_rmse <= self.no_path_limit
            )
        )
        metric_guard_results = self.metric_guard_results(guard_metrics)
        metric_guards_ok = all(
            result['passed'] for result in metric_guard_results.values())
        improved = score_finite and score < self.best_score - self.min_improvement
        accepted = (
            improved and ordinary_ok and no_path_ok and metric_guards_ok)
        if accepted:
            self.best_score = score
            self.best_epoch = int(epoch)
            self.counter = 0
            self.early_stop = False
            reason = 'accepted_improvement'
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience
            if not score_finite:
                reason = 'non_finite_selection_score'
            elif not improved:
                reason = 'no_validation_improvement'
            elif not ordinary_ok:
                reason = 'ordinary_guard_failed'
            elif not no_path_ok:
                reason = 'no_path_guard_failed'
            else:
                failed_name = next(
                    name for name, result in metric_guard_results.items()
                    if not result['passed'])
                reason = f'{failed_name}_guard_failed'
        return {
            'accepted': accepted,
            'improved': improved,
            'ordinary_ok': ordinary_ok,
            'no_path_ok': no_path_ok,
            'metric_guards_ok': metric_guards_ok,
            'metric_guard_results': metric_guard_results,
            'reason': reason,
        }

    def reset_patience(self):
        """Start a fresh rejection window without changing the best model."""
        self.counter = 0
        self.early_stop = False


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args, args_img, args_weather):
        self.args_img = args_img
        self.args_weather = args_weather
        self.loss_type = args.loss_type
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
        self.device = self._acquire_device()
        
        # 构建模型
        self.model, self.model_teacher = self._build_model()
        self.model.to(self.device)
        if self.model_teacher is not None:
            self.model_teacher.to(self.device)
        self.monitor = None
        self._last_validation_monitor = None
        self._monitor_teacher_reference = None
        self._monitor_quality_recorded = set()

        # 添加投影层
        # self.projection_layer = nn.Linear(args_img.c_out + args_weather.c_out, args.c_out).to(self.device)

    def _build_model(self):
        model = self.model_dict[self.args_img.model].Model(self.args, self.args_img, self.args_weather).float()
        if getattr(self.args, 'student_init_path', ''):
            if self.args.data == 'LuoyangParquet':
                validate_checkpoint_contract(self.args.student_init_path, self.args.luoyang_config)
            elif self.args.data == 'YLJParquet':
                validate_ylj_checkpoint_contract(self.args.student_init_path, self.args.ylj_config)
            self._load_student_initialization(model, self.args.student_init_path)
        if getattr(self.args, 'student_only', False):
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            return model, None
        teacher_args = copy.copy(self.args)
        teacher_image_encoder_type = getattr(self.args, 'teacher_image_encoder_type', None)
        if teacher_image_encoder_type:
            teacher_args.image_encoder_type = teacher_image_encoder_type
        teacher_args_img = copy.copy(self.args_img)
        if getattr(self.args, 'teacher_legacy_image_order', False):
            teacher_args_img.history_order = 'past_first'
        if getattr(self.args, 'image_encoder_type', '') == 'cnn_solar_advection':
            teacher_args_img.stanford_image_mode = 'gray'
        model_teacher = self.model_dict[self.args.teacher_model].Model(
            teacher_args, teacher_args_img, self.args_weather).float()

        if not os.path.exists(self.args.teacher_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.args.teacher_path}")
        if self.args.data == 'LuoyangParquet':
            validate_checkpoint_contract(self.args.teacher_path, self.args.luoyang_config)
        elif self.args.data == 'YLJParquet':
            validate_ylj_checkpoint_contract(self.args.teacher_path, self.args.ylj_config)
        teacher_state = torch.load(self.args.teacher_path, map_location='cpu')
        if any(key.startswith('module.') for key in teacher_state.keys()):
            teacher_state = {key.replace('module.', '', 1): value for key, value in teacher_state.items()}
        model_teacher.load_state_dict(teacher_state)


        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            model_teacher = nn.DataParallel(model_teacher, device_ids=self.args.device_ids)

        return model, model_teacher

    @staticmethod
    def _load_student_initialization(model, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Student initialization checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location='cpu')
        if any(key.startswith('module.') for key in state):
            state = {key.replace('module.', '', 1): value for key, value in state.items()}

        model_state = model.state_dict()
        adapted = {}
        for key, value in state.items():
            if key in model_state and model_state[key].shape == value.shape:
                adapted[key] = value
            if key.startswith('branch_img.') and getattr(model, 'dual_image_encoder', False):
                suffix = key[len('branch_img.'):]
                for branch in ['appearance', 'motion']:
                    mapped = f'branch_img.{branch}.{suffix}'
                    if mapped in model_state and model_state[mapped].shape == value.shape:
                        adapted[mapped] = value

        incompatible = model.load_state_dict(adapted, strict=False)
        print(
            f"student_init={checkpoint_path}, loaded_tensors={len(adapted)}, "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}")

    def _set_ts_branch_trainable(self, trainable):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        branch = getattr(model, 'branch_ts', None)
        if branch is None:
            return
        for parameter in branch.parameters():
            parameter.requires_grad = trainable
        branch.train(trainable)

    def _ts_must_stay_frozen(self):
        return bool(
            getattr(self.args, 'freeze_ts_permanently', False)
            or getattr(self.args, 'train_image_residual_only', False)
        )

    def _apply_student_training_policy(self):
        model = self._student_model()
        if getattr(self.args, 'train_image_residual_only', False):
            strategy = self._student_strategy()
            if strategy not in [
                'causal_residual_appearance', 'causal_residual_motion',
                'causal_soft_gate', 'causal_solar_event',
                'causal_solar_token_tail', 'causal_phase_field_tail'
            ]:
                raise ValueError(
                    'train_image_residual_only requires a causal residual image strategy')
            if not getattr(model, 'dual_image_encoder', False):
                raise ValueError(
                    'train_image_residual_only requires a dual causal image encoder')
            if strategy == 'causal_solar_event':
                allowed_prefixes = [
                    'solar_event_encoder.',
                    'solar_color_projection.',
                    'event_state_head.',
                    'event_up_correction_head.',
                    'event_down_correction_head.',
                    'spatial_kd_projection.',
                ]
            elif strategy == 'causal_solar_token_tail':
                if float(getattr(
                    self.args, 'event_residual_aux_weight', 0.0)) <= 0:
                    raise ValueError(
                        'causal_solar_token_tail requires a positive '
                        'event_residual_aux_weight to start zero-initialized '
                        'up/down heads')
                allowed_prefixes = [
                    'solar_token_encoder.',
                    'tail_event_head.',
                    'tail_direction_head.',
                    'tail_up_head.',
                    'tail_down_head.',
                    'tail_prior_',
                ]
            elif strategy == 'causal_phase_field_tail':
                if float(getattr(
                    self.args, 'event_residual_aux_weight', 0.0)) <= 0:
                    raise ValueError(
                        'causal_phase_field_tail requires a positive '
                        'event_residual_aux_weight for zero-initialized heads')
                if not bool(getattr(
                    self.args, 'stanford_return_phase_path', False)):
                    raise ValueError(
                        'causal_phase_field_tail requires '
                        '--stanford_return_phase_path')
                allowed_prefixes = ['phase_']
            else:
                allowed_prefixes = [
                    'branch_img.appearance.',
                    'appearance_correction_head.',
                ]
            if strategy in ['causal_residual_motion', 'causal_soft_gate']:
                allowed_prefixes.append('branch_img.motion.')
                allowed_prefixes.append('motion_correction_head.')
            if strategy == 'causal_soft_gate':
                allowed_prefixes.append('soft_expert_gate.')
            for name, parameter in model.named_parameters():
                parameter.requires_grad = any(
                    name.startswith(prefix) for prefix in allowed_prefixes)
            self._trainable_prefixes = tuple(allowed_prefixes)
        elif getattr(self.args, 'freeze_ts_permanently', False):
            self._set_ts_branch_trainable(False)
            self._trainable_prefixes = tuple()

        if self._ts_must_stay_frozen():
            self._set_ts_branch_trainable(False)

        self._trainable_parameter_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        trainable_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f'student_trainable_parameters={trainable_count}/{total_count} '
            f'train_image_residual_only={getattr(self.args, "train_image_residual_only", False)}')
        if not self._trainable_parameter_names:
            raise ValueError('student training policy left no trainable parameters')

    def _enforce_frozen_ts_mode(self):
        if self._ts_must_stay_frozen():
            self._set_ts_branch_trainable(False)
        if (
            getattr(self.args, 'train_image_residual_only', False)
            and self._student_strategy() in [
                'causal_residual_appearance', 'causal_solar_event',
                'causal_solar_token_tail', 'causal_phase_field_tail']
        ):
            branch_img = getattr(self._student_model(), 'branch_img', None)
            motion_branch = getattr(branch_img, 'motion', None)
            if motion_branch is not None:
                motion_branch.eval()
            if self._student_strategy() in [
                'causal_solar_event', 'causal_solar_token_tail'
                , 'causal_phase_field_tail'
            ]:
                appearance_branch = getattr(branch_img, 'appearance', None)
                if appearance_branch is not None:
                    appearance_branch.eval()
        if self._student_strategy() in [
            'causal_solar_token_tail', 'causal_phase_field_tail'
        ]:
            model = self._student_model()
            for name in [
                'solar_event_encoder',
                'solar_color_projection',
                'event_state_head',
                'event_up_correction_head',
                'event_down_correction_head',
                'appearance_correction_head',
                'motion_correction_head',
            ]:
                module = getattr(model, name, None)
                if module is not None:
                    module.eval()

    def _set_tail_magnitude_trainable(self, trainable):
        if self._student_strategy() != 'causal_solar_token_tail':
            return
        model = self._student_model()
        for name, parameter in model.named_parameters():
            if name.startswith(('tail_up_head.', 'tail_down_head.')):
                parameter.requires_grad = bool(trainable)

    def _set_phase_family_calibration(self, stable_scale, active_scale):
        if self._student_strategy() != 'causal_phase_field_tail':
            return
        head = getattr(
            self._student_model(), 'phase_family_stack_head', None)
        calibration = getattr(head, 'family_calibration', None)
        if calibration is None:
            raise RuntimeError('phase family stack is missing calibration state')
        values = calibration.new_tensor([stable_scale, active_scale])
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError('phase family calibration must be finite')
        if bool(((values < 0) | (values > 1)).any().item()):
            raise ValueError('phase family calibration must be within [0, 1]')
        calibration.copy_(values)

    def _calibrate_phase_families(self, diagnostic_path):
        with np.load(diagnostic_path) as diagnostics:
            result = solve_phase_family_calibration(
                diagnostics,
                min_tail_improvement=float(getattr(
                    self.args, 'val_tail_min_improvement', 0.0)),
            )
        self._set_phase_family_calibration(
            result['stable_scale'], result['active_scale'])
        return result

    def _set_phase_training_stage(self, stage):
        if self._student_strategy() != 'causal_phase_field_tail':
            return
        if stage not in ['state', 'magnitude', 'benefit', 'active']:
            raise ValueError(f'unsupported phase training stage: {stage}')
        model = self._student_model()
        for name, parameter in model.named_parameters():
            if name.startswith('phase_field_encoder.'):
                # The shared field is learned only from multi-lead state/path
                # labels.  Freezing it afterwards keeps the occurrence
                # representation separate from conditional tail magnitude.
                parameter.requires_grad = stage == 'state'
            elif name.startswith('phase_class_correction_heads.'):
                parameter.requires_grad = stage == 'magnitude'
            elif name.startswith('phase_class_benefit_heads.'):
                parameter.requires_grad = stage in ['benefit', 'active']
            elif name.startswith('phase_family_stack_head.'):
                parameter.requires_grad = stage in ['benefit', 'active']
        model.set_phase_route_enabled(stage == 'active')
        model.phase_field_encoder.train(stage == 'state')
        model.phase_class_correction_heads.train(stage == 'magnitude')
        model.phase_class_benefit_heads.train(stage in ['benefit', 'active'])
        model.phase_family_stack_head.train(stage in ['benefit', 'active'])
        self._phase_training_stage = stage

    def _phase_stage_for_epoch(self, epoch):
        durations = self._phase_stage_durations()
        state_epochs = durations['state']
        magnitude_epochs = durations['magnitude']
        benefit_epochs = durations['benefit']
        if epoch < state_epochs:
            return 'state'
        if epoch < state_epochs + magnitude_epochs:
            return 'magnitude'
        if epoch < state_epochs + magnitude_epochs + benefit_epochs:
            return 'benefit'
        return 'active'

    def _phase_stage_durations(self):
        durations = {
            'state': int(getattr(
                self.args, 'phase_state_pretrain_epochs', 0)),
            'magnitude': int(getattr(
                self.args, 'phase_magnitude_pretrain_epochs', 0)),
            'benefit': int(getattr(
                self.args, 'phase_benefit_pretrain_epochs', 0)),
        }
        if min(durations.values()) < 0:
            raise ValueError('phase pretraining epoch counts must be non-negative')
        resume_from = getattr(self.args, 'phase_resume_from', 'none')
        order = ['state', 'magnitude', 'benefit']
        if resume_from not in ['none'] + order:
            raise ValueError(f'unsupported phase resume stage: {resume_from}')
        if resume_from != 'none':
            completed = order.index(resume_from)
            for stage in order[:completed + 1]:
                durations[stage] = 0
        return durations

    @staticmethod
    def _phase_pretrain_selection_score(stage, validation_loss, metrics):
        if stage != 'state':
            return float(validation_loss)
        representation_metrics = np.asarray([
            metrics.get('phase_hazard_event_auprc', np.nan),
            metrics.get('phase_macro_accuracy', np.nan),
        ], dtype=np.float64)
        if not np.isfinite(representation_metrics).all():
            return float('inf')
        # State pretraining exists to learn occurrence and phase
        # representations while routing is disabled.  Parent task loss is
        # constant in this stage and reconstruction loss is only an auxiliary
        # regularizer, so neither should decide the representation checkpoint.
        return float(1.0 - representation_metrics.mean())

    def _ts_branch_digest(self):
        branch = getattr(self._student_model(), 'branch_ts', None)
        if branch is None:
            return None
        digest = hashlib.sha256()
        for name, tensor in branch.state_dict().items():
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode('utf-8'))
            digest.update(str(value.dtype).encode('ascii'))
            digest.update(str(tuple(value.shape)).encode('ascii'))
            digest.update(value.reshape(-1).view(torch.uint8).numpy())
        return digest.hexdigest()

    def _frozen_parent_digest(self):
        """Hash every state tensor outside the explicitly trainable child."""
        model = self._student_model()
        trainable_prefixes = tuple(getattr(self, '_trainable_prefixes', ()))
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            if any(name.startswith(prefix) for prefix in trainable_prefixes):
                continue
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode('utf-8'))
            digest.update(str(value.dtype).encode('ascii'))
            digest.update(str(tuple(value.shape)).encode('ascii'))
            digest.update(value.reshape(-1).view(torch.uint8).numpy())
        return digest.hexdigest()

    def _get_data(self, flag):
        # if self.args.data == 'Folsom':
            # data_provider = data_provider_fast
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError('no trainable student parameters were selected')
        model_optim = optim.Adam(parameters, lr=self.args.learning_rate)
        return model_optim

    @staticmethod
    def _optimizer_parameters(optimizer):
        return [
            parameter
            for group in optimizer.param_groups
            for parameter in group['params']
        ]

    def _select_criterion(self):
        if self.loss_type == 'MSE':
            criterion = nn.MSELoss()
        elif self.loss_type == 'Huber':
            criterion = nn.HuberLoss()
        return criterion

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 10:
            return batch
        if len(batch) == 9:
            return (*batch, None)
        if len(batch) == 8:
            return (*batch, None, None)
        raise ValueError(f'unsupported batch tuple length: {len(batch)}')

    def _phase_targets_to_device(self, phase_targets):
        if phase_targets is None:
            return None
        return {
            name: (
                value.to(self.device, non_blocking=True)
                if torch.is_tensor(value) else value
            )
            for name, value in phase_targets.items()
        }

    def _criterion_loss(self, outputs, target, criterion, sample_weight=None,
                        target_mask=None):
        if sample_weight is None and target_mask is None:
            return criterion(outputs, target)
        if isinstance(criterion, nn.HuberLoss):
            per_sample_loss = torch.nn.functional.huber_loss(outputs, target, reduction='none')
        else:
            per_sample_loss = (outputs - target).pow(2)
        if target_mask is not None:
            mask = target_mask.to(per_sample_loss.device, dtype=per_sample_loss.dtype)
            if mask.dim() == per_sample_loss.dim() - 1:
                mask = mask.unsqueeze(-1)
            weights = mask.expand_as(per_sample_loss)
            if sample_weight is not None:
                weights = weights * sample_weight.to(per_sample_loss).view(-1, 1, 1)
            return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1.0)
        per_sample_loss = per_sample_loss.flatten(start_dim=1).mean(dim=1)
        weights = sample_weight.float().to(per_sample_loss.device).view(-1)
        return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _reduce_sample_loss(per_sample_loss, sample_weight):
        if sample_weight is None:
            return per_sample_loss.mean()
        weights = sample_weight.float().to(per_sample_loss.device).view(-1)
        return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    @staticmethod
    def _reduce_masked_sample_loss(per_sample_loss, mask, sample_weight=None):
        mask = mask.to(per_sample_loss.device, dtype=per_sample_loss.dtype).view(-1)
        weights = mask
        if sample_weight is not None:
            weights = weights * sample_weight.float().to(per_sample_loss.device).view(-1)
        denominator = weights.sum()
        if denominator.item() == 0:
            return per_sample_loss.sum() * 0.0
        return (per_sample_loss * weights).sum() / denominator

    @classmethod
    def _balanced_binary_loss(cls, per_sample_loss, positive_mask,
                              sample_weight=None, eligible_mask=None):
        """Give each present class equal mass, independent of event frequency."""
        positive_mask = positive_mask.to(dtype=torch.bool).view(-1)
        if eligible_mask is None:
            eligible_mask = torch.ones_like(positive_mask)
        else:
            eligible_mask = eligible_mask.to(dtype=torch.bool).view(-1)
        class_masks = [
            eligible_mask & positive_mask,
            eligible_mask & ~positive_mask,
        ]
        class_losses = [
            cls._reduce_masked_sample_loss(per_sample_loss, mask, sample_weight)
            for mask in class_masks if bool(mask.any().item())
        ]
        if not class_losses:
            return per_sample_loss.sum() * 0.0
        return torch.stack(class_losses).mean()

    def _tail_supervision_targets(self, target, current_pv):
        model = self._student_model()
        parent_pred = getattr(model, 'last_tail_parent_pred', None)
        if parent_pred is None:
            raise RuntimeError(
                'tail supervision requires the frozen parent prediction')
        ramp_threshold = float(getattr(
            self.args, 'expert_gate_threshold_kw', 5.0))
        error_threshold = float(getattr(
            self.args, 'tail_error_threshold_kw', 3.0))
        correction_target = target - parent_pred.detach()
        ramp_event = (
            (target - current_pv).abs().flatten(start_dim=1).max(dim=1).values
            >= ramp_threshold
        )
        parent_needs_tail = (
            correction_target.abs().flatten(start_dim=1).max(dim=1).values
            >= error_threshold
        )
        return ramp_event & parent_needs_tail, correction_target

    def _phase_auxiliary_loss(self, target, current_pv, phase_targets,
                              sample_weight=None):
        if phase_targets is None:
            raise RuntimeError('phase-field training requires future PV path labels')
        model = self._student_model()
        future_path = phase_targets['pv_path'].to(target)
        valid_mask = phase_targets['valid_mask'].to(dtype=torch.bool)
        signed_path = future_path - current_pv
        path_targets = build_stanford_path_targets(signed_path, valid_mask)
        episode_weight = phase_targets.get('episode_weight')
        combined_weight = sample_weight
        if episode_weight is not None:
            episode_weight = episode_weight.to(target).view(-1)
            combined_weight = (
                episode_weight if sample_weight is None
                else episode_weight * sample_weight.to(target).view(-1)
            )

        components = {}
        total = torch.zeros((), device=target.device)
        stage = getattr(self, '_phase_training_stage', 'active')
        motion_weight = float(getattr(
            self.args, 'phase_motion_consistency_weight', 0.0))
        motion_consistency = getattr(
            model, 'last_phase_motion_consistency_loss', None)
        if stage == 'state' and motion_weight > 0:
            if motion_consistency is None:
                raise RuntimeError(
                    'phase motion consistency requires historical flow fields')
            if motion_consistency.dim() != 0:
                raise RuntimeError(
                    'phase motion consistency loss must be a scalar')
            component = motion_weight * motion_consistency
            components['phase_motion_consistency'] = component
            total = total + component
        state_weight = float(getattr(
            self.args, 'phase_state_aux_weight', 0.0))
        hazard_logits = getattr(model, 'last_phase_hazard_logits', None)
        phase_logits = getattr(model, 'last_phase_state_logits', None)
        if stage == 'state' and state_weight > 0:
            if hazard_logits is None or phase_logits is None:
                raise RuntimeError('phase state loss requires hazard/phase logits')
            hazard_per_sample = discrete_hazard_survival_nll(
                hazard_logits,
                path_targets.first_crossing_bin,
                path_targets.hazard_at_risk_mask,
                reduction='none',
            )
            observed_path = path_targets.hazard_at_risk_mask.any(dim=1)
            event_path = path_targets.first_crossing_bin >= 0
            hazard_loss = self._balanced_binary_loss(
                hazard_per_sample,
                event_path,
                combined_weight,
                observed_path,
            )
            phase_element_weight = None
            if combined_weight is not None:
                phase_element_weight = combined_weight.view(-1, 1).expand_as(
                    path_targets.phase)
            phase_loss = macro_balanced_phase_loss(
                phase_logits,
                path_targets.phase,
                valid_mask=path_targets.lead_valid_mask,
                element_weight=phase_element_weight,
            )
            endpoint_phase_per = torch.nn.functional.cross_entropy(
                phase_logits[:, -1],
                path_targets.phase[:, -1],
                ignore_index=-100,
                reduction='none',
            )
            endpoint_phase_loss = self._reduce_masked_sample_loss(
                endpoint_phase_per,
                path_targets.lead_valid_mask[:, -1],
                combined_weight,
            )
            phase_loss = 0.5 * (phase_loss + endpoint_phase_loss)
            component = state_weight * 0.5 * (hazard_loss + phase_loss)
            components['phase_state'] = component
            total = total + component

        path_weight = float(getattr(
            self.args, 'phase_path_aux_weight', 0.0))
        predicted_path = getattr(model, 'last_phase_path_delta_kw', None)
        class_predicted_path = getattr(
            model, 'last_phase_class_path_delta_kw', None)
        if stage == 'state' and path_weight > 0:
            if predicted_path is None or class_predicted_path is None:
                raise RuntimeError(
                    'phase path loss requires class-conditional fixed-lead '
                    'predictions')
            capacity = float(getattr(
                self.args, 'stanford_capacity_kw', 30.1))
            endpoint_target = path_targets.lead_values[:, -1]
            endpoint_valid = path_targets.lead_valid_mask[:, -1]
            endpoint_phase = path_targets.phase[:, -1]
            boundary = 5.0
            class_path_losses = []
            for phase_value in range(5):
                class_path = class_predicted_path[:, :, phase_value, :]
                class_mask = (
                    path_targets.lead_valid_mask
                    & (path_targets.phase == phase_value))
                if not bool(class_mask.any().item()):
                    continue
                path_element_weight = None
                if combined_weight is not None:
                    path_element_weight = combined_weight.view(
                        -1, 1, 1).expand_as(class_path)
                class_path_loss = capacity_normalized_path_huber(
                    class_path,
                    path_targets.lead_values.unsqueeze(-1),
                    class_mask.unsqueeze(-1),
                    capacity_kw=capacity,
                    element_weight=path_element_weight,
                )
                endpoint_pred = class_path[:, -1, 0]
                endpoint_mask = endpoint_valid & (
                    endpoint_phase == phase_value)
                endpoint_regression_per = torch.nn.functional.smooth_l1_loss(
                    endpoint_pred / capacity,
                    endpoint_target / capacity,
                    reduction='none',
                    beta=0.1,
                )
                if phase_value == ACTIVE_DOWN:
                    endpoint_boundary_per = torch.relu(
                        endpoint_pred + boundary) / capacity
                elif phase_value == ACTIVE_UP:
                    endpoint_boundary_per = torch.relu(
                        boundary - endpoint_pred) / capacity
                else:
                    endpoint_boundary_per = torch.relu(
                        endpoint_pred.abs() - boundary) / capacity
                endpoint_regression = self._reduce_masked_sample_loss(
                    endpoint_regression_per, endpoint_mask, combined_weight)
                endpoint_consistency = self._reduce_masked_sample_loss(
                    endpoint_boundary_per, endpoint_mask, combined_weight)
                class_path_losses.append(0.5 * (
                    class_path_loss
                    + 0.5 * (endpoint_regression + endpoint_consistency)))
            path_loss = (
                torch.stack(class_path_losses).mean()
                if class_path_losses
                else class_predicted_path.sum() * 0.0)
            component = path_weight * path_loss
            components['phase_path'] = component
            total = total + component

        parent_pred = getattr(model, 'last_phase_parent_pred', None)
        class_raw = getattr(model, 'last_phase_class_raw_correction', None)
        correction_target = (
            target - parent_pred.detach() if parent_pred is not None else None)
        residual_weight = float(getattr(
            self.args, 'event_residual_aux_weight', 0.0))
        if stage == 'magnitude' and residual_weight > 0:
            if parent_pred is None or class_raw is None:
                raise RuntimeError(
                    'phase magnitude loss requires parent and phase-class heads')
            magnitude_tensors = {
                'target': target,
                'parent_pred': parent_pred,
                'class_raw': class_raw,
            }
            non_finite = [
                name for name, value in magnitude_tensors.items()
                if not bool(torch.isfinite(value).all().item())
            ]
            if non_finite:
                raise FloatingPointError(
                    'non-finite phase magnitude tensors: ' + ','.join(non_finite))
            error_threshold = float(getattr(
                self.args, 'tail_error_threshold_kw', 3.0))
            capacity = float(getattr(
                self.args, 'stanford_capacity_kw', 30.1))
            normalized_target = correction_target / capacity
            endpoint_phase = path_targets.phase[:, -1]
            endpoint_valid = path_targets.lead_valid_mask[:, -1]
            correction_needed = (
                correction_target.abs().flatten(start_dim=1).amax(dim=1)
                >= error_threshold)
            class_losses = []
            stable_expert_all_samples = bool(int(getattr(
                self.args, 'phase_stable_expert_all_samples', 0)))
            specialist_abstention = int(getattr(
                self.args, 'phase_specialist_abstention', 0))
            for class_index, phase_value in enumerate(range(5)):
                target_per = torch.nn.functional.smooth_l1_loss(
                    class_raw[..., class_index:class_index + 1] / capacity,
                    normalized_target,
                    reduction='none',
                    beta=0.1,
                ).flatten(start_dim=1).mean(dim=1)
                target_mask = endpoint_valid & (endpoint_phase == phase_value)
                if not (
                    stable_expert_all_samples and class_index in [0, 3, 4]
                ):
                    target_mask = target_mask & correction_needed
                if bool(target_mask.any().item()):
                    target_loss = self._reduce_masked_sample_loss(
                        target_per, target_mask, combined_weight)
                    if specialist_abstention > 0:
                        normalized_class_output = (
                            class_raw[..., class_index:class_index + 1]
                            / capacity)
                        use_sparse_abstention = (
                            specialist_abstention == 2
                            or (
                                specialist_abstention == 3
                                and class_index in [1, 2]
                            )
                        )
                        if use_sparse_abstention:
                            zero_per = normalized_class_output.abs().flatten(
                                start_dim=1).mean(dim=1)
                        else:
                            zero_per = torch.nn.functional.smooth_l1_loss(
                                normalized_class_output,
                                torch.zeros_like(normalized_target),
                                reduction='none', beta=0.1,
                            ).flatten(start_dim=1).mean(dim=1)
                        abstention_terms = []
                        for other_phase in range(5):
                            if other_phase == phase_value:
                                continue
                            other_mask = (
                                endpoint_valid
                                & (endpoint_phase == other_phase))
                            if bool(other_mask.any().item()):
                                abstention_terms.append(
                                    self._reduce_masked_sample_loss(
                                        zero_per, other_mask,
                                        combined_weight))
                        if abstention_terms:
                            target_loss = 0.5 * (
                                target_loss
                                + torch.stack(abstention_terms).mean())
                    class_losses.append(target_loss)
            magnitude_loss = (
                torch.stack(class_losses).mean()
                if class_losses else class_raw.sum() * 0.0)
            component = residual_weight * magnitude_loss
            components['phase_magnitude'] = component
            total = total + component

        benefit_weight = float(getattr(
            self.args, 'phase_benefit_aux_weight', 0.0))
        benefit_logit = getattr(model, 'last_phase_benefit_logit', None)
        scale_logit = getattr(model, 'last_phase_scale_logit', None)
        class_benefit_logits = getattr(
            model, 'last_phase_class_benefit_logits', None)
        class_scale_logits = getattr(
            model, 'last_phase_class_scale_logits', None)
        candidate = getattr(model, 'last_phase_candidate_correction', None)
        route_support = getattr(model, 'last_phase_route_support', None)
        if stage in ['benefit', 'active'] and benefit_weight > 0:
            if any(value is None for value in [
                benefit_logit, scale_logit, class_benefit_logits,
                class_scale_logits, class_raw, candidate, parent_pred,
                route_support
            ]):
                raise RuntimeError('phase benefit loss requires a frozen candidate')
            candidate = candidate.detach()
            class_raw = class_raw.detach()
            correction_target = target - parent_pred.detach()
            endpoint_phase = path_targets.phase[:, -1]
            endpoint_valid = path_targets.lead_valid_mask[:, -1]
            parent_error = correction_target.square().flatten(
                start_dim=1).mean(dim=1)
            class_losses = []
            benefit_mode = str(getattr(
                self.args, 'phase_benefit_mode', 'winner'))
            if benefit_mode not in [
                'winner', 'net_risk', 'conditional_mean', 'family_stack',
                'constrained_stack'
            ]:
                raise ValueError(
                    f'unsupported phase benefit mode: {benefit_mode}')
            winner_matrix = torch.zeros_like(class_benefit_logits, dtype=torch.bool)
            eligible_matrix = torch.zeros_like(class_benefit_logits, dtype=torch.bool)
            for class_index in range(5):
                class_candidate = class_raw[..., class_index:class_index + 1]
                optimal_scale = (
                    class_candidate * correction_target
                    / class_candidate.square().clamp_min(1e-4)
                ).clamp(0.0, 1.0)
                oracle_error = (
                    optimal_scale * class_candidate - correction_target
                ).square().flatten(start_dim=1).mean(dim=1)
                analytic_winner = oracle_error < parent_error
                # The winner head answers whether this candidate is useful on
                # the current sample, including phase-classification mistakes.
                # Restricting it to true class members made active priors ~0.98
                # and suppressed every deployable active/return correction.
                benefit_mask = endpoint_valid
                if benefit_mode in [
                    'conditional_mean', 'family_stack', 'constrained_stack'
                ]:
                    winner_matrix[..., class_index] = (
                        benefit_mask & analytic_winner).view(-1, 1)
                    eligible_matrix[..., class_index] = benefit_mask.view(-1, 1)
                    continue
                terms = []
                if benefit_mode == 'net_risk':
                    full_candidate_error = (
                        class_candidate - correction_target
                    ).square().flatten(start_dim=1).mean(dim=1)
                    capacity = float(getattr(
                        self.args, 'stanford_capacity_kw', 30.1))
                    excess_risk_target = (
                        full_candidate_error - parent_error
                    ) / (capacity ** 2)
                    excess_risk_pred = class_benefit_logits[
                        ..., class_index:class_index + 1
                    ].flatten(start_dim=1).mean(dim=1)
                    risk_per = torch.nn.functional.mse_loss(
                        excess_risk_pred,
                        excess_risk_target.detach(),
                        reduction='none',
                    )
                    terms.append(self._reduce_masked_sample_loss(
                        risk_per, benefit_mask, combined_weight))
                else:
                    winner_target = analytic_winner.to(
                        class_benefit_logits.dtype).view(-1, 1, 1)
                    winner_per = (
                        torch.nn.functional.binary_cross_entropy_with_logits(
                            class_benefit_logits[
                                ..., class_index:class_index + 1],
                            winner_target,
                            reduction='none',
                        ).flatten(start_dim=1).mean(dim=1)
                    )
                    winner_regime_losses = []
                    for phase_value in range(5):
                        regime_mask = (
                            benefit_mask & (endpoint_phase == phase_value))
                        if bool(regime_mask.any().item()):
                            winner_regime_losses.append(
                                self._balanced_binary_loss(
                                    winner_per, analytic_winner,
                                    combined_weight, regime_mask))
                    if winner_regime_losses:
                        terms.append(torch.stack(winner_regime_losses).mean())
                scale_per = torch.nn.functional.smooth_l1_loss(
                    torch.sigmoid(
                        class_scale_logits[..., class_index:class_index + 1]),
                    optimal_scale,
                    reduction='none',
                    beta=0.1,
                ).flatten(start_dim=1).mean(dim=1)
                scale_mask = (
                    benefit_mask
                    if benefit_mode == 'net_risk'
                    else benefit_mask & analytic_winner)
                scale_regime_losses = []
                if benefit_mode == 'net_risk':
                    scale_regime_losses.append(
                        self._reduce_masked_sample_loss(
                            scale_per, scale_mask, combined_weight))
                else:
                    for phase_value in range(5):
                        regime_scale_mask = (
                            scale_mask & (endpoint_phase == phase_value))
                        if bool(regime_scale_mask.any().item()):
                            scale_regime_losses.append(
                                self._reduce_masked_sample_loss(
                                    scale_per, regime_scale_mask,
                                    combined_weight))
                if scale_regime_losses:
                    terms.append(torch.stack(scale_regime_losses).mean())
                if terms:
                    class_losses.append(torch.stack(terms).mean())
                winner_matrix[..., class_index] = (
                    benefit_mask & analytic_winner).view(-1, 1)
                eligible_matrix[..., class_index] = benefit_mask.view(-1, 1)
            if benefit_mode in [
                'conditional_mean', 'family_stack', 'constrained_stack'
            ]:
                direct_correction = getattr(
                    model, 'last_phase_safety_correction', None)
                if direct_correction is None:
                    raise RuntimeError(
                        'conditional-mean phase routing requires mixture correction')
                corrected_error_kw2 = (
                    direct_correction - correction_target
                ).square().flatten(start_dim=1).mean(dim=1)
                capacity = float(getattr(
                    self.args, 'stanford_capacity_kw', 30.1))
                corrected_error = corrected_error_kw2 / (capacity ** 2)
                regime_terms = []
                # Each endpoint regime gets equal influence regardless of its
                # empirical prior. Stable experts can therefore improve the
                # parent while active experts retain the rare ramp objective.
                for phase_value in range(5):
                    group_mask = endpoint_valid & (endpoint_phase == phase_value)
                    if bool(group_mask.any().item()):
                        regime_error = (
                            corrected_error_kw2
                            if benefit_mode == 'constrained_stack'
                            else corrected_error)
                        regime_terms.append(self._reduce_masked_sample_loss(
                            regime_error, group_mask, combined_weight))
                if regime_terms:
                    class_losses.append(torch.stack(regime_terms).mean())
                if benefit_mode == 'constrained_stack':
                    parent_error_kw2 = correction_target.square().flatten(
                        start_dim=1).mean(dim=1)
                    endpoint_ramp = path_targets.lead_values[:, -1].abs()
                    constraint_masks = {
                        'ordinary': endpoint_valid & (endpoint_ramp < 5.0),
                        'no_path': path_targets.no_path_mask,
                        'round_trip': path_targets.round_trip_mask,
                        'ge8': endpoint_valid & (endpoint_ramp >= 8.0),
                    }
                    violations = {}
                    for name, group_mask in constraint_masks.items():
                        if not bool(group_mask.any().item()):
                            continue
                        # Checkpoint guards are unweighted per-window RMSEs.
                        # Episode weights remain appropriate for the phase
                        # objective, but using them here protects a different
                        # metric and can silently violate the deployed guard.
                        corrected_group = corrected_error_kw2[group_mask].mean()
                        parent_group = parent_error_kw2[group_mask].mean()
                        violations[name] = (
                            corrected_group / parent_group.clamp_min(1e-4) - 1.0)

                    duals = getattr(self, '_phase_constraint_duals', None)
                    if duals is None:
                        duals = {name: 0.0 for name in constraint_masks}
                        self._phase_constraint_duals = duals
                    dual_lr = float(getattr(
                        self.args, 'phase_constraint_dual_lr', 0.05))
                    penalty = float(getattr(
                        self.args, 'phase_constraint_penalty', 1.0))
                    if dual_lr < 0 or penalty < 0:
                        raise ValueError(
                            'phase constraint dual rate and penalty must be non-negative')
                    constraint_terms = []
                    for name, violation in violations.items():
                        if model.training:
                            duals[name] = max(
                                0.0,
                                duals.get(name, 0.0)
                                + dual_lr * float(violation.detach().item()),
                            )
                        dual = violation.new_tensor(duals.get(name, 0.0))
                        constraint_terms.append(
                            dual * violation
                            + 0.5 * penalty * torch.relu(violation).square())
                    if constraint_terms:
                        class_losses.append(torch.stack(constraint_terms).sum())
                    self._last_phase_constraint_violations = {
                        name: float(value.detach().item())
                        for name, value in violations.items()
                    }
            prior_updater = getattr(model, 'update_phase_winner_prior', None)
            if stage == 'benefit' and callable(prior_updater):
                prior_updater(winner_matrix, eligible_matrix)
            benefit_loss = (
                torch.stack(class_losses).mean()
                if class_losses else class_benefit_logits.sum() * 0.0)
            component = benefit_weight * benefit_loss
            components['phase_benefit'] = component
            total = total + component

        stable_weight = float(getattr(
            self.args, 'stable_correction_aux_weight', 0.0))
        safety_strength = getattr(
            model, 'last_phase_safety_strength', None)
        safety_correction = getattr(
            model, 'last_phase_safety_correction', None)
        if (
            stage in ['benefit', 'active']
            and stable_weight > 0
            and benefit_logit is not None
            and scale_logit is not None
            and candidate is not None
            and route_support is not None
            and (safety_correction is not None or safety_strength is not None)
        ):
            # Penalize only corrections that are worse than the frozen parent
            # on endpoint-ordinary paths.  This protects no-path and return
            # samples without forbidding a demonstrably useful correction.
            gated_candidate = (
                safety_correction
                if safety_correction is not None
                else safety_strength * candidate.detach())
            correction_target = target - parent_pred.detach()
            per_sample = torch.relu(
                (gated_candidate - correction_target).pow(2)
                - correction_target.pow(2)
            ).flatten(start_dim=1).mean(dim=1)
            endpoint_ordinary = (
                path_targets.lead_valid_mask[:, -1]
                & (path_targets.lead_values[:, -1].abs() < 5.0))
            component = stable_weight * self._reduce_masked_sample_loss(
                per_sample, endpoint_ordinary, combined_weight)
            components['phase_no_path_stable'] = component
            total = total + component

        self._last_phase_path_targets = path_targets
        if not bool(torch.isfinite(total).item()):
            bad_components = [
                name for name, value in components.items()
                if not bool(torch.isfinite(value).item())
            ]
            raise FloatingPointError(
                f'non-finite phase auxiliary loss at stage={stage}: '
                + ','.join(bad_components))
        return total, components

    def _student_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    def _phase_winner_prior_statistics(self):
        """Return phase statistics only for a phase-field Student model."""
        model = self._student_model()
        callback = getattr(model, 'phase_winner_prior_statistics', None)
        if not hasattr(model, 'phase_winner_total') or not callable(callback):
            return None
        return callback()

    def _teacher_model(self):
        if self.model_teacher is None:
            raise RuntimeError('Teacher is unavailable in student-only mode')
        return (
            self.model_teacher.module
            if hasattr(self.model_teacher, 'module') else self.model_teacher
        )

    def _current_pv(self, batch_x):
        if getattr(self.args, 'history_order', 'current_first') == 'current_first':
            return batch_x[:, :1, :1]
        return batch_x[:, -1:, :1]

    def _auxiliary_loss(self, target, current_pv, sample_weight=None,
                        phase_targets=None):
        model = self._student_model()
        if self._student_strategy() == 'causal_phase_field_tail':
            loss, components = self._phase_auxiliary_loss(
                target, current_pv, phase_targets, sample_weight)
            self._last_auxiliary_components = {
                name: value.detach() for name, value in components.items()
            }
            return loss
        loss = torch.zeros((), device=target.device)
        components = {}
        ramp_target = target - current_pv
        tail_event_mask = None
        tail_correction_target = None
        tail_positive_direction = None
        if self._student_strategy() == 'causal_solar_token_tail':
            tail_event_mask, tail_correction_target = self._tail_supervision_targets(
                target, current_pv)
            tail_positive_direction = (
                tail_correction_target.flatten(start_dim=1).mean(dim=1) > 0)
            prior_updater = getattr(model, 'update_tail_priors', None)
            if callable(prior_updater):
                prior_updater(tail_event_mask, tail_positive_direction)

        ramp_weight = float(getattr(self.args, 'ramp_aux_weight', 0.0))
        ramp_pred = getattr(model, 'last_ramp_pred', None)
        if ramp_weight > 0 and ramp_pred is not None:
            per_sample = (ramp_pred - ramp_target).pow(2).flatten(start_dim=1).mean(dim=1)
            loss = loss + ramp_weight * self._reduce_sample_loss(per_sample, sample_weight)

        direction_weight = float(getattr(self.args, 'ramp_direction_aux_weight', 0.0))
        direction_logit = getattr(model, 'last_ramp_direction_logit', None)
        if direction_weight > 0 and direction_logit is not None:
            direction_target = (ramp_target > 0).to(target.dtype)
            per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                direction_logit, direction_target, reduction='none')
            per_sample = per_sample.flatten(start_dim=1).mean(dim=1)
            loss = loss + direction_weight * self._reduce_sample_loss(per_sample, sample_weight)

        residual_weight = float(getattr(self.args, 'residual_aux_weight', 0.0))
        residual_pred = getattr(model, 'last_residual_pred', None)
        stable_pred = getattr(model, 'last_stable_pred', None)
        if residual_weight > 0 and residual_pred is not None and stable_pred is not None:
            residual_target = target - stable_pred.detach()
            per_sample = (residual_pred - residual_target).pow(2).flatten(start_dim=1).mean(dim=1)
            loss = loss + residual_weight * self._reduce_sample_loss(per_sample, sample_weight)

        stable_weight = float(getattr(self.args, 'stable_correction_aux_weight', 0.0))
        raw_correction = getattr(model, 'last_raw_applied_correction', None)
        if self._student_strategy() in [
            'causal_solar_event', 'causal_solar_token_tail'
        ]:
            raw_correction = getattr(model, 'last_raw_event_correction', raw_correction)
        if stable_weight > 0 and raw_correction is not None:
            threshold = float(getattr(self.args, 'stable_ramp_threshold_kw', 5.0))
            stable_mask = (
                ramp_target.abs().flatten(start_dim=1).max(dim=1).values < threshold
            )
            per_sample = raw_correction.pow(2).flatten(start_dim=1).mean(dim=1)
            stable_component = stable_weight * self._reduce_masked_sample_loss(
                per_sample, stable_mask, sample_weight)
            components['stable_correction'] = stable_component
            loss = loss + stable_component

        gate_weight = float(getattr(self.args, 'expert_gate_aux_weight', 0.0))
        gate_logit = getattr(model, 'last_expert_gate_logit', None)
        if gate_weight > 0 and gate_logit is not None:
            gate_target_mode = getattr(self.args, 'expert_gate_target', 'extreme')
            ramp_expert_pred = getattr(model, 'last_ramp_expert_pred', None)
            if gate_target_mode == 'winner' and stable_pred is not None and ramp_expert_pred is not None:
                stable_error = (stable_pred.detach() - target).pow(2)
                ramp_error = (ramp_expert_pred.detach() - target).pow(2)
                gate_target = (ramp_error < stable_error).to(target.dtype)
            else:
                threshold = float(getattr(self.args, 'expert_gate_threshold_kw', 5.0))
                gate_target = (ramp_target.abs() > threshold).to(target.dtype)
            per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                gate_logit, gate_target, reduction='none')
            per_sample = per_sample.flatten(start_dim=1).mean(dim=1)
            loss = loss + gate_weight * self._reduce_sample_loss(per_sample, sample_weight)

        event_weight = float(getattr(self.args, 'event_state_aux_weight', 0.0))
        event_logits = getattr(model, 'last_event_state_logits', None)
        if event_weight > 0 and event_logits is not None:
            if self._student_strategy() == 'causal_solar_token_tail':
                event_logit = getattr(model, 'last_tail_event_logit', None)
                direction_logit = getattr(model, 'last_tail_direction_logit', None)
                if event_logit is None or direction_logit is None:
                    raise RuntimeError('tail supervision requires event and direction logits')
                event_mask = tail_event_mask
                correction_target = tail_correction_target
                event_target = event_mask.to(target.dtype).view_as(event_logit)
                event_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    event_logit, event_target, reduction='none')
                event_bce = event_bce.flatten(start_dim=1).mean(dim=1)
                event_component = event_weight * self._balanced_binary_loss(
                    event_bce, event_mask, sample_weight)
                components['tail_event'] = event_component
                loss = loss + event_component

                direction_target = (correction_target > 0).to(target.dtype)
                direction_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                    direction_logit, direction_target, reduction='none')
                direction_bce = direction_bce.flatten(start_dim=1).mean(dim=1)
                direction_component = event_weight * self._balanced_binary_loss(
                    direction_bce, tail_positive_direction, sample_weight, event_mask)
                components['tail_direction'] = direction_component
                loss = loss + direction_component
            else:
                threshold = float(getattr(
                    self.args, 'expert_gate_threshold_kw', 5.0))
                state_value = ramp_target.flatten(start_dim=1).mean(dim=1)
                event_target = torch.zeros_like(state_value, dtype=torch.long)
                event_target[state_value <= -threshold] = 1
                event_target[state_value >= threshold] = 2
                logits = event_logits.reshape(event_logits.shape[0], -1, 3).mean(dim=1)
                per_sample = torch.nn.functional.cross_entropy(
                    logits, event_target, reduction='none')
                # Focal weighting lets rare transitions train the state encoder
                # without changing the regression loss of ordinary samples.
                probability = torch.exp(-per_sample)
                per_sample = (1.0 - probability).pow(2) * per_sample
                loss = loss + event_weight * self._reduce_sample_loss(
                    per_sample, sample_weight)

        event_residual_weight = float(getattr(
            self.args, 'event_residual_aux_weight', 0.0))
        event_correction = getattr(model, 'last_event_correction', None)
        if self._student_strategy() == 'causal_solar_token_tail':
            event_correction = getattr(
                model, 'last_tail_expert_correction', event_correction)
        event_stable_pred = getattr(model, 'last_stable_pred', None)
        if (
            event_residual_weight > 0
            and event_correction is not None
            and event_stable_pred is not None
        ):
            correction_target = target - event_stable_pred.detach()
            if self._student_strategy() == 'causal_solar_token_tail':
                event_mask = tail_event_mask
                correction_target = tail_correction_target
                capacity = float(getattr(
                    self.args, 'stanford_capacity_kw', 30.1))
                if capacity <= 0:
                    raise ValueError('stanford_capacity_kw must be positive')
                up_raw = getattr(model, 'last_tail_up_raw_correction', None)
                down_raw = getattr(model, 'last_tail_down_raw_correction', None)
                if up_raw is None or down_raw is None:
                    raise RuntimeError(
                        'tail residual supervision requires separate up/down heads')
                correction_normalized = correction_target / capacity
                up_per_sample = (
                    up_raw / capacity - correction_normalized
                ).pow(2).flatten(start_dim=1).mean(dim=1)
                down_per_sample = (
                    down_raw / capacity - correction_normalized
                ).pow(2).flatten(start_dim=1).mean(dim=1)
                positive_direction = tail_positive_direction
                directional_masks = [
                    (up_per_sample, event_mask & positive_direction),
                    (down_per_sample, event_mask & ~positive_direction),
                ]
                directional_losses = [
                    self._reduce_masked_sample_loss(values, mask, sample_weight)
                    for values, mask in directional_masks
                    if bool(mask.any().item())
                ]
                if directional_losses:
                    residual_component = event_residual_weight * torch.stack(
                        directional_losses).mean()
                else:
                    residual_component = event_residual_weight * up_per_sample.sum() * 0.0
            else:
                threshold = float(getattr(
                    self.args, 'expert_gate_threshold_kw', 5.0))
                event_mask = (
                    ramp_target.abs().flatten(start_dim=1).max(dim=1).values
                    >= threshold
                )
                per_sample = (
                    event_correction - correction_target
                ).pow(2).flatten(start_dim=1).mean(dim=1)
                residual_component = event_residual_weight * self._reduce_masked_sample_loss(
                    per_sample, event_mask, sample_weight)
            components['event_residual'] = residual_component
            loss = loss + residual_component

        clear_sky_weight = float(getattr(
            self.args, 'clear_sky_aux_weight', 0.0))
        clear_sky_kw = getattr(model, 'last_clear_sky_kw', None)
        if clear_sky_weight > 0 and clear_sky_kw is not None:
            quantile = float(getattr(self.args, 'clear_sky_quantile', 0.95))
            if not 0.5 < quantile < 1.0:
                raise ValueError('clear_sky_quantile must be between 0.5 and 1.0')
            capacity = float(getattr(
                self.args, 'stanford_capacity_kw', 30.1))
            if capacity <= 0:
                raise ValueError('stanford_capacity_kw must be positive')
            envelope_error = (target - clear_sky_kw) / capacity
            per_sample = torch.where(
                envelope_error >= 0,
                quantile * envelope_error,
                (1.0 - quantile) * (-envelope_error),
            ).flatten(start_dim=1).mean(dim=1)
            clear_sky_component = clear_sky_weight * self._reduce_sample_loss(
                per_sample, sample_weight)
            components['clear_sky_pinball'] = clear_sky_component
            loss = loss + clear_sky_component
        self._last_auxiliary_components = {
            name: value.detach() for name, value in components.items()
        }
        return loss

    def _student_strategy(self):
        return (
            getattr(self.args, 'student_fuse_strategy', None)
            or getattr(self.args, 'fuse_strategy', None)
        )

    def _uses_phase_validation_targets(self, phase_targets):
        return (
            phase_targets is not None
            and self._student_strategy() == 'causal_phase_field_tail'
        )

    def _teacher_strategy(self):
        return getattr(self.args, 'teacher_fuse_strategy', None) or self.args.fuse_strategy

    def _student_forward(
            self, batch_x, batch_x_img, batch_y_img, batch_x_weather,
            batch_y_weather, batch_x_mark=None, batch_y_mark=None,
            phase_lead_marks=None, image_mask=None, strategy=None):
        strategy = strategy or self._student_strategy()
        if getattr(self.args, 'strict_causal_student', False):
            if strategy not in [
                'causal_img', 'causal_gated', 'causal_residual_appearance',
                'causal_residual_motion', 'causal_soft_gate', 'causal_solar_event',
                'causal_solar_token_tail', 'causal_phase_field_tail',
                'no_img', 'ts_only', 'ts']:
                raise ValueError(
                    'strict causal student requires causal_img, causal_gated, no_img, ts_only, or ts')
            batch_y_img = torch.zeros_like(batch_y_img)
            batch_y_weather = torch.zeros_like(batch_y_weather)
        model_args = (
            batch_x, batch_x_img, batch_y_img, batch_x_weather,
            batch_y_weather, strategy)
        if strategy in [
            'causal_solar_event', 'causal_solar_token_tail',
            'causal_phase_field_tail'
        ]:
            extra_args = {}
            if image_mask is not None:
                extra_args['image_mask'] = image_mask
            if strategy == 'causal_phase_field_tail':
                if phase_lead_marks is None:
                    raise ValueError(
                        'causal_phase_field_tail requires label-free lead marks')
                extra_args['phase_lead_marks'] = phase_lead_marks
            model_outputs = self.model(
                *model_args, x_mark_h=batch_x_mark, x_mark_f=batch_y_mark,
                **extra_args)
        else:
            model_outputs = (
                self.model(*model_args, image_mask=image_mask)
                if image_mask is not None else self.model(*model_args)
            )
        if not isinstance(model_outputs, tuple):
            zeros = torch.zeros((), device=batch_x.device)
            return model_outputs, zeros, zeros, zeros, zeros, model_outputs
        if len(model_outputs) == 6:
            return model_outputs
        if len(model_outputs) != 4:
            raise ValueError(f'unsupported student output tuple length: {len(model_outputs)}')
        outputs, sim_img_ts, sim_weather_ts, feat = model_outputs
        image_enabled = strategy not in ['no_img', 'ts_only', 'ts']
        weather_enabled = strategy not in [
            'no_weather', 'ts_only', 'ts', 'causal_img', 'causal_gated',
            'causal_residual_appearance', 'causal_residual_motion',
            'causal_soft_gate', 'causal_solar_event',
            'causal_solar_token_tail', 'causal_phase_field_tail']
        drop_img = torch.tensor(float(image_enabled), device=outputs.device)
        drop_weather = torch.tensor(float(weather_enabled), device=outputs.device)
        return outputs, sim_img_ts, sim_weather_ts, drop_img, drop_weather, feat

    @staticmethod
    def _teacher_image_mask(metadata):
        if metadata is None:
            return None
        history = metadata.get('image_mask')
        future = metadata.get('teacher_future_image_mask')
        if history is None:
            return future
        if future is None:
            return history
        return torch.cat([history.bool(), future.bool()], dim=1)

    @staticmethod
    def _privileged_derangement(batch_size, device):
        if batch_size < 2:
            return None
        shift = int(torch.randint(1, batch_size, (), device=device).item())
        return torch.roll(torch.arange(batch_size, device=device), shifts=shift)

    def _teacher_forward(self, batch_x, batch_x_img, batch_y_img,
                         batch_x_weather, batch_y_weather, metadata=None,
                         strategy=None):
        if self.model_teacher is None:
            raise RuntimeError('Teacher forward is unavailable in student-only mode')
        image_mask = self._teacher_image_mask(metadata)
        model_args = (
            batch_x, batch_x_img, batch_y_img, batch_x_weather,
            batch_y_weather, strategy or self._teacher_strategy())
        return (
            self.model_teacher(*model_args, image_mask=image_mask)
            if image_mask is not None else self.model_teacher(*model_args)
        )

    @staticmethod
    def _monitor_metadata_batch(metadata):
        if metadata is None:
            return {}
        allowed = {
            'image_mask', 'image_minute_offsets', 'timeseries_mask',
            'timeseries_source', 'history_source', 'forecast_source',
            'power_history_mask', 'target_mask', 'current_power_valid',
            'teacher_future_image_mask',
            'teacher_history_timeseries_mask',
            'teacher_future_timeseries_mask',
            'teacher_history_timeseries_source',
            'teacher_future_timeseries_source', 'issue_time_ns',
        }
        result = {}
        for key in allowed:
            value = metadata.get(key)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            result[key] = np.asarray(value)
        return result

    @staticmethod
    def _monitor_source_codes(metadata):
        if metadata is None:
            return None
        source_keys = (
            'timeseries_source', 'history_source', 'forecast_source',
            'teacher_history_timeseries_source',
            'teacher_future_timeseries_source',
        )
        result = {
            key: metadata[key]
            for key in source_keys if metadata.get(key) is not None
        }
        return result or None

    @staticmethod
    def _monitor_coverage_masks(metadata):
        if metadata is None:
            return None
        key_map = {
            'forecast_mask': 'forecast_own_product',
            'power_history_mask': 'power_history',
            'teacher_history_timeseries_mask': 'teacher_history_timeseries',
            'teacher_future_timeseries_mask': 'teacher_future_timeseries',
            'teacher_future_image_mask': 'teacher_future_images',
        }
        result = {
            destination: metadata[source]
            for source, destination in key_map.items()
            if metadata.get(source) is not None
        }
        return result or None

    @staticmethod
    def _monitor_image_age(metadata):
        if metadata is None:
            return None
        value = metadata.get('image_age_minutes')
        return value if value is not None else metadata.get('image_minute_offsets')

    @classmethod
    def _append_monitor_batch(cls, store, predictions, targets, current_power,
                              target_mask, metadata):
        store['predictions'].append(
            predictions.detach().cpu().numpy().astype(np.float32, copy=False))
        store['targets'].append(
            targets.detach().cpu().numpy().astype(np.float32, copy=False))
        store['current_power'].append(
            current_power.detach().cpu().numpy().astype(np.float32, copy=False))
        store['target_mask'].append(
            target_mask.detach().cpu().numpy().astype(bool, copy=False))
        for key, value in cls._monitor_metadata_batch(metadata).items():
            store['metadata'].setdefault(key, []).append(value)

    @staticmethod
    def _finalize_monitor_batches(store):
        if not store['predictions']:
            return None
        metadata = {}
        for key, values in store['metadata'].items():
            try:
                metadata[key] = np.concatenate(values, axis=0)
            except ValueError:
                continue
        return {
            'predictions': np.concatenate(store['predictions'], axis=0),
            'targets': np.concatenate(store['targets'], axis=0),
            'current_power': np.concatenate(store['current_power'], axis=0),
            'target_mask': np.concatenate(store['target_mask'], axis=0),
            'metadata': metadata,
        }

    @staticmethod
    def _new_monitor_store():
        return {
            'predictions': [], 'targets': [], 'current_power': [],
            'target_mask': [], 'metadata': {},
        }

    def _gradient_group_statistics(self):
        totals = {}
        counts = {}
        for name, parameter in self._student_model().named_parameters():
            if parameter.grad is None:
                continue
            group = name.split('.', 1)[0]
            gradient = parameter.grad.detach().float()
            totals[group] = totals.get(group, 0.0) + float(
                gradient.square().sum().item())
            counts[group] = counts.get(group, 0) + int(gradient.numel())
        return {
            group: {
                'grad_norm': float(value ** 0.5),
                'gradient_elements': counts[group],
            }
            for group, value in totals.items()
        }

    def _collect_monitor_predictions(self, data_loader, role='student',
                                     strategy=None):
        if role not in {'student', 'teacher'}:
            raise ValueError(f'unsupported monitoring role: {role}')
        student_was_training = self.model.training
        if role == 'teacher' and self.model_teacher is None:
            raise RuntimeError('Teacher monitoring is unavailable in student-only mode')
        teacher_was_training = (
            self.model_teacher.training if self.model_teacher is not None else None)
        self.model.eval()
        if self.model_teacher is not None:
            self.model_teacher.eval()
        self._enforce_frozen_ts_mode()
        store = self._new_monitor_store()
        with torch.no_grad():
            for index, batch in enumerate(data_loader):
                max_val_batches = int(getattr(
                    self.args, 'max_val_batches', 0))
                if max_val_batches > 0 and index >= max_val_batches:
                    break
                (batch_x, batch_y, batch_x_mark, batch_y_mark,
                 batch_x_img, batch_y_img, batch_x_weather, batch_y_weather,
                 _, metadata) = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(
                    self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(
                    self.device, non_blocking=True)
                batch_x_img = batch_x_img.float().to(
                    self.device, non_blocking=True)
                batch_y_img = batch_y_img.float().to(
                    self.device, non_blocking=True)
                batch_x_weather = batch_x_weather.float().to(
                    self.device, non_blocking=True)
                batch_y_weather = batch_y_weather.float().to(
                    self.device, non_blocking=True)
                metadata = self._phase_targets_to_device(metadata)
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    if role == 'teacher':
                        outputs = self._teacher_forward(
                            batch_x, batch_x_img, batch_y_img,
                            batch_x_weather, batch_y_weather, metadata,
                            strategy=strategy)[0]
                    else:
                        phase_lead_marks = (
                            metadata.get('lead_marks')
                            if metadata is not None else None)
                        outputs = self._student_forward(
                            batch_x, batch_x_img, batch_y_img,
                            batch_x_weather, batch_y_weather,
                            batch_x_mark, batch_y_mark, phase_lead_marks,
                            None if metadata is None else metadata.get('image_mask'),
                            strategy=strategy)[0]
                f_dim = (
                    -42 if self.args.data == 'Folsom'
                    and self.args.features == 'MS'
                    else (-1 if self.args.features == 'MS' else 0)
                )
                predictions = outputs[:, -self.args.pred_len:, f_dim:]
                targets = batch_y[:, -self.args.pred_len:, f_dim:]
                target_mask = (
                    None if metadata is None else metadata.get('target_mask'))
                if target_mask is None:
                    target_mask = torch.ones(
                        targets.shape[:2], dtype=torch.bool,
                        device=targets.device)
                else:
                    target_mask = target_mask.to(
                        targets.device, dtype=torch.bool)
                self._append_monitor_batch(
                    store, predictions, targets, self._current_pv(batch_x),
                    target_mask, metadata)
        self.model.train(student_was_training)
        if self.model_teacher is not None:
            self.model_teacher.train(teacher_was_training)
        self._enforce_frozen_ts_mode()
        return self._finalize_monitor_batches(store)

    def _record_monitor_forecast(self, payload, epoch, split, role,
                                 phase='epoch'):
        if self.monitor is None or not self.monitor.enabled or payload is None:
            return None
        return self.monitor.record_forecast(
            epoch=epoch,
            split=split,
            role=role,
            predictions=payload['predictions'],
            targets=payload['targets'],
            current_power=payload['current_power'],
            target_mask=payload['target_mask'],
            metadata=payload['metadata'],
            phase=phase,
        )

    @staticmethod
    def _distillation_monitor_statistics(
            student_payload, teacher_payload, min_count=1, min_days=1):
        if student_payload is None or teacher_payload is None:
            return {}
        student = np.asarray(student_payload['predictions'], np.float64)
        teacher = np.asarray(teacher_payload['predictions'], np.float64)
        target = np.asarray(student_payload['targets'], np.float64)
        mask = np.asarray(student_payload['target_mask'], bool)
        if student.shape != teacher.shape or student.shape != target.shape:
            raise RuntimeError(
                'monitor Teacher/Student predictions are not aligned')
        if mask.ndim == student.ndim - 1:
            mask = mask[..., None]
        mask = np.broadcast_to(mask, student.shape)
        mask = (
            mask
            & np.isfinite(student)
            & np.isfinite(teacher)
            & np.isfinite(target)
        )
        student_values = student[mask]
        teacher_values = teacher[mask]
        target_values = target[mask]
        if not target_values.size:
            return {'valid_count': 0}
        distinct_day_count = None
        metadata = student_payload.get('metadata', {})
        issue_time = metadata.get('issue_time_ns') if isinstance(
            metadata, dict) else None
        if issue_time is not None:
            try:
                issue_time = np.asarray(issue_time).reshape(student.shape[0], -1)[:, 0]
                issue_dates = issue_time.astype(np.int64).astype('datetime64[ns]').astype(
                    'datetime64[D]')
                valid_samples = np.any(mask.reshape(student.shape[0], -1), axis=1)
                selected_dates = issue_dates[valid_samples]
                selected_dates = selected_dates[~np.isnat(selected_dates)]
                distinct_day_count = int(np.unique(selected_dates).size)
            except (TypeError, ValueError, OverflowError):
                distinct_day_count = None
        if (
            target_values.size < max(1, int(min_count))
            or (
                int(min_days) > 1
                and (
                    distinct_day_count is None
                    or distinct_day_count < int(min_days)
                )
            )
        ):
            return {
                'valid_count': int(target_values.size),
                'distinct_day_count': distinct_day_count,
                'privacy_suppressed': True,
            }
        student_error = np.abs(student_values - target_values)
        teacher_error = np.abs(teacher_values - target_values)
        disagreement = student_values - teacher_values
        equal_error = np.isclose(student_error, teacher_error)
        teacher_better = (teacher_error < student_error) & ~equal_error
        student_better = (student_error < teacher_error) & ~equal_error
        correlation = None
        if (
            student_values.size > 1
            and np.std(student_values) > 0
            and np.std(teacher_values) > 0
        ):
            correlation = float(np.corrcoef(
                student_values, teacher_values)[0, 1])
        return {
            'valid_count': int(target_values.size),
            'distinct_day_count': distinct_day_count,
            'student_teacher_rmse_fraction': float(
                np.sqrt(np.mean(np.square(disagreement)))),
            'student_teacher_mae_fraction': float(
                np.mean(np.abs(disagreement))),
            'teacher_better_fraction': float(
                np.mean(teacher_better)),
            'student_better_fraction': float(
                np.mean(student_better)),
            'equal_error_fraction': float(
                np.mean(equal_error)),
            'student_teacher_prediction_correlation': correlation,
        }

    def _check_student_future_image_invariance(
            self, batch_x, batch_x_img, batch_y_img, batch_x_weather,
            batch_y_weather, batch_x_mark=None, batch_y_mark=None,
            phase_lead_marks=None):
        if not (
            getattr(self.args, 'stanford_privileged_teacher', False)
            or getattr(self.args, 'strict_causal_student', False)
        ):
            return
        was_training = self.model.training
        branch = getattr(self._student_model(), 'branch_ts', None)
        branch_was_training = branch.training if branch is not None else None
        self.model.eval()
        try:
            with torch.no_grad():
                original = self._student_forward(
                    batch_x, batch_x_img, batch_y_img, batch_x_weather,
                    batch_y_weather, batch_x_mark, batch_y_mark,
                    phase_lead_marks)[0]
                variants = [
                    (batch_y_img[torch.randperm(
                        batch_y_img.shape[0], device=batch_y_img.device)],
                     batch_y_weather[torch.randperm(
                        batch_y_weather.shape[0], device=batch_y_weather.device)]),
                    (torch.zeros_like(batch_y_img),
                     torch.zeros_like(batch_y_weather)),
                    (torch.full_like(batch_y_img, 1e3),
                     torch.full_like(batch_y_weather, 1e3)),
                ]
                for image_variant, weather_variant in variants:
                    perturbed = self._student_forward(
                        batch_x, batch_x_img, image_variant,
                        batch_x_weather, weather_variant,
                        batch_x_mark, batch_y_mark, phase_lead_marks)[0]
                    if not torch.equal(original, perturbed):
                        difference = float((original - perturbed).abs().max().item())
                        raise RuntimeError(
                            'student output depends on privileged future inputs; '
                            f'max difference={difference}')
        finally:
            if was_training:
                self.model.train()
            else:
                self.model.eval()
            if branch is not None:
                branch.train(branch_was_training)
            self._enforce_frozen_ts_mode()
        print('student_privileged_future_invariance=passed variants=shuffle,zero,large_constant')
        if getattr(self.args, 'strict_causal_student', False):
            print(f'student_seq_y_img=disabled strategy={self._student_strategy()}')

    def _quiet_latent_mask(self, batch_x):
        """Historical-PV definition used by the dataset audit.

        Quiet-latent windows have less than 1 kW five-minute displacement and
        no one-minute jump of 1 kW or more in the latest five transitions.
        This mask is causal and contains no endpoint or future-path label.
        """
        if batch_x.shape[1] < 6:
            return torch.zeros(
                batch_x.shape[0], device=batch_x.device, dtype=torch.bool)
        pv = batch_x[..., :1]
        if getattr(self.args, 'history_order', 'current_first') == 'current_first':
            recent = pv[:, :6]
            five_minute_change = recent[:, 0] - recent[:, 5]
        else:
            recent = pv[:, -6:]
            five_minute_change = recent[:, -1] - recent[:, 0]
        largest_jump = (recent[:, 1:] - recent[:, :-1]).abs().amax(dim=(1, 2))
        return (
            five_minute_change.abs().flatten(start_dim=1).amax(dim=1) < 1.0
        ) & (largest_jump < 1.0)

    @staticmethod
    def _binary_average_precision(labels, scores):
        labels = np.asarray(labels, dtype=bool).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        finite = np.isfinite(scores)
        labels = labels[finite]
        scores = scores[finite]
        positives = int(labels.sum())
        if positives == 0:
            return float('nan')
        order = np.argsort(-scores, kind='mergesort')
        ordered = labels[order]
        precision = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
        return float(precision[ordered].sum() / positives)

    @staticmethod
    def _binary_balanced_accuracy(labels, predictions):
        labels = np.asarray(labels, dtype=bool).reshape(-1)
        predictions = np.asarray(predictions, dtype=bool).reshape(-1)
        recalls = []
        for value in [False, True]:
            mask = labels == value
            if np.any(mask):
                recalls.append(float((predictions[mask] == value).mean()))
        return float(np.mean(recalls)) if recalls else float('nan')
 

    def vali(self, vali_data, vali_loader, criterion, collect_monitor=False,
             monitor_split='val'):
        total_loss = []
        total_task_loss = []
        loss_weights = []
        squared_errors = []
        parent_squared_errors = []
        candidate_squared_errors = []
        oracle_squared_errors = []
        absolute_ramps = []
        signed_ramps = []
        no_path_masks = []
        round_trip_masks = []
        quiet_latent_masks = []
        hazard_event_labels = []
        hazard_event_scores = []
        phase_labels = []
        phase_predictions = []
        direction_scores = []
        direction_labels = []
        direction_eligible_masks = []
        benefit_strengths = []
        benefit_scores = []
        scale_scores = []
        benefit_labels = []
        route_supports = []
        diagnostic_targets = []
        diagnostic_parents = []
        diagnostic_candidates = []
        diagnostic_class_raw_corrections = []
        diagnostic_class_winner_probabilities = []
        diagnostic_class_scale_probabilities = []
        diagnostic_path_supports = []
        diagnostic_phase_probabilities = []
        diagnostic_endpoint_phase_labels = []
        diagnostic_endpoint_path_deltas = []
        diagnostic_class_endpoint_path_deltas = []
        diagnostic_class_route_supports = []
        monitor_store = self._new_monitor_store() if collect_monitor else None
        validation_sample_dates = []
        validation_target_dates = []
        phase_label_dates = []
        monitor_quality = None
        if (
            collect_monitor
            and self.monitor is not None
            and self.monitor.enabled
            and monitor_split not in self._monitor_quality_recorded
        ):
            fields = getattr(vali_data, 'fields', {})
            feature_names = (
                fields.get('timeseries_columns')
                or fields.get('time_series_columns'))
            monitor_quality = self.monitor.new_data_quality_accumulator(
                monitor_split, feature_names=feature_names)
        
        was_training = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                max_val_batches = int(getattr(
                    self.args, 'max_val_batches', 0))
                if max_val_batches > 0 and i >= max_val_batches:
                    break
                (batch_x, batch_y, batch_x_mark, batch_y_mark,
                 batch_x_img, batch_y_img, batch_x_weather, batch_y_weather,
                 sample_weight, phase_targets) = self._unpack_batch(batch)
                # 数据预处理
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                batch_x_img = batch_x_img.float().to(self.device, non_blocking=True)
                batch_y_img = batch_y_img.float().to(self.device, non_blocking=True)
                batch_x_weather = batch_x_weather.float().to(self.device, non_blocking=True)
                batch_y_weather = batch_y_weather.float().to(self.device, non_blocking=True)
                if sample_weight is not None:
                    sample_weight = sample_weight.float().to(self.device, non_blocking=True)
                phase_targets = self._phase_targets_to_device(phase_targets)
                phase_lead_marks = (
                    phase_targets.get('lead_marks') if phase_targets is not None else None)
                
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self._student_forward(
                        batch_x, batch_x_img, batch_y_img, batch_x_weather,
                        batch_y_weather, batch_x_mark, batch_y_mark,
                        phase_lead_marks,
                        None if phase_targets is None else phase_targets.get('image_mask'))[0]
                
                if self.args.data == 'Folsom':
                    f_dim = -42 if self.args.features == 'MS' else 0
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:]
                target_mask = None if phase_targets is None else phase_targets.get('target_mask')
                task_loss = self._criterion_loss(
                    outputs, batch_y_target, criterion, sample_weight, target_mask)
                auxiliary_loss = self._auxiliary_loss(
                    batch_y_target, self._current_pv(batch_x), sample_weight,
                    phase_targets)
                loss = task_loss + auxiliary_loss

                pred = outputs.detach().cpu()
                true = batch_y_target.detach().cpu()

                # 记录损失
                total_loss.append(loss.item())
                total_task_loss.append(task_loss.item())
                validation_mask = (
                    torch.ones_like(batch_y_target, dtype=torch.bool)
                    if target_mask is None
                    else target_mask.to(batch_y_target.device, dtype=torch.bool).unsqueeze(-1)
                )
                batch_size = int(batch_y_target.shape[0])
                batch_issue_dates = np.full(
                    batch_size, np.datetime64('NaT', 'D'), dtype='datetime64[D]')
                issue_time = (
                    None if phase_targets is None
                    else phase_targets.get('issue_time_ns'))
                if issue_time is not None:
                    try:
                        if torch.is_tensor(issue_time):
                            issue_time = issue_time.detach().cpu().numpy()
                        issue_time = np.asarray(issue_time).reshape(
                            batch_size, -1)[:, 0]
                        if np.issubdtype(issue_time.dtype, np.datetime64):
                            batch_issue_dates = issue_time.astype('datetime64[D]')
                        else:
                            batch_issue_dates = issue_time.astype(
                                np.int64).astype('datetime64[ns]').astype(
                                    'datetime64[D]')
                    except (TypeError, ValueError, OverflowError):
                        pass
                validation_sample_dates.append(batch_issue_dates)
                validation_mask_numpy = validation_mask.detach().cpu().numpy()
                target_date_grid = np.broadcast_to(
                    batch_issue_dates.reshape(batch_size, 1, 1),
                    validation_mask_numpy.shape)
                validation_target_dates.append(
                    target_date_grid[validation_mask_numpy])
                loss_weights.append(int(validation_mask.sum().item()))
                if monitor_store is not None:
                    monitor_target_mask = validation_mask.squeeze(-1)
                    self._append_monitor_batch(
                        monitor_store, outputs, batch_y_target,
                        self._current_pv(batch_x), monitor_target_mask,
                        phase_targets)
                    if monitor_quality is not None:
                        monitor_quality.update(
                            features=batch_x,
                            feature_mask=(
                                None if phase_targets is None
                                else phase_targets.get('timeseries_mask')),
                            target=batch_y_target,
                            current_power=self._current_pv(batch_x),
                            target_mask=monitor_target_mask,
                            image_mask=(
                                None if phase_targets is None
                                else phase_targets.get('image_mask')),
                            image_age_minutes=(
                                self._monitor_image_age(phase_targets)),
                            source_codes=(
                                self._monitor_source_codes(phase_targets)),
                            issue_time_ns=(
                                None if phase_targets is None
                                else phase_targets.get('issue_time_ns')),
                            coverage_masks=(
                                self._monitor_coverage_masks(phase_targets)),
                            privileged_teacher_enabled=(
                                None if phase_targets is None
                                else phase_targets.get(
                                    'privileged_teacher_enabled')),
                            history_images_enabled=(
                                None if phase_targets is None
                                else phase_targets.get('history_images_enabled')),
                            current_power_valid=(
                                None if phase_targets is None
                                else phase_targets.get('current_power_valid')),
                        )
                squared_errors.append(
                    (outputs - batch_y_target).pow(2)[validation_mask]
                    .detach().cpu().numpy().reshape(-1))
                signed_ramp = batch_y_target - self._current_pv(batch_x)
                absolute_ramps.append(
                    signed_ramp.abs()[validation_mask].detach().cpu().numpy().reshape(-1))
                signed_ramps.append(
                    signed_ramp[validation_mask].detach().cpu().numpy().reshape(-1))

                # Phase diagnostics belong solely to the opt-in phase-field
                # Student.  Ordinary causal-image runs still carry generic
                # dataset metadata, but do not construct phase paths.
                path_targets = getattr(self, '_last_phase_path_targets', None)
                if (self._uses_phase_validation_targets(phase_targets)
                        and path_targets is not None):
                    no_path_masks.append(
                        path_targets.no_path_mask.detach().cpu().numpy())
                    round_trip_masks.append(
                        path_targets.round_trip_mask.detach().cpu().numpy())
                    quiet_latent_masks.append(
                        self._quiet_latent_mask(batch_x).detach().cpu().numpy())

                    model = self._student_model()
                    hazard_logits = getattr(
                        model, 'last_phase_hazard_logits', None)
                    if hazard_logits is not None:
                        event_score = 1.0 - (
                            1.0 - torch.sigmoid(hazard_logits)
                        ).prod(dim=1)
                        diagnostic_path_supports.append(
                            event_score.detach().cpu().numpy())
                        hazard_event_scores.append(
                            event_score.detach().cpu().numpy())
                        identifiable = (
                            (path_targets.first_crossing_bin >= 0)
                            | path_targets.no_path_mask)
                        hazard_event_labels.append(np.stack([
                            (path_targets.first_crossing_bin >= 0)
                            .detach().cpu().numpy(),
                            identifiable.detach().cpu().numpy(),
                        ], axis=-1))
                    phase_logits = getattr(model, 'last_phase_state_logits', None)
                    if phase_logits is not None:
                        diagnostic_phase_probabilities.append(
                            torch.softmax(phase_logits, dim=-1).detach().cpu()
                            .numpy())
                        diagnostic_endpoint_phase_labels.append(
                            path_targets.phase[:, -1].detach().cpu().numpy())
                        predicted_path = getattr(
                            model, 'last_phase_path_delta_kw', None)
                        if predicted_path is None:
                            raise RuntimeError(
                                'phase probabilities require endpoint path prediction')
                        diagnostic_endpoint_path_deltas.append(
                            predicted_path[:, -1].detach().cpu().numpy().reshape(-1))
                        class_predicted_path = getattr(
                            model, 'last_phase_class_path_delta_kw', None)
                        if class_predicted_path is not None:
                            diagnostic_class_endpoint_path_deltas.append(
                                class_predicted_path[:, -1, :, 0].detach()
                                .cpu().numpy().reshape(-1, 5))
                        valid_phase = path_targets.lead_valid_mask
                        phase_predictions.append(
                            phase_logits.argmax(dim=-1)[valid_phase]
                            .detach().cpu().numpy())
                        phase_labels.append(
                            path_targets.phase[valid_phase].detach().cpu().numpy())
                        valid_phase_numpy = valid_phase.detach().cpu().numpy()
                        phase_date_grid = np.broadcast_to(
                            batch_issue_dates.reshape(
                                batch_size,
                                *([1] * (valid_phase_numpy.ndim - 1))),
                            valid_phase_numpy.shape)
                        phase_label_dates.append(
                            phase_date_grid[valid_phase_numpy])
                    benefit_logit = getattr(
                        model, 'last_phase_benefit_logit', None)
                    if benefit_logit is not None:
                        route_support = getattr(
                            model, 'last_phase_route_support', None)
                        benefit_strength = getattr(
                            model, 'last_phase_benefit_strength', None)
                        if route_support is None or benefit_strength is None:
                            raise RuntimeError(
                                'phase benefit logits require routed strength')
                        benefit_strengths.append(
                            benefit_strength.detach().cpu().numpy().reshape(-1))
                        benefit_scores.append(
                            torch.sigmoid(benefit_logit).detach().cpu()
                            .numpy().reshape(-1))
                        scale_logit = getattr(
                            model, 'last_phase_scale_logit', None)
                        if scale_logit is None:
                            raise RuntimeError(
                                'phase winner logits require scale logits')
                        scale_scores.append(
                            torch.sigmoid(scale_logit).detach().cpu()
                            .numpy().reshape(-1))
                        class_benefit_logits = getattr(
                            model, 'last_phase_class_benefit_logits', None)
                        class_scale_logits = getattr(
                            model, 'last_phase_class_scale_logits', None)
                        if (
                            class_benefit_logits is None
                            or class_scale_logits is None
                        ):
                            raise RuntimeError(
                                'phase diagnostics require all class-conditional '
                                'winner and scale logits')
                        diagnostic_class_winner_probabilities.append(
                            torch.sigmoid(class_benefit_logits).detach().cpu()
                            .numpy().reshape(-1, 5))
                        diagnostic_class_scale_probabilities.append(
                            torch.sigmoid(class_scale_logits).detach().cpu()
                            .numpy().reshape(-1, 5))
                        route_supports.append(
                            route_support.detach().cpu().numpy().reshape(-1))
                        class_route_support = getattr(
                            model, 'last_phase_class_route_support', None)
                        if class_route_support is not None:
                            diagnostic_class_route_supports.append(
                                class_route_support.detach().cpu().numpy()
                                .reshape(-1, 5))
                    parent_pred = getattr(model, 'last_phase_parent_pred', None)
                    if parent_pred is not None:
                        diagnostic_targets.append(
                            batch_y_target.detach().cpu().numpy().reshape(-1))
                        diagnostic_parents.append(
                            parent_pred.detach().cpu().numpy().reshape(-1))
                        correction_target = batch_y_target - parent_pred
                        correction_size = correction_target.abs().flatten(
                            start_dim=1).amax(dim=1)
                        endpoint_phase = path_targets.phase[:, -1]
                        endpoint_active = path_targets.lead_valid_mask[:, -1] & (
                            (endpoint_phase == ACTIVE_DOWN)
                            | (endpoint_phase == ACTIVE_UP))
                        direction_labels.append(
                            (correction_target.flatten(start_dim=1).mean(dim=1) > 0)
                            .detach().cpu().numpy())
                        direction_eligible_masks.append((
                            endpoint_active
                            & (correction_size >= float(getattr(
                                self.args, 'tail_error_threshold_kw', 3.0)))
                        ).detach().cpu().numpy())
                        parent_squared_errors.append(
                            (parent_pred - batch_y_target).pow(2).detach()
                            .cpu().numpy().reshape(-1))
                        candidate = getattr(
                            model, 'last_phase_candidate_correction', None)
                        if candidate is not None:
                            class_raw = getattr(
                                model, 'last_phase_class_raw_correction', None)
                            if class_raw is None:
                                raise RuntimeError(
                                    'phase candidate diagnostics require all '
                                    'phase-class raw corrections')
                            diagnostic_class_raw_corrections.append(
                                class_raw.detach().cpu().numpy().reshape(-1, 5))
                            candidate = candidate.detach()
                            direction_scores.append(
                                (candidate.flatten(start_dim=1).mean(dim=1) > 0)
                                .to(candidate.dtype).cpu().numpy())
                            diagnostic_candidates.append(
                                candidate.cpu().numpy().reshape(-1))
                            candidate_squared_errors.append(
                                (parent_pred + candidate - batch_y_target)
                                .pow(2).cpu().numpy().reshape(-1))
                            optimal_scale = (
                                candidate * correction_target
                                / candidate.square().clamp_min(1e-4)
                            ).clamp(0.0, 1.0)
                            oracle_error = (
                                optimal_scale * candidate - correction_target
                            ).square().flatten(start_dim=1).mean(dim=1)
                            parent_error = correction_target.square().flatten(
                                start_dim=1).mean(dim=1)
                            benefit_labels.append(
                                (oracle_error < parent_error)
                                .detach().cpu().numpy())
                            oracle_squared_errors.append(
                                (parent_pred + optimal_scale * candidate
                                 - batch_y_target).pow(2).detach()
                                .cpu().numpy().reshape(-1))
                
        # 计算平均损失
        if not total_loss or not sum(loss_weights):
            raise RuntimeError('validation produced no valid target points')
        avg_total_loss = float(np.average(total_loss, weights=loss_weights))
        avg_task_loss = float(np.average(total_task_loss, weights=loss_weights))
        squared_errors = np.concatenate(squared_errors)
        absolute_ramps = np.concatenate(absolute_ramps)
        signed_ramps = np.concatenate(signed_ramps)
        target_dates = np.concatenate(validation_target_dates)
        sample_dates = np.concatenate(validation_sample_dates)
        if target_dates.shape != squared_errors.shape:
            raise RuntimeError(
                'validation dates do not match valid prediction points: '
                f'{target_dates.shape} vs {squared_errors.shape}')

        def masked_distinct_day_count(mask, dates=target_dates):
            selected = np.asarray(mask, dtype=bool).reshape(-1)
            date_values = np.asarray(dates).reshape(-1).astype('datetime64[D]')
            if selected.shape != date_values.shape:
                return 0
            selected_dates = date_values[selected]
            selected_dates = selected_dates[~np.isnat(selected_dates)]
            return int(np.unique(selected_dates).size)

        def minimum_class_privacy_support(labels, dates, mask=None):
            label_values = np.asarray(labels).reshape(-1)
            date_values = np.asarray(dates).reshape(-1).astype('datetime64[D]')
            selected = (
                np.ones(label_values.shape, dtype=bool)
                if mask is None else np.asarray(mask, dtype=bool).reshape(-1))
            if (
                selected.shape != label_values.shape
                or date_values.shape != label_values.shape
                or not np.any(selected)
            ):
                return 0, 0
            selected_labels = label_values[selected]
            selected_dates = date_values[selected]
            class_counts = []
            class_day_counts = []
            for value in np.unique(selected_labels):
                class_mask = selected_labels == value
                class_counts.append(int(class_mask.sum()))
                class_dates = selected_dates[class_mask]
                class_dates = class_dates[~np.isnat(class_dates)]
                class_day_counts.append(int(np.unique(class_dates).size))
            return min(class_counts), min(class_day_counts)

        is_capacity_normalized = self.args.data in {
            'LuoyangParquet', 'YLJParquet'}
        cloud_default = (
            np.zeros_like(squared_errors, dtype=bool)
            if is_capacity_normalized
            else np.ones_like(squared_errors, dtype=bool)
        )
        cloud_event_day = np.asarray(
            getattr(vali_data, 'cloud_event_day_mask', cloud_default),
            dtype=bool,
        )[:squared_errors.size]
        if cloud_event_day.shape != squared_errors.shape:
            raise RuntimeError(
                'cloud-event validation mask does not match prediction order: '
                f'{cloud_event_day.shape} vs {squared_errors.shape}')
        if is_capacity_normalized:
            power_config = vali_data.config['power']
            direction_threshold = float(
                power_config.get('direction_min_delta_fraction', 0.02))
            hard_threshold = float(power_config['hard_delta_fraction'])
            ramp_lt5 = absolute_ramps < direction_threshold
            ramp_ge5 = absolute_ramps >= direction_threshold
            ramp_ge8 = absolute_ramps >= hard_threshold
        else:
            ramp_lt5 = absolute_ramps < 5.0
            ramp_ge5 = absolute_ramps >= 5.0
            ramp_ge8 = absolute_ramps >= 8.0
        cloud_event_lt5 = cloud_event_day & ramp_lt5
        cloud_event_ge5 = cloud_event_day & ramp_ge5
        cloud_event_ge8 = cloud_event_day & ramp_ge8

        def masked_rmse(mask):
            return (
                float(np.sqrt(np.mean(squared_errors[mask])))
                if np.any(mask) else float('nan')
            )

        validation_metrics = {
            'rmse': float(np.sqrt(np.mean(squared_errors))),
            'ramp_lt5_rmse': masked_rmse(ramp_lt5),
            'ramp_ge5_rmse': masked_rmse(ramp_ge5),
            'ramp_ge8_rmse': masked_rmse(ramp_ge8),
            'count': int(squared_errors.size),
            'ramp_lt5_count': int(ramp_lt5.sum()),
            'ramp_ge5_count': int(ramp_ge5.sum()),
            'ramp_ge8_count': int(ramp_ge8.sum()),
            'distinct_day_count': masked_distinct_day_count(
                np.ones_like(squared_errors, dtype=bool)),
            'ramp_lt5_distinct_day_count': masked_distinct_day_count(ramp_lt5),
            'ramp_ge5_distinct_day_count': masked_distinct_day_count(ramp_ge5),
            'ramp_ge8_distinct_day_count': masked_distinct_day_count(ramp_ge8),
            'cloud_event_rmse': masked_rmse(cloud_event_day),
            'cloud_event_ramp_lt5_rmse': masked_rmse(cloud_event_lt5),
            'cloud_event_ramp_ge5_rmse': masked_rmse(cloud_event_ge5),
            'cloud_event_ramp_ge8_rmse': masked_rmse(cloud_event_ge8),
            'cloud_event_count': int(cloud_event_day.sum()),
            'cloud_event_ramp_lt5_count': int(cloud_event_lt5.sum()),
            'cloud_event_ramp_ge5_count': int(cloud_event_ge5.sum()),
            'cloud_event_ramp_ge8_count': int(cloud_event_ge8.sum()),
            'cloud_event_distinct_day_count': masked_distinct_day_count(
                cloud_event_day),
            'cloud_event_ramp_lt5_distinct_day_count': (
                masked_distinct_day_count(cloud_event_lt5)),
            'cloud_event_ramp_ge5_distinct_day_count': (
                masked_distinct_day_count(cloud_event_ge5)),
            'cloud_event_ramp_ge8_distinct_day_count': (
                masked_distinct_day_count(cloud_event_ge8)),
        }
        if is_capacity_normalized:
            validation_metrics.update({
                'ramp_stable_rmse': validation_metrics['ramp_lt5_rmse'],
                'ramp_directional_rmse': validation_metrics['ramp_ge5_rmse'],
                'ramp_hard_rmse': validation_metrics['ramp_ge8_rmse'],
                'ramp_stable_count': validation_metrics['ramp_lt5_count'],
                'ramp_directional_count': validation_metrics['ramp_ge5_count'],
                'ramp_hard_count': validation_metrics['ramp_ge8_count'],
                'ramp_stable_distinct_day_count': validation_metrics[
                    'ramp_lt5_distinct_day_count'],
                'ramp_directional_distinct_day_count': validation_metrics[
                    'ramp_ge5_distinct_day_count'],
                'ramp_hard_distinct_day_count': validation_metrics[
                    'ramp_ge8_distinct_day_count'],
                'direction_threshold_fraction': direction_threshold,
                'hard_threshold_fraction': hard_threshold,
            })

        if no_path_masks:
            no_path = np.concatenate(no_path_masks).astype(bool)
            round_trip = np.concatenate(round_trip_masks).astype(bool)
            quiet_latent = np.concatenate(quiet_latent_masks).astype(bool)
            if no_path.shape != squared_errors.shape:
                raise RuntimeError(
                    'phase validation masks do not match prediction order: '
                    f'{no_path.shape} vs {squared_errors.shape}')
            up_ge5 = ramp_ge5 & (signed_ramps > 0)
            down_ge5 = ramp_ge5 & (signed_ramps < 0)
            quiet_latent_ge5 = quiet_latent & ramp_ge5
            strata = {
                'ramp_lt5': ramp_lt5,
                'ramp_ge5': ramp_ge5,
                'ramp_ge8': ramp_ge8,
                'no_path': no_path,
                'round_trip': round_trip,
                'quiet_latent_ge5': quiet_latent_ge5,
                'ramp_up_ge5': up_ge5,
                'ramp_down_ge5': down_ge5,
                'cloud_event_no_path': cloud_event_day & no_path,
                'cloud_event_ramp_lt5': cloud_event_lt5,
                'cloud_event_ramp_ge5': cloud_event_ge5,
                'cloud_event_ramp_ge8': cloud_event_ge8,
                'cloud_event_round_trip': cloud_event_day & round_trip,
                'cloud_event_quiet_latent_ge5': (
                    cloud_event_day & quiet_latent_ge5),
                'cloud_event_ramp_up_ge5': cloud_event_day & up_ge5,
                'cloud_event_ramp_down_ge5': cloud_event_day & down_ge5,
            }
            for name, mask in strata.items():
                validation_metrics[f'{name}_rmse'] = masked_rmse(mask)
                validation_metrics[f'{name}_count'] = int(mask.sum())
                validation_metrics[f'{name}_distinct_day_count'] = (
                    masked_distinct_day_count(mask))

            if hazard_event_scores:
                hazard_target = np.concatenate(hazard_event_labels)
                identifiable = hazard_target[:, 1].astype(bool)
                hazard_count, hazard_days = minimum_class_privacy_support(
                    hazard_target[:, 0], sample_dates, identifiable)
                validation_metrics['phase_hazard_event_auprc'] = (
                    self._binary_average_precision(
                        hazard_target[identifiable, 0],
                        np.concatenate(hazard_event_scores)[identifiable]))
                validation_metrics['phase_hazard_event_count'] = hazard_count
                validation_metrics[
                    'phase_hazard_event_distinct_day_count'] = hazard_days
            if phase_labels:
                labels = np.concatenate(phase_labels)
                predictions = np.concatenate(phase_predictions)
                label_values = np.unique(labels)
                class_accuracy = [
                    float((predictions[labels == value] == value).mean())
                    for value in label_values
                ]
                validation_metrics['phase_macro_accuracy'] = float(
                    np.mean(class_accuracy))
                # Macro accuracy exposes each class equally, so its privacy
                # support is the smallest contributing class rather than the
                # total number of phase labels.
                phase_class_count, phase_class_days = (
                    minimum_class_privacy_support(
                        labels, np.concatenate(phase_label_dates)))
                validation_metrics['phase_macro_accuracy_count'] = (
                    phase_class_count)
                validation_metrics[
                    'phase_macro_accuracy_distinct_day_count'] = phase_class_days
            if diagnostic_phase_probabilities:
                endpoint_probability = np.concatenate(
                    diagnostic_phase_probabilities)[:, -1, :].astype(
                        np.float64)
                endpoint_label = np.concatenate(
                    diagnostic_endpoint_phase_labels)

                def add_endpoint_factor_metrics(prefix, subset):
                    valid = subset & (endpoint_label >= 0)
                    event_label = endpoint_label > 0
                    event_score = 1.0 - endpoint_probability[:, 0]
                    event_count, event_days = minimum_class_privacy_support(
                        event_label, sample_dates, valid)
                    validation_metrics[f'{prefix}event_bacc'] = (
                        self._binary_balanced_accuracy(
                            event_label[valid], event_score[valid] >= 0.5))
                    validation_metrics[f'{prefix}event_count'] = event_count
                    validation_metrics[
                        f'{prefix}event_distinct_day_count'] = event_days
                    event_valid = valid & event_label
                    active_label = np.isin(endpoint_label, [1, 2])
                    event_mass = event_score.clip(1e-8, None)
                    active_score = (
                        endpoint_probability[:, 1]
                        + endpoint_probability[:, 2]) / event_mass
                    active_prediction = active_score >= 0.5
                    active_count, active_days = minimum_class_privacy_support(
                        active_label, sample_dates, event_valid)
                    validation_metrics[f'{prefix}active_bacc'] = (
                        self._binary_balanced_accuracy(
                            active_label[event_valid],
                            active_prediction[event_valid]))
                    validation_metrics[f'{prefix}active_count'] = active_count
                    validation_metrics[
                        f'{prefix}active_distinct_day_count'] = active_days
                    active_samples = event_valid & active_label
                    return_samples = event_valid & ~active_label
                    validation_metrics[f'{prefix}active_recall'] = (
                        float(active_prediction[active_samples].mean())
                        if np.any(active_samples) else float('nan'))
                    validation_metrics[f'{prefix}active_recall_count'] = int(
                        active_samples.sum())
                    validation_metrics[
                        f'{prefix}active_recall_distinct_day_count'] = (
                            masked_distinct_day_count(active_samples, sample_dates))
                    validation_metrics[f'{prefix}return_specificity'] = (
                        float((~active_prediction[return_samples]).mean())
                        if np.any(return_samples) else float('nan'))
                    validation_metrics[
                        f'{prefix}return_specificity_count'] = int(
                            return_samples.sum())
                    validation_metrics[
                        f'{prefix}return_specificity_distinct_day_count'] = (
                            masked_distinct_day_count(return_samples, sample_dates))
                    up_label = np.isin(endpoint_label, [2, 4])
                    up_score = (
                        endpoint_probability[:, 2]
                        + endpoint_probability[:, 4]) / event_mass
                    direction_count, direction_days = (
                        minimum_class_privacy_support(
                            up_label, sample_dates, event_valid))
                    validation_metrics[f'{prefix}direction_bacc'] = (
                        self._binary_balanced_accuracy(
                            up_label[event_valid],
                            up_score[event_valid] >= 0.5))
                    validation_metrics[f'{prefix}direction_count'] = (
                        direction_count)
                    validation_metrics[
                        f'{prefix}direction_distinct_day_count'] = direction_days

                add_endpoint_factor_metrics(
                    'phase_endpoint_', np.ones_like(endpoint_label, dtype=bool))
                add_endpoint_factor_metrics(
                    'cloud_phase_endpoint_', cloud_event_day)
            if direction_scores:
                direction_score = np.concatenate(direction_scores)
                direction_label = np.concatenate(direction_labels)
                direction_eligible = np.concatenate(
                    direction_eligible_masks).astype(bool)
                direction_count, direction_days = minimum_class_privacy_support(
                    direction_label, sample_dates, direction_eligible)
                validation_metrics['phase_correction_direction_bacc'] = (
                    self._binary_balanced_accuracy(
                        direction_label[direction_eligible],
                        direction_score[direction_eligible] >= 0.5))
                validation_metrics['phase_correction_direction_count'] = (
                    direction_count)
                validation_metrics[
                    'phase_correction_direction_distinct_day_count'] = (
                        direction_days)
            if benefit_strengths:
                benefit = np.concatenate(benefit_strengths)
                validation_metrics['phase_gate_mean'] = float(benefit.mean())
                validation_metrics['phase_gate_count'] = int(benefit.size)
                validation_metrics['phase_gate_distinct_day_count'] = (
                    masked_distinct_day_count(
                        np.ones(benefit.shape, dtype=bool), sample_dates))
                validation_metrics['phase_gate_no_path_mean'] = (
                    float(benefit[no_path].mean())
                    if np.any(no_path) else float('nan'))
                validation_metrics['phase_gate_no_path_count'] = int(
                    no_path.sum())
                validation_metrics[
                    'phase_gate_no_path_distinct_day_count'] = (
                        masked_distinct_day_count(no_path, sample_dates))
                validation_metrics['phase_gate_ge5_mean'] = (
                    float(benefit[ramp_ge5].mean())
                    if np.any(ramp_ge5) else float('nan'))
                validation_metrics['phase_gate_ge5_count'] = int(
                    ramp_ge5.sum())
                validation_metrics['phase_gate_ge5_distinct_day_count'] = (
                    masked_distinct_day_count(ramp_ge5, sample_dates))
                if benefit_labels:
                    benefit_label = np.concatenate(benefit_labels)
                    benefit_count, benefit_days = minimum_class_privacy_support(
                        benefit_label, sample_dates)
                    validation_metrics['phase_benefit_auprc'] = (
                        self._binary_average_precision(
                            benefit_label,
                            np.concatenate(benefit_scores)))
                    validation_metrics['phase_benefit_count'] = benefit_count
                    validation_metrics[
                        'phase_benefit_distinct_day_count'] = benefit_days
                support = np.concatenate(route_supports)
                validation_metrics['phase_route_support_mean'] = float(
                    support.mean())
                validation_metrics['phase_route_support_count'] = int(
                    support.size)
                validation_metrics[
                    'phase_route_support_distinct_day_count'] = (
                        masked_distinct_day_count(
                            np.ones(support.shape, dtype=bool), sample_dates))
                validation_metrics['phase_route_support_no_path_mean'] = (
                    float(support[no_path].mean())
                    if np.any(no_path) else float('nan'))
                validation_metrics['phase_route_support_no_path_count'] = int(
                    no_path.sum())
                validation_metrics[
                    'phase_route_support_no_path_distinct_day_count'] = (
                        masked_distinct_day_count(no_path, sample_dates))
                validation_metrics['phase_route_support_ge5_mean'] = (
                    float(support[ramp_ge5].mean())
                    if np.any(ramp_ge5) else float('nan'))
                validation_metrics['phase_route_support_ge5_count'] = int(
                    ramp_ge5.sum())
                validation_metrics[
                    'phase_route_support_ge5_distinct_day_count'] = (
                        masked_distinct_day_count(ramp_ge5, sample_dates))

            if parent_squared_errors:
                parent_errors = np.concatenate(parent_squared_errors)
                for name, mask in strata.items():
                    parent_rmse = (
                        float(np.sqrt(np.mean(parent_errors[mask])))
                        if np.any(mask) else float('nan'))
                    validation_metrics[f'parent_{name}_rmse'] = parent_rmse
                    child_rmse = validation_metrics[f'{name}_rmse']
                    validation_metrics[f'{name}_rmse_delta_vs_parent'] = (
                        child_rmse - parent_rmse
                        if np.isfinite(child_rmse) and np.isfinite(parent_rmse)
                        else float('nan'))

            if candidate_squared_errors:
                diagnostic_errors = {
                    'candidate': np.concatenate(candidate_squared_errors),
                    'oracle_gate': np.concatenate(oracle_squared_errors),
                }
                for prefix, errors in diagnostic_errors.items():
                    for name, mask in strata.items():
                        validation_metrics[f'{prefix}_{name}_rmse'] = (
                            float(np.sqrt(np.mean(errors[mask])))
                            if np.any(mask) else float('nan'))

            dates = target_dates
            if dates.shape != squared_errors.shape:
                raise RuntimeError(
                    'validation dates do not match prediction order: '
                    f'{dates.shape} vs {squared_errors.shape}')
            date_rmse = []
            date_tail_rmse = []
            for date in np.unique(dates[cloud_event_day]):
                day = cloud_event_day & (dates == date)
                date_rmse.append(masked_rmse(day))
                day_tail = day & ramp_ge5
                if np.any(day_tail):
                    date_tail_rmse.append(masked_rmse(day_tail))
            for name, values in [
                ('cloud_event_date_rmse', date_rmse),
                ('cloud_event_date_ge5_rmse', date_tail_rmse),
            ]:
                values = np.asarray(values, dtype=np.float64)
                validation_metrics[f'{name}_mean'] = (
                    float(values.mean()) if values.size else float('nan'))
                validation_metrics[f'{name}_median'] = (
                    float(np.median(values)) if values.size else float('nan'))
                validation_metrics[f'{name}_worst'] = (
                    float(values.max()) if values.size else float('nan'))
                validation_metrics[f'{name}_count'] = int(values.size)

            diagnostic_dir = getattr(
                self, '_phase_validation_diagnostic_dir', None)
            if diagnostic_dir and diagnostic_candidates:
                diagnostic_label = str(getattr(
                    self, '_phase_validation_diagnostic_label', 'latest'))
                diagnostic_path = os.path.join(
                    diagnostic_dir,
                    f'phase_validation_{diagnostic_label}.npz')
                phase_winner_prior = getattr(
                    self._student_model(),
                    'phase_winner_prior_probability', None)
                if not callable(phase_winner_prior):
                    raise RuntimeError(
                        'phase diagnostics require a train-only winner prior')
                np.savez_compressed(
                    diagnostic_path,
                    target_kw=np.concatenate(diagnostic_targets),
                    parent_kw=np.concatenate(diagnostic_parents),
                    candidate_kw=np.concatenate(diagnostic_candidates),
                    class_raw_correction_kw=np.concatenate(
                        diagnostic_class_raw_corrections),
                    signed_ramp_kw=signed_ramps,
                    path_support=np.concatenate(diagnostic_path_supports),
                    endpoint_phase_probability=np.concatenate(
                        diagnostic_phase_probabilities)[:, -1, :],
                    endpoint_phase_label=np.concatenate(
                        diagnostic_endpoint_phase_labels),
                    endpoint_path_delta_kw=np.concatenate(
                        diagnostic_endpoint_path_deltas),
                    class_endpoint_path_delta_kw=(
                        np.concatenate(diagnostic_class_endpoint_path_deltas)
                        if diagnostic_class_endpoint_path_deltas
                        else np.empty((len(no_path), 0), dtype=np.float32)),
                    winner_probability=np.concatenate(benefit_scores),
                    scale_probability=np.concatenate(scale_scores),
                    class_winner_probability=np.concatenate(
                        diagnostic_class_winner_probabilities),
                    class_scale_probability=np.concatenate(
                        diagnostic_class_scale_probabilities),
                    class_route_support=(
                        np.concatenate(diagnostic_class_route_supports)
                        if diagnostic_class_route_supports
                        else np.empty((len(no_path), 0), dtype=np.float32)),
                    phase_winner_prior=np.asarray(
                        phase_winner_prior().detach().cpu(), dtype=np.float64),
                    no_path=no_path,
                    round_trip=round_trip,
                    cloud_event_day=cloud_event_day,
                    sample_date=dates.astype('datetime64[D]').astype(str),
                )
        
        if was_training:
            self.model.train()
        else:
            self.model.eval()
        self._enforce_frozen_ts_mode()
        self._last_validation_monitor = (
            self._finalize_monitor_batches(monitor_store)
            if monitor_store is not None else None)
        if monitor_quality is not None:
            self.monitor.record_data_quality(
                monitor_split, monitor_quality.summary())
            self._monitor_quality_recorded.add(monitor_split)
        
        return avg_total_loss, avg_task_loss, validation_metrics

    def _validation_selection_score(self, validation_loss, validation_metrics):
        metric = getattr(self.args, 'val_selection_metric', 'loss')
        if metric == 'rmse':
            return float(validation_metrics['rmse'])
        if metric == 'cloud_event_rmse':
            return float(validation_metrics['cloud_event_rmse'])
        if metric == 'ramp_composite':
            extreme_rmse = float(validation_metrics['ramp_ge8_rmse'])
            if not np.isfinite(extreme_rmse):
                extreme_rmse = 0.0
            return float(
                validation_metrics['rmse']
                + float(getattr(self.args, 'val_extreme_metric_weight', 0.25))
                * extreme_rmse
            )
        return float(validation_loss)

    def _ordinary_validation_rmse(self, validation_metrics):
        return float(validation_metrics['ramp_lt5_rmse'])

    @staticmethod
    def _no_path_validation_rmse(validation_metrics):
        return float(validation_metrics.get('no_path_rmse', float('nan')))

    @staticmethod
    def _json_metrics(metrics, score=None, epoch=None):
        record = {}
        if epoch is not None:
            record['epoch'] = int(epoch)
        if score is not None:
            record['selection_score'] = float(score) if np.isfinite(score) else None
        for key, value in metrics.items():
            if isinstance(value, (int, np.integer)):
                record[key] = int(value)
            else:
                value = float(value)
                record[key] = value if np.isfinite(value) else None
        return record

    @staticmethod
    def _json_guard_results(results):
        payload = {}
        for name, result in results.items():
            payload[name] = {
                'value': (
                    float(result['value'])
                    if np.isfinite(result['value']) else None),
                'limit': (
                    float(result['limit'])
                    if np.isfinite(result['limit']) else None),
                'passed': bool(result['passed']),
                'mode': result['mode'],
            }
        return payload

    @staticmethod
    def _write_json(path, payload):
        temporary_path = path + '.tmp'
        with open(temporary_path, 'w') as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        os.replace(temporary_path, path)

    def _save_model_checkpoint(self, model, path):
        temporary_path = path + '.tmp'
        torch.save(model.state_dict(), temporary_path)
        os.replace(temporary_path, path)
        if self.args.data == 'LuoyangParquet':
            write_checkpoint_contract(path, self.args.luoyang_config)
        elif self.args.data == 'YLJParquet':
            write_ylj_checkpoint_contract(path, self.args.ylj_config)

    def train(self, setting):
        self._checkpoint_accepted = False
        conservative = bool(getattr(self.args, 'conservative_checkpointing', False))
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data = None
        test_loader = None
        include_training_test = bool(getattr(
            self.args, 'monitor_include_test_during_training', False))
        if include_training_test:
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        history_path = os.path.join(path, 'loss_history.csv')
        if os.path.exists(history_path):
            os.remove(history_path)

        self._apply_student_training_policy()
        self.monitor = TrainingMonitor.from_args(
            self.args, run_dir=path, stage='student', model=self.model)
        if self.monitor.enabled:
            monitored_datasets = {'train': train_data, 'val': vali_data}
            if test_data is not None:
                monitored_datasets['test'] = test_data
            self.monitor.start_run(self.model, monitored_datasets)

        if conservative and not getattr(self.args, 'student_init_path', ''):
            raise ValueError('conservative checkpointing requires --student_init_path')

        phase_strategy = self._student_strategy() == 'causal_phase_field_tail'
        self._phase_validation_diagnostic_dir = path if phase_strategy else None
        if phase_strategy:
            if not conservative:
                raise ValueError(
                    'causal_phase_field_tail requires conservative checkpointing')
            phase_stage_durations = self._phase_stage_durations()
            phase_pretrain_total = sum(phase_stage_durations.values())
            if phase_pretrain_total >= int(self.args.train_epochs):
                raise ValueError(
                    'phase-field training requires at least one active-route epoch')
            resume_from = getattr(self.args, 'phase_resume_from', 'none')
            if resume_from != 'none' and not getattr(
                self.args, 'student_init_path', ''):
                raise ValueError(
                    'phase resume requires a stage checkpoint as student initialization')
            if resume_from == 'none' and phase_stage_durations['state'] <= 0:
                raise ValueError('phase-field training requires state/path pretraining')
            learned_phase_residual = str(getattr(
                self.args, 'phase_correction_mode', 'learned_residual'
            )) == 'learned_residual'
            if (
                learned_phase_residual
                and resume_from in ['none', 'state']
                and phase_stage_durations['magnitude'] <= 0
            ):
                raise ValueError('phase-field training requires magnitude pretraining')
            if resume_from in ['none', 'state', 'magnitude'] and phase_stage_durations[
                'benefit'
            ] <= 0:
                raise ValueError('phase-field training requires benefit pretraining')
            if float(getattr(self.args, 'phase_state_aux_weight', 0.0)) <= 0:
                raise ValueError('phase state auxiliary weight must be positive')
            if float(getattr(self.args, 'phase_path_aux_weight', 0.0)) <= 0:
                raise ValueError('phase path auxiliary weight must be positive')
            if float(getattr(self.args, 'phase_benefit_aux_weight', 0.0)) <= 0:
                raise ValueError('phase benefit auxiliary weight must be positive')
            if getattr(
                self.args, 'val_selection_metric', 'loss'
            ) != 'cloud_event_rmse':
                raise ValueError(
                    'phase-field checkpoints require cloud_event_rmse selection')
            if not bool(getattr(self.args, 'val_ordinary_guard', False)):
                raise ValueError('phase-field training requires the endpoint ordinary guard')
            if not bool(getattr(self.args, 'val_no_path_guard', False)):
                raise ValueError('phase-field training requires the no-path guard')
            self._set_phase_training_stage(self._phase_stage_for_epoch(0))
        configured_tail_pretrain_epochs = int(getattr(
            self.args, 'tail_gate_pretrain_epochs', 0))
        if (
            self._student_strategy() == 'causal_solar_token_tail'
            and getattr(self._student_model(), 'tail_routing_mode', '') == 'posterior'
            and configured_tail_pretrain_epochs == 0
        ):
            prior_stats = self._student_model().tail_prior_statistics()
            if (
                prior_stats.get('event_total_count', 0.0) <= 0
                or not prior_stats.get('frozen', False)
            ):
                raise ValueError(
                    'posterior tail routing without gate pretraining requires '
                    'a checkpoint containing frozen train-only priors')
        ts_digest_before = self._ts_branch_digest() if self._ts_must_stay_frozen() else None
        parent_digest_before = (
            self._frozen_parent_digest()
            if self._student_strategy() in [
                'causal_solar_token_tail', 'causal_phase_field_tail'
            ]
            else None
        )

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = None
        if not conservative:
            checkpoint_callback = None
            if self.args.data == 'LuoyangParquet':
                checkpoint_callback = lambda checkpoint: write_checkpoint_contract(
                    checkpoint, self.args.luoyang_config)
            elif self.args.data == 'YLJParquet':
                checkpoint_callback = lambda checkpoint: write_ylj_checkpoint_contract(
                    checkpoint, self.args.ylj_config)
            early_stopping = EarlyStopping(
                patience=self.args.patience, verbose=True,
                on_save=checkpoint_callback)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 记录第一个epoch的时间用于预估
        first_epoch_time = None
        estimated_total_time = None

        if self.args.kd_type == 'inter_causal':
            criterion_kd = inter_causal_kd()
        else:
            criterion_kd = response_kd()
        causal_boundary_checked = False
        self._last_epoch_phase_components = {
            'phase_motion_consistency': float('nan'),
            'phase_state': float('nan'),
            'phase_path': float('nan'),
            'phase_magnitude': float('nan'),
            'phase_benefit': float('nan'),
            'phase_no_path_stable': float('nan'),
        }
        teacher_required = self.model_teacher is not None and any([
            float(getattr(self.args, 'kd_loss_weight_sim', 0.0)) > 0,
            float(getattr(self.args, 'kd_loss_weight_inter', 0.0)) > 0,
            float(getattr(self.args, 'kd_extreme_response_weight', 0.0)) > 0,
        ])
        print(f'train_teacher_forward_enabled={teacher_required}')
        if (
            self.monitor.enabled
            and self.args.data in {'LuoyangParquet', 'YLJParquet'}
        ):
            initial_student_payload = self._collect_monitor_predictions(
                vali_loader, role='student')
            self._record_monitor_forecast(
                initial_student_payload, epoch=0, split='val',
                role='student', phase='initial')
            if self.model_teacher is not None:
                self._monitor_teacher_reference = self._collect_monitor_predictions(
                    vali_loader, role='teacher')
                self._record_monitor_forecast(
                    self._monitor_teacher_reference, epoch=0, split='val',
                    role='teacher', phase='reference')
                initial_distillation = self._distillation_monitor_statistics(
                    initial_student_payload, self._monitor_teacher_reference,
                    min_count=self.monitor.min_slice_count,
                    min_days=self.monitor.min_slice_days)
                initial_distillation.update({
                    'epoch': 0,
                    'phase': 'initial',
                    'teacher_strategy': self._teacher_strategy(),
                    'student_strategy': self._student_strategy(),
                })
                self.monitor.record_distillation(initial_distillation)

                teacher_ablations = (
                    ['ts_only', 'no_img', 'no_weather']
                    if self.args.data == 'LuoyangParquet'
                    else ['ts_only']
                )
                for ablation in teacher_ablations:
                    if ablation == self._teacher_strategy():
                        continue
                    ablation_payload = self._collect_monitor_predictions(
                        vali_loader, role='teacher', strategy=ablation)
                    self._record_monitor_forecast(
                        ablation_payload, epoch=0, split='val',
                        role=f'teacher_{ablation}', phase='ablation')
            if self._student_strategy() not in {'ts_only', 'ts'}:
                student_ts_payload = self._collect_monitor_predictions(
                    vali_loader, role='student', strategy='ts_only')
                self._record_monitor_forecast(
                    student_ts_payload, epoch=0, split='val',
                    role='student_ts_only', phase='ablation')

        def history_record(epoch, train_values, vali_values, test_values,
                           selection_score, eligible, decision_reason,
                           early_stop_counter, best_score, epoch_time_sec):
            vali_loss, vali_ts_loss, vali_metrics = vali_values
            test_loss, test_ts_loss, test_metrics = test_values
            prior_getter = getattr(
                self._student_model(), 'tail_prior_statistics', None)
            prior_stats = prior_getter() if callable(prior_getter) else {}
            record = {
                "epoch": epoch,
                "train_steps": train_steps,
                "train_samples": len(train_data),
                "vali_samples": len(vali_data),
                "test_samples": len(test_data) if test_data is not None else 0,
                "train_loss": train_values[0],
                "train_task_loss": train_values[1],
                "train_kd_sim_loss": train_values[2],
                "train_kd_task_loss": train_values[3],
                "train_auxiliary_loss": train_values[4],
                "train_kd_extreme_response_loss": train_values[5],
                "train_aux_tail_event_loss": train_values[6],
                "train_aux_tail_direction_loss": train_values[7],
                "train_aux_event_residual_loss": train_values[8],
                "train_aux_clear_sky_pinball_loss": train_values[9],
                "tail_train_event_prior": prior_stats.get(
                    'event_prior', float('nan')),
                "tail_train_up_given_event_prior": prior_stats.get(
                    'up_given_event_prior', float('nan')),
                "tail_train_event_count": prior_stats.get(
                    'event_positive_count', float('nan')),
                "tail_train_seen_count": prior_stats.get(
                    'event_total_count', float('nan')),
                "vali_loss": vali_loss,
                "vali_ts_loss": vali_ts_loss,
                "vali_rmse": vali_metrics['rmse'],
                "vali_ramp_lt5_rmse": vali_metrics['ramp_lt5_rmse'],
                "vali_ramp_ge5_rmse": vali_metrics['ramp_ge5_rmse'],
                "vali_ramp_ge8_rmse": vali_metrics['ramp_ge8_rmse'],
                "vali_cloud_event_rmse": vali_metrics['cloud_event_rmse'],
                "vali_cloud_event_ramp_lt5_rmse": vali_metrics['cloud_event_ramp_lt5_rmse'],
                "vali_cloud_event_ramp_ge5_rmse": vali_metrics['cloud_event_ramp_ge5_rmse'],
                "vali_cloud_event_ramp_ge8_rmse": vali_metrics['cloud_event_ramp_ge8_rmse'],
                "vali_count": vali_metrics['count'],
                "vali_ramp_lt5_count": vali_metrics['ramp_lt5_count'],
                "vali_ramp_ge5_count": vali_metrics['ramp_ge5_count'],
                "vali_ramp_ge8_count": vali_metrics['ramp_ge8_count'],
                "vali_cloud_event_count": vali_metrics['cloud_event_count'],
                "vali_cloud_event_ramp_lt5_count": vali_metrics['cloud_event_ramp_lt5_count'],
                "vali_cloud_event_ramp_ge5_count": vali_metrics['cloud_event_ramp_ge5_count'],
                "vali_cloud_event_ramp_ge8_count": vali_metrics['cloud_event_ramp_ge8_count'],
                "selection_score": selection_score,
                "checkpoint_eligible": eligible,
                "checkpoint_decision": decision_reason,
                "test_loss": test_loss,
                "test_ts_loss": test_ts_loss,
                "test_rmse": test_metrics['rmse'],
                "test_ramp_lt5_rmse": test_metrics['ramp_lt5_rmse'],
                "test_ramp_ge5_rmse": test_metrics['ramp_ge5_rmse'],
                "test_ramp_ge8_rmse": test_metrics['ramp_ge8_rmse'],
                "test_cloud_event_rmse": test_metrics['cloud_event_rmse'],
                "test_cloud_event_ramp_lt5_rmse": test_metrics['cloud_event_ramp_lt5_rmse'],
                "test_cloud_event_ramp_ge5_rmse": test_metrics['cloud_event_ramp_ge5_rmse'],
                "test_cloud_event_ramp_ge8_rmse": test_metrics['cloud_event_ramp_ge8_rmse'],
                "learning_rate": model_optim.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time_sec,
                "early_stop_counter": early_stop_counter,
                "best_validation_score": best_score,
            }
            record['phase_training_stage'] = getattr(
                self, '_phase_training_stage', None)
            for name, value in getattr(
                self, '_last_epoch_phase_components', {}
            ).items():
                record[f'train_aux_{name}_loss'] = value
            for name, value in vali_metrics.items():
                record.setdefault(f'vali_{name}', value)
            return record

        empty_metrics = {
            'rmse': float('nan'),
            'ramp_lt5_rmse': float('nan'),
            'ramp_ge5_rmse': float('nan'),
            'ramp_ge8_rmse': float('nan'),
            'count': 0,
            'ramp_lt5_count': 0,
            'ramp_ge5_count': 0,
            'ramp_ge8_count': 0,
            'cloud_event_rmse': float('nan'),
            'cloud_event_ramp_lt5_rmse': float('nan'),
            'cloud_event_ramp_ge5_rmse': float('nan'),
            'cloud_event_ramp_ge8_rmse': float('nan'),
            'cloud_event_count': 0,
            'cloud_event_ramp_lt5_count': 0,
            'cloud_event_ramp_ge5_count': 0,
            'cloud_event_ramp_ge8_count': 0,
        }
        skipped_test_values = (float('nan'), float('nan'), empty_metrics)
        checkpoint_path = os.path.join(path, 'checkpoint.pth')
        input_checkpoint_path = os.path.join(path, 'input_checkpoint.pth')
        checkpoint_guard = None
        input_metrics = None
        input_score = None
        best_eligible_metrics = None
        best_diagnostic_metrics = None
        best_diagnostic_score = float('inf')
        best_diagnostic_epoch = None

        if conservative:
            if (
                phase_strategy
                and self._phase_stage_for_epoch(0) == 'active'
                and str(getattr(
                    self.args, 'phase_benefit_mode', 'winner'))
                == 'constrained_stack'
                and bool(int(getattr(
                    self.args, 'phase_validation_calibration', 0)))
            ):
                # A resumed benefit checkpoint contains a trained residual
                # head.  Epoch 0 must still measure the immutable frozen
                # parent, otherwise all guards inherit pre-existing leakage.
                self._set_phase_family_calibration(0.0, 0.0)
            self.model.eval()
            self._enforce_frozen_ts_mode()
            self._phase_validation_diagnostic_label = 'input'
            input_vali_loss, input_vali_ts_loss, input_metrics = self.vali(
                vali_data, vali_loader, criterion)
            self._enforce_frozen_ts_mode()
            input_score = self._validation_selection_score(input_vali_loss, input_metrics)
            phase_metric_guards = None
            if phase_strategy:
                phase_metric_guards = {
                    'cloud_event_ramp_lt5_rmse': {
                        'baseline': input_metrics[
                            'cloud_event_ramp_lt5_rmse'],
                        'mode': 'non_degrade',
                        'relative_tolerance': float(getattr(
                            self.args,
                            'val_cloud_ordinary_relative_tolerance', 0.0)),
                    },
                    'cloud_event_no_path_rmse': {
                        'baseline': input_metrics['cloud_event_no_path_rmse'],
                        'mode': 'non_degrade',
                        'relative_tolerance': float(getattr(
                            self.args,
                            'val_cloud_no_path_relative_tolerance', 0.0)),
                    },
                    'cloud_event_ramp_ge5_rmse': {
                        'baseline': input_metrics['cloud_event_ramp_ge5_rmse'],
                        'mode': 'improve',
                        'min_improvement': float(getattr(
                            self.args, 'val_tail_min_improvement', 0.0)),
                    },
                    'cloud_event_ramp_ge8_rmse': {
                        'baseline': input_metrics['cloud_event_ramp_ge8_rmse'],
                        'mode': 'non_degrade',
                        'relative_tolerance': float(getattr(
                            self.args,
                            'val_extreme_relative_tolerance', 0.0)),
                    },
                }
            checkpoint_guard = ConservativeCheckpointGuard(
                input_score=input_score,
                input_ordinary_rmse=self._ordinary_validation_rmse(input_metrics),
                patience=self.args.patience,
                min_improvement=getattr(self.args, 'val_min_improvement', 0.0),
                ordinary_guard=getattr(self.args, 'val_ordinary_guard', False),
                ordinary_relative_tolerance=getattr(
                    self.args, 'val_ordinary_relative_tolerance', 0.0),
                input_no_path_rmse=self._no_path_validation_rmse(input_metrics),
                no_path_guard=getattr(self.args, 'val_no_path_guard', False),
                no_path_relative_tolerance=getattr(
                    self.args, 'val_no_path_relative_tolerance', 0.0),
                metric_guards=phase_metric_guards,
            )
            self._save_model_checkpoint(self.model, input_checkpoint_path)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            try:
                os.link(input_checkpoint_path, checkpoint_path)
            except OSError:
                shutil.copy2(input_checkpoint_path, checkpoint_path)
            if self.args.data == 'LuoyangParquet':
                write_checkpoint_contract(checkpoint_path, self.args.luoyang_config)
            elif self.args.data == 'YLJParquet':
                write_ylj_checkpoint_contract(checkpoint_path, self.args.ylj_config)
            print(
                'Epoch 0 input checkpoint | '
                f'Vali RMSE: {input_metrics["rmse"]:.7f}, '
                f'<5kW RMSE: {input_metrics["ramp_lt5_rmse"]:.7f}, '
                f'no-path RMSE: {input_metrics.get("no_path_rmse", float("nan")):.7f}, '
                f'Cloud-event RMSE: {input_metrics["cloud_event_rmse"]:.7f}, '
                f'>=8kW RMSE: {input_metrics["ramp_ge8_rmse"]:.7f}, '
                f'selection: {input_score:.7f}')
            append_metrics_row(history_path, history_record(
                0,
                (float('nan'),) * 10,
                (input_vali_loss, input_vali_ts_loss, input_metrics),
                skipped_test_values,
                input_score,
                True,
                'input_baseline_saved',
                0,
                input_score,
                0.0,
            ))
        
        epochs_completed = 0
        selected_monitor_epoch = 0
        phase_stage_best = {
            stage: {
                'score': float('inf'),
                'validation_loss': float('inf'),
                'epoch': None,
            }
            for stage in ['state', 'magnitude', 'benefit']
        }
        previous_phase_stage = (
            self._phase_stage_for_epoch(0) if phase_strategy else None)
        for epoch in range(self.args.train_epochs):
            epochs_completed = epoch + 1
            iter_count = 0
            train_loss = []
            train_task_loss = []
            train_kd_sim_loss = []
            train_kd_task_loss = []
            train_auxiliary_loss = []
            train_kd_extreme_response_loss = []
            valid_privileged_interventions = 0
            skipped_privileged_interventions = 0
            effective_privileged_intervention_batches = 0
            valid_extreme_response_samples = 0
            train_aux_tail_event_loss = []
            train_aux_tail_direction_loss = []
            train_aux_event_residual_loss = []
            train_aux_clear_sky_pinball_loss = []
            train_valid_counts = []
            train_gradient_norms = []
            train_gradient_group_samples = {}
            clipped_gradient_steps = 0
            nonfinite_gradient_steps = 0
            amp_skipped_steps = 0
            train_sim_img_means = []
            train_sim_weather_means = []
            train_student_feature_stds = []
            train_teacher_feature_stds = []
            train_positive_cosines = []
            train_negative_cosines = []
            train_intervention_input_deltas = []
            train_intervention_feature_deltas = []
            train_image_modality_active = []
            train_weather_modality_active = []
            processed_train_samples = 0
            student_feature_dim = None
            train_phase_component_losses = {
                'phase_motion_consistency': [],
                'phase_state': [],
                'phase_path': [],
                'phase_magnitude': [],
                'phase_benefit': [],
                'phase_no_path_stable': [],
            }
            
            phase_stage = (
                self._phase_stage_for_epoch(epoch) if phase_strategy else None)
            phase_stage_changed = phase_strategy and phase_stage != previous_phase_stage
            if phase_stage_changed:
                self._set_phase_training_stage(phase_stage)
                self.model.zero_grad(set_to_none=True)
                model_optim = self._select_optimizer()
                previous_phase_stage = phase_stage
                if phase_stage == 'active':
                    if checkpoint_guard is not None:
                        checkpoint_guard.reset_patience()
                    if early_stopping is not None:
                        early_stopping.counter = 0
                        early_stopping.early_stop = False
                print(
                    f'student_phase_stage={phase_stage} epoch={epoch + 1} '
                    'optimizer_reset=true checkpoint_patience_reset='
                    f'{phase_stage == "active"}')

            self.model.train()
            if self.model_teacher is not None:
                self.model_teacher.eval()
            self._enforce_frozen_ts_mode()
            if phase_strategy:
                # model.train() recursively enables every child; restore the
                # stage-specific train/eval and gradient boundary afterwards.
                self._set_phase_training_stage(phase_stage)
                if (
                    phase_stage == 'active'
                    and str(getattr(
                        self.args, 'phase_benefit_mode', 'winner'))
                    == 'constrained_stack'
                    and bool(int(getattr(
                        self.args, 'phase_validation_calibration', 0)))
                ):
                    # Train the conditional head in its canonical scale.  The
                    # two validation projections are model-selection state and
                    # are recomputed after each epoch.
                    self._set_phase_family_calibration(1.0, 1.0)
                if epoch == 0:
                    route_enabled = bool(getattr(
                        self._student_model(), 'phase_route_enabled',
                        torch.tensor(False)).item())
                    print(
                        f'student_phase_stage={phase_stage} epoch=1 '
                        f'route_enabled={str(route_enabled).lower()}')
            tail_pretrain_epochs = int(getattr(
                self.args, 'tail_gate_pretrain_epochs', 0))
            tail_gate_pretraining = (
                self._student_strategy() == 'causal_solar_token_tail'
                and epoch < tail_pretrain_epochs
            )
            self._set_tail_magnitude_trainable(not tail_gate_pretraining)
            tail_magnitude_unlocked = (
                self._student_strategy() == 'causal_solar_token_tail'
                and tail_pretrain_epochs > 0
                and epoch == tail_pretrain_epochs
            )
            if tail_magnitude_unlocked:
                prior_freezer = getattr(
                    self._student_model(), 'freeze_tail_priors', None)
                if callable(prior_freezer):
                    prior_freezer()
                if checkpoint_guard is not None:
                    checkpoint_guard.reset_patience()
                if early_stopping is not None:
                    early_stopping.counter = 0
                    early_stopping.early_stop = False
                print(
                    'student_tail_magnitude_unlocked=true '
                    f'epoch={epoch + 1} checkpoint_patience_reset=true '
                    'train_tail_priors_frozen=true')
            freeze_ts = (
                self._ts_must_stay_frozen()
                or epoch < int(getattr(self.args, 'freeze_ts_epochs', 0))
            )
            self._set_ts_branch_trainable(not freeze_ts)
            if epoch == 0 or (
                not self._ts_must_stay_frozen()
                and epoch == int(getattr(self.args, 'freeze_ts_epochs', 0))
            ):
                print(f"student_ts_branch_frozen={freeze_ts} epoch={epoch + 1}")
            if self._student_strategy() == 'causal_solar_token_tail' and (
                epoch == 0 or epoch == tail_pretrain_epochs
            ):
                print(
                    'student_tail_gate_pretraining='
                    f'{tail_gate_pretraining} epoch={epoch + 1} '
                    f'pretrain_epochs={tail_pretrain_epochs}')
            epoch_time = time.time()
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            parameter_snapshot = (
                self.monitor.start_parameter_snapshot(self.model)
                if self.monitor.enabled else None)
            train_quality = None
            if self.monitor.enabled and epoch == 0:
                fields = getattr(train_data, 'fields', {})
                feature_names = (
                    fields.get('timeseries_columns')
                    or fields.get('time_series_columns'))
                train_quality = self.monitor.new_data_quality_accumulator(
                    'train', feature_names=feature_names)
            
            for i, batch in enumerate(train_loader):
                max_train_batches = int(getattr(
                    self.args, 'max_train_batches', 0))
                if max_train_batches > 0 and i >= max_train_batches:
                    break
                (batch_x, batch_y, batch_x_mark, batch_y_mark,
                 batch_x_img, batch_y_img, batch_x_weather, batch_y_weather,
                 sample_weight, phase_targets) = self._unpack_batch(batch)
                iter_count += 1
                
                # 数据预处理
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                processed_train_samples += int(batch_x.shape[0])
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                batch_x_img = batch_x_img.float().to(self.device, non_blocking=True)
                batch_y_img = batch_y_img.float().to(self.device, non_blocking=True)
                batch_x_weather = batch_x_weather.float().to(self.device, non_blocking=True)
                batch_y_weather = batch_y_weather.float().to(self.device, non_blocking=True)
                if sample_weight is not None:
                    sample_weight = sample_weight.float().to(self.device, non_blocking=True)
                phase_targets = self._phase_targets_to_device(phase_targets)
                phase_lead_marks = (
                    phase_targets.get('lead_marks') if phase_targets is not None else None)
                
                # 单层训练
                model_optim.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    if not causal_boundary_checked:
                        self._check_student_future_image_invariance(
                            batch_x, batch_x_img, batch_y_img, batch_x_weather,
                            batch_y_weather, batch_x_mark, batch_y_mark,
                            phase_lead_marks)
                        causal_boundary_checked = True
                    # 前向传播
                    outputs, sim_img_ts, sim_weather_ts, drop_img, drop_weather, feat = self._student_forward(
                        batch_x, batch_x_img, batch_y_img, batch_x_weather,
                        batch_y_weather, batch_x_mark, batch_y_mark,
                        phase_lead_marks,
                        None if phase_targets is None else phase_targets.get('image_mask'))
                    if teacher_required:
                        with torch.no_grad():
                            (outputs_teacher, sim_img_ts_teacher,
                             sim_weather_ts_teacher, feat_teacher) = self._teacher_forward(
                                batch_x, batch_x_img, batch_y_img,
                                batch_x_weather, batch_y_weather, phase_targets)
                    else:
                        outputs_teacher = outputs.detach()
                        sim_img_ts_teacher = sim_img_ts.detach()
                        sim_weather_ts_teacher = sim_weather_ts.detach()
                        feat_teacher = feat.detach()

                    train_sim_img_means.append(float(
                        sim_img_ts.detach().float().mean().item()))
                    train_sim_weather_means.append(float(
                        sim_weather_ts.detach().float().mean().item()))
                    train_student_feature_stds.append(float(
                        feat.detach().float().std(unbiased=False).item()))
                    student_feature_dim = int(feat.shape[-1])
                    train_teacher_feature_stds.append(float(
                        feat_teacher.detach().float().std(unbiased=False).item()))
                    train_image_modality_active.append(float(
                        drop_img.detach().float().mean().item()))
                    train_weather_modality_active.append(float(
                        drop_weather.detach().float().mean().item()))
                    if teacher_required:
                        paired_positive = torch.nn.functional.cosine_similarity(
                            feat.reshape(-1, feat.shape[-1]).float(),
                            feat_teacher.detach().reshape(
                                -1, feat_teacher.shape[-1]).float(),
                            dim=-1,
                        )
                        train_positive_cosines.append(float(
                            paired_positive.mean().item()))

                    if self.args.data == 'Folsom':
                        f_dim = -42 if self.args.features == 'MS' else 0
                    else:
                        f_dim = -1 if self.args.features == 'MS' else 0

                    outputs_target = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:]
                    outputs_teacher_target = outputs_teacher[:, -self.args.pred_len:, f_dim:]

                    # 计算损失
                    target_mask = None if phase_targets is None else phase_targets.get('target_mask')
                    valid_target_count = (
                        int(batch_y_target.numel())
                        if target_mask is None
                        else int(target_mask.to(dtype=torch.bool).sum().item())
                    )
                    if train_quality is not None:
                        train_quality.update(
                            features=batch_x,
                            feature_mask=(
                                None if phase_targets is None
                                else phase_targets.get('timeseries_mask')),
                            target=batch_y_target,
                            current_power=self._current_pv(batch_x),
                            target_mask=target_mask,
                            image_mask=(
                                None if phase_targets is None
                                else phase_targets.get('image_mask')),
                            image_age_minutes=(
                                self._monitor_image_age(phase_targets)),
                            source_codes=(
                                self._monitor_source_codes(phase_targets)),
                            issue_time_ns=(
                                None if phase_targets is None
                                else phase_targets.get('issue_time_ns')),
                            coverage_masks=(
                                self._monitor_coverage_masks(phase_targets)),
                            privileged_teacher_enabled=(
                                None if phase_targets is None
                                else phase_targets.get(
                                    'privileged_teacher_enabled')),
                            history_images_enabled=(
                                None if phase_targets is None
                                else phase_targets.get('history_images_enabled')),
                            current_power_valid=(
                                None if phase_targets is None
                                else phase_targets.get('current_power_valid')),
                        )
                    task_loss = self._criterion_loss(
                        outputs_target, batch_y_target, criterion, sample_weight,
                        target_mask)
                    auxiliary_loss = self._auxiliary_loss(
                        batch_y_target, self._current_pv(batch_x), sample_weight,
                        phase_targets)
                    if self.args.kd_loss_weight_sim > 0:
                        loss_kd_sim = self.args.kd_loss_weight_sim * (
                            criterion(sim_img_ts, sim_img_ts_teacher) * drop_img.mean()
                            + criterion(sim_weather_ts, sim_weather_ts_teacher) * drop_weather.mean()
                        ) / 2
                    else:
                        loss_kd_sim = torch.zeros((), device=self.device)
                    loss_kd_task = torch.zeros((), device=self.device)
                    loss_kd_extreme_response = torch.zeros((), device=self.device)

                    if (
                        self.args.kd_loss_weight_inter > 0
                        and self.args.kd_type == 'inter_causal'
                    ):
                        intervention_attempt_count = int(batch_y_img.shape[0])
                        intervention_valid_count = 0
                        with torch.no_grad():
                            shuffle_indices = self._privileged_derangement(
                                batch_y_img.shape[0], batch_y_img.device)
                            if shuffle_indices is None:
                                inter_feat_img = None
                            else:
                                perturb_batch_y_img = batch_y_img[shuffle_indices]
                                perturb_batch_y_weather = batch_y_weather[shuffle_indices]
                                perturb_metadata = phase_targets
                                if phase_targets is not None:
                                    perturb_metadata = dict(phase_targets)
                                    future_mask = phase_targets.get(
                                        'teacher_future_image_mask')
                                    if future_mask is not None:
                                        perturb_metadata['teacher_future_image_mask'] = (
                                            future_mask[shuffle_indices])
                                image_input_delta = (
                                    perturb_batch_y_img - batch_y_img
                                ).abs().flatten(start_dim=1).amax(dim=1)
                                weather_input_delta = (
                                    perturb_batch_y_weather - batch_y_weather
                                ).abs().flatten(start_dim=1).amax(dim=1)
                                input_deltas = torch.maximum(
                                    image_input_delta, weather_input_delta)
                                train_intervention_input_deltas.extend(
                                    input_deltas.detach().cpu().tolist()
                                )
                                if not bool((input_deltas > 0).any().item()):
                                    inter_feat_img = None
                                else:
                                    _, _, _, inter_feat_img = self._teacher_forward(
                                        batch_x, batch_x_img, perturb_batch_y_img,
                                        batch_x_weather, perturb_batch_y_weather,
                                        perturb_metadata)
                                    feature_deltas = (
                                        inter_feat_img - feat_teacher
                                    ).abs().flatten(start_dim=1).amax(dim=1)
                                    train_intervention_feature_deltas.extend(
                                        feature_deltas.detach().cpu().tolist())
                                    valid_intervention_mask = (
                                        (input_deltas > 0)
                                        & (feature_deltas > float(
                                            self.args.kd_intervention_min_feature_delta))
                                    )
                                    intervention_valid_count = int(
                                        valid_intervention_mask.sum().item())
                                    if float(feature_deltas.max().item()) <= float(
                                            self.args.kd_intervention_min_feature_delta):
                                        inter_feat_img = None
                        if inter_feat_img is None:
                            loss_kd_task = torch.zeros((), device=self.device)
                        else:
                            effective_privileged_intervention_batches += 1
                            loss_kd_task = criterion_kd(
                                feat, feat_teacher.detach(), inter_feat_img.detach()
                            ) * self.args.kd_loss_weight_inter
                            paired_negative = torch.nn.functional.cosine_similarity(
                                feat.reshape(-1, feat.shape[-1]).float(),
                                inter_feat_img.detach().reshape(
                                    -1, inter_feat_img.shape[-1]).float(),
                                dim=-1,
                            )
                            train_negative_cosines.append(float(
                                paired_negative[valid_intervention_mask]
                                .mean().item()))
                        valid_privileged_interventions += intervention_valid_count
                        skipped_privileged_interventions += (
                            intervention_attempt_count - intervention_valid_count)
                    elif (
                        self.args.kd_loss_weight_inter > 0
                        and self.args.kd_type == 'spatial_causal'
                    ):
                        student_spatial = getattr(
                            self._student_model(),
                            'last_spatial_forecast_projected', None)
                        teacher_spatial = getattr(
                            self._teacher_model(),
                            'last_image_spatial_target', None)
                        if student_spatial is None or teacher_spatial is None:
                            raise RuntimeError(
                                'spatial_causal KD requires student forecast and '
                                'teacher future-image spatial maps')
                        if student_spatial.shape != teacher_spatial.shape:
                            raise RuntimeError(
                                'spatial_causal KD shape mismatch: '
                                f'{tuple(student_spatial.shape)} vs '
                                f'{tuple(teacher_spatial.shape)}')
                        student_spatial = torch.nn.functional.normalize(
                            student_spatial, dim=1, eps=1e-6)
                        teacher_spatial = torch.nn.functional.normalize(
                            teacher_spatial.detach(), dim=1, eps=1e-6)
                        loss_kd_task = self.args.kd_loss_weight_inter * (
                            1.0 - (student_spatial * teacher_spatial).sum(dim=1)
                        ).mean()
                    elif (
                        self.args.kd_loss_weight_inter > 0
                        and self.args.kd_type == 'solar_token_causal'
                    ):
                        student_model = self._student_model()
                        student_token = getattr(
                            student_model, 'last_solar_token', None)
                        solar_logits = getattr(
                            student_model, 'last_solar_attention_logits', None)
                        teacher_spatial = getattr(
                            self._teacher_model(), 'last_image_spatial_target', None)
                        if (
                            student_token is None
                            or solar_logits is None
                            or teacher_spatial is None
                        ):
                            raise RuntimeError(
                                'solar_token_causal KD requires the causal student '
                                'token, calibrated solar attention and teacher future map')
                        if teacher_spatial.dim() != 4:
                            raise RuntimeError(
                                'teacher future map must have shape [B,C,H,W]')
                        spatial_size = teacher_spatial.shape[-2] * teacher_spatial.shape[-1]
                        if solar_logits.shape != (teacher_spatial.shape[0], spatial_size):
                            raise RuntimeError(
                                'solar attention does not align with the teacher map: '
                                f'{tuple(solar_logits.shape)} vs {tuple(teacher_spatial.shape)}')
                        solar_attention = torch.softmax(solar_logits, dim=-1)
                        teacher_token = (
                            teacher_spatial.detach().flatten(start_dim=2)
                            * solar_attention.detach().unsqueeze(1)
                        ).sum(dim=-1)
                        if student_token.shape != teacher_token.shape:
                            raise RuntimeError(
                                'solar token KD shape mismatch: '
                                f'{tuple(student_token.shape)} vs {tuple(teacher_token.shape)}')
                        student_token = torch.nn.functional.normalize(
                            student_token, dim=-1, eps=1e-6)
                        teacher_token = torch.nn.functional.normalize(
                            teacher_token, dim=-1, eps=1e-6)
                        loss_kd_task = self.args.kd_loss_weight_inter * (
                            1.0 - (student_token * teacher_token).sum(dim=-1)
                        ).mean()
                    elif (
                        self.args.kd_loss_weight_inter > 0
                        and self.args.kd_type == 'response'
                    ):
                        loss_kd_task = criterion_kd(
                            outputs_target, outputs_teacher_target.detach()
                        ) * self.args.kd_loss_weight_inter

                    extreme_response_weight = float(
                        getattr(self.args, 'kd_extreme_response_weight', 0.0))
                    if extreme_response_weight > 0:
                        threshold = float(
                            getattr(self.args, 'kd_extreme_response_threshold_kw', 5.0))
                        current_pv = self._current_pv(batch_x)
                        extreme_mask = (
                            (batch_y_target - current_pv).abs().flatten(start_dim=1).max(dim=1).values
                            >= threshold
                        ).to(outputs_target.dtype)
                        valid_extreme_response_samples += int(
                            extreme_mask.sum().item())
                        per_sample_response = (
                            outputs_target - outputs_teacher_target.detach()
                        ).pow(2).flatten(start_dim=1).mean(dim=1)
                        loss_kd_extreme_response = extreme_response_weight * (
                            per_sample_response * extreme_mask
                        ).sum() / extreme_mask.sum().clamp_min(1.0)

                    loss = (
                        task_loss + auxiliary_loss + loss_kd_sim + loss_kd_task
                        + loss_kd_extreme_response
                    )
                
                # 反向传播
                detailed_gradient_stats = None
                if self.args.use_amp:
                    amp_scale_before = float(scaler.get_scale())
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    optimized_parameters = self._optimizer_parameters(model_optim)
                    gradients_finite = all(
                        parameter.grad is None
                        or bool(torch.isfinite(parameter.grad).all().item())
                        for parameter in optimized_parameters
                    )
                    if gradients_finite:
                        if self.monitor.enabled and (
                            (i + 1) % int(getattr(
                                self.args, 'monitor_gradient_interval', 50)) == 0
                        ):
                            detailed_gradient_stats = (
                                self._gradient_group_statistics())
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            optimized_parameters, max_norm=1.0)
                    else:
                        gradient_norm = torch.tensor(
                            float('nan'), device=self.device)
                        nonfinite_gradient_steps += 1
                    scaler.step(model_optim)
                    scaler.update()
                    amp_scale_after = float(scaler.get_scale())
                    if not gradients_finite or amp_scale_after < amp_scale_before:
                        amp_skipped_steps += 1
                else:
                    loss.backward()
                    optimized_parameters = self._optimizer_parameters(model_optim)
                    if self.monitor.enabled and (
                        (i + 1) % int(getattr(
                            self.args, 'monitor_gradient_interval', 50)) == 0
                    ):
                        detailed_gradient_stats = self._gradient_group_statistics()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        optimized_parameters, max_norm=1.0,
                        error_if_nonfinite=True)
                    model_optim.step()
                gradient_norm_value = float(gradient_norm.detach().item())
                if np.isfinite(gradient_norm_value):
                    train_gradient_norms.append(gradient_norm_value)
                    if gradient_norm_value > 1.0:
                        clipped_gradient_steps += 1
                if detailed_gradient_stats is not None:
                    for group, values in detailed_gradient_stats.items():
                        train_gradient_group_samples.setdefault(
                            group, []).append(values['grad_norm'])
                non_finite_parameters = [
                    name for name, parameter in self._student_model().named_parameters()
                    if parameter.requires_grad
                    and not bool(torch.isfinite(parameter).all().item())
                ]
                if non_finite_parameters:
                    raise FloatingPointError(
                        'optimizer produced non-finite student parameters: '
                        + ','.join(non_finite_parameters))
                
                train_loss.append(loss.item())
                train_valid_counts.append(valid_target_count)
                train_task_loss.append(task_loss.item())
                train_kd_sim_loss.append(loss_kd_sim.item())
                train_kd_task_loss.append(loss_kd_task.item())
                train_auxiliary_loss.append(auxiliary_loss.item())
                train_kd_extreme_response_loss.append(loss_kd_extreme_response.item())
                auxiliary_components = getattr(
                    self, '_last_auxiliary_components', {})
                train_aux_tail_event_loss.append(float(
                    auxiliary_components.get(
                        'tail_event', torch.zeros((), device=self.device)).item()))
                train_aux_tail_direction_loss.append(float(
                    auxiliary_components.get(
                        'tail_direction', torch.zeros((), device=self.device)).item()))
                train_aux_event_residual_loss.append(float(
                    auxiliary_components.get(
                        'event_residual', torch.zeros((), device=self.device)).item()))
                train_aux_clear_sky_pinball_loss.append(float(
                    auxiliary_components.get(
                        'clear_sky_pinball', torch.zeros((), device=self.device)).item()))
                for name in train_phase_component_losses:
                    train_phase_component_losses[name].append(float(
                        auxiliary_components.get(
                            name, torch.zeros((), device=self.device)).item()))
                
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            epoch_duration = time.time() - epoch_time
            if (
                self.args.kd_loss_weight_inter > 0
                and self.args.kd_type == 'inter_causal'
            ):
                if effective_privileged_intervention_batches == 0:
                    raise RuntimeError(
                        'inter_causal KD found no effective privileged Teacher '
                        f'intervention in epoch {epoch + 1}; skipped='
                        f'{skipped_privileged_interventions}')
                print(
                    'privileged_interventions='
                    f'{valid_privileged_interventions} effective samples, '
                    f'{skipped_privileged_interventions} skipped samples, '
                    f'{effective_privileged_intervention_batches} effective batches')
            
            # 记录第一个epoch的时间并预估总时间
            if epoch == 0:
                first_epoch_time = epoch_duration
                estimated_total_time = first_epoch_time * self.args.train_epochs
                print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, epoch_duration))
                print("📊 时间预估:")
                print(f"   第一个epoch耗时: {first_epoch_time:.2f}秒")
                print(f"   预估总训练时间: {estimated_total_time:.2f}秒 ({estimated_total_time/3600:.2f}小时)")
                print(f"   预估剩余时间: {(estimated_total_time - first_epoch_time):.2f}秒 ({(estimated_total_time - first_epoch_time)/3600:.2f}小时)")
            else:
                print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, epoch_duration))
                if estimated_total_time:
                    elapsed_time = epoch_duration * (epoch + 1)
                    remaining_time = estimated_total_time - elapsed_time
                    print(f"   已用时间: {elapsed_time:.2f}秒 ({elapsed_time/3600:.2f}小时)")
                    print(f"   预估剩余时间: {remaining_time:.2f}秒 ({remaining_time/3600:.2f}小时)")
                    print(f"   预估完成时间: {remaining_time/3600:.2f}小时")
            
            if not train_loss or not sum(train_valid_counts):
                raise RuntimeError('training produced no valid target points')

            def weighted_train_mean(values):
                return float(np.average(values, weights=train_valid_counts))

            train_loss = weighted_train_mean(train_loss)
            train_task_loss = weighted_train_mean(train_task_loss)
            train_kd_sim_loss = weighted_train_mean(train_kd_sim_loss)
            train_kd_task_loss = weighted_train_mean(train_kd_task_loss)
            train_auxiliary_loss = weighted_train_mean(train_auxiliary_loss)
            train_kd_extreme_response_loss = weighted_train_mean(
                train_kd_extreme_response_loss)
            train_aux_tail_event_loss = weighted_train_mean(
                train_aux_tail_event_loss)
            train_aux_tail_direction_loss = weighted_train_mean(
                train_aux_tail_direction_loss)
            train_aux_event_residual_loss = weighted_train_mean(
                train_aux_event_residual_loss)
            train_aux_clear_sky_pinball_loss = weighted_train_mean(
                train_aux_clear_sky_pinball_loss)
            parameter_update_statistics = (
                self.monitor.finish_parameter_snapshot(
                    self.model, parameter_snapshot)
                if self.monitor.enabled else {})
            if train_quality is not None:
                self.monitor.record_data_quality(
                    'train', train_quality.summary())
            self._last_epoch_phase_components = {
                name: float(np.average(values))
                for name, values in train_phase_component_losses.items()
            }
            diagnostic_label = (
                f'epoch_{epoch + 1}_{phase_stage}'
                if phase_strategy else f'epoch_{epoch + 1}')
            calibration_enabled = (
                phase_strategy
                and phase_stage == 'active'
                and str(getattr(
                    self.args, 'phase_benefit_mode', 'winner'))
                == 'constrained_stack'
                and bool(int(getattr(
                    self.args, 'phase_validation_calibration', 0)))
            )
            if calibration_enabled:
                self._set_phase_family_calibration(1.0, 1.0)
                self._phase_validation_diagnostic_label = (
                    diagnostic_label + '_raw')
                self.vali(vali_data, vali_loader, criterion)
                raw_diagnostic_path = os.path.join(
                    path,
                    f'phase_validation_{diagnostic_label}_raw.npz')
                calibration_result = self._calibrate_phase_families(
                    raw_diagnostic_path)
                delta_text = ', '.join(
                    f'{name}={value:+.5f}'
                    for name, value in calibration_result[
                        'rmse_deltas'].items())
                print(
                    'phase_family_validation_projection '
                    f'feasible={str(calibration_result["feasible"]).lower()} '
                    f'stable={calibration_result["stable_scale"]:.4f} '
                    f'active={calibration_result["active_scale"]:.4f} '
                    f'deltas: {delta_text}')
            self._phase_validation_diagnostic_label = diagnostic_label
            monitor_epoch_due = (
                self.monitor.enabled
                and (epoch + 1) % int(getattr(
                    self.args, 'monitor_epoch_interval', 1)) == 0
            )
            vali_loss, vali_ts_loss, vali_metrics = self.vali(
                vali_data, vali_loader, criterion,
                collect_monitor=monitor_epoch_due)
            validation_monitor_payload = self._last_validation_monitor
            training_test_monitor_payload = None
            if test_loader is None:
                test_values = skipped_test_values
            else:
                test_values = self.vali(
                    test_data, test_loader, criterion,
                    collect_monitor=monitor_epoch_due,
                    monitor_split='test')
                training_test_monitor_payload = self._last_validation_monitor
            self._enforce_frozen_ts_mode()

            validation_forecast_summary = self._record_monitor_forecast(
                validation_monitor_payload, epoch=epoch + 1, split='val',
                role='student', phase='epoch')
            training_test_forecast_summary = self._record_monitor_forecast(
                training_test_monitor_payload, epoch=epoch + 1, split='test',
                role='student', phase='training_opt_in')
            if monitor_epoch_due and self._monitor_teacher_reference is not None:
                distillation_statistics = self._distillation_monitor_statistics(
                    validation_monitor_payload,
                    self._monitor_teacher_reference,
                    min_count=self.monitor.min_slice_count,
                    min_days=self.monitor.min_slice_days)
                distillation_statistics.update({
                    'epoch': epoch + 1,
                    'phase': 'epoch',
                    'teacher_strategy': self._teacher_strategy(),
                    'student_strategy': self._student_strategy(),
                    'valid_privileged_interventions': (
                        valid_privileged_interventions),
                    'skipped_privileged_interventions': (
                        skipped_privileged_interventions),
                    'privileged_intervention_count_unit': 'sample',
                    'effective_privileged_intervention_batches': (
                        effective_privileged_intervention_batches),
                })
                self.monitor.record_distillation(distillation_statistics)

            selection_score = self._validation_selection_score(vali_loss, vali_metrics)
            phase_pretraining = phase_strategy and phase_stage != 'active'
            if phase_strategy and phase_pretraining:
                phase_checkpoint_path = os.path.join(
                    path, f'phase_{phase_stage}_checkpoint.pth')
                stage_loss = float(vali_loss)
                stage_score = self._phase_pretrain_selection_score(
                    phase_stage, stage_loss, vali_metrics)
                if (
                    np.isfinite(stage_score)
                    and stage_score < phase_stage_best[phase_stage]['score']
                ):
                    phase_stage_best[phase_stage] = {
                        'score': stage_score,
                        'validation_loss': stage_loss,
                        'epoch': epoch + 1,
                    }
                    self._save_model_checkpoint(
                        self.model, phase_checkpoint_path)
                phase_stage_ends = {
                    'state': phase_stage_durations['state'],
                    'magnitude': phase_stage_durations['state']
                    + phase_stage_durations['magnitude'],
                    'benefit': phase_pretrain_total,
                }
                if epoch + 1 == phase_stage_ends[phase_stage]:
                    selected_state = torch.load(
                        phase_checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(selected_state)
                    if phase_stage == 'benefit':
                        prior_freezer = getattr(
                            self._student_model(),
                            'freeze_phase_winner_prior', None)
                        if callable(prior_freezer):
                            prior_freezer()
                    self._save_model_checkpoint(
                        self.model, phase_checkpoint_path)
                    print(
                        f'phase_pretrain_checkpoint_saved stage={phase_stage} '
                        f'best_epoch={phase_stage_best[phase_stage]["epoch"]} '
                        f'best_val_score={phase_stage_best[phase_stage]["score"]:.7f} '
                        f'best_val_loss={phase_stage_best[phase_stage]["validation_loss"]:.7f} '
                        f'path={phase_checkpoint_path}')
            if (
                self._student_strategy() == 'causal_solar_token_tail'
                and tail_gate_pretraining
                and epoch + 1 == tail_pretrain_epochs
            ):
                prior_freezer = getattr(
                    self._student_model(), 'freeze_tail_priors', None)
                if callable(prior_freezer):
                    prior_freezer()
                gate_checkpoint_path = os.path.join(
                    path, 'gate_pretrain_checkpoint.pth')
                self._save_model_checkpoint(self.model, gate_checkpoint_path)
                print(
                    'student_tail_gate_checkpoint_saved=true '
                    f'path={gate_checkpoint_path} train_tail_priors_frozen=true')
            if (
                not tail_gate_pretraining
                and not phase_pretraining
                and np.isfinite(selection_score)
                and selection_score < best_diagnostic_score
            ):
                best_diagnostic_score = float(selection_score)
                best_diagnostic_epoch = epoch + 1
                best_diagnostic_metrics = dict(vali_metrics)

            if conservative:
                if phase_pretraining:
                    decision = {
                        'accepted': False,
                        'improved': False,
                        'ordinary_ok': True,
                        'no_path_ok': True,
                        'reason': f'phase_{phase_stage}_pretraining_not_eligible',
                    }
                elif tail_gate_pretraining:
                    decision = {
                        'accepted': False,
                        'improved': False,
                        'ordinary_ok': True,
                        'no_path_ok': True,
                        'reason': 'tail_gate_pretraining_not_eligible',
                    }
                else:
                    decision = checkpoint_guard.consider(
                        epoch + 1, selection_score,
                        self._ordinary_validation_rmse(vali_metrics),
                        self._no_path_validation_rmse(vali_metrics),
                        vali_metrics)
                checkpoint_eligible = decision['accepted']
                decision_reason = decision['reason']
                if checkpoint_eligible:
                    self._save_model_checkpoint(self.model, checkpoint_path)
                    best_eligible_metrics = dict(vali_metrics)
                    print(
                        f'checkpoint_promoted=passed epoch={epoch + 1} '
                        f'selection={selection_score:.7f} reason={decision_reason}')
                else:
                    ordinary_limit = checkpoint_guard.ordinary_limit
                    limit_text = 'disabled' if ordinary_limit is None else f'{ordinary_limit:.7f}'
                    print(
                        f'checkpoint_promoted=rejected epoch={epoch + 1} '
                        f'reason={decision_reason} ordinary_limit={limit_text}')
                if phase_pretraining:
                    checkpoint_guard.reset_patience()
                early_stop_counter = checkpoint_guard.counter
                best_validation_score = checkpoint_guard.best_score
                stop_training = (
                    False if phase_pretraining else checkpoint_guard.early_stop)
            else:
                previous_best = early_stopping.val_loss_min
                early_stopping(selection_score, self.model, path, model_name='checkpoint.pth')
                checkpoint_eligible = early_stopping.val_loss_min < previous_best
                decision_reason = (
                    'legacy_checkpoint_saved' if checkpoint_eligible else 'legacy_no_improvement')
                early_stop_counter = early_stopping.counter
                best_validation_score = early_stopping.val_loss_min
                stop_training = early_stopping.early_stop

            if checkpoint_eligible:
                selected_monitor_epoch = epoch + 1

            test_loss, test_ts_loss, test_metrics = test_values
            test_loss_text = (
                'skipped' if test_loader is None else f'{test_loss:.7f}')
            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss_text}")
            print("Vali Details - TS: {:.7f}".format(vali_ts_loss))
            if test_loader is not None:
                print("Test Details - TS: {:.7f}".format(test_ts_loss))
            if self.args.data in {'LuoyangParquet', 'YLJParquet'}:
                print(
                    "Vali RMSE: {:.7f}, stable RMSE: {:.7f}, directional RMSE: {:.7f}, "
                    "hard-delta RMSE: {:.7f}, selection: {:.7f}".format(
                        vali_metrics['rmse'], vali_metrics['ramp_stable_rmse'],
                        vali_metrics['ramp_directional_rmse'],
                        vali_metrics['ramp_hard_rmse'], selection_score))
            else:
                print(
                    "Vali RMSE: {:.7f}, <5kW RMSE: {:.7f}, >=5kW RMSE: {:.7f}, "
                    ">=8kW RMSE: {:.7f}, Cloud-event RMSE: {:.7f}, selection: {:.7f}".format(
                        vali_metrics['rmse'], vali_metrics['ramp_lt5_rmse'],
                        vali_metrics['ramp_ge5_rmse'],
                        vali_metrics['ramp_ge8_rmse'],
                        vali_metrics['cloud_event_rmse'], selection_score))
            if self._student_strategy() == 'causal_solar_token_tail':
                print(
                    'Tail auxiliary - event: {:.7f}, direction: {:.7f}, '
                    'residual: {:.7f}, clear-sky pinball: {:.7f}'.format(
                        train_aux_tail_event_loss,
                        train_aux_tail_direction_loss,
                        train_aux_event_residual_loss,
                        train_aux_clear_sky_pinball_loss,
                    ))
            if phase_strategy:
                phase_text = ', '.join(
                    f'{name}={value:.7f}'
                    for name, value in self._last_epoch_phase_components.items())
                print(
                    f'Phase auxiliary [{phase_stage}] - {phase_text}; '
                    f'hazard AUPRC={vali_metrics["phase_hazard_event_auprc"]:.4f}, '
                    f'phase macro acc={vali_metrics["phase_macro_accuracy"]:.4f}, '
                    f'no-path={vali_metrics["no_path_rmse"]:.4f}, '
                    f'round-trip={vali_metrics["round_trip_rmse"]:.4f}, '
                    f'quiet-latent >=5={vali_metrics["quiet_latent_ge5_rmse"]:.4f}')
                constraint_violations = getattr(
                    self, '_last_phase_constraint_violations', {})
                if constraint_violations:
                    violation_text = ', '.join(
                        f'{name}={value:+.5f}'
                        for name, value in constraint_violations.items())
                    dual_text = ', '.join(
                        f'{name}={value:.4f}'
                        for name, value in getattr(
                            self, '_phase_constraint_duals', {}).items())
                    print(
                        f'Phase group-risk constraints - {violation_text}; '
                        f'duals: {dual_text}')

            append_metrics_row(history_path, history_record(
                epoch + 1,
                (
                    train_loss, train_task_loss, train_kd_sim_loss,
                    train_kd_task_loss, train_auxiliary_loss,
                    train_kd_extreme_response_loss,
                    train_aux_tail_event_loss,
                    train_aux_tail_direction_loss,
                    train_aux_event_residual_loss,
                    train_aux_clear_sky_pinball_loss,
                ),
                (vali_loss, vali_ts_loss, vali_metrics),
                test_values,
                selection_score,
                checkpoint_eligible,
                decision_reason,
                early_stop_counter,
                best_validation_score,
                epoch_duration,
            ))
            if self.monitor.enabled:
                def monitor_mean(values):
                    return float(np.mean(values)) if values else None

                validation_privacy_suppressed = (
                    validation_forecast_summary is None
                    or validation_forecast_summary.get(
                        'privacy_suppressed', False))
                gradient_group_summary = {
                    group: {
                        'grad_norm_mean': monitor_mean(values),
                        'sample_count': len(values),
                    }
                    for group, values in train_gradient_group_samples.items()
                }
                mean_positive = monitor_mean(train_positive_cosines)
                mean_negative = monitor_mean(train_negative_cosines)
                kd_intervention_metric_count = (
                    valid_privileged_interventions
                    if self.args.kd_type == 'inter_causal'
                    else (
                        processed_train_samples
                        if self.args.kd_loss_weight_inter > 0 else 0))
                monitor_warnings = []
                if (
                    self.args.kd_type == 'inter_causal'
                    and student_feature_dim is not None
                    and student_feature_dim <= 1
                ):
                    monitor_warnings.append(
                        'inter_causal_feature_dimension_is_one')
                self.monitor.record_epoch({
                    'epoch': epoch + 1,
                    'stage': 'student',
                    'train_loss': train_loss,
                    'train_task_loss': train_task_loss,
                    'train_kd_similarity_loss_weighted': train_kd_sim_loss,
                    'train_kd_intervention_loss_weighted': train_kd_task_loss,
                    'train_kd_intervention_loss_weighted_count': (
                        kd_intervention_metric_count),
                    'train_auxiliary_loss': train_auxiliary_loss,
                    'train_kd_extreme_response_loss_weighted': (
                        train_kd_extreme_response_loss),
                    'train_kd_extreme_response_loss_weighted_count': (
                        valid_extreme_response_samples),
                    'loss_component_fraction': {
                        'task': train_task_loss / max(abs(train_loss), 1e-12),
                        'task_count': processed_train_samples,
                        'kd_similarity': train_kd_sim_loss / max(abs(train_loss), 1e-12),
                        'kd_similarity_count': (
                            processed_train_samples
                            if self.args.kd_loss_weight_sim > 0 else 0),
                        'kd_intervention': train_kd_task_loss / max(abs(train_loss), 1e-12),
                        'kd_intervention_count': kd_intervention_metric_count,
                        'kd_extreme_response': (
                            train_kd_extreme_response_loss
                            / max(abs(train_loss), 1e-12)),
                        'kd_extreme_response_count': (
                            valid_extreme_response_samples),
                        'auxiliary': train_auxiliary_loss / max(abs(train_loss), 1e-12),
                        'auxiliary_count': processed_train_samples,
                    },
                    'validation_loss': (
                        None if validation_privacy_suppressed else vali_loss),
                    'validation_task_loss': (
                        None if validation_privacy_suppressed else vali_ts_loss),
                    'validation_metrics': (
                        None if validation_privacy_suppressed else vali_metrics),
                    'validation_forecast': validation_forecast_summary,
                    'training_test_forecast': training_test_forecast_summary,
                    'learning_rate': model_optim.param_groups[0]['lr'],
                    'gradient_norm_mean': monitor_mean(train_gradient_norms),
                    'gradient_norm_max': (
                        float(np.max(train_gradient_norms))
                        if train_gradient_norms else None),
                    'gradient_clipped_fraction': (
                        clipped_gradient_steps / max(len(train_gradient_norms), 1)),
                    'nonfinite_gradient_steps': nonfinite_gradient_steps,
                    'gradient_groups': gradient_group_summary,
                    'amp_scale': (
                        float(scaler.get_scale()) if self.args.use_amp else None),
                    'amp_skipped_steps': amp_skipped_steps,
                    'parameter_updates': parameter_update_statistics,
                    'student_feature_dimension': student_feature_dim,
                    'student_feature_std': monitor_mean(
                        train_student_feature_stds),
                    'teacher_feature_std': monitor_mean(
                        train_teacher_feature_stds),
                    'positive_cosine_mean': mean_positive,
                    'positive_cosine_count': (
                        processed_train_samples if mean_positive is not None else 0),
                    'negative_cosine_mean': mean_negative,
                    'negative_cosine_count': valid_privileged_interventions,
                    'positive_negative_margin': (
                        mean_positive - mean_negative
                        if mean_positive is not None
                        and mean_negative is not None else None),
                    'positive_negative_margin_count': min(
                        processed_train_samples,
                        valid_privileged_interventions),
                    'image_similarity_mean': monitor_mean(train_sim_img_means),
                    'weather_similarity_mean': monitor_mean(
                        train_sim_weather_means),
                    'image_branch_enabled_batch_fraction': monitor_mean(
                        train_image_modality_active),
                    'weather_branch_enabled_batch_fraction': monitor_mean(
                        train_weather_modality_active),
                    'intervention_input_delta_mean': monitor_mean(
                        train_intervention_input_deltas),
                    'intervention_input_delta_count': len(
                        train_intervention_input_deltas),
                    'intervention_feature_delta_mean': monitor_mean(
                        train_intervention_feature_deltas),
                    'intervention_feature_delta_count': len(
                        train_intervention_feature_deltas),
                    'valid_privileged_interventions': (
                        valid_privileged_interventions),
                    'skipped_privileged_interventions': (
                        skipped_privileged_interventions),
                    'privileged_intervention_count_unit': 'sample',
                    'effective_privileged_intervention_batches': (
                        effective_privileged_intervention_batches),
                    'processed_samples': processed_train_samples,
                    'samples_per_second': (
                        processed_train_samples / max(epoch_duration, 1e-9)),
                    'epoch_time_sec': epoch_duration,
                    'cuda_peak_memory_bytes': (
                        int(torch.cuda.max_memory_allocated(self.device))
                        if self.device.type == 'cuda' else None),
                    'checkpoint_eligible': checkpoint_eligible,
                    'checkpoint_decision': decision_reason,
                    'selection_score': (
                        None if validation_privacy_suppressed
                        else selection_score),
                    'early_stop_counter': early_stop_counter,
                    'warnings': monitor_warnings,
                })

            if stop_training:
                print("Early stopping")
                break

            scheduler_epoch = epoch + 1
            if self._student_strategy() == 'causal_solar_token_tail':
                # Classifier-only pretraining must not consume the magnitude
                # branch's learning-rate schedule.
                scheduler_epoch = max(1, scheduler_epoch - tail_pretrain_epochs)
            if phase_strategy:
                stage_starts = {
                    'state': 0,
                    'magnitude': phase_stage_durations['state'],
                    'benefit': phase_stage_durations['state']
                    + phase_stage_durations['magnitude'],
                    'active': phase_pretrain_total,
                }
                scheduler_epoch = epoch - stage_starts[phase_stage] + 1
            adjust_learning_rate(model_optim, scheduler_epoch, self.args)

        encoder_unchanged_during_training = True
        if ts_digest_before is not None:
            encoder_unchanged_during_training = (
                self._ts_branch_digest() == ts_digest_before)
            print(
                'student_ts_training_state_unchanged='
                f'{"passed" if encoder_unchanged_during_training else "failed"}')
            if not encoder_unchanged_during_training:
                raise RuntimeError(
                    'frozen student branch_ts changed before checkpoint selection')
        frozen_parent_unchanged_during_training = True
        if parent_digest_before is not None:
            frozen_parent_unchanged_during_training = (
                self._frozen_parent_digest() == parent_digest_before)
            print(
                'student_frozen_parent_training_state_unchanged='
                f'{"passed" if frozen_parent_unchanged_during_training else "failed"}')
            if not frozen_parent_unchanged_during_training:
                raise RuntimeError(
                    'frozen Stage-1 parent changed before checkpoint selection')

        # The helper exists on every Student model, but phase-field buffers are
        # created only by the opt-in solar-advection route.  YLJ ts_only has no
        # phase prior to serialize.
        trained_phase_winner_prior = self._phase_winner_prior_statistics()

        # Load either the best eligible candidate or the untouched epoch-0 model.
        best_model_path = checkpoint_path
        if self.args.data == 'LuoyangParquet':
            validate_checkpoint_contract(best_model_path, self.args.luoyang_config)
        elif self.args.data == 'YLJParquet':
            validate_ylj_checkpoint_contract(best_model_path, self.args.ylj_config)
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self._enforce_frozen_ts_mode()

        encoder_unchanged = True
        if ts_digest_before is not None:
            encoder_unchanged = (
                encoder_unchanged_during_training
                and self._ts_branch_digest() == ts_digest_before)
            print(
                'student_ts_checkpoint_unchanged='
                f'{"passed" if encoder_unchanged else "failed"}')
        frozen_parent_unchanged = True
        if parent_digest_before is not None:
            frozen_parent_unchanged = (
                frozen_parent_unchanged_during_training
                and self._frozen_parent_digest() == parent_digest_before)
            print(
                'student_frozen_parent_checkpoint_unchanged='
                f'{"passed" if frozen_parent_unchanged else "failed"}')

        if conservative:
            self._phase_validation_diagnostic_label = 'selected'
            selected_vali_loss, selected_vali_ts_loss, selected_metrics = self.vali(
                vali_data, vali_loader, criterion)
            self._enforce_frozen_ts_mode()
            selected_score = self._validation_selection_score(
                selected_vali_loss, selected_metrics)
            if not np.isclose(
                selected_score, checkpoint_guard.best_score,
                rtol=1e-6, atol=1e-7,
            ):
                raise RuntimeError(
                    'reloaded checkpoint validation score does not match the '
                    f'selected score: {selected_score} vs {checkpoint_guard.best_score}')
            if (
                checkpoint_guard.ordinary_guard
                and self._ordinary_validation_rmse(selected_metrics)
                > checkpoint_guard.ordinary_limit + 1e-7
            ):
                raise RuntimeError(
                    'reloaded checkpoint violates the ordinary validation guard')
            if (
                checkpoint_guard.no_path_guard
                and self._no_path_validation_rmse(selected_metrics)
                > checkpoint_guard.no_path_limit + 1e-7
            ):
                raise RuntimeError(
                    'reloaded checkpoint violates the no-path validation guard')
            selected_metric_guard_results = (
                checkpoint_guard.metric_guard_results(selected_metrics))
            if not all(
                result['passed']
                for result in selected_metric_guard_results.values()
            ) and checkpoint_guard.best_epoch > 0:
                failed = [
                    name for name, result
                    in selected_metric_guard_results.items()
                    if not result['passed']
                ]
                raise RuntimeError(
                    'reloaded checkpoint violates phase validation guards: '
                    + ','.join(failed))
            accepted = checkpoint_guard.best_epoch > 0
            diagnostic_improved = (
                best_diagnostic_epoch is not None
                and best_diagnostic_score
                < input_score - checkpoint_guard.min_improvement
            )
            if checkpoint_guard.ordinary_guard and best_diagnostic_metrics is not None:
                diagnostic_ordinary_rmse = self._ordinary_validation_rmse(
                    best_diagnostic_metrics)
                diagnostic_ordinary_ok = (
                    np.isfinite(diagnostic_ordinary_rmse)
                    and diagnostic_ordinary_rmse <= checkpoint_guard.ordinary_limit
                )
            else:
                diagnostic_ordinary_ok = True
            if checkpoint_guard.no_path_guard and best_diagnostic_metrics is not None:
                diagnostic_no_path_rmse = self._no_path_validation_rmse(
                    best_diagnostic_metrics)
                diagnostic_no_path_ok = (
                    np.isfinite(diagnostic_no_path_rmse)
                    and diagnostic_no_path_rmse <= checkpoint_guard.no_path_limit
                )
            else:
                diagnostic_no_path_ok = True
            diagnostic_metric_guard_results = (
                checkpoint_guard.metric_guard_results(best_diagnostic_metrics)
                if best_diagnostic_metrics is not None else {})
            failed_diagnostic_metric_guards = [
                name for name, result
                in diagnostic_metric_guard_results.items()
                if not result['passed']
            ]
            if accepted:
                rejection_reason = None
                selected_source = 'trained_candidate'
            elif diagnostic_improved and not diagnostic_ordinary_ok:
                rejection_reason = 'ordinary_guard_failed'
                selected_source = 'input_rollback'
            elif diagnostic_improved and not diagnostic_no_path_ok:
                rejection_reason = 'no_path_guard_failed'
                selected_source = 'input_rollback'
            elif diagnostic_improved and failed_diagnostic_metric_guards:
                rejection_reason = (
                    f'{failed_diagnostic_metric_guards[0]}_guard_failed')
                selected_source = 'input_rollback'
            else:
                rejection_reason = 'no_validation_improvement'
                selected_source = 'input_rollback'

            config = {
                'seed': int(getattr(self.args, 'seed', 0)),
                'student_fuse_strategy': self._student_strategy(),
                'parent_fuse_strategy': getattr(self.args, 'parent_fuse_strategy', ''),
                'student_init_path': os.path.abspath(self.args.student_init_path),
                'sample_weight_mode': getattr(
                    self.args, 'stanford_sample_weight_mode', 'none'),
                'ramp_weight_5': float(getattr(self.args, 'stanford_ramp_weight_5', 1.5)),
                'ramp_weight_8': float(getattr(self.args, 'stanford_ramp_weight_8', 2.0)),
                'stable_correction_aux_weight': float(getattr(
                    self.args, 'stable_correction_aux_weight', 0.0)),
                'stable_ramp_threshold_kw': float(getattr(
                    self.args, 'stable_ramp_threshold_kw', 5.0)),
                'event_state_aux_weight': float(getattr(
                    self.args, 'event_state_aux_weight', 0.0)),
                'event_residual_aux_weight': float(getattr(
                    self.args, 'event_residual_aux_weight', 0.0)),
                'event_routing_mode': getattr(
                    self.args, 'event_routing_mode', 'soft'),
                'tail_routing_mode': getattr(
                    self.args, 'tail_routing_mode', 'confidence'),
                'tail_confidence_threshold': float(getattr(
                    self.args, 'tail_confidence_threshold', 0.5)),
                'tail_error_threshold_kw': float(getattr(
                    self.args, 'tail_error_threshold_kw', 3.0)),
                'tail_gate_pretrain_epochs': int(getattr(
                    self.args, 'tail_gate_pretrain_epochs', 0)),
                'clear_sky_aux_weight': float(getattr(
                    self.args, 'clear_sky_aux_weight', 0.0)),
                'clear_sky_quantile': float(getattr(
                    self.args, 'clear_sky_quantile', 0.95)),
                'phase_state_aux_weight': float(getattr(
                    self.args, 'phase_state_aux_weight', 0.0)),
                'phase_path_aux_weight': float(getattr(
                    self.args, 'phase_path_aux_weight', 0.0)),
                'phase_motion_consistency_weight': float(getattr(
                    self.args, 'phase_motion_consistency_weight', 0.0)),
                'phase_correction_mode': str(getattr(
                    self.args, 'phase_correction_mode', 'learned_residual')),
                'phase_benefit_aux_weight': float(getattr(
                    self.args, 'phase_benefit_aux_weight', 0.0)),
                'phase_state_pretrain_epochs': int(getattr(
                    self.args, 'phase_state_pretrain_epochs', 0)),
                'phase_magnitude_pretrain_epochs': int(getattr(
                    self.args, 'phase_magnitude_pretrain_epochs', 0)),
                'phase_benefit_pretrain_epochs': int(getattr(
                    self.args, 'phase_benefit_pretrain_epochs', 0)),
                'phase_resume_from': getattr(
                    self.args, 'phase_resume_from', 'none'),
                'phase_route_classes': getattr(
                    self.args, 'phase_route_classes',
                    'active_down,active_up'),
                'kd_type': self.args.kd_type,
                'kd_loss_weight_inter': float(self.args.kd_loss_weight_inter),
                'kd_extreme_response_weight': float(getattr(
                    self.args, 'kd_extreme_response_weight', 0.0)),
                'selection_metric': getattr(self.args, 'val_selection_metric', 'loss'),
                'selection_extreme_weight': float(getattr(
                    self.args, 'val_extreme_metric_weight', 0.25)),
                'ordinary_guard': bool(getattr(self.args, 'val_ordinary_guard', False)),
                'ordinary_relative_tolerance': float(getattr(
                    self.args, 'val_ordinary_relative_tolerance', 0.0)),
                'no_path_guard': bool(getattr(
                    self.args, 'val_no_path_guard', False)),
                'no_path_relative_tolerance': float(getattr(
                    self.args, 'val_no_path_relative_tolerance', 0.0)),
                'cloud_ordinary_relative_tolerance': float(getattr(
                    self.args, 'val_cloud_ordinary_relative_tolerance', 0.0)),
                'cloud_no_path_relative_tolerance': float(getattr(
                    self.args, 'val_cloud_no_path_relative_tolerance', 0.0)),
                'tail_min_improvement': float(getattr(
                    self.args, 'val_tail_min_improvement', 0.0)),
                'extreme_relative_tolerance': float(getattr(
                    self.args, 'val_extreme_relative_tolerance', 0.0)),
                'min_improvement': float(getattr(
                    self.args, 'val_min_improvement', 0.0)),
                'freeze_ts_permanently': bool(getattr(
                    self.args, 'freeze_ts_permanently', False)),
                'train_image_residual_only': bool(getattr(
                    self.args, 'train_image_residual_only', False)),
                'strict_causal_student': bool(getattr(
                    self.args, 'strict_causal_student', False)),
                'max_train_batches': int(getattr(
                    self.args, 'max_train_batches', 0)),
                'max_val_batches': int(getattr(
                    self.args, 'max_val_batches', 0)),
            }
            config_fingerprint = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()
            if config['parent_fuse_strategy'] == 's14_timeseries_encoder':
                epoch0_semantics = (
                    'adapted original s14 branch_ts with a zero-initialized image '
                    'residual; this is not the legacy full-fusion s14 output')
            else:
                epoch0_semantics = (
                    'parent checkpoint evaluated under the candidate strategy; '
                    'motion and soft gate are identity-preserving Appearance children')
            manifest = {
                'schema_version': 1,
                'status': 'completed',
                'tag': getattr(self.args, 'extra_tag', '') or self.args.model_id,
                'setting': setting,
                'seed': int(getattr(self.args, 'seed', 0)),
                'config_fingerprint': config_fingerprint,
                'config': config,
                'input_checkpoint': os.path.abspath(self.args.student_init_path),
                'epoch0_checkpoint': os.path.abspath(input_checkpoint_path),
                'epoch0_semantics': epoch0_semantics,
                'checkpoint': os.path.abspath(checkpoint_path),
                'result_predictions': os.path.abspath(os.path.join(
                    './results', setting, 'sample_predictions.npz')),
                'input': self._json_metrics(input_metrics, input_score, epoch=0),
                'best_trained': (
                    self._json_metrics(
                        best_diagnostic_metrics,
                        best_diagnostic_score,
                        epoch=best_diagnostic_epoch)
                    if best_diagnostic_metrics is not None else None
                ),
                'best_eligible': (
                    self._json_metrics(
                        best_eligible_metrics,
                        checkpoint_guard.best_score,
                        epoch=checkpoint_guard.best_epoch)
                    if accepted else None
                ),
                'selected': self._json_metrics(
                    selected_metrics, selected_score,
                    epoch=checkpoint_guard.best_epoch),
                'selected_source': selected_source,
                'accepted': accepted,
                'improved': accepted,
                'ordinary_guard_passed': (
                    (True if accepted else diagnostic_ordinary_ok)
                    if checkpoint_guard.ordinary_guard else None),
                'ordinary_limit_rmse': (
                    checkpoint_guard.ordinary_limit
                    if checkpoint_guard.ordinary_limit is not None
                    and np.isfinite(checkpoint_guard.ordinary_limit)
                    else None),
                'no_path_guard_passed': (
                    (True if accepted else diagnostic_no_path_ok)
                    if checkpoint_guard.no_path_guard else None),
                'no_path_limit_rmse': (
                    checkpoint_guard.no_path_limit
                    if checkpoint_guard.no_path_limit is not None
                    and np.isfinite(checkpoint_guard.no_path_limit)
                    else None),
                'phase_metric_guards': self._json_guard_results(
                    checkpoint_guard.metric_guard_results(selected_metrics)
                    if accepted else diagnostic_metric_guard_results),
                'rejection_reason': rejection_reason,
                'epochs_completed': epochs_completed,
                'encoder_unchanged': encoder_unchanged,
                'encoder_unchanged_during_training': encoder_unchanged_during_training,
                'frozen_parent_unchanged': frozen_parent_unchanged,
                'frozen_parent_unchanged_during_training': (
                    frozen_parent_unchanged_during_training),
                'trainable_parameter_names': self._trainable_parameter_names,
                'validation_proxy_note': (
                    'cloud_event metrics use only validation-day PV volatility '
                    'and no weather/test labels; test Cloudy metrics are diagnostic only'),
                'student_seq_y_img_disabled': bool(getattr(
                    self.args, 'strict_causal_student', False)),
                'student_future_image_invariance_passed': causal_boundary_checked,
                'gate_pretrain_checkpoint': (
                    os.path.abspath(os.path.join(
                        path, 'gate_pretrain_checkpoint.pth'))
                    if os.path.exists(os.path.join(
                        path, 'gate_pretrain_checkpoint.pth')) else None
                ),
                'phase_pretrain_checkpoints': {
                    stage: (
                        os.path.abspath(os.path.join(
                            path, f'phase_{stage}_checkpoint.pth'))
                        if os.path.exists(os.path.join(
                            path, f'phase_{stage}_checkpoint.pth')) else None)
                    for stage in ['state', 'magnitude', 'benefit']
                },
                'learned_train_tail_priors': (
                    self._student_model().tail_prior_statistics()
                    if callable(getattr(
                        self._student_model(), 'tail_prior_statistics', None))
                    else None
                ),
                'learned_train_phase_winner_prior': (
                    trained_phase_winner_prior
                ),
            }
            manifest_path = os.path.join(path, 'checkpoint_selection.json')
            self._write_json(manifest_path, manifest)
            self._checkpoint_accepted = bool(accepted)
            print(
                f'checkpoint_decision={"accepted" if accepted else "rejected_rollback"} '
                f'best_epoch={checkpoint_guard.best_epoch} '
                f'input_selection={input_score:.7f} selected_selection={selected_score:.7f} '
                f'manifest={manifest_path}')

        if self._ts_must_stay_frozen() and not encoder_unchanged:
            raise RuntimeError('frozen student branch_ts parameters or buffers changed')
        if parent_digest_before is not None and not frozen_parent_unchanged:
            raise RuntimeError('frozen Stage-1 parent parameters or buffers changed')

        if not conservative:
            self._checkpoint_accepted = True
        self.model.eval()
        self._enforce_frozen_ts_mode()

        if (
            self.monitor.enabled
            and self.args.data in {'LuoyangParquet', 'YLJParquet'}
        ):
            selected_payload = self._collect_monitor_predictions(
                vali_loader, role='student')
            selected_forecast = self._record_monitor_forecast(
                selected_payload, epoch=selected_monitor_epoch,
                split='val', role='student', phase='best')
            selected_distillation = None
            if self._monitor_teacher_reference is not None:
                selected_distillation = self._distillation_monitor_statistics(
                    selected_payload, self._monitor_teacher_reference,
                    min_count=self.monitor.min_slice_count,
                    min_days=self.monitor.min_slice_days)
                selected_distillation.update({
                    'epoch': selected_monitor_epoch,
                    'phase': 'best',
                    'teacher_strategy': self._teacher_strategy(),
                    'student_strategy': self._student_strategy(),
                })
                self.monitor.record_distillation(selected_distillation)
            if self._student_strategy() not in {'ts_only', 'ts'}:
                selected_ts_payload = self._collect_monitor_predictions(
                    vali_loader, role='student', strategy='ts_only')
                self._record_monitor_forecast(
                    selected_ts_payload, epoch=selected_monitor_epoch,
                    split='val', role='student_ts_only', phase='best_ablation')
            summary_warnings = []
            if getattr(self.args, 'lradj', '') == 'type1':
                summary_warnings.append('learning_rate_halves_each_epoch')
            if self.args.kd_type == 'inter_causal' and self.args.c_out == 1:
                summary_warnings.append(
                    'inter_causal_operates_on_one_dimensional_predictions')
            if self.args.data == 'YLJParquet':
                configured_shift = int(
                    vali_data.config['time'].get(
                        'forecast_window_shift_minutes', 0))
                if configured_shift:
                    summary_warnings.append(
                        'forecast_window_shift_semantics_require_manual_confirmation')
            self.monitor.finalize({
                'status': 'completed',
                'stage': 'student',
                'epochs_completed': epochs_completed,
                'best_epoch': selected_monitor_epoch,
                'checkpoint_accepted': self._checkpoint_accepted,
                'selected_validation': selected_forecast,
                'selected_distillation': selected_distillation,
                'test_during_training': test_loader is not None,
                'warnings': summary_warnings,
            })

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading models')
            checkpoint_path = (
                self.args.test_checkpoint
                if getattr(self.args, 'test_checkpoint', '')
                else os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            )
            if self.args.data == 'LuoyangParquet':
                validate_checkpoint_contract(checkpoint_path, self.args.luoyang_config)
            elif self.args.data == 'YLJParquet':
                validate_ylj_checkpoint_contract(checkpoint_path, self.args.ylj_config)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            # self.model.load_state_dict(torch.load('checkpoints/long_term_forecast_Folsom_48_24_MTS_31_Folsom_ftM_sl48_ll0_pl24_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_fusionsimilarity_imgdim512_weatherdim6_Exp_kd_loss_weight_sim_0.001_inter_0.1_0/checkpoint.pth'))

        if getattr(self.args, 'monitor', False) and (
            self.monitor is None or not self.monitor.enabled
        ):
            explicit_checkpoint = getattr(self.args, 'test_checkpoint', '')
            monitor_run_dir = (
                os.path.dirname(os.path.abspath(explicit_checkpoint))
                if explicit_checkpoint
                else os.path.join(self.args.checkpoints, setting)
            )
            self.monitor = TrainingMonitor.from_args(
                self.args, run_dir=monitor_run_dir,
                stage='student', model=self.model)
            if self.monitor.enabled:
                self.monitor.start_run(self.model, {'test': test_data})

        if self.monitor is not None and self.monitor.enabled:
            self.monitor.register_datasets({'test': test_data})

        preds = []
        trues = []
        monitor_store = (
            self._new_monitor_store()
            if self.monitor is not None and self.monitor.enabled else None)
        test_quality = None
        if self.monitor is not None and self.monitor.enabled:
            fields = getattr(test_data, 'fields', {})
            feature_names = (
                fields.get('timeseries_columns')
                or fields.get('time_series_columns'))
            test_quality = self.monitor.new_data_quality_accumulator(
                'test', feature_names=feature_names)
        criterion = self._select_criterion()

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                (batch_x, batch_y, batch_x_mark, batch_y_mark,
                 batch_x_img, batch_y_img, batch_x_weather, batch_y_weather,
                 _, phase_targets) = self._unpack_batch(batch)
                # 数据预处理
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                batch_x_img = batch_x_img.float().to(self.device, non_blocking=True)
                batch_y_img = batch_y_img.float().to(self.device, non_blocking=True)
                batch_x_weather = batch_x_weather.float().to(self.device, non_blocking=True)
                batch_y_weather = batch_y_weather.float().to(self.device, non_blocking=True)
                phase_targets = self._phase_targets_to_device(phase_targets)
                phase_lead_marks = (
                    phase_targets.get('lead_marks') if phase_targets is not None else None)

                # 前向传播
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._student_forward(
                            batch_x, batch_x_img, batch_y_img, batch_x_weather,
                            batch_y_weather, batch_x_mark, batch_y_mark,
                            phase_lead_marks,
                            None if phase_targets is None else phase_targets.get('image_mask'))[0]
                else:
                    outputs = self._student_forward(
                        batch_x, batch_x_img, batch_y_img, batch_x_weather,
                        batch_y_weather, batch_x_mark, batch_y_mark,
                        phase_lead_marks,
                        None if phase_targets is None else phase_targets.get('image_mask'))[0]

                if self.args.data == 'Folsom':
                    f_dim = -42 if self.args.features == 'MS' else 0
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                if monitor_store is not None:
                    monitor_predictions = outputs[
                        :, -self.args.pred_len:, f_dim:]
                    monitor_targets = batch_y[
                        :, -self.args.pred_len:, f_dim:]
                    monitor_target_mask = (
                        None if phase_targets is None
                        else phase_targets.get('target_mask'))
                    if monitor_target_mask is None:
                        monitor_target_mask = torch.ones(
                            monitor_targets.shape[:2], dtype=torch.bool,
                            device=monitor_targets.device)
                    else:
                        monitor_target_mask = monitor_target_mask.to(
                            monitor_targets.device, dtype=torch.bool)
                    self._append_monitor_batch(
                        monitor_store, monitor_predictions, monitor_targets,
                        self._current_pv(batch_x), monitor_target_mask,
                        phase_targets)
                    test_quality.update(
                        features=batch_x,
                        feature_mask=(
                            None if phase_targets is None
                            else phase_targets.get('timeseries_mask')),
                        target=monitor_targets,
                        current_power=self._current_pv(batch_x),
                        target_mask=monitor_target_mask,
                        image_mask=(
                            None if phase_targets is None
                            else phase_targets.get('image_mask')),
                        image_age_minutes=(
                            self._monitor_image_age(phase_targets)),
                        source_codes=self._monitor_source_codes(phase_targets),
                        issue_time_ns=(
                            None if phase_targets is None
                            else phase_targets.get('issue_time_ns')),
                        coverage_masks=(
                            self._monitor_coverage_masks(phase_targets)),
                        privileged_teacher_enabled=(
                            None if phase_targets is None
                            else phase_targets.get('privileged_teacher_enabled')),
                        history_images_enabled=(
                            None if phase_targets is None
                            else phase_targets.get('history_images_enabled')),
                        current_power_valid=(
                            None if phase_targets is None
                            else phase_targets.get('current_power_valid')),
                    )

                # 准备最终输出
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
      
        print('test shape:', preds.shape, trues.shape)

        test_monitor_payload = (
            self._finalize_monitor_batches(monitor_store)
            if monitor_store is not None else None)
        test_monitor_summary = self._record_monitor_forecast(
            test_monitor_payload, epoch=None, split='test',
            role='student', phase='final')
        if self.monitor is not None and self.monitor.enabled:
            self.monitor.record_data_quality('test', test_quality.summary())
            self.monitor.finalize({
                'status': 'completed',
                'stage': 'student',
                'final_test': test_monitor_summary,
                'test_during_training': bool(getattr(
                    self.args, 'monitor_include_test_during_training', False)),
            })

        # result save
        folder_path = os.path.join(self.args.results_root, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'
        
        if self.args.data == 'Stanford':
            split_metrics = stanford_sunny_cloudy_metrics(
                preds, trues, test_data.sample_times, self.args.stanford_capacity_kw)
            metrics_text = format_stanford_split_metrics(split_metrics)
            npz_path, csv_path = save_stanford_sample_predictions(
                preds, trues, test_data, folder_path, self.args.stanford_capacity_kw)
            print(f"sample predictions saved: {npz_path}")
            print(f"sample predictions csv: {csv_path}")
            print(metrics_text)
            print('dtw:{}'.format(dtw))
        elif self.args.data == 'LuoyangParquet':
            preds = test_data.inverse_transform_power(preds)
            trues = test_data.inverse_transform_power(trues)
            predictions_path, metrics_path = build_official_outputs(
                preds, trues, test_data, folder_path, self.args.luoyang_config)
            metrics_text = f"official predictions: {predictions_path}; metrics: {metrics_path}"
            print(metrics_text)
        elif self.args.data == 'YLJParquet':
            preds = test_data.inverse_transform_power(preds)
            trues = test_data.inverse_transform_power(trues)
            predictions_path, metrics_path = build_ylj_outputs(
                preds, trues, test_data, folder_path, self.args.ylj_config)
            metrics_text = f"official predictions: {predictions_path}; metrics: {metrics_path}"
            print(metrics_text)
        else:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            metrics_text = "mse:{}, mae:{}".format(mse, mae)
            print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))

        f = open("result_long_term_forecasting_student.txt", 'a')
        f.write(setting + "  \n")
        if self.args.data == 'Stanford':
            f.write(metrics_text)
        else:
            f.write(metrics_text)
        f.write('\n')
        f.write('\n')
        f.close()

        return
