from data_provider.data_factory import data_provider
from exp.exp_basic_teacher import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, append_metrics_row
from utils.metrics import metric, stanford_sunny_cloudy_metrics, format_stanford_split_metrics, Weighted_MSE_MAE
from utils.stanford_outputs import save_stanford_sample_predictions
from utils.luoyang_outputs import build_official_outputs
from utils.ylj_outputs import build_ylj_outputs
from utils.checkpoint_contract import write_checkpoint_contract, validate_checkpoint_contract
from utils.ylj_checkpoint_contract import write_ylj_checkpoint_contract, validate_ylj_checkpoint_contract
from utils.training_monitor import TrainingMonitor
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from models.utils import loss_function
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args, args_img, args_weather):
        self.args_img = args_img  
        self.args_weather = args_weather
        self.loss_type = args.loss_type
        self.training_monitor = None
        self._monitor_best_epoch = None
        self._last_validation_monitor = {}
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args_img.model].Model(self.args, self.args_img, self.args_weather).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
   
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = loss_function
       
        return criterion

    @staticmethod
    def _privacy_safe_validation_diagnostics(diagnostics):
        """Retain validation metrics only when the day-aware forecast is safe."""
        if diagnostics is None:
            return None
        forecast = diagnostics.get('forecast')
        if (
            isinstance(forecast, dict)
            and not forecast.get('privacy_suppressed', False)
        ):
            return diagnostics
        return {
            'split': diagnostics.get('split'),
            'phase': diagnostics.get('phase'),
            'valid_target_count': diagnostics.get('valid_target_count', 0),
            'forecast': forecast,
            'privacy_suppressed': True,
        }

    @staticmethod
    def _privacy_safe_validation_value(value, diagnostics):
        return (
            None
            if diagnostics is None
            or diagnostics.get('privacy_suppressed', False)
            else value
        )

    @classmethod
    def _privacy_safe_similarity(cls, image_stats, weather_stats, forecast):
        if (
            not isinstance(forecast, dict)
            or forecast.get('privacy_suppressed', False)
        ):
            return {'privacy_suppressed': True}
        return {
            'image': cls._finish_scalar_stats(image_stats),
            'weather': cls._finish_scalar_stats(weather_stats),
        }

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 10:
            return batch
        if len(batch) == 9:
            return (*batch, None)
        if len(batch) == 8:
            return (*batch, None, None)
        raise ValueError(f'unsupported batch tuple length: {len(batch)}')

    def _metadata_to_device(self, metadata):
        if metadata is None:
            return None
        return {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in metadata.items()
        }

    def _forward(self, batch_x, batch_x_img, batch_y_img, batch_x_weather,
                 batch_y_weather, metadata):
        image_mask = None if metadata is None else metadata.get('image_mask')
        if metadata is not None and metadata.get('teacher_future_image_mask') is not None:
            future_mask = metadata['teacher_future_image_mask']
            image_mask = future_mask if image_mask is None else torch.cat(
                [image_mask.bool(), future_mask.bool()], dim=1)
        model_args = (
            batch_x, batch_x_img, batch_y_img, batch_x_weather,
            batch_y_weather, self.args.fuse_strategy)
        return (
            self.model(*model_args, image_mask=image_mask)
            if image_mask is not None else self.model(*model_args)
        )

    def _orthogonality_loss(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(model, "bop_img") or model.bop_img is None:
            return torch.zeros((), device=self.device)
        return model.bop_img.get_orthogonality_loss()

    def _reduce_sample_loss(self, per_sample_loss, sample_weight):
        if sample_weight is None:
            return per_sample_loss.mean()
        weights = sample_weight.float().to(per_sample_loss.device).view(-1)
        return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def _monitor_current_power(self, batch_x):
        if self.args.data == 'Stanford' and self.args.history_order == 'current_first':
            return batch_x[:, :1, :1]
        return batch_x[:, -1:, :1]

    @staticmethod
    def _monitor_feature_names(dataset):
        config = getattr(dataset, 'config', None)
        if isinstance(config, dict):
            fields = config.get('fields', {})
            names = fields.get(
                'timeseries_columns', fields.get('time_series_columns'))
            if names:
                return list(names)
        return None

    @staticmethod
    def _new_scalar_stats():
        return {
            'count': 0, 'sum': 0.0, 'sum_sq': 0.0,
        }

    @staticmethod
    def _update_scalar_stats(stats, values):
        if values is None:
            return
        array = values.detach().float().cpu().numpy().reshape(-1)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return
        stats['count'] += int(array.size)
        stats['sum'] += float(array.sum(dtype=np.float64))
        stats['sum_sq'] += float(np.square(array, dtype=np.float64).sum())

    @staticmethod
    def _finish_scalar_stats(stats):
        count = stats['count']
        if count == 0:
            return {'count': 0}
        mean = stats['sum'] / count
        variance = max(stats['sum_sq'] / count - mean * mean, 0.0)
        return {
            'count': count,
            'mean': mean,
            'std': variance ** 0.5,
        }

    @staticmethod
    def _gradient_branch_norms(model):
        squared_norms = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            pieces = name.split('.')
            if pieces and pieces[0] == 'module':
                pieces = pieces[1:]
            branch = pieces[0] if pieces else 'model'
            gradient = parameter.grad.detach().float()
            norm_squared = float(torch.sum(gradient * gradient).item())
            if np.isfinite(norm_squared):
                squared_norms[branch] = (
                    squared_norms.get(branch, 0.0) + norm_squared)
        return {
            branch: value ** 0.5
            for branch, value in squared_norms.items()
        }

    @staticmethod
    def _monitor_array(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _append_monitor_metadata(self, storage, metadata, batch_size):
        if metadata is None:
            return
        key_map = {
            'image_mask': 'image_mask',
            'image_minute_offsets': 'image_age_minutes',
            'timeseries_mask': 'timeseries_mask',
            'forecast_mask': 'forecast_mask',
            'power_history_mask': 'power_history_mask',
            'current_power_valid': 'current_power_valid',
            'issue_time_ns': 'issue_time_ns',
        }
        for source, destination in key_map.items():
            value = self._monitor_array(metadata.get(source))
            if value is None:
                continue
            if value.ndim == 0:
                value = np.repeat(value.reshape(1), batch_size, axis=0)
            storage.setdefault(destination, []).append(value)
        timeseries_mask = metadata.get('timeseries_mask')
        if timeseries_mask is not None:
            mask = self._monitor_array(timeseries_mask).astype(bool, copy=False)
            axes = tuple(range(1, mask.ndim))
            missing = 1.0 - mask.mean(axis=axes) if axes else 1.0 - mask
            storage.setdefault('input_missing_fraction', []).append(missing)

    @staticmethod
    def _concatenate_monitor_metadata(storage):
        return {
            key: np.concatenate(values, axis=0)
            for key, values in storage.items() if values
        }

    def _update_data_quality(self, accumulator, batch_x, batch_y,
                             current_power, target_mask, metadata):
        if accumulator is None:
            return
        feature_mask = None if metadata is None else metadata.get('timeseries_mask')
        image_mask = None if metadata is None else metadata.get('image_mask')
        image_age = None
        if metadata is not None:
            image_age = metadata.get('image_age_minutes')
            if image_age is None:
                image_age = metadata.get('image_minute_offsets')
        source_codes = None
        coverage_masks = None
        if metadata is not None:
            source_keys = (
                'timeseries_source', 'history_source', 'forecast_source',
                'teacher_history_timeseries_source',
                'teacher_future_timeseries_source',
            )
            source_codes = {
                key: metadata[key]
                for key in source_keys if metadata.get(key) is not None
            } or None
            coverage_key_map = {
                'forecast_mask': 'forecast_own_product',
                'power_history_mask': 'power_history',
                'teacher_history_timeseries_mask': 'teacher_history_timeseries',
                'teacher_future_timeseries_mask': 'teacher_future_timeseries',
                'teacher_future_image_mask': 'teacher_future_images',
            }
            coverage_masks = {
                destination: metadata[source]
                for source, destination in coverage_key_map.items()
                if metadata.get(source) is not None
            } or None
        accumulator.update(
            features=batch_x,
            feature_mask=feature_mask,
            target=batch_y,
            current_power=current_power,
            target_mask=target_mask,
            image_mask=image_mask,
            image_age_minutes=image_age,
            source_codes=source_codes,
            issue_time_ns=(
                None if metadata is None else metadata.get('issue_time_ns')),
            coverage_masks=coverage_masks,
            privileged_teacher_enabled=(
                None if metadata is None
                else metadata.get('privileged_teacher_enabled')),
            history_images_enabled=(
                None if metadata is None
                else metadata.get('history_images_enabled')),
            current_power_valid=(
                None if metadata is None
                else metadata.get('current_power_valid')),
        )

    def _compute_loss(self, outputs, target, sample_weight=None, current_pv=None,
                      target_mask=None):
        loss_target = target
        if target_mask is not None:
            valid_target = target_mask.to(target.device, dtype=torch.bool)
            if valid_target.dim() == target.dim() - 1:
                valid_target = valid_target.unsqueeze(-1)
            valid_target = valid_target.expand_as(target)
            loss_target = torch.where(
                valid_target, target, outputs.detach())
        if self.loss_type == 'Huber':
            element_loss = torch.nn.functional.huber_loss(
                outputs, loss_target, reduction='none')
        elif self.loss_type == 'Weighted_MSE_MAE':
            element_loss = (
                (outputs - loss_target).pow(2) * self.args.alpha_weight
                + (outputs - loss_target).abs()
                * (1 - self.args.alpha_weight))
        elif self.loss_type == 'log_cosh':
            element_loss = torch.log(torch.cosh(outputs - loss_target))
        else:
            element_loss = (outputs - loss_target).pow(2)
        if target_mask is None:
            per_sample_loss = element_loss.flatten(start_dim=1).mean(dim=1)
            if sample_weight is None:
                task_weight = float(per_sample_loss.numel())
            else:
                task_weight = float(sample_weight.detach().float().sum().item())
            task_loss = self._reduce_sample_loss(per_sample_loss, sample_weight)
        else:
            mask = target_mask.to(element_loss.device, dtype=element_loss.dtype)
            if mask.dim() == element_loss.dim() - 1:
                mask = mask.unsqueeze(-1)
            weights = mask.expand_as(element_loss)
            if sample_weight is not None:
                weights = weights * sample_weight.to(element_loss).view(-1, 1, 1)
            task_weight = float(weights.detach().sum().item())
            valid_element_loss = torch.where(
                weights > 0, element_loss, torch.zeros_like(element_loss))
            task_loss = (
                (valid_element_loss * weights).sum()
                / weights.sum().clamp_min(1.0)
            )
        ramp_component = torch.zeros((), device=outputs.device)
        direction_component = torch.zeros((), device=outputs.device)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        ramp_aux_weight = float(getattr(self.args, 'ramp_aux_weight', 0.0))
        direction_aux_weight = float(getattr(self.args, 'ramp_direction_aux_weight', 0.0))
        if current_pv is not None and ramp_aux_weight > 0 and getattr(model, 'last_ramp_pred', None) is not None:
            ramp_target = loss_target - current_pv
            ramp_element_loss = (model.last_ramp_pred - ramp_target).pow(2)
            if target_mask is None:
                ramp_loss = ramp_element_loss.flatten(start_dim=1).mean(dim=1)
                ramp_component = ramp_aux_weight * self._reduce_sample_loss(
                    ramp_loss, sample_weight)
            else:
                ramp_weights = target_mask.to(
                    ramp_element_loss.device,
                    dtype=ramp_element_loss.dtype)
                if ramp_weights.dim() == ramp_element_loss.dim() - 1:
                    ramp_weights = ramp_weights.unsqueeze(-1)
                ramp_weights = ramp_weights.expand_as(ramp_element_loss)
                if sample_weight is not None:
                    ramp_weights = ramp_weights * sample_weight.to(
                        ramp_element_loss).view(-1, 1, 1)
                valid_ramp_loss = torch.where(
                    ramp_weights > 0, ramp_element_loss,
                    torch.zeros_like(ramp_element_loss))
                ramp_component = ramp_aux_weight * (
                    (valid_ramp_loss * ramp_weights).sum()
                    / ramp_weights.sum().clamp_min(1.0))
        if current_pv is not None and direction_aux_weight > 0 and getattr(model, 'last_ramp_direction_logit', None) is not None:
            direction_target = (loss_target > current_pv).to(target.dtype)
            direction_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model.last_ramp_direction_logit, direction_target, reduction='none')
            if target_mask is None:
                direction_loss = direction_loss.flatten(start_dim=1).mean(dim=1)
                direction_component = direction_aux_weight * self._reduce_sample_loss(
                    direction_loss, sample_weight)
            else:
                direction_weights = target_mask.to(
                    direction_loss.device, dtype=direction_loss.dtype)
                if direction_weights.dim() == direction_loss.dim() - 1:
                    direction_weights = direction_weights.unsqueeze(-1)
                direction_weights = direction_weights.expand_as(direction_loss)
                if sample_weight is not None:
                    direction_weights = direction_weights * sample_weight.to(
                        direction_loss).view(-1, 1, 1)
                valid_direction_loss = torch.where(
                    direction_weights > 0, direction_loss,
                    torch.zeros_like(direction_loss))
                direction_component = direction_aux_weight * (
                    (valid_direction_loss * direction_weights).sum()
                    / direction_weights.sum().clamp_min(1.0))
        orthogonality_component = torch.zeros((), device=outputs.device)
        if self.args.fuse_strategy not in ['no_img', 'ts_only', 'ts']:
            orthogonality_component = self._orthogonality_loss()
        loss = (
            task_loss + ramp_component + direction_component
            + orthogonality_component
        )
        self._last_loss_components = {
            'task_loss': float(task_loss.detach().item()),
            'task_weight': task_weight,
            'ramp_aux_loss': float(ramp_component.detach().item()),
            'direction_aux_loss': float(direction_component.detach().item()),
            'orthogonality_loss': float(orthogonality_component.detach().item()),
            'auxiliary_loss': float((
                ramp_component + direction_component + orthogonality_component
            ).detach().item()),
        }
        return loss
 

    def vali(self, vali_data, vali_loader, criterion, monitor=None, epoch=0,
             split='val', phase='epoch', collect_monitor=False,
             data_quality_accumulator=None):
        task_loss_numerator = 0.0
        task_loss_denominator = 0.0
        auxiliary_loss_numerator = 0.0
        component_losses = {
            'ramp_aux_loss': 0.0,
            'direction_aux_loss': 0.0,
            'orthogonality_loss': 0.0,
        }
        predictions = []
        targets = []
        current_powers = []
        target_masks = []
        monitor_metadata = {}
        image_similarity = self._new_scalar_stats()
        weather_similarity = self._new_scalar_stats()
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_img, batch_y_img, batch_x_weather, batch_y_weather, _, metadata = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                batch_x_img = batch_x_img.float().to(self.device, non_blocking=True)
                batch_y_img = batch_y_img.float().to(self.device, non_blocking=True)
                
                batch_x_weather = batch_x_weather.float().to(self.device, non_blocking=True)
                batch_y_weather = batch_y_weather.float().to(self.device, non_blocking=True)
                metadata = self._metadata_to_device(metadata)

                # encoder - decoder
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs, sim_img_ts, sim_weather_ts, _ = self._forward(
                        batch_x, batch_x_img, batch_y_img, batch_x_weather,
                        batch_y_weather, metadata)
                
                if self.args.data == 'Folsom':
                    f_dim = -42 if self.args.features == 'MS' else 0
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device, non_blocking=True)

                current_pv = batch_x[:, :1, :1] if self.args.history_order == 'current_first' else batch_x[:, -1:, :1]
                monitor_current_pv = self._monitor_current_power(batch_x)
                target_mask = None if metadata is None else metadata.get('target_mask')
                loss = self._compute_loss(
                    outputs, batch_y, current_pv=current_pv, target_mask=target_mask)
                components = self._last_loss_components
                task_loss_numerator += (
                    components['task_loss'] * components['task_weight'])
                task_loss_denominator += components['task_weight']
                auxiliary_loss_numerator += (
                    components['auxiliary_loss'] * components['task_weight'])
                for name in component_losses:
                    component_losses[name] += (
                        components[name] * components['task_weight'])

                self._update_data_quality(
                    data_quality_accumulator, batch_x, batch_y,
                    monitor_current_pv, target_mask, metadata)
                if collect_monitor and monitor is not None and monitor.enabled:
                    predictions.append(outputs.detach().cpu().numpy())
                    targets.append(batch_y.detach().cpu().numpy())
                    current_powers.append(
                        monitor_current_pv.detach().cpu().numpy())
                    if target_mask is None:
                        mask = np.ones(batch_y.shape[:-1], dtype=bool)
                    else:
                        mask = target_mask.detach().cpu().numpy().astype(bool)
                    target_masks.append(mask)
                    self._append_monitor_metadata(
                        monitor_metadata, metadata, batch_y.shape[0])
                    self._update_scalar_stats(image_similarity, sim_img_ts)
                    self._update_scalar_stats(weather_similarity, sim_weather_ts)

        task_loss = (
            task_loss_numerator / task_loss_denominator
            if task_loss_denominator > 0 else float('nan'))
        auxiliary_loss = (
            auxiliary_loss_numerator / task_loss_denominator
            if task_loss_denominator > 0 else 0.0)
        total_loss = task_loss + auxiliary_loss
        self._last_validation_monitor = {
            'split': split,
            'phase': phase,
            'total_loss': total_loss,
            'task_loss': task_loss,
            'auxiliary_loss': auxiliary_loss,
            'valid_target_count': int(task_loss_denominator),
            **{
                name: value / task_loss_denominator
                if task_loss_denominator > 0 else 0.0
                for name, value in component_losses.items()
            },
            'image_similarity': self._finish_scalar_stats(image_similarity),
            'weather_similarity': self._finish_scalar_stats(weather_similarity),
        }
        if collect_monitor and predictions:
            self._last_validation_monitor['forecast'] = monitor.record_forecast(
                epoch=epoch,
                split=split,
                role='teacher',
                predictions=np.concatenate(predictions, axis=0),
                targets=np.concatenate(targets, axis=0),
                current_power=np.concatenate(current_powers, axis=0),
                target_mask=np.concatenate(target_masks, axis=0),
                metadata=self._concatenate_monitor_metadata(monitor_metadata),
                phase=phase,
            )

        if was_training:
            self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        include_training_test = bool(getattr(
            self.args, 'monitor_include_test_during_training', False))
        if include_training_test:
            test_data, test_loader = self._get_data(flag='test')
        else:
            test_data, test_loader = None, None

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        history_path = os.path.join(path, 'loss_history.csv')
        if os.path.exists(history_path):
            os.remove(history_path)

        time_now = time.time()

        train_steps = len(train_loader)
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

        self.training_monitor = TrainingMonitor.from_args(
            self.args, path, 'teacher', self.model)
        monitored_datasets = {'train': train_data, 'val': vali_data}
        if test_data is not None:
            monitored_datasets['test'] = test_data
        self.training_monitor.start_run(self.model, monitored_datasets)
        train_quality = (
            self.training_monitor.new_data_quality_accumulator(
                'train', self._monitor_feature_names(train_data))
            if self.training_monitor.enabled else None)
        val_quality = (
            self.training_monitor.new_data_quality_accumulator(
                'val', self._monitor_feature_names(vali_data))
            if self.training_monitor.enabled else None)
        test_quality = (
            self.training_monitor.new_data_quality_accumulator(
                'test', self._monitor_feature_names(test_data))
            if self.training_monitor.enabled and test_data is not None else None)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_task_numerator = 0.0
            train_task_denominator = 0.0
            train_auxiliary_numerator = 0.0
            train_components = {
                'ramp_aux_loss': 0.0,
                'direction_aux_loss': 0.0,
                'orthogonality_loss': 0.0,
            }
            image_similarity = self._new_scalar_stats()
            weather_similarity = self._new_scalar_stats()
            gradient_norms = []
            branch_gradient_norms = {}
            clipped_gradients = 0
            nonfinite_gradients = 0
            amp_skipped_steps = 0
            amp_scales = []
            processed_samples = 0
            parameter_snapshot = self.training_monitor.start_parameter_snapshot(
                self.model)

            self.model.train()
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_img, batch_y_img, batch_x_weather, batch_y_weather, sample_weight, metadata = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                batch_x_img = batch_x_img.float().to(self.device, non_blocking=True)
                batch_y_img = batch_y_img.float().to(self.device, non_blocking=True)
                
                batch_x_weather = batch_x_weather.float().to(self.device, non_blocking=True)
                batch_y_weather = batch_y_weather.float().to(self.device, non_blocking=True)
                metadata = self._metadata_to_device(metadata)
                if sample_weight is not None:
                    sample_weight = sample_weight.float().to(self.device, non_blocking=True)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs, sim_img_ts, sim_weather_ts, _ = self._forward(
                        batch_x, batch_x_img, batch_y_img, batch_x_weather,
                        batch_y_weather, metadata)

                    if self.args.data == 'Folsom':
                        f_dim = -42 if self.args.features == 'MS' else 0
                    else:
                        f_dim = -1 if self.args.features == 'MS' else 0

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:]
                    current_pv = batch_x[:, :1, :1] if self.args.history_order == 'current_first' else batch_x[:, -1:, :1]
                    target_mask = None if metadata is None else metadata.get('target_mask')
                    loss = self._compute_loss(
                        outputs, batch_y_target, sample_weight, current_pv=current_pv,
                        target_mask=target_mask)

                components = self._last_loss_components
                train_task_numerator += (
                    components['task_loss'] * components['task_weight'])
                train_task_denominator += components['task_weight']
                train_auxiliary_numerator += (
                    components['auxiliary_loss'] * components['task_weight'])
                for name in train_components:
                    train_components[name] += (
                        components[name] * components['task_weight'])
                processed_samples += int(batch_y_target.shape[0])
                if self.training_monitor.enabled:
                    self._update_scalar_stats(image_similarity, sim_img_ts)
                    self._update_scalar_stats(weather_similarity, sim_weather_ts)
                    if epoch == 0:
                        self._update_data_quality(
                            train_quality, batch_x, batch_y_target,
                            self._monitor_current_power(batch_x), target_mask,
                            metadata)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scale_before = float(scaler.get_scale())
                    amp_scales.append(scale_before)
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                else:
                    loss.backward()
                gradient_interval = int(getattr(
                    self.args, 'monitor_gradient_interval', 50))
                if (
                    self.training_monitor.enabled
                    and (i == 0 or (i + 1) % gradient_interval == 0)
                ):
                    for branch, norm in self._gradient_branch_norms(
                        self.model).items():
                        branch_gradient_norms.setdefault(branch, []).append(norm)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)
                if self.args.use_amp:
                    scaler.step(model_optim)
                    scaler.update()
                    if float(scaler.get_scale()) < scale_before:
                        amp_skipped_steps += 1
                else:
                    model_optim.step()
                grad_norm = float(grad_norm.detach().item())
                if np.isfinite(grad_norm):
                    gradient_norms.append(grad_norm)
                    clipped_gradients += int(grad_norm > 1.0)
                else:
                    nonfinite_gradients += 1

            epoch_duration = time.time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_duration))
            train_task_loss = (
                train_task_numerator / train_task_denominator
                if train_task_denominator > 0 else float('nan'))
            train_auxiliary_loss = (
                train_auxiliary_numerator / train_task_denominator
                if train_task_denominator > 0 else 0.0)
            train_loss = train_task_loss + train_auxiliary_loss
            parameter_updates = self.training_monitor.finish_parameter_snapshot(
                self.model, parameter_snapshot)
            collect_epoch = (
                self.training_monitor.enabled
                and (epoch + 1) % int(getattr(
                    self.args, 'monitor_epoch_interval', 1)) == 0)
            vali_loss = self.vali(
                vali_data, vali_loader, criterion,
                monitor=self.training_monitor,
                epoch=epoch + 1,
                split='val',
                phase='epoch',
                collect_monitor=collect_epoch,
                data_quality_accumulator=val_quality if epoch == 0 else None,
            )
            val_diagnostics = self._privacy_safe_validation_diagnostics(
                dict(self._last_validation_monitor))
            if test_loader is not None:
                test_loss = self.vali(
                    test_data, test_loader, criterion,
                    monitor=self.training_monitor,
                    epoch=epoch + 1,
                    split='test',
                    phase='training_opt_in',
                    collect_monitor=collect_epoch,
                    data_quality_accumulator=test_quality if epoch == 0 else None,
                )
                test_diagnostics = self._privacy_safe_validation_diagnostics(
                    dict(self._last_validation_monitor))
            else:
                test_loss = None
                test_diagnostics = None

            if test_loss is None:
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                    "Vali Loss: {3:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss))
            else:
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                    "Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss,
                        test_loss))

            previous_best = early_stopping.val_loss_min
            early_stopping(vali_loss, self.model, path, model_name='checkpoint.pth')
            checkpoint_saved = bool(early_stopping.val_loss_min < previous_best)
            if checkpoint_saved:
                self._monitor_best_epoch = epoch + 1
            if epoch == 0 and self.training_monitor.enabled:
                self.training_monitor.record_data_quality(
                    'train', train_quality.summary())
                self.training_monitor.record_data_quality(
                    'val', val_quality.summary())
                if test_quality is not None:
                    self.training_monitor.record_data_quality(
                        'test', test_quality.summary())

            peak_memory_mb = None
            if self.device.type == 'cuda':
                peak_memory_mb = float(
                    torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
            self.training_monitor.record_epoch({
                'epoch': epoch + 1,
                'stage': 'teacher',
                'losses': {
                    'total_loss': train_loss,
                    'task_loss': train_task_loss,
                    'auxiliary_loss': train_auxiliary_loss,
                    **{
                        name: value / train_task_denominator
                        if train_task_denominator > 0 else 0.0
                        for name, value in train_components.items()
                    },
                },
                'validation': val_diagnostics,
                'test_during_training': test_diagnostics,
                'similarity': {
                    'image': self._finish_scalar_stats(image_similarity),
                    'weather': self._finish_scalar_stats(weather_similarity),
                },
                'gradients': {
                    'pre_clip_norm_mean': (
                        float(np.mean(gradient_norms))
                        if gradient_norms else None),
                    'pre_clip_norm_std': (
                        float(np.std(gradient_norms))
                        if gradient_norms else None),
                    'clip_rate': (
                        clipped_gradients / len(gradient_norms)
                        if gradient_norms else None),
                    'nonfinite_count': nonfinite_gradients,
                    'branches': {
                        name: {
                            'sample_count': len(values),
                            'pre_clip_norm_mean': float(np.mean(values)),
                            'pre_clip_norm_std': float(np.std(values)),
                        }
                        for name, values in branch_gradient_norms.items()
                    },
                },
                'amp': {
                    'enabled': bool(self.args.use_amp),
                    'scale_mean': (
                        float(np.mean(amp_scales))
                        if amp_scales else None),
                    'scale_final': (
                        float(scaler.get_scale())
                        if self.args.use_amp else None),
                    'skipped_steps': amp_skipped_steps,
                },
                'parameter_updates': parameter_updates,
                'optimizer': {
                    'learning_rate': model_optim.param_groups[0]['lr'],
                },
                'performance': {
                    'epoch_time_sec': epoch_duration,
                    'samples_per_sec': (
                        processed_samples / epoch_duration
                        if epoch_duration > 0 else None),
                    'steps': train_steps,
                    'samples': processed_samples,
                    'peak_cuda_memory_mb': peak_memory_mb,
                },
                'checkpoint': {
                    'saved': checkpoint_saved,
                    'best_epoch': self._monitor_best_epoch,
                    'best_validation_loss': (
                        self._privacy_safe_validation_value(
                            early_stopping.val_loss_min, val_diagnostics)),
                    'early_stop_counter': early_stopping.counter,
                    'early_stop_triggered': early_stopping.early_stop,
                },
            })
            append_metrics_row(history_path, {
                "epoch": epoch + 1,
                "train_steps": train_steps,
                "train_samples": len(train_data),
                "vali_samples": len(vali_data),
                "test_samples": len(test_data) if test_data is not None else 0,
                "train_loss": train_loss,
                "train_task_loss": train_task_loss,
                "train_auxiliary_loss": train_auxiliary_loss,
                "vali_loss": vali_loss,
                "test_loss": test_loss,
                "learning_rate": model_optim.param_groups[0]["lr"],
                "epoch_time_sec": epoch_duration,
                "early_stop_counter": early_stopping.counter,
                "best_vali_loss": early_stopping.val_loss_min,
            })

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        if self.args.data == 'LuoyangParquet':
            write_checkpoint_contract(best_model_path, self.args.luoyang_config)
            validate_checkpoint_contract(best_model_path, self.args.luoyang_config)
        elif self.args.data == 'YLJParquet':
            write_ylj_checkpoint_contract(best_model_path, self.args.ylj_config)
            validate_ylj_checkpoint_contract(best_model_path, self.args.ylj_config)

        best_epoch = self._monitor_best_epoch or 0
        best_vali_loss = self.vali(
            vali_data, vali_loader, criterion,
            monitor=self.training_monitor,
            epoch=best_epoch,
            split='val',
            phase='best_checkpoint',
            collect_monitor=self.training_monitor.enabled,
        )
        best_forecast = self._last_validation_monitor.get('forecast')
        safe_best_vali_loss = (
            None if isinstance(best_forecast, dict)
            and best_forecast.get('privacy_suppressed')
            else best_vali_loss)
        self.training_monitor.finalize({
            'stage': 'teacher',
            'status': 'trained',
            'best_epoch': best_epoch,
            'best_validation_loss': safe_best_vali_loss,
            'test_during_training_enabled': include_training_test,
        })

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        run_path = os.path.join(self.args.checkpoints, setting)
        if self.training_monitor is None:
            self.training_monitor = TrainingMonitor.from_args(
                self.args, run_path, 'teacher', self.model)
            self.training_monitor.start_run(
                self.model, {'test': test_data})
        self.training_monitor.register_datasets({'test': test_data})
        if self._monitor_best_epoch is None:
            stored_best_epoch = self.training_monitor.result_value('best_epoch')
            if isinstance(stored_best_epoch, int) and not isinstance(
                stored_best_epoch, bool
            ):
                self._monitor_best_epoch = stored_best_epoch
        test_quality = (
            self.training_monitor.new_data_quality_accumulator(
                'test', self._monitor_feature_names(test_data))
            if self.training_monitor.enabled else None)
        if test:
            print('loading models')
            checkpoint_path = os.path.join(run_path, 'checkpoint.pth')
            if self.args.data == 'LuoyangParquet':
                validate_checkpoint_contract(checkpoint_path, self.args.luoyang_config)
            elif self.args.data == 'YLJParquet':
                validate_ylj_checkpoint_contract(checkpoint_path, self.args.ylj_config)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))


        preds = []
        trues = []
        monitor_current_powers = []
        monitor_target_masks = []
        monitor_metadata = {}
        image_similarity = self._new_scalar_stats()
        weather_similarity = self._new_scalar_stats()

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_x_img, batch_y_img, batch_x_weather, batch_y_weather, _, metadata = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                batch_x_img = batch_x_img.float().to(self.device, non_blocking=True)
                batch_y_img = batch_y_img.float().to(self.device, non_blocking=True)
                
                batch_x_weather = batch_x_weather.float().to(self.device, non_blocking=True)
                batch_y_weather = batch_y_weather.float().to(self.device, non_blocking=True)
                metadata = self._metadata_to_device(metadata)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs, sim_img_ts, sim_weather_ts, _ = self._forward(
                        batch_x, batch_x_img, batch_y_img, batch_x_weather,
                        batch_y_weather, metadata)

                if self.args.data == 'Folsom':
                    f_dim = -42 if self.args.features == 'MS' else 0
                else:
                    f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(
                    self.device, non_blocking=True)
                current_power = self._monitor_current_power(batch_x)
                target_mask = None if metadata is None else metadata.get('target_mask')
                self._update_data_quality(
                    test_quality, batch_x, batch_y, current_power,
                    target_mask, metadata)
                if self.training_monitor.enabled:
                    monitor_current_powers.append(
                        current_power.detach().cpu().numpy())
                    if target_mask is None:
                        mask = np.ones(batch_y.shape[:-1], dtype=bool)
                    else:
                        mask = target_mask.detach().cpu().numpy().astype(bool)
                    monitor_target_masks.append(mask)
                    self._append_monitor_metadata(
                        monitor_metadata, metadata, batch_y.shape[0])
                    self._update_scalar_stats(image_similarity, sim_img_ts)
                    self._update_scalar_stats(weather_similarity, sim_weather_ts)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

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

        test_forecast_summary = None
        if self.training_monitor.enabled:
            self.training_monitor.record_data_quality(
                'test', test_quality.summary())
            test_forecast_summary = self.training_monitor.record_forecast(
                epoch=self._monitor_best_epoch,
                split='test',
                role='teacher',
                predictions=preds,
                targets=trues,
                current_power=np.concatenate(
                    monitor_current_powers, axis=0),
                target_mask=np.concatenate(
                    monitor_target_masks, axis=0),
                metadata=self._concatenate_monitor_metadata(
                    monitor_metadata),
                phase='final_checkpoint',
            )

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

        f = open("result_long_term_forecasting_teacher.txt", 'a')   
        f.write(setting + "  \n")
        if self.args.data == 'Stanford':
            f.write(metrics_text)
        else:
            f.write(metrics_text)
        f.write('\n')
        f.write('\n')
        f.close()

        test_similarity = self._privacy_safe_similarity(
            image_similarity, weather_similarity, test_forecast_summary
        )
        final_monitor_payload = {
            'stage': 'teacher',
            'status': 'tested',
            'final_test_recorded': bool(self.training_monitor.enabled),
            'final_test': test_forecast_summary,
            'test_similarity': test_similarity,
        }
        if self._monitor_best_epoch is not None:
            final_monitor_payload['best_epoch'] = self._monitor_best_epoch
        self.training_monitor.finalize(final_monitor_payload)

        return
