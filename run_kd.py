import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting_kd import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import copy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Teacher Network Training for Multi-Modal Time Series Forecasting')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--config_only', action='store_true', default=False,
                        help='validate and print the derived configuration, then exit')
    parser.add_argument('--model_id', type=str, default='student_MTS_31', help='model id')
    parser.add_argument('--model', type=str, default='MTS_31',
                        help='student model name, should be MTS_31')
    parser.add_argument('--tsl_model_name', type=str, default='PatchTST',
                        help='External Time-Series-Library model name when --model TSL_Generic')

    # data loader
    parser.add_argument('--data', type=str, default='Folsom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/Folsom/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='Target_intra-hour.csv', help='data file')
    parser.add_argument('--luoyang_config', type=str, default='configs/luoyang_parquet.json',
                        help='configuration file for data=LuoyangParquet')
    parser.add_argument('--ylj_config', type=str, default='configs/datasets/ylj.yaml',
                        help='configuration file for data=YLJParquet')
    parser.add_argument('--results_root', type=str, default='./results', help='prediction and metric output root')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='ghi_5min', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='5t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--times_trainval_path', type=str, default='times_trainval.npy',
                        help='Stanford timestamp file for trainval split')
    parser.add_argument('--times_test_path', type=str, default='times_test.npy',
                        help='Stanford timestamp file for test split')
    parser.add_argument('--train_ratio', type=float, default=0.85,
                        help='train/validation split ratio inside Stanford trainval split when split_strategy=ratio')
    parser.add_argument('--split_strategy', type=str, default='day_block',
                        choices=['day_block', 'ratio'],
                        help='Stanford train/validation split strategy')
    parser.add_argument('--num_fold', type=int, default=10,
                        help='number of folds for Stanford day-block cross validation')
    parser.add_argument('--fold_index', type=int, default=0,
                        help='fold index for Stanford day-block cross validation')
    parser.add_argument('--forecast_horizon_minutes', type=int, default=15,
                        help='Stanford forecast target horizon in minutes')
    parser.add_argument('--sample_interval_minutes', type=int, default=1,
                        help='minimum interval between Stanford forecast samples')
    parser.add_argument('--history_order', type=str, default='current_first',
                        choices=['current_first', 'past_first'],
                        help='order of Stanford image_log/pv_log history samples')
    parser.add_argument('--stanford_capacity_kw', type=float, default=30.1,
                        help='Stanford PV array rated capacity used for NRMSE/NMAE normalization')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=216, help='input sequence length (18 hours * 12 points/hour)')
    parser.add_argument('--label_len', type=int, default=36, help='start token length (3 hours * 12 points/hour)')
    parser.add_argument('--pred_len', type=int, default=72, help='prediction sequence length (6 hours * 12 points/hour)')
    parser.add_argument('--seasonal_patterns', type=str, default='Daily', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=42, help='encoder input size (time series features)')
    parser.add_argument('--dec_in', type=int, default=42, help='decoder input size (time series features)')
    parser.add_argument('--c_out', type=int, default=42, help='output size (time series features)')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')
    
    # Folsom dataset specific parameters
    parser.add_argument('--seq_len_hours', type=float, default=4, help='input sequence length in hours')
    parser.add_argument('--pred_len_hours', type=float, default=2, help='prediction sequence length in hours')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='number of batches prefetched by each DataLoader worker when num_workers > 0')
    parser.add_argument('--max_train_batches', type=int, default=0,
                        help='optional smoke-test cap; 0 uses the full training loader')
    parser.add_argument('--max_val_batches', type=int, default=0,
                        help='optional smoke-test cap; 0 uses the full validation loader')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='teacher_multimodal', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='write aggregate-only training diagnostics')
    parser.add_argument('--monitor_level', choices=['basic', 'standard', 'debug'],
                        default='standard', help='monitoring detail level')
    parser.add_argument('--monitor_epoch_interval', type=int, default=1,
                        help='validation monitoring interval in epochs')
    parser.add_argument('--monitor_gradient_interval', type=int, default=50,
                        help='step interval for detailed gradient diagnostics')
    parser.add_argument('--monitor_min_slice_count', type=int, default=100,
                        help='minimum valid points required for a reported slice')
    parser.add_argument('--monitor_min_slice_days', type=int, default=5,
                        help='minimum distinct days required for a reported temporal slice')
    parser.add_argument('--monitor_dir', type=str, default='',
                        help='optional monitoring output directory; defaults under the run checkpoint')
    parser.add_argument('--monitor_include_test_during_training', action='store_true',
                        default=False, help='explicitly allow aggregate test diagnostics each epoch')
    parser.add_argument('--monitor_save_raw_predictions', action='store_true',
                        default=False, help='reserved; raw prediction monitoring is prohibited')
    
    # Teacher network specific parameters (for future knowledge distillation)
    parser.add_argument('--kd_type', type=str, default='response', 
                        help='knowledge distillation type: response, relation, attention, contrastive, causal')
    parser.add_argument('--kd_loss_weight_sim', type=float, default=0.001, help='knowledge distillation loss weight for similarity')
    parser.add_argument('--kd_loss_weight_inter', type=float, default=0.1, help='knowledge distillation loss weight for inter-causal')
    
    # Multi-modal fusion parameters
    parser.add_argument('--fusion_type', type=str, default='attention', 
                        help='fusion method: attention, gate, concat')
    parser.add_argument('--img_feature_dim', type=int, default=512, help='image feature dimension')
    parser.add_argument('--weather_feature_dim', type=int, default=6, help='weather feature dimension')
    parser.add_argument('--privileged_teacher', action='store_true', default=False,
                        help='load configured future modalities for Teacher while masking Student')
    parser.add_argument('--image_height', type=int, default=64, help='Stanford sky image height')
    parser.add_argument('--image_width', type=int, default=64, help='Stanford sky image width')
    parser.add_argument('--image_proj_h', type=int, default=8, help='BOP projected image height')
    parser.add_argument('--image_proj_w', type=int, default=8, help='BOP projected image width')
    parser.add_argument('--stanford_image_mode', type=str, default='gray', choices=['gray', 'rgb'],
                        help='Stanford image branch input mode')
    parser.add_argument('--image_encoder_type', type=str, default='bop',
                        choices=['bop', 'cnn', 'cnn_motion', 'cnn_latest', 'cnn_dual_motion',
                                 'cnn_solar_advection'],
                        help='Stanford image encoder: BOP+FACTS encoder or CNN temporal encoder')
    parser.add_argument('--teacher_image_encoder_type', type=str, default=None,
                        choices=['bop', 'cnn', 'cnn_motion'],
                        help='teacher-specific image encoder matching its checkpoint')
    parser.add_argument('--image_cnn_dim', type=int, default=64,
                        help='frame feature width for CNN image encoders')
    parser.add_argument('--image_temporal_hidden_dim', type=int, default=64,
                        help='GRU hidden size for CNN image encoders')
    parser.add_argument('--stanford_sample_weight_mode', type=str, default='none',
                        choices=['none', 'pv_ramp', 'ramp_gt5', 'ramp_gt8', 'ramp_piecewise', 'cloudy_dates'],
                        help='Stanford training sample weighting mode')
    parser.add_argument('--stanford_sample_weight_scale', type=float, default=1.0,
                        help='strength for Stanford sample weighting')
    parser.add_argument('--stanford_sample_weight_clip', type=float, default=3.0,
                        help='maximum Stanford sample weight; <=0 disables clipping')
    parser.add_argument('--stanford_ramp_weight_5', type=float, default=1.5,
                        help='sample weight for 5 <= |ramp| < 8 kW in ramp_piecewise mode')
    parser.add_argument('--stanford_ramp_weight_8', type=float, default=2.0,
                        help='sample weight for |ramp| >= 8 kW in ramp_piecewise mode')
    parser.add_argument('--stanford_privileged_teacher', action='store_true', default=False,
                        help='load seq_y_img for the teacher while keeping the student causal')
    parser.add_argument('--stanford_return_phase_path', action='store_true', default=False,
                        help='return t+1..t+15 PV path targets for causal phase losses')
    parser.add_argument('--ramp_aux_weight', type=float, default=0.0,
                        help='weight for auxiliary PV change regression loss')
    parser.add_argument('--ramp_direction_aux_weight', type=float, default=0.0,
                        help='weight for auxiliary PV change direction classification loss')
    parser.add_argument('--expert_gate_bias', type=float, default=-1.0,
                        help='initial bias for the causal two-expert gate')
    parser.add_argument('--expert_gate_aux_weight', type=float, default=0.0,
                        help='weight for extreme-ramp gate supervision')
    parser.add_argument('--expert_gate_threshold_kw', type=float, default=5.0,
                        help='ramp threshold used to supervise the expert gate')
    parser.add_argument('--expert_gate_target', type=str, default='extreme', choices=['extreme', 'winner'],
                        help='supervise the gate using extreme ramp or the lower-error expert')
    parser.add_argument('--expert_gate_hidden_dim', type=int, default=32,
                        help='hidden width for the causal soft gate')
    parser.add_argument('--event_state_aux_weight', type=float, default=0.0,
                        help='weight for stable/cloud-enter/cloud-exit state supervision')
    parser.add_argument('--event_residual_aux_weight', type=float, default=0.0,
                        help='weight for event-only correction regression')
    parser.add_argument('--event_routing_mode', type=str, default='soft',
                        choices=['soft', 'hard', 'confidence'],
                        help='event expert routing used for validation and inference')
    parser.add_argument('--tail_routing_mode', type=str, default='confidence',
                        choices=['soft', 'hard', 'confidence', 'posterior'],
                        help='routing for the frozen-parent solar-token child')
    parser.add_argument('--tail_confidence_threshold', type=float, default=0.5,
                        help='minimum up/down probability for confidence tail routing')
    parser.add_argument('--tail_error_threshold_kw', type=float, default=3.0,
                        help='parent absolute-error threshold defining tail state supervision')
    parser.add_argument('--tail_gate_pretrain_epochs', type=int, default=0,
                        help='epochs that learn event/direction tokens while tail magnitude stays zero')
    parser.add_argument('--clear_sky_aux_weight', type=float, default=0.0,
                        help='weight for the target-time clear-sky envelope objective')
    parser.add_argument('--clear_sky_quantile', type=float, default=0.95,
                        help='upper conditional quantile learned by the clear-sky head')
    parser.add_argument('--phase_state_aux_weight', type=float, default=0.0,
                        help='weight for multi-lead hazard and return-phase supervision')
    parser.add_argument('--phase_path_aux_weight', type=float, default=0.0,
                        help='weight for capacity-normalized multi-lead PV path regression')
    parser.add_argument('--phase_motion_consistency_weight', type=float, default=0.0,
                        help='weight for causal adjacent-frame cloud-motion reconstruction')
    parser.add_argument('--phase_correction_mode', type=str,
                        default='learned_residual',
                        choices=['learned_residual', 'physics_path'],
                        help='free residual heads or current PV plus predicted phase path')
    parser.add_argument('--phase_benefit_aux_weight', type=float, default=0.0,
                        help='weight for endpoint candidate-benefit scale supervision')
    parser.add_argument('--phase_benefit_mode', type=str, default='winner',
                        choices=['winner', 'net_risk', 'conditional_mean',
                                 'family_stack', 'constrained_stack'],
                        help=('winner classifies any oracle improvement; net_risk '
                              'regresses signed excess squared error and abstains '
                              'unless the candidate has negative predicted risk'))
    parser.add_argument('--phase_constraint_dual_lr', type=float, default=0.05,
                        help='projected-ascent rate for group non-degradation constraints')
    parser.add_argument('--phase_constraint_penalty', type=float, default=1.0,
                        help='quadratic penalty for positive group-risk violations')
    parser.add_argument('--phase_validation_calibration', type=int, default=0,
                        help='project stable/active family scales on validation before checkpointing')
    parser.add_argument('--phase_risk_scale_kw', type=float, default=3.0,
                        help='RMSE scale mapping negative excess risk to route strength')
    parser.add_argument('--phase_transport_mode', type=str, default='mean_flow',
                        choices=['mean_flow', 'multi_hypothesis',
                                 'hypothesis_tokens', 'set_attention'],
                        help='retain one mean flow or multiple local correspondence modes')
    parser.add_argument('--phase_transport_hypotheses', type=int, default=3,
                        help='number of local motion modes transported to each forecast lead')
    parser.add_argument('--phase_stable_expert_all_samples', type=int, default=0,
                        help='train no-event/return residual experts on every matching sample')
    parser.add_argument('--phase_specialist_abstention', type=int, default=0,
                        help='train each phase residual expert to output zero outside its own regime')
    parser.add_argument('--phase_state_pretrain_epochs', type=int, default=0,
                        help='classifier-only causal phase pretraining epochs')
    parser.add_argument('--phase_magnitude_pretrain_epochs', type=int, default=0,
                        help='conditional residual pretraining epochs with routing disabled')
    parser.add_argument('--phase_benefit_pretrain_epochs', type=int, default=0,
                        help='benefit-gate pretraining epochs with routing disabled')
    parser.add_argument('--phase_resume_from', type=str, default='none',
                        choices=['none', 'state', 'magnitude', 'benefit'],
                        help='stages already completed by the student initialization checkpoint')
    parser.add_argument('--phase_route_classes', type=str,
                        default='active_down,active_up',
                        help='comma-separated endpoint phase experts enabled at inference')
    parser.add_argument('--residual_aux_weight', type=float, default=0.0,
                        help='weight for predicting the residual relative to the stable s14 branch')
    parser.add_argument('--residual_correction_clip_kw', type=float, default=10.0,
                        help='absolute clamp applied to image-driven residual corrections')
    parser.add_argument('--stable_correction_aux_weight', type=float, default=0.0,
                        help='penalty weight for image corrections on |ramp| below the stable threshold')
    parser.add_argument('--stable_ramp_threshold_kw', type=float, default=5.0,
                        help='strict upper ramp threshold used by the stable correction penalty')
    parser.add_argument('--kd_extreme_response_weight', type=float, default=0.0,
                        help='response KD weight applied only to high-ramp samples')
    parser.add_argument('--kd_extreme_response_threshold_kw', type=float, default=5.0,
                        help='ramp threshold for conditional response KD')
    parser.add_argument('--val_selection_metric', type=str, default='loss',
                        choices=['loss', 'rmse', 'ramp_composite', 'cloud_event_rmse'],
                        help='checkpoint selection metric; cloud_event_rmse uses internally volatile validation days')
    parser.add_argument('--val_extreme_metric_weight', type=float, default=0.25,
                        help='weight of >=8 kW validation RMSE in ramp_composite selection')
    parser.add_argument('--strict_causal_student', action='store_true', default=False,
                        help='zero future image/weather inputs before every student forward')
    parser.add_argument('--conservative_checkpointing', action='store_true', default=False,
                        help='treat epoch 0 as the immutable checkpoint baseline and rollback on no improvement')
    parser.add_argument('--val_min_improvement', type=float, default=0.0,
                        help='minimum strict validation-score decrease required for checkpoint promotion')
    parser.add_argument('--val_ordinary_guard', action='store_true', default=False,
                        help='reject checkpoints whose validation |ramp|<5 kW RMSE exceeds the epoch-0 limit')
    parser.add_argument('--val_ordinary_relative_tolerance', type=float, default=0.0,
                        help='relative tolerance for the ordinary-region validation guard')
    parser.add_argument('--val_no_path_guard', action='store_true', default=False,
                        help='reject checkpoints that degrade complete no-event paths')
    parser.add_argument('--val_no_path_relative_tolerance', type=float, default=0.0,
                        help='relative tolerance for the all-validation no-path guard')
    parser.add_argument('--val_cloud_ordinary_relative_tolerance', type=float, default=0.0,
                        help='relative tolerance for cloud-event endpoint ordinary RMSE')
    parser.add_argument('--val_cloud_no_path_relative_tolerance', type=float, default=0.0,
                        help='relative tolerance for cloud-event no-path RMSE')
    parser.add_argument('--val_tail_min_improvement', type=float, default=0.0,
                        help='absolute validation RMSE decrease required for cloud-event >=5 kW')
    parser.add_argument('--val_extreme_relative_tolerance', type=float, default=0.0,
                        help='relative tolerance for cloud-event >=8 kW RMSE')
    parser.add_argument('--skip_test_after_train', action='store_true', default=False,
                        help='finish after validation-based checkpoint selection')
    parser.add_argument('--test_only_if_accepted', action='store_true', default=False,
                        help='run test only when conservative validation accepted a trained checkpoint')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--use_trend', default=False, action="store_true")
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    parser.add_argument('--trail', type=str, default='1', help="trail number")
    
    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--tors', type=str, default='student', help='teacher or student')
    parser.add_argument('--teacher_model', type=str, default='MTS_31', help='teacher model name')
    parser.add_argument('--teacher_path', type=str, default='./checkpoints/long_term_forecast_Folsom_48_24_MTS_31F_Folsom_ftM_sl48_ll0_pl24_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_fusionattention_imgdim512_weatherdim6_Exp_MTS_31F_no_fusion_0/checkpoint.pth', help='teacher model path')
    parser.add_argument('--student_only', action='store_true', default=False,
                        help='train the Student with supervised task loss only; do not build or load a Teacher')
    parser.add_argument('--student_init_path', type=str, default='',
                        help='optional checkpoint used to initialize the student before training')
    parser.add_argument('--test_checkpoint', type=str, default='',
                        help='explicit checkpoint for standalone Luoyang inference')
    parser.add_argument('--freeze_ts_epochs', type=int, default=0,
                        help='freeze the original s14 time-series branch for the first N epochs')
    parser.add_argument('--freeze_ts_permanently', action='store_true', default=False,
                        help='keep the original time-series branch frozen and in eval mode for all epochs')
    parser.add_argument('--train_image_residual_only', action='store_true', default=False,
                        help='optimize only the causal image residual encoder/heads')
    parser.add_argument('--parent_fuse_strategy', type=str, default='',
                        help='parent checkpoint inference strategy recorded for branch auditing')
    parser.add_argument('--modality_dropout_rate', type=float, default=0.5, help='modality dropout rate')
    parser.add_argument('--loss_type', type=str, default='MSE', help='loss type', choices=['MSE', 'Huber'])
    parser.add_argument('--fuse_strategy', type=str, default=None,
                        choices=['no_img', 'no_weather', 'ts_only', 'ts', 'causal_img', 'causal_gated',
                                 'causal_residual_appearance', 'causal_residual_motion', 'causal_soft_gate',
                                 'causal_solar_event', 'causal_solar_token_tail',
                                 'causal_phase_field_tail'],
                        help='ablation/fusion strategy for multimodal branches')
    parser.add_argument('--student_fuse_strategy', type=str, default=None,
                        choices=['no_img', 'no_weather', 'ts_only', 'ts', 'causal_img', 'causal_gated',
                                 'causal_residual_appearance', 'causal_residual_motion', 'causal_soft_gate',
                                 'causal_solar_event', 'causal_solar_token_tail',
                                 'causal_phase_field_tail'],
                        help='student-specific causal fusion strategy')
    parser.add_argument('--teacher_fuse_strategy', type=str, default=None,
                        choices=['full', 'no_img', 'no_weather', 'ts_only', 'ts'],
                        help='teacher-specific privileged fusion strategy')
    parser.add_argument('--kd_intervention_min_feature_delta', type=float, default=1e-8,
                        help='minimum Teacher feature change required from privileged intervention')
    parser.add_argument('--teacher_legacy_image_order', action='store_true', default=False,
                        help='preserve the historical image order used to train legacy t11 checkpoints')

    args = parser.parse_args()

    if args.student_only:
        if args.privileged_teacher:
            parser.error('--student_only cannot be combined with --privileged_teacher')
        if any(weight > 0 for weight in (
                args.kd_loss_weight_sim, args.kd_loss_weight_inter,
                args.kd_extreme_response_weight)):
            parser.error('--student_only requires every KD loss weight to be zero')

    if args.monitor_save_raw_predictions:
        parser.error('monitoring is aggregate-only; raw predictions cannot be saved')
    if args.monitor_epoch_interval <= 0 or args.monitor_gradient_interval <= 0:
        parser.error('monitor intervals must be positive')
    if args.monitor_min_slice_count < 100 or args.monitor_min_slice_days < 5:
        parser.error(
            'monitor privacy thresholds require at least 100 values and 5 days')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f'Random seed: {args.seed}')
    
    # 设备配置
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU: cuda:{}'.format(args.gpu))
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using CPU or MPS')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        print(
            f"DataParallel device_ids={args.device_ids} "
            f"visible_cuda_devices={torch.cuda.device_count()}")

    print('Arguments for Teacher Network Training:')
    print_args(args)

    # 配置多模态参数
    # 图像分支参数配置
    args_img = copy.deepcopy(args)
    if args.data in ['Stanford', 'LuoyangParquet', 'YLJParquet']:
        args_img.seq_len = args.seq_len
        image_channels = 3 if args.stanford_image_mode == 'rgb' else 1
        if args.image_encoder_type == 'cnn_solar_advection':
            # The solar encoder consumes RGB tokens but keeps the legacy
            # 64-dimensional Appearance latent/checkpoint contract.
            args.img_feature_dim = args.image_cnn_dim
        else:
            args.img_feature_dim = image_channels * args.image_proj_h * args.image_proj_w
        args_img.enc_in = args.img_feature_dim
        args_img.dec_in = args.img_feature_dim
        args_img.c_out = args.img_feature_dim
        args_img.H = args.image_height
        args_img.W = args.image_width
        args_img.r_h = args.image_proj_h
        args_img.r_w = args.image_proj_w
        args_img.stanford_image_mode = args.stanford_image_mode
    else:
        args_img.enc_in = args.img_feature_dim  # 512维图像特征
        args_img.dec_in = args.img_feature_dim
        args_img.c_out = args.img_feature_dim
        args_img.H = 64
        args_img.W = 64
        args_img.r_h = 8
        args_img.r_w = 8
    args_img.task_name = 'long_term_forecast'
    
    # 天气分支参数配置
    args_weather = copy.deepcopy(args)
    args_weather.seq_len = args.seq_len + args.pred_len
    args_weather.enc_in = args.weather_feature_dim  # 42维天气特征
    args_weather.dec_in = args.weather_feature_dim
    args_weather.c_out = args.weather_feature_dim
    args_weather.task_name = 'long_term_forecast'

    # 确保主分支（时序）配置正确
    if args.features == 'MS':
        args.c_out = 1  # 单变量预测
    elif args.features == 'M':
        args.c_out = 42  # 多变量预测
    elif args.features == 'S':
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1

    # 验证配置
    print(f"\n=== Multi-Modal Configuration ===")
    print(f"Main branch (time series): enc_in={args.enc_in}, c_out={args.c_out}")
    print(f"Image branch: enc_in={args_img.enc_in}, c_out={args_img.c_out}")
    print(f"Weather branch: enc_in={args_weather.enc_in}, c_out={args_weather.c_out}")
    print(f"Task: {args.task_name}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Sequence Length: {args.seq_len} ({args.seq_len_hours} hours)")
    print(f"Prediction Length: {args.pred_len} ({args.pred_len_hours} hours)")
    print("=" * 50)

    if args.config_only:
        print('Configuration validation completed; training was not started.')
        raise SystemExit(0)

    # 教师网络训练只支持长期预测任务
    if args.task_name != 'long_term_forecast':
        print("Warning: Teacher network training is designed for long_term_forecast task.")
        print("Setting task_name to 'long_term_forecast'")
        args.task_name = 'long_term_forecast'

    # 实验设置
    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # 创建实验实例
            exp = Exp(args, args_img, args_weather)
            
            # 生成实验设置标识符
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_fusion{}_imgdim{}_weatherdim{}_{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.fusion_type,
                args.img_feature_dim,
                args.weather_feature_dim,
                args.des, 
                args.trail,
                ii)

            print('\n' + '='*100)
            print('>>>>>>>>>> Starting Student Network Training: {} <<<<<<<<<<'.format(setting))
            print('='*100)
            
            # 训练教师网络
            exp.train(setting)

            should_test = not args.skip_test_after_train
            if args.test_only_if_accepted:
                should_test = should_test and bool(getattr(
                    exp, '_checkpoint_accepted', False))
            if should_test:
                print('\n' + '='*100)
                print('>>>>>>>>>> Testing Student Network: {} <<<<<<<<<<'.format(setting))
                print('='*100)
                exp.test(setting)
            else:
                print(
                    'Test skipped after training: validation-only mode or '
                    'no accepted trained checkpoint.')
            
            if args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
                
            print(f"Student network training iteration {ii+1}/{args.itr} completed.\n")
            
    else:
        # 测试模式
        exp = Exp(args, args_img, args_weather)
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_fusion{}_imgdim{}_weatherdim{}_{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.fusion_type,
            args.img_feature_dim,
            args.weather_feature_dim,
            args.des, 
            args.trail,
            ii)

        print('\n' + '='*100)
        print('>>>>>>>>>> Testing Trained Student Network: {} <<<<<<<<<<'.format(setting))
        print('='*100)
        
        exp.test(setting, test=1)
        
        if args.gpu_type == 'cuda':
            torch.cuda.empty_cache()

    print("\n🎉 Student Network Training/Testing completed successfully!")
    print("📁 Model checkpoints saved in:", args.checkpoints)
    print("📊 Results saved in: ./results/")
