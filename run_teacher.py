import argparse
import torch
import torch.backends
from exp.exp_long_term_forecasting_teacher import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import copy

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Teacher Network Training for Multi-Modal Time Series Forecasting')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name, options:[long_term_forecast]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--config_only', action='store_true', default=False,
                        help='validate and print the derived configuration, then exit')
    parser.add_argument('--model_id', type=str, default='teacher_MTS_3', help='model id')
    parser.add_argument('--model', type=str, default='MTS_3', help='teacher model name, should be MTS_3')
    parser.add_argument('--tsl_model_name', type=str, default='PatchTST',
                        help='External Time-Series-Library model name when --model TSL_Generic')

    # data loader
    parser.add_argument('--data', type=str, default='Folsom', help='dataset type')
    parser.add_argument('--root_path', type=str, help='root path of the data file')
    parser.add_argument('--data_path', type=str, help='data file')
    parser.add_argument('--luoyang_config', type=str, default='configs/luoyang_parquet.json',
                        help='configuration file for data=LuoyangParquet')
    parser.add_argument('--ylj_config', type=str, default='configs/datasets/ylj.yaml',
                        help='configuration file for data=YLJParquet')
    parser.add_argument('--results_root', type=str, default='./results', help='prediction and metric output root')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='ghi_5min', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='5t', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
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
    parser.add_argument('--image_height', type=int, default=64, help='Stanford sky image height')
    parser.add_argument('--image_width', type=int, default=64, help='Stanford sky image width')
    parser.add_argument('--image_proj_h', type=int, default=8, help='BOP projected image height')
    parser.add_argument('--image_proj_w', type=int, default=8, help='BOP projected image width')
    parser.add_argument('--stanford_image_mode', type=str, default='gray', choices=['gray', 'rgb'],
                        help='Stanford image branch input mode')
    parser.add_argument('--image_encoder_type', type=str, default='bop', choices=['bop', 'cnn', 'cnn_motion'],
                        help='Stanford image encoder: BOP+FACTS encoder or CNN temporal encoder')
    parser.add_argument('--image_cnn_dim', type=int, default=64,
                        help='frame feature width for CNN image encoders')
    parser.add_argument('--image_temporal_hidden_dim', type=int, default=64,
                        help='GRU hidden size for CNN image encoders')
    parser.add_argument('--stanford_capacity_kw', type=float, default=30.1,
                        help='Stanford PV array rated capacity used for NRMSE/NMAE normalization')
    parser.add_argument('--stanford_sample_weight_mode', type=str, default='none',
                        choices=['none', 'pv_ramp', 'ramp_gt5', 'ramp_gt8', 'cloudy_dates'],
                        help='Stanford training sample weighting mode')
    parser.add_argument('--stanford_sample_weight_scale', type=float, default=1.0,
                        help='strength for Stanford sample weighting')
    parser.add_argument('--stanford_sample_weight_clip', type=float, default=3.0,
                        help='maximum Stanford sample weight; <=0 disables clipping')
    parser.add_argument('--ramp_aux_weight', type=float, default=0.0,
                        help='weight for signed-ramp kW regression auxiliary loss')
    parser.add_argument('--ramp_direction_aux_weight', type=float, default=0.0,
                        help='weight for ramp-direction classification auxiliary loss')
    parser.add_argument('--expert_gate_bias', type=float, default=-1.0,
                        help='initial bias for the transition-expert gate')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Daily', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
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
    parser.add_argument('--factor', type=int, default=1, help='attention factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='TimeMixer/FreTS channel independence flag')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        choices=['moving_avg', 'dft_decomp'],
                        help='TimeMixer decomposition method')
    parser.add_argument('--down_sampling_layers', type=int, default=0,
                        help='TimeMixer down-sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1,
                        help='TimeMixer down-sampling window')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        choices=[None, 'avg', 'max', 'conv'],
                        help='TimeMixer down-sampling method')
    parser.add_argument('--seg_len', type=int, default=8,
                        help='SegRNN segment length')
    parser.add_argument('--patch_len', type=int, default=8,
                        help='Patch length for PatchTST/PAttn/TimeXer/TimeFilter')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='DLinear individual channel flag')
    parser.add_argument('--channel_individual', action='store_true', default=False,
                        help='Reserved compatibility flag for channel-wise TSL models')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='FiLM spectral ratio')
    parser.add_argument('--subgraph_size', type=int, default=10,
                        help='MSGNet subgraph size')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='TimeFilter graph construction alpha')
    parser.add_argument('--top_p', type=float, default=0.5,
                        help='TimeFilter dynamic routing top_p')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1,
                        help='TimeFilter positional embedding switch')
    
    # Folsom dataset specific parameters
    parser.add_argument('--seq_len_hours', type=float, default=4, help='input sequence length in hours')
    parser.add_argument('--pred_len_hours', type=float, default=2, help='prediction sequence length in hours')
    
    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='number of batches prefetched by each DataLoader worker when num_workers > 0')
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
                        default=False,
                        help='explicitly allow aggregate test diagnostics each epoch')
    parser.add_argument('--monitor_save_raw_predictions', action='store_true',
                        default=False,
                        help='reserved; raw prediction monitoring is prohibited')
    
    # Teacher network specific parameters (for future knowledge distillation)
    parser.add_argument('--kd_type', type=str, default='response', 
                        help='knowledge distillation type: response, relation, attention, contrastive, causal')
    parser.add_argument('--kd_loss_weight', type=float, default=0.1, help='knowledge distillation loss weight')
    
    # Multi-modal fusion parameters
    parser.add_argument('--fusion_type', type=str, default='attention', 
                        help='fusion method: attention, gate, concat')
    parser.add_argument('--img_feature_dim', type=int, default=512, help='image feature dimension')
    parser.add_argument('--weather_feature_dim', type=int, default=6, help='weather feature dimension')
    parser.add_argument('--privileged_teacher', action='store_true', default=False,
                        help='expose configured future modalities to Teacher only')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers of projector')


    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Loss function parameters
    parser.add_argument('--loss_type', type=str, default='MSE', help='loss type', choices=['MSE', 'Huber', 'log_cosh', 'Weighted_MSE_MAE'])
    parser.add_argument('--alpha_weight', type=float, default=1.0, help='loss weight')
    parser.add_argument(
        '--fuse_strategy',
        type=str,
        default=None,
        help='ablation/fusion strategy',
        choices=['full', 'no_img', 'no_weather', 'ts_only', 'causal_img', 'causal_gated'])
    parser.add_argument('--tors', type=str, default='teacher', help='teacher or student')

    args = parser.parse_args()

    if args.monitor_save_raw_predictions:
        parser.error('monitoring is aggregate-only; raw predictions cannot be saved')
    if args.monitor_epoch_interval <= 0 or args.monitor_gradient_interval <= 0:
        parser.error('monitor intervals must be positive')
    if args.monitor_min_slice_count < 100 or args.monitor_min_slice_days < 5:
        parser.error(
            'monitor privacy thresholds require at least 100 values and 5 days')
    
    # Device configuration
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

    # Configure multi-modal parameters
    # Image branch parameter configuration
    args_img = copy.deepcopy(args)
    if args.data in ['Stanford', 'LuoyangParquet', 'YLJParquet']:
        args_img.seq_len = args.seq_len
        image_channels = 3 if args.stanford_image_mode == 'rgb' else 1
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
        args_img.seq_len = args.seq_len + args.label_len + args.pred_len
        args_img.enc_in = args.img_feature_dim  # image feature dimension
        args_img.dec_in = args.img_feature_dim
        args_img.c_out = args.img_feature_dim
        args_img.H = args.seq_len + args.label_len + args.pred_len
        args_img.W = args.img_feature_dim
        args_img.r_h = 8
        args_img.r_w = 8
    args_img.task_name = 'long_term_forecast'
    
    # Weather branch parameter configuration
    args_weather = copy.deepcopy(args)
    args_weather.seq_len = args.seq_len + args.pred_len
    args_weather.enc_in = args.weather_feature_dim  # 6-dimensional weather features
    args_weather.dec_in = args.weather_feature_dim
    args_weather.c_out = args.weather_feature_dim
    args_weather.task_name = 'long_term_forecast'

    # Ensure main branch (time series) configuration is correct
    if args.features == 'MS':
        args.c_out = 1  # Univariate prediction
    elif args.features == 'M':
        args.c_out = 42  # Multivariate prediction
    elif args.features == 'S':
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1

    # Validate configuration
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

    # Teacher network training only supports long-term forecasting task
    if args.task_name != 'long_term_forecast':
        print("Warning: Teacher network training is designed for long_term_forecast task.")
        print("Setting task_name to 'long_term_forecast'")
        args.task_name = 'long_term_forecast'

    # Experiment setup
    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # Create experiment instance
            exp = Exp(args, args_img, args_weather)
            
            # Generate experiment setting identifier
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_fusion{}_imgdim{}_weatherdim{}_{}_{}'.format(
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
                args.embed,
                args.fusion_type,
                args.img_feature_dim,
                args.weather_feature_dim,
                args.des, 
                ii)

            print('\n' + '='*100)
            print('>>>>>>>>>> Starting Teacher Network Training: {} <<<<<<<<<<'.format(setting))
            print('='*100)
            
            # Train teacher network
            exp.train(setting)

            print('\n' + '='*100)
            print('>>>>>>>>>> Testing Teacher Network: {} <<<<<<<<<<'.format(setting))
            print('='*100)
            
            # Test teacher network
            exp.test(setting)
            
            if args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
                
            print(f"Teacher network training iteration {ii+1}/{args.itr} completed.\n")
            
    else:
        # Test mode
        exp = Exp(args, args_img, args_weather)
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_fusion{}_imgdim{}_weatherdim{}_{}_{}'.format(
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
            args.embed,
            args.fusion_type,
            args.img_feature_dim,
            args.weather_feature_dim,
            args.des, 
            ii)

        print('\n' + '='*100)
        print('>>>>>>>>>> Testing Trained Teacher Network: {} <<<<<<<<<<'.format(setting))
        print('='*100)
        
        exp.test(setting, test=1)
        
        if args.gpu_type == 'cuda':
            torch.cuda.empty_cache()

    print("\n🎉 Teacher Network Training/Testing completed successfully!")
    print("📁 Model checkpoints saved in:", args.checkpoints)
    print("📊 Results saved in: ./results/")
