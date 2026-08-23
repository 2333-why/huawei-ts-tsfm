from torch.utils.data import DataLoader

from data_provider.data_loader_stanford import StanfordSolarForecastDataset
from data_provider.data_loader_luoyang import LuoyangParquetDataset
from data_provider.data_loader_ylj import YLJParquetDataset


_NO_FUTURE_IMAGE_STRATEGIES = {
    "no_img",
    "ts_only",
    "ts",
    "causal_img",
    "causal_gated",
    "causal_residual_appearance",
    "causal_residual_motion",
    "causal_soft_gate",
    "causal_solar_event",
    "causal_solar_token_tail",
    "causal_phase_field_tail",
}


def _loader_options(args):
    num_workers = int(getattr(args, "num_workers", 0))
    use_cuda = bool(getattr(args, "use_gpu", False)) and getattr(
        args, "gpu_type", "cuda"
    ) == "cuda"
    options = {"num_workers": num_workers, "pin_memory": use_cuda}
    if num_workers > 0:
        options.update(
            persistent_workers=True,
            prefetch_factor=int(getattr(args, "prefetch_factor", 4)),
        )
    return options


def _get_data_class(data_name):
    if data_name == "Stanford":
        return StanfordSolarForecastDataset
    if data_name == "LuoyangParquet":
        return LuoyangParquetDataset
    if data_name == "YLJParquet":
        return YLJParquetDataset
    raise ValueError(f"FACTS_NEW supports Stanford, LuoyangParquet or YLJParquet, got {data_name}")


def _stanford_load_future_images(args, flag):
    # KD exposes this flag explicitly: privileged images exist only on train.
    if hasattr(args, "stanford_privileged_teacher"):
        return flag == "train" and bool(args.stanford_privileged_teacher)
    return getattr(args, "fuse_strategy", None) not in _NO_FUTURE_IMAGE_STRATEGIES


def data_provider(args, flag):
    data_class = _get_data_class(args.data)
    if args.task_name != "long_term_forecast":
        raise ValueError(
            "FACTS_NEW only supports task_name=long_term_forecast, got "
            f"{args.task_name}"
        )

    is_train = flag == "train"
    privileged_teacher = bool(getattr(args, "privileged_teacher", False)) and (
        flag != "test" or getattr(args, "tors", "student") == "teacher"
    )
    if args.data == "LuoyangParquet":
        dataset = data_class(
            config_path=args.luoyang_config,
            flag=flag,
            load_images=getattr(args, "fuse_strategy", None)
            not in {"no_img", "ts_only", "ts"},
            privileged_teacher=privileged_teacher,
        )
    elif args.data == "YLJParquet":
        dataset = data_class(
            config_path=args.ylj_config, flag=flag,
            privileged_teacher=privileged_teacher,
        )
    else:
        dataset = data_class(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            features=args.features,
            target=args.target,
            scale=False,
            timeenc=0 if args.embed != "timeF" else 1,
            freq=args.freq,
            train_ratio=args.train_ratio,
            split_strategy=getattr(args, "split_strategy", "day_block"),
            num_fold=getattr(args, "num_fold", 10),
            fold_index=getattr(args, "fold_index", 0),
            forecast_horizon_minutes=args.forecast_horizon_minutes,
            sample_interval_minutes=args.sample_interval_minutes,
            history_order=args.history_order,
            times_trainval_path=args.times_trainval_path,
            times_test_path=args.times_test_path,
            weather_feature_dim=args.weather_feature_dim,
            image_mode=getattr(args, "stanford_image_mode", "gray"),
            stanford_capacity_kw=getattr(args, "stanford_capacity_kw", 30.1),
            sample_weight_mode=getattr(args, "stanford_sample_weight_mode", "none"),
            sample_weight_scale=getattr(args, "stanford_sample_weight_scale", 1.0),
            sample_weight_clip=getattr(args, "stanford_sample_weight_clip", 3.0),
            ramp_weight_5=getattr(args, "stanford_ramp_weight_5", 1.5),
            ramp_weight_8=getattr(args, "stanford_ramp_weight_8", 2.0),
            load_images=getattr(args, "fuse_strategy", None)
            not in {"no_img", "ts_only", "ts"},
            load_future_images=_stanford_load_future_images(args, flag),
            return_phase_path=bool(getattr(args, "stanford_return_phase_path", False)),
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=is_train,
        drop_last=is_train,
        **_loader_options(args),
    )
    return dataset, loader
