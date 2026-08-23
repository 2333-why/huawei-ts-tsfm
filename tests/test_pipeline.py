from pathlib import Path

from scripts import pipeline


def arg_value(command, name):
    return command[command.index(name) + 1]


def commands():
    config = pipeline.load_config()
    return {
        stage: pipeline.command_for_stage(config, stage, dry_run=True)
        for stage in pipeline.STAGE_ORDER
    }


def test_privileged_images_exist_only_where_teacher_distillation_is_enabled():
    stage_commands = commands()
    for stage in ("s14", "appearance", "solar"):
        assert "--stanford_privileged_teacher" in stage_commands[stage]
        assert "--strict_causal_student" in stage_commands[stage]
    for stage in ("phase_pretrain", "physics"):
        command = stage_commands[stage]
        assert "--stanford_privileged_teacher" not in command
        assert arg_value(command, "--kd_loss_weight_sim") == "0.0"
        assert arg_value(command, "--kd_loss_weight_inter") == "0.0"


def test_standard_student_commands_do_not_enable_legacy_teacher_image_order():
    stage_commands = commands()
    for stage in ("s14", "appearance", "solar", "phase_pretrain", "physics"):
        assert "--teacher_legacy_image_order" not in stage_commands[stage]


def test_phase_pretraining_emits_the_checkpoint_consumed_by_physics():
    stage_commands = commands()
    phase = stage_commands["phase_pretrain"]
    assert arg_value(phase, "--phase_state_pretrain_epochs") == "5"
    assert arg_value(phase, "--phase_magnitude_pretrain_epochs") == "3"
    assert arg_value(phase, "--phase_transport_mode") == "multi_hypothesis"
    assert arg_value(phase, "--phase_route_classes") == "all"

    physics = stage_commands["physics"]
    assert Path(arg_value(physics, "--student_init_path")).name == (
        "phase_magnitude_checkpoint.pth"
    )
    assert arg_value(physics, "--phase_resume_from") == "magnitude"
    assert arg_value(physics, "--phase_correction_mode") == "physics_path"
    assert arg_value(physics, "--phase_benefit_mode") == "constrained_stack"


def test_physics_requires_all_validation_guards_and_skips_test_during_training():
    command = commands()["physics"]
    for flag in (
        "--conservative_checkpointing",
        "--val_ordinary_guard",
        "--val_no_path_guard",
        "--skip_test_after_train",
        "--test_only_if_accepted",
        "--stanford_return_phase_path",
    ):
        assert flag in command
    assert arg_value(command, "--val_selection_metric") == "cloud_event_rmse"
    assert arg_value(command, "--val_cloud_ordinary_relative_tolerance") == "0.0"
    assert arg_value(command, "--val_cloud_no_path_relative_tolerance") == "0.0"
    assert arg_value(command, "--val_extreme_relative_tolerance") == "0.0"


def test_default_pipeline_has_no_legacy_facts_path_dependency():
    config_text = pipeline.CONFIG_PATH.read_text(encoding="utf-8")
    script_text = Path(pipeline.__file__).read_text(encoding="utf-8")
    legacy = "/opt/data/private/code/Stanford-solar-forecasting-dataset/" + "FACTS/"
    assert legacy not in config_text
    assert legacy not in script_text
