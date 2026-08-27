from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from models import MODEL_NAMES, RUN_MODES, iter_experiment_tasks
from models.tasks import ExperimentTask


def test_task_matrix_has_expected_cardinality_and_is_deterministic():
    first = list(iter_experiment_tasks())
    second = list(iter_experiment_tasks())

    assert len(first) == 68
    assert first == second
    assert len(set(first)) == 68
    assert all(isinstance(task, ExperimentTask) for task in first)


def test_task_matrix_maps_each_setting_to_dataset_specific_horizon():
    tasks = list(iter_experiment_tasks())
    expected_task_shapes = {
        ("seq48_pred1", "skippd_luoyang"): (48, 1),
        ("seq48_pred1", "pvod_station00_ylj"): (48, 1),
        ("seq96_h4", "skippd_luoyang"): (96, 48),
        ("seq96_h4", "pvod_station00_ylj"): (96, 16),
    }

    assert {
        (task.setting, task.dataset, task.seq_len, task.pred_len)
        for task in tasks
    } == {
        (setting, dataset, seq_len, pred_len)
        for (setting, dataset), (seq_len, pred_len) in expected_task_shapes.items()
    }
    assert [(task.model, task.mode) for task in tasks[:17]] == [
        ("Sundial", "zero_shot"),
        ("Sundial", "adapter"),
        ("Sundial", "full"),
        ("Sundial", "last_layer"),
        ("TimeMoE", "zero_shot"),
        ("TimeMoE", "adapter"),
        ("TimeMoE", "full"),
        ("TimeMoE", "last_layer"),
        ("Chronos2", "zero_shot"),
        ("Chronos2", "adapter"),
        ("Chronos2", "full"),
        ("Chronos2", "last_layer"),
        ("TiRex", "zero_shot"),
        ("TimesFM", "zero_shot"),
        ("TimesFM", "adapter"),
        ("TimesFM", "full"),
        ("TimesFM", "last_layer"),
    ]
    assert {(task.model, task.mode) for task in tasks} == {
        ("Sundial", "zero_shot"),
        ("Sundial", "adapter"),
        ("Sundial", "full"),
        ("Sundial", "last_layer"),
        ("TimeMoE", "zero_shot"),
        ("TimeMoE", "adapter"),
        ("TimeMoE", "full"),
        ("TimeMoE", "last_layer"),
        ("Chronos2", "zero_shot"),
        ("Chronos2", "adapter"),
        ("Chronos2", "full"),
        ("Chronos2", "last_layer"),
        ("TiRex", "zero_shot"),
        ("TimesFM", "zero_shot"),
        ("TimesFM", "adapter"),
        ("TimesFM", "full"),
        ("TimesFM", "last_layer"),
    }

    for task in tasks:
        assert (task.seq_len, task.pred_len) == expected_task_shapes[
            (task.setting, task.dataset)
        ]


def test_experiment_task_is_immutable_and_has_declared_field_order():
    task = ExperimentTask(
        "seq48_pred1", "skippd_luoyang", "Sundial", "zero_shot", 48, 1
    )

    assert task == ExperimentTask(
        "seq48_pred1", "skippd_luoyang", "Sundial", "zero_shot", 48, 1
    )
    with pytest.raises(FrozenInstanceError):
        task.pred_len = 2
