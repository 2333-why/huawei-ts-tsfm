import datetime as dt

import numpy as np
import pandas as pd
import pytest

from data_provider.stanford_sunset_preprocessing import (
    apply_sunset_filters,
    interpolate_to_sunset_10_seconds,
    validate_sunset_processed_data,
)


def test_interpolates_each_second_then_takes_ten_second_samples():
    times = [dt.datetime(2019, 2, 1, 12, 0, 0), dt.datetime(2019, 2, 1, 12, 0, 20)]
    result = interpolate_to_sunset_10_seconds(times, [1.0, 3.0])
    assert list(result.index) == [pd.Timestamp(times[0]), pd.Timestamp(2019, 2, 1, 12, 0, 10)]
    np.testing.assert_allclose(result.to_numpy(), [1.0, 2.0])


def test_exactly_one_hour_gap_is_interpolated_and_not_missing():
    start = dt.datetime(2019, 2, 1, 12, 0, 0)
    end = start + dt.timedelta(hours=1)
    pv_10_seconds = interpolate_to_sunset_10_seconds([start, end], [1.0, 2.0])
    cleaned, audit = apply_sunset_filters(pv_10_seconds, [start, end])
    assert len(cleaned) == 360
    assert audit["raw_gap_gt_1h_count"] == 0


def test_manual_interval_and_zero_are_removed():
    times = pd.to_datetime(
        [
            "2017-11-05 12:00:00",
            "2019-02-01 12:00:00",
            "2019-02-01 12:00:10",
        ]
    )
    values = pd.Series([5.0, 0.0, 2.0], index=times)
    raw_times = [dt.datetime(2019, 2, 1, 12, 0, 0), dt.datetime(2019, 2, 1, 12, 0, 10)]
    cleaned, audit = apply_sunset_filters(values, raw_times)
    assert list(cleaned.to_numpy()) == [2.0]
    assert audit["manual_invalid_10s_count"] == 1
    assert audit["nonpositive_10s_count"] == 1


def test_processed_validation_requires_naive_positive_aligned_data():
    times = np.asarray([dt.datetime(2019, 2, 1, 12, 0, 0)], dtype=object)
    audit = validate_sunset_processed_data(times, np.asarray([1.0]), "test")
    assert audit["strictly_positive"] is True

    with pytest.raises(ValueError, match="PV > 0"):
        validate_sunset_processed_data(times, np.asarray([0.0]), "test")

    aware = np.asarray([dt.datetime(2019, 2, 1, 12, 0, tzinfo=dt.timezone.utc)], dtype=object)
    with pytest.raises(ValueError, match="naive"):
        validate_sunset_processed_data(aware, np.asarray([1.0]), "test")
