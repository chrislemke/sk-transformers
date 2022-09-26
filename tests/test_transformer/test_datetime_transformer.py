# -*- coding: utf-8 -*-
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline

from feature_reviser import DurationCalculatorTransformer, TimestampTransformer

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_duration_calculator_transformer_in_pipeline_seconds(X_time_values) -> None:
    pipeline = make_pipeline(
        DurationCalculatorTransformer(
            ("b", "c"), new_column_name="duration", unit="seconds"
        )
    )
    X = pipeline.fit_transform(X_time_values)
    expected = np.array(
        [
            0.0,
            0.0,
            31536000.0,
            31536000.0,
            2678400.0,
            2678400.0,
            86400.0,
            86400.0,
            -1957305600.0,
            2460672000.0,
        ]
    )
    assert np.array_equal(X["duration"].values, expected)
    assert pipeline.steps[0][0] == "durationcalculatortransformer"
    assert pipeline.steps[0][1].unit == "seconds"


def test_duration_calculator_transformer_in_pipeline_days(X_time_values) -> None:
    pipeline = make_pipeline(
        DurationCalculatorTransformer(
            ("b", "c"), new_column_name="duration", unit="days"
        )
    )
    X = pipeline.fit_transform(X_time_values)
    expected = np.array([0, 0, 365, 365, 31, 31, 1, 1, -22654, 28480])
    assert np.array_equal(X["duration"].values, expected)
    assert pipeline.steps[0][0] == "durationcalculatortransformer"
    assert pipeline.steps[0][1].unit == "days"


def test_duration_calculator_transformer_exception() -> None:
    with pytest.raises(ValueError) as error:
        DurationCalculatorTransformer(
            ("b", "c"), new_column_name="duration", unit="not_supported"
        )
    assert "Unsupported unit. Should be either `days` or `seconds`!" == str(error.value)


def test_timestamp_transformer_in_pipeline(X_time_values) -> None:
    pipeline = make_pipeline(TimestampTransformer(["b"]))
    result = pipeline.fit_transform(X_time_values)["b"].values
    expected = np.array(
        [
            -3.1561920e08,
            0.0000000e00,
            8.6400000e04,
            1.6412544e09,
            1.6413408e09,
            1.6414272e09,
            1.6415136e09,
            1.6416000e09,
            1.6416864e09,
            1.6417728e09,
        ]
    )
    assert np.array_equal(result, expected)
    assert pipeline.steps[0][0] == "timestamptransformer"
