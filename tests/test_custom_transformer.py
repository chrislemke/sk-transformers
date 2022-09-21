# -*- coding: utf-8 -*-

import numpy as np
from sklearn.pipeline import make_pipeline

from feature_reviser.transformer.custom_transformer import (
    DurationCalculatorTransformer,
    NaNTransformer,
    TimestampTransformer,
)

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_duration_calculator_transformer_in_pipeline(X_time_values) -> None:
    pipeline = make_pipeline(
        DurationCalculatorTransformer(
            columns=("b", "c"), new_column_name="duration", unit="days"
        )
    )
    X = pipeline.transform(X_time_values)
    expected = np.array([0, 0, 365, 365, 31, 31, 1, 1, -22654, 28480])
    assert np.array_equal(X["duration"].values, expected)
    assert pipeline.steps[0][0] == "durationcalculatortransformer"
    assert pipeline.steps[0][1].columns == ("b", "c")


def test_nan_transform_in_pipeline(X_nan_values) -> None:
    pipeline = make_pipeline(NaNTransformer({"a": -1, "b": -1, "c": "missing"}))
    X = pipeline.transform(X_nan_values)

    assert X.isnull().sum().sum() == 0
    assert X["a"][1] == -1
    assert X["b"][2] == -1
    assert X["c"][6] == "missing"
    assert pipeline.steps[0][0] == "nantransformer"
    assert pipeline.steps[0][1].values["a"] == -1


def test_timestamp_transformer_in_pipeline(X_time_values) -> None:
    pipeline = make_pipeline(TimestampTransformer(columns=["b"]))
    result = pipeline.transform(X_time_values)["b"].values
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
    assert pipeline.steps[0][1].columns == ["b"]
