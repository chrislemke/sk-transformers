# -*- coding: utf-8 -*-

import numpy as np
from sklearn.pipeline import make_pipeline

from feature_reviser.transformer.custom_transformer import (
    DurationCalculatorTransformer,
    IPAddressEncoderTransformer,
    NaNTransformer,
    TimestampTransformer,
)

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_ip_address_encoder_transformer_in_pipeline(X_numbers) -> None:
    pipeline = make_pipeline(IPAddressEncoderTransformer())

    X = pipeline.transform(X_numbers[["ip_address"]])
    expected = np.array(
        [
            0.3405803971,
            0.3232235777,
            4.254076642994478e-11,
            4.254076645264119e-11,
            -1.0,
        ]
    )

    assert pipeline.steps[0][0] == "ipaddressencodertransformer"
    assert np.array_equal(X["ip_address"].values, expected)
    print(X)


def test_duration_calculator_transformer_in_pipeline(X_time_values) -> None:
    pipeline = make_pipeline(
        DurationCalculatorTransformer(new_column_name="duration", unit="days")
    )
    X = pipeline.transform(X_time_values[["b", "c"]])
    expected = np.array([0, 0, 365, 365, 31, 31, 1, 1, -22654, 28480])
    assert np.array_equal(X["duration"].values, expected)
    assert pipeline.steps[0][0] == "durationcalculatortransformer"
    assert pipeline.steps[0][1].unit == "days"


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
    pipeline = make_pipeline(TimestampTransformer())
    result = pipeline.transform(X_time_values[["b"]])["b"].values
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
