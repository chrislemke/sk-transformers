# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

from feature_reviser.transformer.custom_transformer import (
    ColumnDropperTransformer,
    DurationCalculatorTransformer,
    EmailTransformer,
    IPAddressEncoderTransformer,
    NaNTransformer,
    TimestampTransformer,
)

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_column_dropper_transformer_in_pipeline(X) -> None:
    pipeline = make_pipeline(ColumnDropperTransformer(columns=["a", "b", "c", "d"]))

    result = pipeline.transform(X)
    expected = X.drop(columns=["a", "b", "c", "d"])
    assert result.equals(expected)
    assert pipeline.steps[0][0] == "columndroppertransformer"


def test_email_transformer_in_pipeline(X_strings) -> None:
    pipeline = make_pipeline(EmailTransformer())
    result = pipeline.transform(X_strings[["email"]])
    expected = pd.DataFrame(
        {
            "email": {
                0: "test",
                1: "test123",
                2: "test_123$$",
                3: "test_test",
                4: "ttt",
                5: "test_test_test",
            },
            "email_domain": {
                0: "test1",
                1: "test2",
                2: "test3",
                3: "test4",
                4: "test5",
                5: None,
            },
            "email_num_of_digits": {0: 0, 1: 3, 2: 3, 3: 0, 4: 0, 5: 0},
            "email_num_of_letters": {0: 4, 1: 4, 2: 4, 3: 8, 4: 3, 5: 12},
            "email_num_of_special_chars": {0: 0, 1: 0, 2: 3, 3: 1, 4: 0, 5: 2},
            "email_num_of_repeated_chars": {0: 1, 1: 1, 2: 2, 3: 1, 4: 3, 5: 1},
            "email_num_of_words": {0: 1, 1: 1, 2: 2, 3: 2, 4: 1, 5: 3},
        }
    )
    assert result.equals(expected)
    assert pipeline.steps[0][0] == "emailtransformer"


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
