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
    PhoneTransformer,
    StringSimilarityTransformer,
    TimestampTransformer,
    ValueReplacerTransformer,
)

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_phone_number_transformer(X_numbers):
    pipeline = make_pipeline(PhoneTransformer(["phone_number"]))

    X = pipeline.fit_transform(X_numbers)
    expected_national_number = np.array(
        [1.763456123, -999.0, 4.045654449, -999.0, -999.0]
    )
    expected_country_code = np.array([0.49, -999.0, 0.49, -999.0, -999.0])

    assert pipeline.steps[0][0] == "phonetransformer"
    assert np.array_equal(
        X["phone_number_national_number"].values, expected_national_number
    )
    assert np.array_equal(X["phone_number_country_code"].values, expected_country_code)


def test_string_similarity_transformer_in_pipeline(X_strings):
    pipeline = make_pipeline(StringSimilarityTransformer(["strings_1", "strings_2"]))
    result = pipeline.fit_transform(X_strings)
    expected = np.array(
        [
            0.8888888888888888,
            1.0,
            0.8181818181818182,
            0.8888888888888888,
            0.0,
            0.058823529411764705,
        ]
    )
    assert np.array_equal(result["strings_1_strings_2_similarity"].values, expected)
    assert pipeline.steps[0][0] == "stringsimilaritytransformer"


def test_value_replacer_transformer_in_pipeline(X_time_values) -> None:
    values = [
        (["a", "e"], r"^(?:[1-9][0-9]+|9)$", 99),
        (
            ["dd"],
            r"^(?!(19|20)\d\d[-\/.](0[1-9]|1[012]|[1-9])[-\/.](0[1-9]|[12][0-9]|3[01]|[1-9])$).*",
            "1900-01-01",
        ),
    ]
    pipeline = make_pipeline(ValueReplacerTransformer(values))
    result = pipeline.fit_transform(X_time_values)
    expected_a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 99, 99])
    expected_dd = np.array(
        [
            "1900-01-01",
            "1970-01-01",
            "1900-01-01",
            "1900-01-01",
            "2022.02.05",
            "1900-01-01",
            "2022/01/08",
            "2022-01-09",
            "1960-01-01",
            "1900-01-01",
        ]
    )
    expected_e = np.array([2, 4, 6, 8, 99, 99, 99, 99, 99, 99])

    assert np.array_equal(result["a"].values, expected_a)
    assert np.array_equal(result["dd"].values, expected_dd)
    assert np.array_equal(result["e"].values, expected_e)
    assert pipeline.steps[0][0] == "valuereplacertransformer"


def test_column_dropper_transformer_in_pipeline(X) -> None:
    pipeline = make_pipeline(ColumnDropperTransformer(columns=["a", "b", "c", "d"]))

    result = pipeline.fit_transform(X)
    expected = X.drop(columns=["a", "b", "c", "d"])
    assert result.equals(expected)
    assert pipeline.steps[0][0] == "columndroppertransformer"


def test_email_transformer_in_pipeline(X_strings) -> None:
    pipeline = make_pipeline(EmailTransformer(["email"]))
    result = pipeline.fit_transform(X_strings[["email"]])
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
    pipeline = make_pipeline(IPAddressEncoderTransformer(["ip_address"]))

    X = pipeline.fit_transform(X_numbers[["ip_address"]])
    expected = np.array(
        [
            0.3405803971,
            0.3232235777,
            4.254076642994478e-11,
            4.254076645264119e-11,
            -999,
        ]
    )

    assert pipeline.steps[0][0] == "ipaddressencodertransformer"
    assert np.array_equal(X["ip_address"].values, expected)
    print(X)


def test_duration_calculator_transformer_in_pipeline_days(X_time_values) -> None:
    pipeline = make_pipeline(
        DurationCalculatorTransformer(
            ("b", "c"), new_column_name="duration", unit="days"
        )
    )
    X = pipeline.fit_transform(X_time_values[["b", "c"]])
    expected = np.array([0, 0, 365, 365, 31, 31, 1, 1, -22654, 28480])
    assert np.array_equal(X["duration"].values, expected)
    assert pipeline.steps[0][0] == "durationcalculatortransformer"
    assert pipeline.steps[0][1].unit == "days"


def test_duration_calculator_transformer_in_pipeline_seconds(X_time_values) -> None:
    pipeline = make_pipeline(
        DurationCalculatorTransformer(
            ("b", "c"), new_column_name="duration", unit="seconds"
        )
    )
    X = pipeline.fit_transform(X_time_values[["b", "c"]])
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


def test_nan_transform_in_pipeline(X_nan_values) -> None:
    pipeline = make_pipeline(NaNTransformer({"a": -1, "b": -1, "c": "missing"}))
    X = pipeline.fit_transform(X_nan_values)

    assert X.isnull().sum().sum() == 0
    assert X["a"][1] == -1
    assert X["b"][2] == -1
    assert X["c"][6] == "missing"
    assert pipeline.steps[0][0] == "nantransformer"
    assert pipeline.steps[0][1].values["a"] == -1


def test_timestamp_transformer_in_pipeline(X_time_values) -> None:
    pipeline = make_pipeline(TimestampTransformer(["b"]))
    result = pipeline.fit_transform(X_time_values[["b"]])["b"].values
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
