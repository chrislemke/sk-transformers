import numpy as np
import polars as pl
import pytest
from sklearn.pipeline import make_pipeline

from sk_transformers import (
    DateColumnsTransformer,
    DurationCalculatorTransformer,
    TimestampTransformer,
)

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_date_columns_transformer_in_pipeline(tiny_date_df):
    pipeline = make_pipeline(DateColumnsTransformer(["a"]))
    X = pipeline.fit_transform(tiny_date_df).drop("a", axis=1)
    expected = np.array(
        [
            [
                2021,
                1,
                1,
                4,
                1,
                53,
                1,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                False,
            ],
            [
                2022,
                2,
                2,
                2,
                33,
                5,
                1,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            [
                2023,
                3,
                3,
                4,
                62,
                9,
                1,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        ],
        dtype=object,
    )

    assert np.array_equal(X.to_numpy(), expected)
    assert pipeline.steps[0][0] == "datecolumnstransformer"
    assert pipeline.steps[0][1].features == ["a"]
    assert pipeline.steps[0][1].date_format == "%Y-%m-%d"
    assert pipeline.steps[0][1].date_elements == [
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "quarter",
        "is_leap_year",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_weekend",
    ]


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
    assert np.array_equal(X["duration"].to_numpy(), expected)
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
    assert np.array_equal(X["duration"].to_numpy(), expected)
    assert pipeline.steps[0][0] == "durationcalculatortransformer"
    assert pipeline.steps[0][1].unit == "days"


def test_duration_calculator_transformer_exception() -> None:
    with pytest.raises(ValueError) as error:
        DurationCalculatorTransformer(
            ("b", "c"), new_column_name="duration", unit="not_supported"
        )
    assert "Unsupported unit. Should be either `days` or `seconds`!" == str(error.value)


def test_duration_calculator_transformer_exception_no_column(X) -> None:
    with pytest.raises(ValueError) as error:
        DurationCalculatorTransformer(
            ("non_existing", "c"), new_column_name="duration", unit="days"
        ).fit_transform(X)

    assert """
                DurationCalculatorTransformer:
                Not all provided `features` could be found in `X`! Following columns were not found in the dataframe: `non_existing`.
                """ == str(
        error.value
    )


def test_timestamp_transformer_in_pipeline(X_time_values) -> None:
    pipeline = make_pipeline(TimestampTransformer(["b"]))
    result = pipeline.fit_transform(X_time_values)["b"].to_numpy()
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
