# -*- coding: utf-8 -*-

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline

from feature_reviser.transformer.generic_transformer import (
    ColumnDropperTransformer,
    NaNTransformer,
    QueryTransformer,
    ValueIndicatorTransformer,
    ValueReplacerTransformer,
)

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_query_transformer_in_pipeline(X) -> None:
    pipeline = make_pipeline(QueryTransformer(["a > 6"]))
    X = pipeline.fit_transform(X)

    assert X.shape[0] == 4
    assert X.shape[1] == 5
    assert pipeline.steps[0][0] == "querytransformer"


def test_value_indicator_transformer_in_pipeline(X_nan_values) -> None:
    pipeline = make_pipeline(
        ValueIndicatorTransformer([("d", -999), ("e", "-999")], as_int=True)
    )
    X = pipeline.fit_transform(X_nan_values)

    assert X.loc[5, "d_found_indicator"] == 1
    assert X.loc[6, "d_found_indicator"] == 0
    assert X.loc[6, "e_found_indicator"] == 1
    assert X.loc[7, "e_found_indicator"] == 0

    assert pipeline.steps[0][0] == "valueindicatortransformer"


def test_value_indicator_transformer_in_pipeline_with_non_existing_column(
    X_nan_values,
) -> None:
    with pytest.raises(ValueError) as error:
        pipeline = make_pipeline(
            ValueIndicatorTransformer(
                [("d", -999), ("not_existing", "-999")], as_int=True
            )
        )
        _ = pipeline.fit_transform(X_nan_values)

    assert "Not all provided `features` could be found in `X`!" == str(error.value)


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


def test_nan_transform_in_pipeline(X_nan_values) -> None:
    pipeline = make_pipeline(NaNTransformer({"a": -1, "b": -1, "c": "missing"}))
    X = pipeline.fit_transform(X_nan_values)

    assert X.isnull().sum().sum() == 0
    assert X["a"][1] == -1
    assert X["b"][2] == -1
    assert X["c"][6] == "missing"
    assert pipeline.steps[0][0] == "nantransformer"
    assert pipeline.steps[0][1].values["a"] == -1
