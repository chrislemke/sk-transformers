# -*- coding: utf-8 -*-

import numpy as np
from sklearn.pipeline import make_pipeline

from feature_reviser.transformer.generic_transformer import (
    ColumnDropperTransformer,
    NaNTransformer,
    ValueReplacerTransformer,
)

# pylint: disable=missing-function-docstring, missing-class-docstring


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
