# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from feature_reviser.utils import (
    check_data,
    prepare_categorical_data,
)

# pylint: disable=missing-function-docstring


def test_check_data_x_type() -> None:
    with pytest.raises(TypeError) as error:
        check_data("wrong_type", pd.Series([1, 2, 3]))

    assert "Features must be a pandas.DataFrame!" == str(error.value)


def test_check_data_y_type() -> None:
    with pytest.raises(TypeError) as error:
        check_data(pd.DataFrame([1, 2, 3]), "wrong_type")

    assert "y must be a pandas.Series!" == str(error.value)


def test_check_data_x_nan() -> None:
    with pytest.raises(ValueError) as error:
        check_data(pd.DataFrame([1, None, 3]), pd.Series([1, 2, 3]))

    assert "Features must not contain NaN values!" == str(error.value)


def test_check_data_y_nan() -> None:
    with pytest.raises(ValueError) as error:
        check_data(pd.DataFrame([1, 2, 3]), pd.Series([1, None, 3]))

    assert "y must not contain NaN values!" == str(error.value)


def test_prepare_categorical_data_x_type() -> None:
    with pytest.raises(TypeError) as error:
        prepare_categorical_data("wrong_type", [("a", 1), ("b", 2)])

    assert "features must be a pandas.DataFrame!" == str(error.value)


def test_prepare_categorical_data_x_value() -> None:
    X = pd.DataFrame(
        {
            "c": [1, 2, 3],
            "d": [4, 5, 6],
            "e": [7, 8, 9],
        }
    )

    with pytest.raises(ValueError) as error:
        prepare_categorical_data(X, [("a", 1), ("b", 2)])

    assert "cat_features must be in the dataframe!" == str(error.value)


def test_prepare_categorical_data() -> None:
    X = pd.DataFrame(
        {
            "a": ["A1", "A2", "A2", "A1", "A1", "A2", "A1", "A1"],
            "b": [1, 2, 3, 4, 5, 6, 7, 8],
            "c": [1, 2, 3, 1, 2, 3, 1, 3],
            "d": [1.1, 2, 3, 4, 5, 6, 7, 8],
            "e": ["A", "B", "C", "D", "E", "F", "G", "H"],
        }
    )

    categories = [("a", 2), ("b", 3), ("c", 3), ("d", 3), ("e", 3)]
    result = prepare_categorical_data(X, categories).dtypes
    expected = pd.Series(
        [
            "category",
            "int64",
            "int64",
            "float64",
            "object",
        ],
        index=X.columns
    )
    assert result.equals(expected)