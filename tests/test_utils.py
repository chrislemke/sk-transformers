import numpy as np
import pandas as pd
import pytest

from sk_transformers.utils import (
    check_data,
    check_ready_to_transform,
    prepare_categorical_data,
)

# pylint: disable=missing-function-docstring


def test_check_ready_to_transform_for_empty_df() -> None:
    with pytest.raises(ValueError) as error:
        check_ready_to_transform(pd.DataFrame())

    assert "X must not be empty!" == str(error.value)


def test_check_ready_to_transform_for_not_dataframe() -> None:
    with pytest.raises(ValueError) as error:
        check_ready_to_transform(np.ndarray([1, 2, 3]))

    assert "X must be a Pandas dataframe!" == str(error.value)


def test_check_ready_to_transform_for_wrong_column() -> None:
    with pytest.raises(ValueError) as error:
        check_ready_to_transform(pd.DataFrame({"a": [1, 2, 3]}), "b")

    assert "Column `b` not in dataframe!" == str(error.value)


def test_check_ready_to_transform_for_wrong_columns() -> None:
    with pytest.raises(ValueError) as error:
        check_ready_to_transform(pd.DataFrame({"a": [1, 2, 3]}), ["b", "c"])

    assert (
        "Not all provided `features` could be found in `X`! Following columns were not found in the dataframe: `b`, `c`."
        == str(error.value)
    )


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


def test_prepare_categorical_data_x_value(X_categorical) -> None:
    with pytest.raises(ValueError) as error:
        prepare_categorical_data(X_categorical, [("f", 1)])

    assert "cat_features must be in the dataframe!" == str(error.value)


def test_prepare_categorical_data(X_categorical) -> None:
    categories = [("a", 2), ("b", 3), ("c", 3), ("d", 3), ("e", 3)]
    result = prepare_categorical_data(X_categorical, categories).dtypes
    expected = pd.Series(
        [
            "category",
            "int64",
            "int64",
            "float64",
            "object",
        ],
        index=X_categorical.columns,
    )
    assert result.equals(expected)
