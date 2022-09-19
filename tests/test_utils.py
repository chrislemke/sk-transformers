# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from feature_reviser.utils import check_data

# pylint: disable=missing-function-docstring


def test_check_data_x_type() -> None:
    with pytest.raises(TypeError):
        check_data("wrong_type", pd.Series([1, 2, 3]))


def test_check_data_y_type() -> None:
    with pytest.raises(TypeError):
        check_data(pd.DataFrame([1, 2, 3]), "wrong_type")


def test_check_data_x_nan() -> None:
    with pytest.raises(ValueError):
        check_data(pd.DataFrame([1, None, 3]), pd.Series([1, 2, 3]))


def test_check_data_y_nan() -> None:
    with pytest.raises(ValueError):
        check_data(pd.DataFrame([1, 2, 3]), pd.Series([1, None, 3]))
