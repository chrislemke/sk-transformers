# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

# pylint: disable=missing-function-docstring


@pytest.fixture()
def clf():
    return DecisionTreeClassifier()


@pytest.fixture()
def X():
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
            "c": [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
            "d": [5, 5, 5, 6, 6, 6, 6, 7, 7, 7],
            "e": [5, 5, 5, 5, 5, 5, 5, 5, 7, 7],
        }
    )


@pytest.fixture()
def y():
    return pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
