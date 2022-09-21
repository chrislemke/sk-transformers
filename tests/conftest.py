# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# pylint: disable=missing-function-docstring


@pytest.fixture()
def clf() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=42)


@pytest.fixture()
def ordinal_encoder() -> OrdinalEncoder:
    return OrdinalEncoder()


@pytest.fixture()
def standard_scaler() -> StandardScaler:
    return StandardScaler()


@pytest.fixture()
def X() -> pd.DataFrame:
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
def X_time_values() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [
                "1960-01-01",
                "1970-01-01",
                "1970-01-02",
                "2022-01-04",
                "2022-01-05",
                "2022-01-06",
                "2022-01-07",
                "2022-01-08",
                "2022-01-09",
                "2022-01-10",
            ],
            "c": [
                "1960-01-01",
                "1970-01-01",
                "1971-01-02",
                "2023-01-04",
                "2022-02-05",
                "2022-02-06",
                "2022-01-08",
                "2022-01-09",
                "1960-01-01",
                "2100-01-01",
            ],
        }
    )


@pytest.fixture()
def X_nan_values() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, None, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [1.1, 2.2, None, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
            "c": ["A", "B", "C", "D", "E", "F", None, "H", "I", "J"],
        }
    )


@pytest.fixture()
def y() -> pd.Series:
    return pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
