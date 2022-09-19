# -*- coding: utf-8 -*-

import pandas as pd


def check_data(X: pd.DataFrame, y: pd.Series) -> None:
    """
    Checks if the data has the correct types, shapes and does not contain any missing values.

    Args:
        features (pandas.DataFrame): The dataframe containing the features.
        y (pandas.Series): The target variable.

    Raises:
        TypeError: If the features are not a `pandas.DataFrame` or the target variable is not a `pandas.Series` or `numpy.ndarray`.
        ValueError: If the features or target variable contain missing values.

    Returns:
        None
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("features must be a pandas.DataFrame!")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas.Series!")
    if X.isnull().values.any():
        raise ValueError("features must not contain NaN values!")
    if y.isnull().values.any():
        raise ValueError("y must not contain NaN values!")
