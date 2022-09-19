# -*- coding: utf-8 -*-
from typing import Union

import numpy as np
import pandas as pd


def check_data(features: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> None:
    """
    Checks if the data has the correct types, shapes and does not contain any missing values.

    Args:
        features (pandas.DataFrame): The dataframe containing the features.
        y (Union[pandas.Series, numpy.ndarray]): The target variable.

    Raises:
        TypeError: If the features are not a `pandas.DataFrame` or the target variable is not a `pandas.Series` or `numpy.ndarray`.
        ValueError: If the features or target variable contain missing values.

    Returns:
        None
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas.DataFrame!")
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError("y must be a pandas.Series or numpy.ndarray!")
    if features.isnull().values.any():
        raise ValueError("features must not contain NaN values!")
    if y.isnull().values.any():
        raise ValueError("y must not contain NaN values!")
    if len(features.shape) != 2:
        raise ValueError("features must be 2-dimensional!")
    if len(y.shape) != 1:
        raise ValueError("y must be 1-dimensional!")
