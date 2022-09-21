# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import pandas as pd


def check_data(X: pd.DataFrame, y: pd.Series, check_nans: bool = True) -> None:
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
        raise TypeError("Features must be a pandas.DataFrame!")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas.Series!")
    if check_nans:
        if X.isnull().values.any():
            raise ValueError("Features must not contain NaN values!")
        if y.isnull().values.any():
            raise ValueError("y must not contain NaN values!")


def prepare_categorical_data(
    X: pd.DataFrame, categories: List[Tuple[str, int]]
) -> pd.DataFrame:
    """
    Checks for the validity of the categorical features inside the dataframe.
    And prepares the data for further processing by changing the `dtypes`.

    Args:
        X (pandas.DataFrame): The dataframe containing the categorical features.
        categories (List[Tuple[str, int]]): The list of categorical features and their thresholds.
            If the number of unique values is greater than the threshold, the feature is considered numerical and not categorical.

    Raises:
        TypeError: If the features are not a `pandas.DataFrame` or the categorical features are not a `List[str]`.
        ValueError: If the categorical features are not in the dataframe.
        ValueError: If the dataframe does not contain any categorical features.

    Returns:
        pandas.DataFrame: The original dataframe with the categorical features converted to `category` dtype.
    """
    cat_features = [f[0] for f in categories]

    if not isinstance(X, pd.DataFrame):
        raise TypeError("features must be a pandas.DataFrame!")
    if not set(set(cat_features)).issubset(set(X.columns)):
        raise ValueError("cat_features must be in the dataframe!")

    cont_features = []
    for feature, threshold in categories:
        if X[feature].nunique() > threshold:
            cont_features.append(feature)
            cat_features.remove(feature)
            print(
                f"""{feature} has less unique vlaues that {threshold}.
            So it is not a categorical feature and will be handled as a numerical feature."""
            )

    pd.options.mode.chained_assignment = None
    for column in X.columns:
        if column in cat_features:
            X[column] = X[column].astype("category").copy()
        else:
            X[column] = X[column].astype(np.float32).copy()

    return X
