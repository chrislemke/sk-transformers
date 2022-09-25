# -*- coding: utf-8 -*-

from typing import Union

import pandas as pd
from feature_engine.encoding import MeanEncoder as Me
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# pylint: disable= missing-function-docstring, unused-argument


class MeanEncoder(BaseEstimator, TransformerMixin):
    """
    Scikit-learn API for the [feature-engine MeanEncoder](https://feature-engine.readthedocs.io/en/latest/api_doc/encoding/MeanEncoder.html).

    Args:
        fill_na_value (Union[int, float]): Value to fill NaN values with.
            Those may appear if a category is not present in the set the encoder was not fitted on.
    """

    def __init__(self, fill_na_value: Union[int, float] = -999) -> None:
        self.encoder = Me(ignore_format=False)
        self.fill_na_value = fill_na_value

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanEncoder":
        """
        Fit the MeanEncoder to the data.
        Args:
            X (pandas.DataFrame): DataFrame to fit the MeanEncoder to.
            y (pandas.Series): Target variable.

        Returns:
            MeanEncoder: Fitted MeanEncoder.
        """
        self.encoder.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted MeanEncoder.
        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Transformed data.
        """
        check_is_fitted(self)
        return self.encoder.transform(X).fillna(self.fill_na_value)
