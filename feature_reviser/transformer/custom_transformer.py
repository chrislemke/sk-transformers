# -*- coding: utf-8 -*-

from typing import List, Union

import pandas as pd
from feature_engine.creation import MathFeatures
from feature_engine.encoding import MeanEncoder as Me
from sklearn.base import BaseEstimator, TransformerMixin


class MeanEncoder(BaseEstimator, TransformerMixin):
    """
    Scikit-learn API for the feature-engine MeanEncoder.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Me(ignore_format=False)

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
        return self.encoder.transform(X.copy()).fillna(-1)


class MathFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn API for the feature-engine [MathFeatures](https://feature-engine.readthedocs.io/en/latest/api_doc/creation/MathFeatures.html).
    `MathFeatures(()` applies functions across multiple features returning one or more additional features as a result.
    It uses `pandas.agg()` to create the features, setting `axis=1`.
    For supported aggregation functions, see [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html).
    More details in the Feature-engine [User Guide](https://feature-engine.readthedocs.io/en/latest/user_guide/creation/MathFeatures.html#math-features).

    Args:
        func (Union[str, List[str]]): String of a function (e.g. `sum`) or list of strings of functions to apply to the data.
        num_columns (list[str]): List of numerical columns to apply the functions to.
        drop_original (bool, optional): Whether to drop the original columns. Defaults to False.

    Raises:
        None
    """

    def __init__(
        self,
        func: Union[str, List[str]],
        num_columns: list[str],
        drop_original: bool = False,
    ) -> None:
        super().__init__()
        self.num_columns = num_columns
        self.func = func
        self.drop_original = drop_original
        self.transformer = MathFeatures(
            num_columns,
            func,
            drop_original=drop_original,
        )

    def fit(self, X: pd.DataFrame) -> "MathFeatureTransformer":
        """
        Fit the MathFeatureTransformer to the data.
        Args:
            X (pandas.DataFrame): DataFrame to fit the MathFeatureTransformer to.

        Returns:
            MathFeatureTransformer: Fitted MathFeatureTransformer.
        """
        if "object" in [X[c].dtype for c in self.num_columns]:
            raise ValueError(
                "MathFeaturesTransformer only works with numerical columns!"
            )

        self.transformer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted MathFeatureTransformer.
        Args:
            X (pandas.DataFrame): DataFrame to transform.
        Returns:
            pandas.DataFrame: Transformed data.
        """
        return self.transformer.transform(X.copy())
