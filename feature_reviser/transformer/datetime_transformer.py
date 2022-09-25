# -*- coding: utf-8 -*-
from datetime import datetime
from typing import List, Tuple

import pandas as pd
from feature_engine.dataframe_checks import check_X

from feature_reviser.transformer.base_transformer import BaseTransformer

# pylint: disable= missing-function-docstring, unused-argument


class DurationCalculatorTransformer(BaseTransformer):
    """
    Calculates the duration between to given dates.

    Args:
        features (Tuple[str, str]): The two columns that contain the dates which should be used to calculate the duration.
        unit (str): The unit in which the duration should be returned. Should be either `days` or `seconds`.
        new_column_name (str): The name of the output column.
    """

    def __init__(
        self, features: Tuple[str, str], unit: str, new_column_name: str
    ) -> None:

        if unit not in ["days", "seconds"]:
            raise ValueError("Unsupported unit. Should be either `days` or `seconds`!")

        self.features = features
        self.unit = unit
        self.new_column_name = new_column_name

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method that calculates the duration between two dates.

        Args:
            X (pandas.DataFrame): The input DataFrame.

        Returns:
            pandas.DataFrame: The transformed DataFrame.
        """

        if not all(f in X.columns for f in self.features):
            raise ValueError("Not all provided `features` could be found in `X`!")

        X = check_X(X)

        if X.shape[1] != 2:
            raise ValueError(
                f"Only two columns should be provided! But {X.shape[1]} were given."
            )

        duration_series = pd.to_datetime(
            X[self.features[1]], utc=True, errors="raise"
        ) - pd.to_datetime(X[self.features[0]], utc=True, errors="raise")

        X[self.new_column_name] = (
            duration_series.dt.days
            if self.unit == "days"
            else duration_series.dt.total_seconds()
        )
        return X


class TimestampTransformer(BaseTransformer):
    """
    Transforms a date column with a specified format into a timestamp column.

    Args:
        features (List[str]): List of features which should be transformed.
        format (str): Format of the date column. Defaults to "%Y-%m-%d".
    """

    def __init__(
        self,
        features: List[str],
        date_format: str = "%Y-%m-%d",
    ) -> None:
        self.features = features
        self.date_format = date_format

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms columns from the provided dataframe.

        Args:
            X (pandas.DataFrame): Dataframe with columns to transform.

        Returns:
            pandas.DataFrame: Dataframe with transformed columns.
        """

        if not all(f in X.columns for f in self.features):
            raise ValueError("Not all provided `features` could be found in `X`!")

        X = check_X(X)
        for column in self.features:
            X[column] = pd.to_datetime(
                X[column], format=self.date_format, errors="raise"
            )
            X[column] = (X[column] - datetime(1970, 1, 1)).dt.total_seconds()
        return X
