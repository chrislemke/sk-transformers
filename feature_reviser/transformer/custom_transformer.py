# -*- coding: utf-8 -*-

from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
from feature_engine.encoding import MeanEncoder as Me
from sklearn.base import BaseEstimator, TransformerMixin


class DurationCalculatorTransformer(BaseEstimator, TransformerMixin):
    """
    Calculates the duration between to given dates.

    Args:
        X (pd.DataFrame): The input DataFrame.
        columns (Tuple[str, str]): The two columns that contain the dates which should be used to calculate the duration.
        unit (str): The unit in which the duration should be returned. Should be either `days` or `seconds`.
        new_column_name (str): The name of the output column.
    """

    def __init__(self, columns: Tuple[str, str], unit: str, new_column_name: str):

        if len(columns) != 2:
            raise ValueError("columns must be a tuple of two strings!")

        if unit not in ["days", "seconds"]:
            raise ValueError("Unsupported unit. Should be either `days` or `seconds`!")

        self.columns = columns
        self.unit = unit
        self.new_column_name = new_column_name

    def fit(self) -> "DurationCalculatorTransformer":
        """
        Fit method that does nothing.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method that calculates the duration between two dates.

        Args:
            X (pandas.DataFrame): The input DataFrame.

        Returns:
            pandas.DataFrame: The transformed DataFrame.
        """
        X = X.copy()

        if self.unit == "days":
            X[self.new_column_name] = (
                pd.to_datetime(X[self.columns[1]], utc=True, errors="raise")
                - pd.to_datetime(X[self.columns[0]], utc=True, errors="raise")
            ).dt.days
        else:
            (
                pd.to_datetime(X[self.columns[1]], utc=True, errors="raise")
                - pd.to_datetime(X[self.columns[0]], utc=True, errors="raise")
            ).dt.total_seconds()
        return X


class NaNTransformer(BaseEstimator, TransformerMixin):
    """
    Replace NaN values with a specified value.

    Args:
        X (pandas.DataFrame): Dataframe to transform.
        values (Dict[str, Any]): Dictionary with column names as keys and values to replace NaN with as values.
    """

    def __init__(self, values: Dict[str, Any]):
        self.values = values

    def fit(self) -> "NaNTransformer":
        """
        Fit method that does nothing.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace NaN values with a specified value.

        Args:
            X (pandas.DataFrame): Dataframe to transform.

        Returns:
            pandas.DataFrame: Transformed dataframe.
        """
        X = X.copy()
        return X.fillna(self.values)


class TimestampTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms a date column with a specified format into a timestamp column.

    Args:
        columns (List[str]): List of columns to transform.
        format (str): Format of the date column. Defaults to "%Y-%m-%d".
        errors (str): How to handle errors. Choices are "raise", "coerce", and "ignore". Defaults to "raise".
    """

    def __init__(
        self,
        columns: List[str],
        date_format: str = "%Y-%m-%d",
        errors: str = "raise",
    ):
        self.columns = columns
        self.date_format = date_format
        self.errors = errors

    def fit(self) -> "TimestampTransformer":
        """
        Fit method that does nothing.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms columns from the provided dataframe.

        Args:
            X (pandas.DataFrame): Dataframe with columns to transform.

        Returns:
            pandas.DataFrame: Dataframe with transformed columns.
        """
        X = X.copy()
        for col in self.columns:
            X[col] = pd.to_datetime(X[col], format=self.date_format, errors=self.errors)
            X[col] = (X[col] - datetime(1970, 1, 1)).dt.total_seconds()
        return X


class QueryTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a list of queries to a dataframe.
    Read more about queries [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html#pandas.DataFrame.query).

    Args:
        queries (List[str]): List of queries to apply to the dataframe.

    Returns:
        None

    """

    def __init__(self, queries: List[str]) -> None:
        super().__init__()
        self.queries = queries

    def fit(self) -> "QueryTransformer":
        """
        No need to fit anything.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the list of queries to the dataframe.

        Args:
            X (pd.DataFrame): Dataframe to apply the queries to.

        Returns:
            pd.DataFrame: Dataframe with the queries applied.
        """
        X = X.copy()
        for query in self.queries:
            X = X.query(query)
        return X


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
