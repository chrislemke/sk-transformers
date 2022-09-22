# -*- coding: utf-8 -*-

import functools
import ipaddress
import itertools
import re
from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from feature_engine.encoding import MeanEncoder as Me
from sklearn.base import TransformerMixin

# pylint: disable=unused-argument


class ColumnDropperTransformer(TransformerMixin):
    """
    Drops columns from a dataframe

    Args:
        columns (Union[str, List[str]]): Columns to drop.
        Either a single column name or a list of column names.

    Returns:
        None
    """

    def __init__(self, columns: Union[str, List[str]]) -> None:
        self.columns = columns

    def fit(self, X=None, y=None) -> "columnDropperTransformer":  # type: ignore
        """
        Fit method that does nothing.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops columns from a dataframe.

        Args:
            X (pd.DataFrame): Dataframe to drop columns from.

        Returns:
            pd.DataFrame: Dataframe with columns dropped.
        """
        return X.drop(self.columns, axis=1)


class DurationCalculatorTransformer(TransformerMixin):
    """
    Calculates the duration between to given dates.

    Args:
        X (pd.DataFrame): The input DataFrame.
        columns (Tuple[str, str]): The two columns that contain the dates which should be used to calculate the duration.
        unit (str): The unit in which the duration should be returned. Should be either `days` or `seconds`.
        new_column_name (str): The name of the output column.
    """

    def __init__(self, unit: str, new_column_name: str):

        if unit not in ["days", "seconds"]:
            raise ValueError("Unsupported unit. Should be either `days` or `seconds`!")

        self.unit = unit
        self.new_column_name = new_column_name

    def fit(self, X=None, y=None) -> "DurationCalculatorTransformer":  # type: ignore
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

        if X.shape[1] != 2:
            raise ValueError(
                f"Only two columns should be provided! But {X.shape[1]} were given."
            )

        if self.unit == "days":
            X[self.new_column_name] = (
                pd.to_datetime(X.iloc[:, 1], utc=True, errors="raise")
                - pd.to_datetime(X.iloc[:, 0], utc=True, errors="raise")
            ).dt.days
        else:
            (
                pd.to_datetime(X.iloc[:, 1], utc=True, errors="raise")
                - pd.to_datetime(X.iloc[:, 0], utc=True, errors="raise")
            ).dt.total_seconds()
        return X


class NaNTransformer(TransformerMixin):
    """
    Replace NaN values with a specified value.

    Args:
        X (pandas.DataFrame): Dataframe to transform.
        values (Dict[str, Any]): Dictionary with column names as keys and values to replace NaN with as values.
    """

    def __init__(self, values: Dict[str, Any]):
        self.values = values

    def fit(self, X=None, y=None) -> "NaNTransformer":  # type: ignore
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


class TimestampTransformer(TransformerMixin):
    """
    Transforms a date column with a specified format into a timestamp column.

    Args:
        format (str): Format of the date column. Defaults to "%Y-%m-%d".
        errors (str): How to handle errors. Choices are "raise", "coerce", and "ignore". Defaults to "raise".
    """

    def __init__(
        self,
        date_format: str = "%Y-%m-%d",
        errors: str = "raise",
    ):
        self.date_format = date_format
        self.errors = errors

    def fit(self, X=None, y=None) -> "TimestampTransformer":  # type: ignore
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
        for col in X.columns:
            X[col] = pd.to_datetime(X[col], format=self.date_format, errors=self.errors)
            X[col] = (X[col] - datetime(1970, 1, 1)).dt.total_seconds()
        return X


class QueryTransformer(TransformerMixin):
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

    def fit(self, X=None, y=None) -> "QueryTransformer":  # type: ignore
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


class MeanEncoder(TransformerMixin):
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


class IPAddressEncoderTransformer(TransformerMixin):
    """
    Encodes IPv4 and IPv6 strings addresses to a float representation.
    To shrink the values to a reasonable size IPv4 addresses are divided by 2^10 and IPv6 addresses are divided by 2^48.
    Those values can be changed using the `ipv4_divider` and `ipv6_divider` parameters.
    """

    def __init__(self, ip4_divisor: float = 1e10, ip6_divisor: float = 1e48) -> None:
        super().__init__()
        self.ip4_divisor = ip4_divisor
        self.ip6_divisor = ip6_divisor

    def fit(self, X=None, y=None) -> "IPAddressEncoderTransformer":  # type: ignore
        """
        No need to fit anything.
        """
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transforms the column containing the IP addresses to float column.
        `-1` indicates that the value could not be parsed.

        Args:
            X (Union[pandas.DataFrame, numpy.ndarray]): DataFrame or array to transform.

        Returns:
            Union[pandas.DataFrame, numpy.ndarray]: Transformed dataframe or array.
        """

        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("X must be a numpy.ndarray or pandas.DataFrame!")

        function = functools.partial(
            IPAddressEncoderTransformer.__to_float, self.ip4_divisor, self.ip6_divisor
        )
        if isinstance(X, pd.DataFrame):
            X = X.applymap(function)
        else:
            X = np.vectorize(function)(X)
        return X

    @staticmethod
    def __to_float(ip4_devisor: float, ip6_devisor: float, ip_address: str) -> float:
        try:
            return int(ipaddress.IPv4Address(ip_address)) / int(ip4_devisor)
        except:  # pylint: disable=W0702
            try:
                return int(ipaddress.IPv6Address(ip_address)) / int(ip6_devisor)
            except:  # pylint: disable=W0702
                return -1


class EmailTransformer(TransformerMixin):
    """
    Transforms an email address into multiple features.
    """

    def fit(self, X=None, y=None) -> "EmailTransformer":  # type: ignore
        """
        No need to fit anything.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the one column from X, containing the email addresses, into multiple columns.

        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Transformed dataframe containing the extra columns.
        """

        X = X.copy()

        if X.shape[1] != 1:
            raise ValueError(
                "Only one column is allowed! Please try something like: `df[['email']]`."
            )
        column_name = X.iloc[:, 0].name

        X[f"{column_name}_domain"] = (
            X.iloc[:, 0].str.split("@").str[1].str.split(".").str[0]
        )

        X.iloc[:, 0] = X.iloc[:, 0].str.split("@").str[0]

        X[f"{column_name}_num_of_digits"] = X.iloc[:, 0].map(
            EmailTransformer.__num_of_digits
        )
        X[f"{column_name}_num_of_letters"] = X.iloc[:, 0].map(
            EmailTransformer.__num_of_letters
        )
        X[f"{column_name}_num_of_special_chars"] = X.iloc[:, 0].map(
            EmailTransformer.__num_of_special_characters
        )
        X[f"{column_name}_num_of_repeated_chars"] = X.iloc[:, 0].map(
            EmailTransformer.__num_of_repeated_characters
        )
        X[f"{column_name}_num_of_words"] = X.iloc[:, 0].map(
            EmailTransformer.__num_of_words
        )
        return X

    @staticmethod
    def __num_of_digits(string: str) -> int:
        return sum(map(str.isdigit, string))

    @staticmethod
    def __num_of_letters(string: str) -> int:
        return sum(map(str.isalpha, string))

    @staticmethod
    def __num_of_special_characters(string: str) -> int:
        return len(re.findall(r"[^A-Za-z0-9]", string))

    @staticmethod
    def __num_of_repeated_characters(string: str) -> int:
        return max(len("".join(g)) for _, g in itertools.groupby(string))

    @staticmethod
    def __num_of_words(string: str) -> int:
        return len(re.findall(r"[.\-_]", string)) + 1
