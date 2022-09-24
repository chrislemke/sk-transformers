# -*- coding: utf-8 -*-

import re
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from feature_engine.dataframe_checks import check_X
from sklearn.base import BaseEstimator, TransformerMixin

# pylint: disable=unused-argument, missing-function-docstring


class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    """
    Drops columns from a dataframe using Pandas `drop` method.

    Args:
        columns (Union[str, List[str]]): Columns to drop. Either a single column name or a list of column names.
    """

    def __init__(self, columns: Union[str, List[str]]) -> None:
        self.columns = columns

    def fit(self, X=None, y=None) -> "ColumnDropperTransformer":  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the dataframe with the `columns` dropped.

        Args:
            X (pd.DataFrame): Dataframe to drop columns from.

        Returns:
            pd.DataFrame: Dataframe with columns dropped.
        """
        X = check_X(X)
        return X.drop(self.columns, axis=1)


class NaNTransformer(BaseEstimator, TransformerMixin):
    """
    Replace NaN values with a specified value. Internally Pandas `fillna` method is used.

    Args:
        X (pandas.DataFrame): Dataframe to transform.
        values (Dict[str, Any]): Dictionary with column names as keys and values to replace NaN with as values.
    """

    def __init__(self, values: Dict[str, Any]) -> None:
        self.values = values

    def fit(self, X=None, y=None) -> "NaNTransformer":  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace NaN values with a specified value.

        Args:
            X (pandas.DataFrame): Dataframe to transform.

        Returns:
            pandas.DataFrame: Transformed dataframe.
        """
        X = check_X(X)
        return X.fillna(self.values)


class QueryTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a list of queries to a dataframe.
    If it operates on a dataset used for supervised learning this transformer should
    be applied on the dataframe containing `X` and `y`. So removing of columns by queries
    also removes the corresponding `y` value.
    Read more about queries [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html).

    Args:
        queries (List[str]): List of query string to evaluate to the dataframe.
    """

    def __init__(self, queries: List[str]) -> None:
        self.queries = queries

    def fit(self, X=None, y=None) -> "QueryTransformer":  # type: ignore
        return self

    def transform(self, Xy: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the list of queries to the dataframe.

        Args:
            Xy (pd.DataFrame): Dataframe to apply the queries to. For also operating on the target column `y` - if needed.
                This column should also be part of the dataframe.

        Returns:
            pd.DataFrame: Dataframe with the queries applied.
        """

        Xy = check_X(Xy)
        for query in self.queries:
            Xy = Xy.query(query, inplace=False)
        return Xy


class ValueReplacerTransformer(BaseEstimator, TransformerMixin):
    """
    Uses Pandas `replace` method to replace values in a column. This transformer loops over the `features` and applies
    `replace` to the according columns. If the column is not from type string but a valid regular expression is provided
    the column will be temporarily changed to a string column and after the manipulation by `replace` changed back to its
    original type. It may happen, that this type changing fails if the modified column is not compatible with its original type.

    Args:
        features (List[Tuple[List[str], str, Any]]): List of tuples containing the column names as a list,
            the value to replace (can be a regex), and the replacement value.
    """

    def __init__(self, features: List[Tuple[List[str], str, Any]]) -> None:
        self.features = features

    def fit(self, X=None, y=None) -> "ValueReplacerTransformer":  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            X (pd.DataFrame): Dataframe containing the columns where values should be replaced.

        Returns:
            pd.DataFrame: Dataframe with replaced values.
        """

        X = check_X(X)

        for (columns, value, replacement) in self.features:
            for column in columns:
                print(column)
                is_regex = ValueReplacerTransformer.__check_for_regex(value)
                column_dtype = X[column].dtype

                if column_dtype is not str and is_regex:
                    X[column] = X[column].astype(str)

                X[column] = X[column].replace(value, replacement, regex=True)

                if X[column].dtype != column_dtype:
                    X[column] = X[column].astype(column_dtype)

        return X

    @staticmethod
    def __check_for_regex(string: str) -> bool:
        if not isinstance(string, str):
            return False
        try:
            re.compile(string)
            is_valid = True
        except re.error:  # pylint: disable=W0702
            is_valid = False
        return is_valid
