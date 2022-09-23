# -*- coding: utf-8 -*-

import functools
import ipaddress
import itertools
import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import phonenumbers
from feature_engine.dataframe_checks import check_X
from feature_engine.encoding import MeanEncoder as Me
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# pylint: disable=unused-argument, missing-function-docstring


class ValueReplacerTransformer(BaseEstimator, TransformerMixin):
    """
    Uses Pandas `replace` method to replace values in a column.

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
        Returns the dataframe with the columns dropped.

        Args:
            X (pd.DataFrame): Dataframe to drop columns from.

        Returns:
            pd.DataFrame: Dataframe with columns dropped.
        """
        X = check_X(X)
        return X.drop(self.columns, axis=1)


class DurationCalculatorTransformer(BaseEstimator, TransformerMixin):
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

    def fit(self, X=None, y=None) -> "DurationCalculatorTransformer":  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method that calculates the duration between two dates.

        Args:
            X (pandas.DataFrame): The input DataFrame.

        Returns:
            pandas.DataFrame: The transformed DataFrame.
        """

        if not all(elem in X.columns for elem in self.features):
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


class NaNTransformer(BaseEstimator, TransformerMixin):
    """
    Replace NaN values with a specified value.

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


class TimestampTransformer(BaseEstimator, TransformerMixin):
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

    def fit(self, X=None, y=None) -> "TimestampTransformer":  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms columns from the provided dataframe.

        Args:
            X (pandas.DataFrame): Dataframe with columns to transform.

        Returns:
            pandas.DataFrame: Dataframe with transformed columns.
        """

        if not all(elem in X.columns for elem in self.features):
            raise ValueError("Not all provided `features` could be found in `X`!")

        X = check_X(X)
        for column in self.features:
            X[column] = pd.to_datetime(
                X[column], format=self.date_format, errors="raise"
            )
            X[column] = (X[column] - datetime(1970, 1, 1)).dt.total_seconds()
        return X


class QueryTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a list of queries to a dataframe.
            If it operates on a dataset used for supervised learning this transformer should
            be applied on the dataframe containing `X` and `y`. So removing of columns by queries
            also removes the corresponding `y` value.
    Read more about queries [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html#pandas.DataFrame.query).

    Args:
        queries (List[str]): List of queries to apply to the dataframe.
    """

    def __init__(self, queries: List[str]) -> None:
        self.queries = queries

    def fit(self, X=None, y=None) -> "QueryTransformer":  # type: ignore
        return self

    def transform(self, Xy: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the list of queries to the dataframe.

        Args:
            Xy (pd.DataFrame): Dataframe to apply the queries to.

        Returns:
            pd.DataFrame: Dataframe with the queries applied.
        """

        Xy = check_X(Xy)
        for query in self.queries:
            Xy = Xy.query(query)
        return Xy


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


class IPAddressEncoderTransformer(BaseEstimator, TransformerMixin):
    """
    Encodes IPv4 and IPv6 strings addresses to a float representation.
    To shrink the values to a reasonable size IPv4 addresses are divided by 2^10 and IPv6 addresses are divided by 2^48.
    Those values can be changed using the `ipv4_divider` and `ipv6_divider` parameters.

    Args:
        features (List[str]): List of features which should be transformed.
        ipv4_divider (float): Divider for IPv4 addresses.
        ipv6_divider (float): Divider for IPv6 addresses.
        error_value (Union[int, float]): Value if parsing fails.
    """

    def __init__(
        self,
        features: List[str],
        ip4_divisor: float = 1e10,
        ip6_divisor: float = 1e48,
        error_value: Union[int, float] = -999,
    ) -> None:
        super().__init__()
        self.features = features
        self.ip4_divisor = ip4_divisor
        self.ip6_divisor = ip6_divisor
        self.error_value = error_value

    def fit(self, X=None, y=None) -> "IPAddressEncoderTransformer":  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the column containing the IP addresses to float column.

        Args:
            X (pandas.DataFrame): DataFrame to transform.
            error_value (Union[int, float]): Value if parsing fails.

        Returns:
            pandas.DataFrame: Transformed dataframe.
        """

        if not all(elem in X.columns for elem in self.features):
            raise ValueError("Not all provided `features` could be found in `X`!")

        X = check_X(X)

        function = functools.partial(
            IPAddressEncoderTransformer.__to_float,
            self.ip4_divisor,
            self.ip6_divisor,
            self.error_value,
        )
        for column in self.features:
            X[column] = X[column].map(function)

        return X

    @staticmethod
    def __to_float(
        ip4_devisor: float,
        ip6_devisor: float,
        error_value: Union[int, float],
        ip_address: str,
    ) -> float:
        try:
            return int(ipaddress.IPv4Address(ip_address)) / int(ip4_devisor)
        except:  # pylint: disable=W0702
            try:
                return int(ipaddress.IPv6Address(ip_address)) / int(ip6_devisor)
            except:  # pylint: disable=W0702
                return error_value


class EmailTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms an email address into multiple features.

    Args:
        features (List[str]): List of features which should be transformed.
    """

    def __init__(self, features: List[str]) -> None:
        self.features = features

    def fit(self, X=None, y=None) -> "EmailTransformer":  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the one column from X, containing the email addresses, into multiple columns.

        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Transformed dataframe containing the extra columns.
        """

        if not all(elem in X.columns for elem in self.features):
            raise ValueError("Not all provided `features` could be found in `X`!")

        X = check_X(X)

        for column in self.features:

            X[f"{column}_domain"] = (
                X[column].str.split("@").str[1].str.split(".").str[0]
            )

            X[column] = X[column].str.split("@").str[0]

            X[f"{column}_num_of_digits"] = X[column].map(
                EmailTransformer.__num_of_digits
            )
            X[f"{column}_num_of_letters"] = X[column].map(
                EmailTransformer.__num_of_letters
            )
            X[f"{column}_num_of_special_chars"] = X[column].map(
                EmailTransformer.__num_of_special_characters
            )
            X[f"{column}_num_of_repeated_chars"] = X[column].map(
                EmailTransformer.__num_of_repeated_characters
            )
            X[f"{column}_num_of_words"] = X[column].map(EmailTransformer.__num_of_words)
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


class StringSimilarityTransformer(BaseEstimator, TransformerMixin):
    """
    Calculates the similarity between two strings using the `gestalt pattern matching` algorithm from the `SequenceMatcher` class.
    Args:
        features (Tuple[str, str]): The two columns that contain the strings for which the similarity should be calculated.
    """

    def __init__(self, features: Tuple[str, str]) -> None:
        self.features = features

    def fit(self, X=None, y=None) -> "StringSimilarityTransformer":  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the similarity of two strings provided in `features`.

        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Original dataframe containing the extra column with the calculated similarity.
        """
        if not all(elem in X.columns for elem in self.features):
            raise ValueError("Not all provided `features` could be found in `X`!")

        X = check_X(X)

        X[f"{self.features[0]}_{self.features[1]}_similarity"] = X[
            [self.features[0], self.features[1]]
        ].apply(
            lambda x: StringSimilarityTransformer.__similar(
                StringSimilarityTransformer.__normalize_string(x[self.features[0]]),
                StringSimilarityTransformer.__normalize_string(x[self.features[1]]),
            ),
            axis=1,
        )
        return X

    @staticmethod
    def __similar(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def __normalize_string(string: str) -> str:
        string = string.strip().lower()
        return (
            unicodedata.normalize("NFKD", string)
            .encode("utf8", "strict")
            .decode("utf8")
        )


class PhoneTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms a phone number into multiple features.

    Args:
        features (List[str]): List of features which should be transformed.
        national_number_divisor (float): Divider `national_number`.
        country_code_divisor (flat): Divider for `country_code`.
        error_value (str): Value to use if the phone number is invalid or the parsing fails.
    """

    def __init__(
        self,
        features: List[str],
        national_number_divisor: float = 1e9,
        country_code_divisor: float = 1e2,
        error_value: str = "-999",
    ) -> None:
        self.features = features
        self.national_number_divisor = national_number_divisor
        self.country_code_divisor = country_code_divisor
        self.error_value = error_value

    def fit(self, X=None, y=None) -> "PhoneTransformer":  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the similarity of two strings provided in `features`.

        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Original dataframe containing the extra column with the calculated similarity.
        """

        if not all(elem in X.columns for elem in self.features):
            raise ValueError("Not all provided `features` could be found in `X`!")

        X = check_X(X)

        for column in self.features:

            X[f"{column}_national_number"] = X[column].apply(
                lambda x: PhoneTransformer.__phone_to_float(
                    "national_number",
                    x,
                    int(self.national_number_divisor),
                    self.error_value,
                )
            )
            X[f"{column}_country_code"] = X[column].apply(
                lambda x: PhoneTransformer.__phone_to_float(
                    "country_code", x, int(self.country_code_divisor), self.error_value
                )
            )

        return X

    @staticmethod
    def __phone_to_float(
        attribute: str, phone: str, divisor: int, error_value: str
    ) -> float:
        phone = phone.replace(" ", "")
        phone = re.sub(r"[^0-9+-]", "", phone)
        phone = re.sub(r"^00", "+", phone)
        try:
            return float(getattr(phonenumbers.parse(phone, None), attribute)) / divisor
        except:  # pylint: disable=W0702
            try:
                return float(re.sub(r"(?<!^)[^0-9]", "", error_value))
            except:  # pylint: disable=W0702
                return float(error_value)
