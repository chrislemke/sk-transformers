from datetime import datetime
from typing import List, Tuple

import pandas as pd

from sk_transformers.base_transformer import BaseTransformer
from sk_transformers.utils import check_ready_to_transform


class DateColumnsTransformer(BaseTransformer):
    """Splits a date column into multiple columns.

    Example:
    ```python
    import pandas as pd
    from sk_transformers import DateColumnsTransformer

    X = pd.DataFrame({"foo": ["2021-01-01", "2022-02-02", "2023-03-03"]})
    transformer = DateColumnsTransformer(["foo"])
    transformer.fit_transform(X)
    ```
    ```
             foo  foo_year  ...  foo_is_year_end  foo_is_weekend
    0 2021-01-01      2021  ...            False           False
    1 2022-02-02      2022  ...            False           False
    2 2023-03-03      2023  ...            False           False
    ```

    Args:
        features (List[str]): List of columns to transform.
        date_format (str): Date format. Defaults to `%Y-%m-%d`.
        errors (str): How to handle errors in `pd.to_datetime`. Defaults to `raise`.
            available values: `ignore`, `raise`, `coerce`.
            If `raise`, then invalid parsing will raise an exception.
            If `coerce`, then invalid parsing will be set as `NaT`.
            If `ignore`, then invalid parsing will return the input.
        date_elements ([List[str]]): List of date elements to extract.
    """

    __slots__ = (
        "features",
        "date_format",
        "date_elements",
        "errors",
    )

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        features: List[str],
        date_format: str = "%Y-%m-%d",
        errors: str = "raise",
        date_elements: List[str] = [
            "year",
            "month",
            "day",
            "day_of_week",
            "day_of_year",
            "week_of_year",
            "quarter",
            "is_leap_year",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
            "is_weekend",
        ],
    ) -> None:
        super().__init__()
        self.features = features
        self.date_format = date_format
        self.date_elements = date_elements
        self.errors = errors

    def transform(  # pylint: disable=too-many-branches
        self, X: pd.DataFrame
    ) -> pd.DataFrame:
        """Transforms columns from the provided dataframe.

        Args:
            X (pandas.DataFrame): Dataframe with columns to transform.

        Returns:
            pandas.DataFrame: Dataframe with transformed columns.
        """

        X = check_ready_to_transform(self, X, self.features)

        for column in self.features:
            X[column] = pd.to_datetime(
                X[column], format=self.date_format, errors=self.errors
            )
            if "year" in self.date_elements:
                X[f"{column}_year"] = X[column].dt.year
            if "month" in self.date_elements:
                X[f"{column}_month"] = X[column].dt.month
            if "day" in self.date_elements:
                X[f"{column}_day"] = X[column].dt.day
            if "day_of_week" in self.date_elements:
                X[f"{column}_day_of_week"] = X[column].dt.dayofweek
            if "day_of_year" in self.date_elements:
                X[f"{column}_day_of_year"] = X[column].dt.dayofyear
            if "week_of_year" in self.date_elements:
                X[f"{column}_week_of_year"] = X[column].dt.isocalendar().week
            if "quarter" in self.date_elements:
                X[f"{column}_quarter"] = X[column].dt.quarter
            if "is_leap_year" in self.date_elements:
                X[f"{column}_is_leap_year"] = X[column].dt.is_leap_year
            if "is_month_start" in self.date_elements:
                X[f"{column}_is_month_start"] = X[column].dt.is_month_start
            if "is_month_end" in self.date_elements:
                X[f"{column}_is_month_end"] = X[column].dt.is_month_end
            if "is_quarter_start" in self.date_elements:
                X[f"{column}_is_quarter_start"] = X[column].dt.is_quarter_start
            if "is_quarter_end" in self.date_elements:
                X[f"{column}_is_quarter_end"] = X[column].dt.is_quarter_end
            if "is_year_start" in self.date_elements:
                X[f"{column}_is_year_start"] = X[column].dt.is_year_start
            if "is_year_end" in self.date_elements:
                X[f"{column}_is_year_end"] = X[column].dt.is_year_end
            if "is_weekend" in self.date_elements:
                X[f"{column}_is_weekend"] = X[column].dt.dayofweek.isin([5, 6])
        return X


class DurationCalculatorTransformer(BaseTransformer):
    """Calculates the duration between to given dates.

    Example:
    ```python
    import pandas as pd
    from sk_transformers import DurationCalculatorTransformer

    X = pd.DataFrame(
        {
            "foo": ["1960-01-01", "1970-01-01", "1990-01-01"],
            "bar": ["1960-01-01", "1971-01-01", "1988-01-01"],
        }
    )
    transformer = DurationCalculatorTransformer(("foo", "bar"), "days", "foo_bar_duration")
    transformer.fit_transform(X)
    ```
    ```
              foo         bar  foo_bar_duration
    0  1960-01-01  1960-01-01                 0
    1  1970-01-01  1971-01-01               365
    2  1990-01-01  1988-01-01              -731
    ```

    Args:
        features (Tuple[str, str]): The two columns that contain the dates which should be used to calculate the duration.
        unit (str): The unit in which the duration should be returned. Should be either `days` or `seconds`.
        new_column_name (str): The name of the output column.
    """

    __slots__ = (
        "features",
        "unit",
        "new_column_name",
    )

    def __init__(
        self, features: Tuple[str, str], unit: str, new_column_name: str
    ) -> None:
        super().__init__()
        if unit not in ["days", "seconds"]:
            raise ValueError("Unsupported unit. Should be either `days` or `seconds`!")

        self.features = features
        self.unit = unit
        self.new_column_name = new_column_name

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform method that calculates the duration between two dates.

        Args:
            X (pandas.DataFrame): The input DataFrame.

        Returns:
            pandas.DataFrame: The transformed DataFrame.
        """
        X = check_ready_to_transform(self, X, list(self.features))

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
    """Transforms a date column with a specified format into a timestamp
    column.

    Example:
    ```python
    import pandas as pd
    from sk_transformers import TimestampTransformer

    X = pd.DataFrame({"foo": ["1960-01-01", "1970-01-01", "1990-01-01"]})
    transformer = TimestampTransformer(["foo"])
    transformer.fit_transform(X)
    ```
    ```
               foo
    0 -315619200.0
    1          0.0
    2  631152000.0
    ```

    Args:
        features (List[str]): List of features which should be transformed.
        date_format (str): Format of the date column. Defaults to "%Y-%m-%d".
    """

    __slots__ = (
        "features",
        "date_format",
    )

    def __init__(
        self,
        features: List[str],
        date_format: str = "%Y-%m-%d",
    ) -> None:
        super().__init__()
        self.features = features
        self.date_format = date_format

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms columns from the provided dataframe.

        Args:
            X (pandas.DataFrame): Dataframe with columns to transform.

        Returns:
            pandas.DataFrame: Dataframe with transformed columns.
        """
        X = check_ready_to_transform(self, X, self.features)

        for column in self.features:
            X[column] = pd.to_datetime(
                X[column], format=self.date_format, errors="raise"
            )
            X[column] = (X[column] - datetime(1970, 1, 1)).dt.total_seconds()
        return X
