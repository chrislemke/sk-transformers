from typing import Dict, Tuple

import pandas as pd
import polars as pl

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
        features (list[str]): List of columns to transform.
        date_format (str): Date format. Defaults to `%Y-%m-%d`.
        errors (str): How to handle errors in `pd.to_datetime`. Defaults to `raise`.
            available values: `ignore`, `raise`, `coerce`.
            If `raise`, then invalid parsing will raise an exception.
            If `coerce`, then invalid parsing will be set as `NaT`.
            If `ignore`, then invalid parsing will return the input.
        date_elements ([list[str]]): List of date elements to extract.
    """

    __slots__ = (
        "features",
        "date_format",
        "date_elements",
        "errors",
    )

    def __init__(  # pylint: disable=dangerous-default-value
        self,
        features: list[str],
        date_format: str = "%Y-%m-%d",
        errors: str = "raise",
        date_elements: list[str] = [
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

        X = check_ready_to_transform(self, X, self.features, return_polars=True)

        for column in self.features:  # pylint: disable=duplicate-code
            X = X.with_columns(
                pl.col(column)
                .str.strptime(pl.Datetime, fmt=self.date_format)
                .alias(column + "_datetime")
            )

            date_element_dict: Dict[str, pl.Expr] = {
                "year": pl.col(f"{column}_datetime").dt.year(),
                "month": pl.col(f"{column}_datetime").dt.month(),
                "day": pl.col(f"{column}_datetime").dt.day(),
                "day_of_week": pl.col(f"{column}_datetime").dt.weekday() - 1,
                "day_of_year": pl.col(f"{column}_datetime").dt.ordinal_day(),
                "week_of_year": pl.col(f"{column}_datetime").dt.week(),
                "quarter": pl.col(f"{column}_datetime").dt.quarter(),
                "is_leap_year": pl.col(f"{column}_datetime").dt.year() % 4 == 0,
                "is_month_start": pl.col(f"{column}_datetime").dt.day() == 1,
                "is_month_end": pl.col(f"{column}_datetime")
                .dt.day()
                .is_in([28, 29, 30, 31]),
                "is_quarter_start": pl.col(f"{column}_datetime")
                .dt.ordinal_day()
                .is_in([1, 91, 183, 275]),
                "is_quarter_end": pl.col(f"{column}_datetime")
                .dt.ordinal_day()
                .is_in([90, 182, 274, 365]),
                "is_year_start": pl.col(f"{column}_datetime").dt.ordinal_day() == 1,
                "is_year_end": pl.col(f"{column}_datetime")
                .dt.ordinal_day()
                .is_in([365, 366]),
                "is_weekend": pl.col(f"{column}_datetime").dt.weekday().is_in([6, 7]),
            }

            X = X.with_columns(
                [
                    date_element_dict[element].alias(f"{column}_{element}")
                    for element in self.date_elements
                ]
            ).drop(f"{column}_datetime")

        return X.to_pandas()


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
        X = check_ready_to_transform(self, X, list(self.features), return_polars=True)

        if self.unit == "seconds":
            return X.with_columns(
                (
                    pl.col(self.features[1]).str.strptime(pl.Datetime, fmt="%Y-%m-%d")
                    - pl.col(self.features[0]).str.strptime(pl.Datetime, fmt="%Y-%m-%d")
                )
                .dt.seconds()
                .alias(self.new_column_name)
            ).to_pandas()
        return X.with_columns(
            (
                pl.col(self.features[1]).str.strptime(pl.Datetime, fmt="%Y-%m-%d")
                - pl.col(self.features[0]).str.strptime(pl.Datetime, fmt="%Y-%m-%d")
            )
            .dt.days()
            .alias(self.new_column_name)
        ).to_pandas()


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
        features (list[str]): List of features which should be transformed.
        date_format (str): Format of the date column. Defaults to "%Y-%m-%d".
    """

    __slots__ = (
        "features",
        "date_format",
    )

    def __init__(
        self,
        features: list[str],
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
        X = check_ready_to_transform(self, X, self.features, return_polars=True)

        return X.with_columns(
            [
                pl.col(column)
                .str.strptime(pl.Datetime, self.date_format)
                .dt.timestamp("ms")
                / 1000
                for column in self.features
            ]
        ).to_pandas()
