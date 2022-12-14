import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from sk_transformers.base_transformer import BaseTransformer
from sk_transformers.utils import check_ready_to_transform


class DtypeTransformer(BaseTransformer):
    """
    Transformer that converts a column to a different dtype.

    Example:
    ```python
    import numpy as np
    import pandas as pd
    from sk_transformers.generic_transformer import DtypeTransformer

    X = pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "a", "b"]})
    transformer = DtypeTransformer([("foo", np.float32), ("bar", "category")])
    transformer.fit_transform(X).dtypes
    ```
    ```
    foo     float32
    bar    category
    dtype: object
    ```

    Args:
        features (List[Tuple[str, Union[str, type]]]): List of tuples containing the column name and the dtype (`str` or `type`).
    """

    def __init__(self, features: List[Tuple[str, Union[str, type]]]) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe by converting the columns to the specified dtypes.

        Args:
            X (pandas.DataFrame): dataframe to transform.

        Returns:
            pandas.DataFrame: Transformed dataframe.
        """
        check_ready_to_transform(self, X, [feature[0] for feature in self.features])

        for (column, dtype) in self.features:
            X[column] = X[column].astype(dtype)
        return X


class AggregateTransformer(BaseTransformer):
    """
    This transformer uses Pandas `groupby` method and `aggregate` to apply function on a column grouped by another column.
    Read more about Pandas [`aggregate`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.aggregate.html) method
    to understand how to use function for aggregation. Other than Pandas function this transformer only support functions and string-names.

    Example:
    ```python
    import pandas as pd
    from sk_transformers.generic_transformer import AggregateTransformer

    X = pd.DataFrame(
        {
            "foo": ["mr", "mr", "ms", "ms", "ms", "mr", "mr", "mr", "mr", "ms"],
            "bar": [46, 32, 78, 48, 93, 68, 53, 38, 76, 56],
        }
    )

    transformer = AggregateTransformer([("foo", "bar", ["mean"])])
    transformer.fit_transform(X)
    ```
    ```
      foo  bar  MEAN(foo__bar)
    0  mr   46       52.166668
    1  mr   32       52.166668
    2  ms   78       68.750000
    3  ms   48       68.750000
    4  ms   93       68.750000
    5  mr   68       52.166668
    6  mr   53       52.166668
    7  mr   38       52.166668
    8  mr   76       52.166668
    9  ms   56       68.750000
    ```

    Args:
        features (List[Tuple[str, str, List[str]]]): List of tuples containing the column identifiers and the aggregation function(s).
                The first column identifier (features[0]) is the column that will be used to group the data.
                It can be either numerical or categorical. The second column identifier (features[1]) is the column that will be used
                for aggregations. This column must be numerical.
    """

    def __init__(self, features: List[Tuple[str, str, List[str]]]) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new columns by using Pandas `groupby` method and `aggregate` to apply function on the column.

        Args:
            X (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe. It contains the original columns and the new columns created by this transformer.
        """

        check_ready_to_transform(
            self,
            X,
            [feature[0] for feature in self.features]
            + [feature[1] for feature in self.features],
        )

        for (groupby_column, agg_column, aggs) in self.features:

            agg_df = (
                X.groupby([groupby_column])[agg_column]
                .aggregate(aggs, engine="cython")
                .reset_index()
            )

            agg_df = agg_df.rename(
                columns={
                    agg: f"{agg.upper()}({groupby_column}__{agg_column})"
                    for agg in aggs
                }
            )

            for column in list(np.delete(agg_df.columns, 0)):
                agg_df[column] = agg_df[column].astype(np.float32)

            X = X.merge(agg_df, on=groupby_column, how="left")

        return X


class FunctionsTransformer(BaseTransformer):
    """
    This transformer is a plain wrapper around the [`sklearn.preprocessing.FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html).
    Its main function is to apply multiple functions to different columns. Other than the scikit-learn transformer,
    this transformer *does not* support the `inverse_func`, `accept_sparse`, `feature_names_out` and, `inv_kw_args` parameters.

    Example:
    ```python
    import numpy as np
    import pandas as pd
    from sk_transformers.generic_transformer import FunctionsTransformer

    X = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    transformer = FunctionsTransformer([("foo", np.log1p, None), ("bar", np.sqrt, None)])
    transformer.fit_transform(X)
    ```
    ```
            foo       bar
    0  0.693147  2.000000
    1  1.098612  2.236068
    2  1.386294  2.449490
    ```

    Args:
        features (List[str, Callable, Optional[Dict[str, Any]]]): List of tuples containing the name of the
            column to apply the function on and the function itself.
            As well as a dictionary passed to the function as `kwargs`.
    """

    def __init__(
        self, features: List[Tuple[str, Callable, Optional[Dict[str, Any]]]]
    ) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the functions to the columns, and returns the dataframe with the modified columns.

        Args:
            X (pandas.DataFrame): DataFrame containing the columns to apply the functions on.

        Returns:
            pandas.DataFrame: The original dataframe with the modified columns.
        """

        X = check_ready_to_transform(self, X, [feature[0] for feature in self.features])

        for (column, func, kwargs) in self.features:
            X[column] = FunctionTransformer(
                func, validate=True, kw_args=kwargs
            ).transform(X[[column]].to_numpy())

        return X


class MapTransformer(BaseTransformer):
    """
    This transformer iterates over all columns in the `features` list and applies the given callback to the column.
    For this it uses the [`pandas.Series.map`](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html) method.

    Example:
    ```python
    import pandas as pd
    from sk_transformers.generic_transformer import MapTransformer

    X = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    transformer = MapTransformer([("foo", lambda x: x + 1)])
    transformer.fit_transform(X)
    ```
    ```
       foo  bar
    0    2    4
    1    3    5
    2    4    6
    ```

    Args:
        features (List[Tuple[str, Callable]]): List of tuples containing the name of the
            column to apply the callback on and the callback itself.
    """

    def __init__(self, features: List[Tuple[str, Callable]]) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the callback to the column.

        Args:
            X (pandas.DataFrame): Dataframe containing the the columns to apply the callback on.

        Returns:
            pandas.DataFrame: The dataframe containing
                the new column together with the non-transformed original columns.
        """

        X = check_ready_to_transform(self, X, [feature[0] for feature in self.features])

        for (feature, callback) in self.features:
            X[feature] = X[feature].map(callback)

        return X


class ColumnDropperTransformer(BaseTransformer):
    """
    Drops columns from a dataframe using Pandas `drop` method.

    Example:
    ```python
    import pandas as pd
    from sk_transformers.generic_transformer import ColumnDropperTransformer

    X = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    transformer = ColumnDropperTransformer(["foo"])
    transformer.fit_transform(X)
    ```
    ```
       bar
    0    4
    1    5
    2    6
    ```

    Args:
        columns (Union[str, List[str]]): Columns to drop. Either a single column name or a list of column names.
    """

    def __init__(self, columns: Union[str, List[str]]) -> None:
        super().__init__()
        self.columns = columns

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the dataframe with the `columns` dropped.

        Args:
            X (pd.DataFrame): Dataframe to drop columns from.

        Returns:
            pd.DataFrame: Dataframe with columns dropped.
        """
        X = check_ready_to_transform(self, X, self.columns)
        return X.drop(self.columns, axis=1)


class NaNTransformer(BaseTransformer):
    """
    Replace NaN values with a specified value. Internally Pandas [`fillna`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html) method is used.

    Example:
    ```python
    from sk_transformers.generic_transformer import NaNTransformer
    import pandas as pd
    import numpy as np

    X = pd.DataFrame({"foo": [1, np.NaN, 3], "bar": ["a", np.NaN, "c"]})
    transformer = NaNTransformer([("foo", -999), ("bar", "-999")])
    transformer.fit_transform(X)
    ```
    ```
          foo   bar
    0     1.0     a
    1  -999.0  -999
    2     3.0     c
    ```

    Args:
        features (List[Tuple[str, Any]]): List of tuples where the first element is the column name, and the second is the value to replace NaN with.
    """

    def __init__(self, features: List[Tuple[str, Any]]) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replace NaN values with a specified value.

        Args:
            X (pandas.DataFrame): Dataframe to transform.

        Returns:
            pandas.DataFrame: Transformed dataframe.
        """
        X = check_ready_to_transform(
            self,
            X,
            [feature[0] for feature in self.features],
            force_all_finite="allow-nan",
        )
        return X.fillna(dict(self.features))


class ValueIndicatorTransformer(BaseTransformer):
    """
    Adds a column to a dataframe indicating if a value is equal to a specified value.
    The idea behind this method is, that it is often useful to know if a `NaN` value was
    present in the original data and has been changed by some imputation step.
    Sometimes the present of a `NaN` value is actually important information.
    But obviously this method works with any kind of data.

    `NaN`, `None` or `np.nan` are **Not** caught by this implementation.

    Example:
    ```python
    from sk_transformers.generic_transformer import ValueIndicatorTransformer
    import pandas as pd

    X = pd.DataFrame({"foo": [1, -999, 3], "bar": ["a", "-999", "c"]})
    transformer = ValueIndicatorTransformer([("foo", -999), ("bar", "-999")])
    transformer.fit_transform(X).to_dict()
    ```
    ```
       foo   bar  foo_found_indicator  bar_found_indicator
    0    1     a                False                False
    1 -999  -999                 True                 True
    2    3     c                False                False
    ```

    Args:
        features (List[Tuple[str, Any]]): A list of tuples where the first value in represents the column
            name and the second value represents the value to check for.
    """

    def __init__(self, features: List[Tuple[str, Any]], as_int: bool = False) -> None:
        super().__init__()
        self.features = features
        self.as_int = as_int

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add a column to a dataframe indicating if a value is equal to a specified value.

        Args:
            X (pandas.DataFrame): Dataframe to transform.

        Returns:
            pandas.DataFrame: Transformed dataframe containing columns indicating if a certain value was found.
                Format of the new columns: `"column_name"_nan_indicator`.
        """
        X = check_ready_to_transform(
            self,
            X,
            [feature[0] for feature in self.features],
            force_all_finite="allow-nan",
        )

        for (column, indicator) in self.features:
            X[f"{column}_found_indicator"] = (X[column] == indicator).astype(
                int if self.as_int else bool
            )
        return X


class QueryTransformer(BaseTransformer):
    """
    Applies a list of queries to a dataframe.
    If it operates on a dataset used for supervised learning this transformer should
    be applied on the dataframe containing `X` and `y`. So removing of columns by queries
    also removes the corresponding `y` value.
    Read more about queries [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html).

    Example:
    ```python
    import pandas as pd
    from sk_transformers.generic_transformer import QueryTransformer

    X = pd.DataFrame({"foo": [1, 8, 3, 6, 5, 4, 7, 2]})
    transformer = QueryTransformer(["foo > 4"])
    transformer.fit_transform(X)
    ```
    ```
       foo
    1    8
    3    6
    4    5
    6    7
    ```

    Args:
        queries (List[str]): List of query string to evaluate to the dataframe.
    """

    def __init__(self, queries: List[str]) -> None:
        super().__init__()
        self.queries = queries

    def transform(self, Xy: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the list of queries to the dataframe.

        Args:
            Xy (pd.DataFrame): Dataframe to apply the queries to. For also operating on the target column `y` - if needed.
                This column should also be part of the dataframe.

        Returns:
            pd.DataFrame: Dataframe with the queries applied.
        """

        Xy = check_ready_to_transform(self, Xy)
        for query in self.queries:
            Xy = Xy.query(query, inplace=False)
        return Xy


class ValueReplacerTransformer(BaseTransformer):
    r"""
    Uses Pandas `replace` method to replace values in a column. This transformer loops over the `features` and applies
    `replace` to the according columns. If the column is not from type string but a valid regular expression is provided
    the column will be temporarily changed to a string column and after the manipulation by `replace` changed back to its
    original type. It may happen, that this type changing fails if the modified column is not compatible with its original type.

    Example:
    ```python
    import pandas as pd
    from sk_transformers.generic_transformer import ValueReplacerTransformer

    X = pd.DataFrame(
        {"foo": ["0000-01-01", "2022/01/08", "bar", "1982-12-7", "28-09-2022"]}
    )
    transformer = ValueReplacerTransformer(
        [
            (
                ["foo"],
                r"^(?!(19|20)\d\d[-\/.](0[1-9]|1[012]|[1-9])[-\/.](0[1-9]|[12][0-9]|3[01]|[1-9])$).*",
                "1900-01-01",
            )
        ]
    )

    transformer.fit_transform(X)
    ```
    ```
              foo
    0  1900-01-01
    1  2022/01/08
    2  1900-01-01
    3   1982-12-7
    4  1900-01-01
    ```


    Args:
        features (List[Tuple[List[str], str, Any]]): List of tuples containing the column names as a list,
            the value to replace (can be a regex), and the replacement value.
    """

    def __init__(self, features: List[Tuple[List[str], Any, Any]]) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces a value or regular expression with another value.

        Args:
            X (pd.DataFrame): Dataframe containing the columns where values should be replaced.

        Returns:
            pd.DataFrame: Dataframe with replaced values.
        """

        X = check_ready_to_transform(
            self, X, [feature[0][0] for feature in self.features]
        )

        for (columns, value, replacement) in self.features:
            for column in columns:
                is_regex = ValueReplacerTransformer.__check_for_regex(value)
                column_dtype = X[column].dtype

                if column_dtype is not str and is_regex:
                    X[column] = X[column].astype(str)

                X[column] = X[column].replace(value, replacement, regex=is_regex)

                if X[column].dtype != column_dtype:
                    X[column] = X[column].astype(column_dtype)

        return X

    @staticmethod
    def __check_for_regex(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        try:
            re.compile(value)
            is_valid = True
        except re.error:
            is_valid = False
        return is_valid


class LeftJoinTransformer(BaseTransformer):
    """
    Performs a database-style left-join using `pd.merge`. This transformer is suitable for
    replacing values in a column of a dataframe by looking-up another `pd.DataFrame`
    or `pd.Series`. Note that, the join is based on the index of the right dataframe.

    Example:
    ```python
    import pandas as pd
    from sk_transformers.generic_transformer import LeftJoinTransformer

    X = pd.DataFrame({"foo": ["A", "B", "C", "A", "C"]})
    lookup_df = pd.Series([1, 2, 3], index=["A", "B", "C"], name="values")
    transformer = LeftJoinTransformer([("foo", lookup_df)])
    transformer.fit_transform(X)
    ```
    ```
      foo  foo_values
    0   A           1
    1   B           2
    2   C           3
    3   A           1
    4   C           3
    ```

    Args:
        features (List[Tuple[str, Union[pd.Series, pd.DataFrame]]]): A list of tuples
            where the first element is the name of the column
            and the second element is the look-up dataframe or series.
    """

    def __init__(
        self, features: List[Tuple[str, Union[pd.Series, pd.DataFrame]]]
    ) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform a left-join on the given columns of a dataframe with another cooresponding dataframe.

        Args:
            X (pd.DataFrame): Dataframe containing the columns to be joined on.

        Returns:
            pd.DataFrame: Dataframe joined on the given columns.
        """

        X = check_ready_to_transform(self, X, [feature[0] for feature in self.features])

        for (column, lookup_df) in self.features:
            lookup_df = LeftJoinTransformer.__prefix_df_column_names(lookup_df, column)
            X = pd.merge(X, lookup_df, how="left", left_on=column, right_index=True)

        return X

    @staticmethod
    def __prefix_df_column_names(
        df: Union[pd.Series, pd.DataFrame], prefix: str
    ) -> Union[pd.Series, pd.DataFrame]:
        if isinstance(df, pd.Series):
            df.name = prefix + "_" + (df.name if df.name else "lookup")
        elif isinstance(df, pd.DataFrame):
            df.columns = [prefix + "_" + column for column in df.columns]
        return df
