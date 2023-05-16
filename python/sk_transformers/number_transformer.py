import operator
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import sk_transformers
from sk_transformers.base_transformer import BaseTransformer
from sk_transformers.utils import check_ready_to_transform


class MathExpressionTransformer(BaseTransformer):
    """Applies an function/operation to a column and a given value or column.
    The operation can be a function from NumPy's mathematical functions or
    operator package.

    **Warning!** Some functions/operators may not work as expected. Especially, functions that don't
    belong in [`numpy.ufunc`](https://numpy.org/doc/stable/reference/ufuncs.html) are not supported.
    NumPy functions with return values that don't fit the size of the source column are also not supported.

    Example:
    ```python
    import pandas as pd
    from sk_transformers import MathExpressionTransformer

    X = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    transformer = MathExpressionTransformer([("foo", "np.add", "bar", None)])
    transformer.fit_transform(X)
    ```
    ```
       foo  bar  foo_add_bar
    0    1    4            5
    1    2    5            7
    2    3    6            9
    ```

    Args:
        features (List[str, str, Union[int, float]]): List of tuples containing the name of the column to apply the operation on,
            a string representation of the operation (see list above) and the value to apply the operation on. The value can be
            an number (int or float) or the name of another column in the dataframe. If the value is `None`, it it expected that
            the operation only takes one argument. The fourth entry of the tuple is a dictionary passed as `kwargs` to the operation.
    """

    def __init__(
        self,
        features: List[
            Tuple[str, str, Union[int, float, str, None], Optional[Dict[str, Any]]]
        ],
    ) -> None:
        super().__init__()
        self.features = features

    def __verify_operation(self, operation: str) -> Tuple[bool, Any]:
        if operation.startswith("np"):
            if hasattr(np, operation[3:]):
                op = getattr(np, operation[3:])
                is_np_op = True
                if not isinstance(op, np.ufunc):
                    raise ValueError(
                        f"The function `{operation}` is not a NumPy universal function. If you are using `np.sum` or `np.prod`, please use `np.add` or `np.multiply` instead."
                    )
            else:
                raise AttributeError(f"Operation {operation[3:]} not found in NumPy!")

        elif hasattr(operator, operation) and operation not in [
            "attrgetter",
            "itemgetter",
            "methodcaller",
        ]:
            op = getattr(operator, operation)
            is_np_op = False

        else:
            raise AttributeError(
                f"Invalid operation! `{operation}` is not a valid operation! Please refer to the `numpy` and `operator` package."
            )

        return is_np_op, op

    def __abbreviate_numpy_in_operation(self, operation: str) -> str:
        """Replaces `numpy` at the start of a string with `np`.

        Args:
            operation (str): The operation as a string.

        Returns:
            str: The operation as a string with numpy replaced with np.
        """
        if operation.startswith("numpy"):
            operation = "np" + operation[5:]
        return operation

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies the operation to the column and the value.

        Args:
            X (pandas.DataFrame): DataFrame containing the columns to apply the operation on.

        Returns:
            pandas.DataFrame: The original dataframe with the new columns. The new columns are named as follows:
            '`column_name`_`operation`_`value`' or '`column_name`_`operation`' if `value` is `None`.
        """
        X = check_ready_to_transform(self, X, [feature[0] for feature in self.features])

        for feature, operation, value, kwargs in self.features:
            operation = self.__abbreviate_numpy_in_operation(operation)
            is_np_op, op = self.__verify_operation(operation)

            new_column = f"{feature}_{operation}".replace("np.", "")
            new_column_with_value = f"{feature}_{operation}_{value}".replace("np.", "")

            if is_np_op:
                warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
                if value is None:
                    X[new_column] = op(X[feature], **kwargs or {})
                elif isinstance(value, str):
                    X[new_column_with_value] = op(X[feature], X[value], **kwargs or {})
                else:
                    X[new_column_with_value] = op(X[feature], value, **kwargs or {})
            else:
                if value is None:
                    X[new_column] = op(X[feature])
                elif isinstance(value, str):
                    X[new_column_with_value] = op(X[feature], X[value])
                else:
                    X[new_column_with_value] = op(
                        X[feature], value
                    )  # This created a ragged array in will be deprecated in future.
        return X


class GeoDistanceTransformer(BaseTransformer):
    """Calculates the distance in kilometers between two places on the earth
    using the geographic coordinates.

    Example:
    ```python
    import pandas as pd
    from sk_transformers import GeoDistanceTransformer

    X = pd.DataFrame(
        {
            "lat_1": [48.353802, 51.289501, 53.63040161],
            "long_1": [11.7861, 6.76678, 9.988229752],
            "lat_2": [51.289501, 53.63040161, 48.353802],
            "long_2": [6.76678, 9.988229752, 11.7861],
        }
    )
    transformer = GeoDistanceTransformer([("lat_1", "long_1", "lat_2", "long_2")])
    transformer.fit_transform(X)
    ```
    ```
           lat_1    long_1      lat_2    long_2  distance_lat_1_lat_2
    0  48.353802  11.78610  51.289501   6.76678            485.975293
    1  51.289501   6.76678  53.630402   9.98823            339.730537
    2  53.630402   9.98823  48.353802  11.78610            600.208154
    ```

    Args:
        features: A list of tuples containing the names of four columns, which are coordinates of the
            two points in the following order:
            - latitude of point 1
            - longitude of point 1
            - latitude of point 2
            - longitude of point 2
    """

    def __init__(self, features: List[Tuple[str, str, str, str]]) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Adds new columns containing the distances between two points on the
        earth based on their geographical coordinates.

        Args:
            X (pandas.DataFrame): Dataframe containing the coordinates of the two points as four columns.

        Returns:
            pandas.DataFrame: Dataframe containing the new columns.
        """
        X = check_ready_to_transform(self, X, [feature[0] for feature in self.features])

        for coordinates in self.features:
            GeoDistanceTransformer.__check_latitudes(X[coordinates[0]])
            GeoDistanceTransformer.__check_longitudes(X[coordinates[1]])
            GeoDistanceTransformer.__check_latitudes(X[coordinates[2]])
            GeoDistanceTransformer.__check_longitudes(X[coordinates[3]])

            X[f"distance_{coordinates[0]}_{coordinates[2]}"] = pd.Series(
                sk_transformers.distance_function(  # type: ignore
                    X[coordinates[0]].to_numpy(),
                    X[coordinates[1]].to_numpy(),
                    X[coordinates[2]].to_numpy(),
                    X[coordinates[3]].to_numpy(),
                )
            )

        return X

    @staticmethod
    def __check_latitudes(x: pd.Series) -> None:
        if ((x > 90) | (x < -90)).sum() > 0:
            raise ValueError("Invalid values for latitude.")

    @staticmethod
    def __check_longitudes(x: pd.Series) -> None:
        if ((x > 180) | (x < -180)).sum() > 0:
            raise ValueError("Invalid values for longitude.")
