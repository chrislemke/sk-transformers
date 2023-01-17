import operator
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sk_transformers.base_transformer import BaseTransformer
from sk_transformers.utils import check_ready_to_transform


class MathExpressionTransformer(BaseTransformer):
    """Applies an function/operation to a column and a given value or column.
    The operation can be a function from NumPy's mathematical functions or
    operator package.

    **Warning!** Some functions/operators may not work as expected. Especially not all NumPy methods are supported. For example:
    various NumPy methods return values which are not fitting the size of the source column.

    Example:
    ```python
    import pandas as pd
    from sk_transformers import MathExpressionTransformer

    X = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    transformer = MathExpressionTransformer([("foo", "np.sum", "bar", {"axis": 0})])
    transformer.fit_transform(X)
    ```
    ```
       foo  bar  foo_sum_bar
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

        for (feature, operation, value, kwargs) in self.features:
            operation = self.__abbreviate_numpy_in_operation(operation)
            is_np_op, op = self.__verify_operation(operation)

            new_column = f"{feature}_{operation}".replace("np.", "")
            new_column_with_value = f"{feature}_{operation}_{value}".replace("np.", "")

            if is_np_op:
                warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
                if value is None:
                    X[new_column] = op(X[feature], **kwargs or {})
                elif isinstance(value, str):
                    X[new_column_with_value] = op(
                        [X[feature], X[value]], **kwargs or {}
                    )
                else:
                    X[new_column_with_value] = op([X[feature], value], **kwargs or {})
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
