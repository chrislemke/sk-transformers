# -*- coding: utf-8 -*-


import operator
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from feature_engine.dataframe_checks import check_X

from feature_reviser.transformer.base_transformer import BaseTransformer

# pylint: disable= missing-function-docstring, unused-argument


class MathExpressionTransformer(BaseTransformer):
    """
    Applies an operation to a column and a given value or column.
    The operation can be any operation from the `numpy` or `operator` package.

    **Warning!** Some operators may not work as expected. Especially not all NumPy methods are supported. For example:
    various NumPy methods return values which are not fitting the size of the source column.

    Example:
        >>> X = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
        >>> transformer = MathExpressionTransformer([("foo", "np.sum", "bar", {"axis": 0})])
        >>> transformer.fit_transform(X).values
        array([[1, 4, 5],
               [2, 5, 7],
               [3, 6, 9]])

    Args:
        features (List[str, str, Union[int, float]]): List of tuples containing the name of the column to apply the operation on,
            a string representation of the operation (see list above) and the value to apply the operation on. The value can be
            an number (int or float) or the name of another column in the dataframe. If the value is `None`, it it expected that
            the operation only takes one argument.
    """

    def __init__(
        self,
        features: List[
            Tuple[str, str, Union[int, float, str, None], Optional[Dict[str, Any]]]
        ],
    ) -> None:
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the operation to the column and the value.

        Args:
            X (pandas.DataFrame): DataFrame containing the columns to apply the operation on.

        Returns:
            pandas.DataFrame: The original dataframe with the new columns. The new columns are named as follows:
            '`column_name`_`operation`_`value`' or '`column_name`_`operation`' if `value` is `None`.
        """
        if not all(f in X.columns for f in [f[0] for f in self.features]):
            raise ValueError("Not all provided `features` could be found in `X`!")

        X = check_X(X)

        for (feature, operation, value, kwargs) in self.features:
            is_np_op, op = self.__verify_operation(operation)

            new_column = f"{feature}_{operation}".replace("np.", "")
            new_column_with_value = f"{feature}_{operation}_{value}".replace("np.", "")

            if is_np_op:
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
                    X[new_column_with_value] = op(X[feature], value)
        return X

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
