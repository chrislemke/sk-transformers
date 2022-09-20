# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper


def transform_df_columns(
    X: pd.DataFrame,
    y: pd.Series,
    columns: List[Tuple[str, TransformerMixin]],
    drop_columns: Optional[List[str]] = None,
) -> pd.DataFrame:

    """
    Transform columns of a DataFrame using a list of transformers.

    Args:
        X (pandas.DataFrame): DataFrame to transform.
        y (pandas.Series): Target variable.
        columns (List[Tuple[str, sklearn.base.TransformerMixin]]): List of tuples containing the column name and the transformer.

    Returns:
        pandas.DataFrame: Transformed DataFrame.
    """

    X = X.copy()
    not_transformed_columns = list(set(X.columns) - set([c[0] for c in columns][0]))
    mapper = DataFrameMapper(columns, drop_cols=drop_columns)
    mapper.fit(X, y)
    return pd.DataFrame(
        np.concatenate([X[not_transformed_columns].values, mapper.transform(X)], axis=1)
    )
