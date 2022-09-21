# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper

from feature_reviser.utils import check_data


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
        DataFrameMapper: Fitted mapper object used to transform the DataFrame.
    """

    check_data(X, y, check_nans=False)

    mapper = DataFrameMapper(columns, drop_cols=drop_columns, default=None, df_out=True)
    return mapper.fit_transform(X, y), mapper
