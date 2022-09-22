# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn_pandas import DataFrameMapper

from feature_reviser.utils import check_data


def transform_df_columns(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[Tuple[Union[str, List[str]], TransformerMixin]],
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, DataFrameMapper]:

    """
    Transform columns of a DataFrame using a list of transformers.

    Args:
        X (pandas.DataFrame): DataFrame to transform.
        y (pandas.Series): Target variable.
        columns (List[Tuple[Union[str, List[str]], TransformerMixin]]): List of tuples containing the column name(s) and the transformer.

    Returns:
        Tuple[pd.DataFrame, DataFrameMapper]: Transformed DataFrame and fitted mapper object.
    """

    check_data(X, y, check_nans=False)

    mapper = DataFrameMapper(
        features, drop_cols=drop_columns, default=None, df_out=True
    )
    return mapper.fit_transform(X, y), mapper
