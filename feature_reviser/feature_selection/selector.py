# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif

from feature_reviser.utils import check_data, prepare_categorical_data


def select_with_classifier(
    clf: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    select_k_best_first: bool = False,
    cat_features: Optional[List[Tuple[str, int]]] = None,
    cat_k_best: Optional[int] = None,
    num_k_best: Optional[int] = None,
    drop_correlated: bool = False,
    drop_correlated_threshold: float = 0.6,
    model_select_threshold: Optional[float] = None,
    max_features: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Optionally examens the features of `X` using `SelectKBest` for categorical and numerical features before the features of `X`
    are selected using `SelectFromModel` with the provided classifier.

    Args:
        clf (BaseEstimator): The classifier used for examine the features.
        X (pandas.DataFrame): The dataframe containing the categorical and numerical features.
        y (pandas.Series): The target variable.
        select_k_best_first (bool): If `True` the features are selected using `SelectKBest` first. Defaults to False.
        cat_features (Optional[List[Tuple[str, int]]]): A tuple containing the names of the categorical features and the corresponding threshold.
            If the number of unique values is greater than the threshold, the feature is considered numerical and not categorical.
            This is needed if `select_k_best_first` is `True`.
        cat_k_best (Optional[int]): The max number of categorical features to select using `SelectKBest`.
            Defaults to None. This is needed if `select_k_best_first` is `True`.
        num_k_best (Optional[int]): The max number of numerical features to select using `SelectKBest`.
            Defaults to None. This is needed if `select_k_best_first` is `True`.
        drop_correlated (bool): If `True` the correlated features are dropped. Defaults to False.
        drop_correlated_threshold (float): The threshold used for dropping correlated features. Defaults to 0.6.
        model_select_threshold (Optional[float]): The threshold used for `SelectFromModel`. Defaults to None.
        max_features (Optional[int]): The max number of features to select using `SelectFromModel`. Defaults to None.

    Raises:
        ValueError: if the `cat_features` are not in the dataframe.
        ValueError: If `select_k_best_first` is `True` and `cat_features`, `num_features`, `cat_k_best` or `num_k_best` are `None`.
        TypeError: If the classifier does not contain the `fit` attribute.

    Returns:
        Tuple[pandas.DataFrame, pandas.Series]: Tuple containing the selected features and the target variable.
    """
    X = X.copy()
    check_data(X, y)

    if cat_features:
        # pylint: disable=consider-using-set-comprehension
        if not {f[0] for f in cat_features}.issubset(set(X.columns)):
            raise ValueError("cat_features must be in the dataframe!")
        X = prepare_categorical_data(X, cat_features)

        if len(X.select_dtypes(include=["category"]).columns) == 0:
            raise ValueError(
                "cat_features must be in the dataframe! Check if the threshold is maybe a bit too low."
            )

    if not hasattr(clf, "fit"):
        raise AttributeError("Classifier does not have fit method!")

    if select_k_best_first:
        if cat_features is None or cat_k_best is None or num_k_best is None:
            raise ValueError(
                "If `select_k_best_first` is set to `True`, `cat_features`, `num_features`, `cat_k_best`, and `num_k_best` must be provided!"
            )

        cat_df = X.select_dtypes(include=["category"])
        num_df = X.select_dtypes(include=[np.float64, np.int64])

        print("Selecting categorical features...")
        cat_transformer = SelectKBest(chi2, k=min(cat_k_best, cat_df.shape[1] - 1)).fit(
            cat_df, y
        )
        print("Selecting numerical features...")
        num_transformer = SelectKBest(
            f_classif, k=min(num_k_best, num_df.shape[1] - 1)
        ).fit(num_df, y)

        cat_x = cat_transformer.transform(cat_df)
        num_x = num_transformer.transform(num_df)

        columns = [
            cat_df.columns[i] for i in cat_transformer.get_support(indices=True)
        ] + [num_df.columns[i] for i in num_transformer.get_support(indices=True)]

        print(
            f"The following columns were selected using the `SelectKBest` algorithm: {columns}.".replace(
                "[", ""
            )
            .replace("]", "")
            .replace("'", "")
        )
        X = pd.DataFrame(data=np.column_stack((cat_x, num_x)), columns=columns)

    if drop_correlated:
        print("Dropping correlated features...")
        X = DropCorrelatedFeatures(
            variables=list(num_df.columns),
            method="pearson",
            threshold=drop_correlated_threshold,
        ).fit_transform(X)
        X = DropCorrelatedFeatures(
            variables=list(cat_df.columns),
            method="spearman",
            threshold=drop_correlated_threshold,
        ).fit_transform(X)

    print("Selecting features with classifier...")

    selector = SelectFromModel(
        estimator=clf,
        threshold=model_select_threshold,
        max_features=max_features or X.shape[1],
    ).fit(X, y)
    selected = selector.transform(X)
    columns = [X.columns[i] for i in selector.get_support(indices=True)]
    print(
        f"The following columns were selected using the `{clf.__class__.__name__}`: {columns}.".replace(
            "[", ""
        )
        .replace("]", "")
        .replace("'", "")
    )
    return (
        pd.DataFrame(
            data=selected,
            columns=columns,
        ),
        y,
    )
