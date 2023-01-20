from typing import Any, List, Optional, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted


def check_ready_to_transform(
    transformer: Any,
    X: pd.DataFrame,
    features: Union[str, List[str]],
    force_all_finite: Union[bool, str] = True,
    dtype: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Args:
        transformer (Any): The transformer that calls this function. It must be a subclass of `BaseEstimator` from scikit-learn.
        X (pandas.DataFrame): pandas dataframe or NumPy array. The input to check and copy or transform.
        features (Optional[Union[str, List[str]]]): The features to check if they are in the dataframe.
        force_all_finite (Union[bool, str]): Whether to raise an error on np.inf and np.nan in X. The possibilities are:
            - True: Force all values of array to be finite.
            - False: accepts np.inf, np.nan, pd.NA in array.
            - "allow-nan": accepts only np.nan and pd.NA values in array. Values cannot be infinite.
        dtype (Optional[Union[str, List[str]]]): Data type of result. If None, the `dtype` of the input is preserved.
            If "numeric", `dtype` is preserved unless `array.dtype` is object.
            If dtype is a list of types, conversion on the first type is only performed if the dtype of the input
            is not in the list.

    Raises:
        TypeError: If the input `transformer` is not a subclass of `BaseEstimator`.
        ValueError: If the input `X` is not a Pandas dataframe.
        ValueError: If the input is an empty Pandas dataframe.
        ValueError: If the input `X` does not contain the feature.
        ValueError: if the input `X` does not contain all features.


    Returns:
        pandas.DataFrame: A checked copy of original dataframe.
    """

    if isinstance(features, str):
        features = [features]

    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a Pandas dataframe!")
    if X.empty:
        raise ValueError("X must not be empty!")

    if isinstance(features, list):
        if not all(c in X.columns for c in features):
            not_in_df = (
                str([c for c in features if c not in X.columns])
                .replace("[", "")
                .replace("]", "")
                .replace("'", "`")
            )
            raise ValueError(
                f"Not all provided `features` could be found in `X`! Following columns were not found in the dataframe: {not_in_df}."
            )

    if issubclass(transformer.__class__, BaseEstimator) is False:
        raise TypeError(
            f"""
            `transformer` from type (`{transformer.__class__.__name__}`) is not a subclass of `BaseEstimator`!
            See https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py#L146 for an example/template.
            """
        )
    check_is_fitted(transformer, "fitted_")

    X_tmp = X[
        dict.fromkeys(X[features]).keys()
    ].copy()  # `dict.fromkeys` was chosen instead of `set` to maintain the order of the entries.

    X_tmp_array = check_array(
        X_tmp.to_numpy(),
        dtype=dtype,
        accept_large_sparse=False,
        force_all_finite=force_all_finite,
    )
    X_tmp = pd.DataFrame(X_tmp_array, columns=X_tmp.columns, index=X_tmp.index)

    for column in X_tmp.columns:
        X_tmp[column] = X_tmp[column].astype(X[column].dtype)

    non_included_features = [c for c in X.columns if c not in features]
    if non_included_features:
        X_tmp = pd.concat([X_tmp, X[non_included_features]], axis=1)

    return X_tmp


def check_data(X: pd.DataFrame, y: pd.Series, check_nans: bool = True) -> None:
    """Checks if the data has the correct types, shapes and does not contain
    any missing values.

    Args:
        X (pandas.DataFrame): The features.
        y (pandas.Series): The target variable.
        check_nans (bool): Whether to check for missing values. Defaults to `True`.

    Raises:
        TypeError: If the features are not a `pandas.DataFrame` or the target variable is not a `pandas.Series` or `numpy.ndarray`.
        ValueError: If the features or target variable contain missing values.

    Returns:
        None
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Features must be a pandas.DataFrame!")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas.Series!")
    if check_nans:
        if X.isnull().to_numpy().any():
            raise ValueError("Features must not contain NaN values!")
        if y.isnull().to_numpy().any():
            raise ValueError("y must not contain NaN values!")


def prepare_categorical_data(
    X: pd.DataFrame, categories: List[Tuple[str, int]]
) -> pd.DataFrame:
    """Checks for the validity of the categorical features inside the
    dataframe. And prepares the data for further processing by changing the
    `dtypes`.

    Args:
        X (pandas.DataFrame): The dataframe containing the categorical features.
        categories (List[Tuple[str, int]]): The list of categorical features and their thresholds.
            If the number of unique values is greater than the threshold, the feature is not considered categorical.

    Raises:
        TypeError: If the features are not a `pandas.DataFrame` or the categorical features are not a `List[str]`.
        ValueError: If the categorical features are not in the dataframe.

    Returns:
        pandas.DataFrame: The original dataframe with the categorical features converted to `category` dtype.
    """
    cat_features = [f[0] for f in categories]

    if not isinstance(X, pd.DataFrame):
        raise TypeError("features must be a pandas.DataFrame!")
    if not set(set(cat_features)).issubset(set(X.columns)):
        raise ValueError("cat_features must be in the dataframe!")

    for feature, threshold in categories:
        if (str(X[feature].dtype) != "object") or (X[feature].nunique() > threshold):
            cat_features.remove(feature)
            print(
                f"""{feature} has fewer unique values than {threshold}.
                So it will not be converted to Category dtype."""
            )

    pd.options.mode.chained_assignment = None
    for column in X.columns:
        if column in cat_features:
            X[column] = X[column].astype("category").copy()

    return X
