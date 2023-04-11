from typing import Union

import pandas as pd
from feature_engine.encoding import MeanEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class MeanEncoderTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn API for the [feature-engine MeanEncoder](https://feature-
    engine.readthedocs.io/en/latest/api_doc/encoding/MeanEncoder.html).

    Example:
    ```python
    import pandas as pd
    from sk_transformers import MeanEncoderTransformer

    X = pd.DataFrame({"foo": ["a", "b", "a", "c", "b", "a", "c", "a", "b", "c"]})
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    encoder = MeanEncoderTransformer()
    encoder.fit_transform(X, y)
    ```
    ```
            foo
    0  0.500000
    1  0.666667
    2  0.500000
    3  0.333333
    4  0.666667
    5  0.500000
    6  0.333333
    7  0.500000
    8  0.666667
    9  0.333333
    ```

    Args:
        fill_na_value (Union[int, float]): Value to fill NaN values with.
            Those may appear if a category is not present in the set the encoder was not fitted on.
    """

    def __init__(self, fill_na_value: Union[int, float] = -999) -> None:
        self.encoder = MeanEncoder(ignore_format=False)
        self.fill_na_value = fill_na_value

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanEncoderTransformer":
        """Fit the MeanEncoder to the data.

        Args:
            X (pandas.DataFrame): DataFrame to fit the MeanEncoder to.
            y (pandas.Series): Target variable.

        Returns:
            MeanEncoder: Fitted MeanEncoder.
        """
        self.encoder.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted MeanEncoder.

        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Transformed DataFrame.
        """
        return self.encoder.transform(X).fillna(self.fill_na_value)
