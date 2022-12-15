from sklearn.base import BaseEstimator, TransformerMixin

# pylint: disable= missing-function-docstring, unused-argument


class BaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for all custom transformers. This class inherits from BaseEstimator and TransformerMixin.
    Its main purpose is to provide an implementation of the `fit` method that does nothing except setting the `self.fitted_` to `True`.
    Since most custom transformers do not need to implement a fit method, this class
    can be used as a base class for all transformers not needing a `fit` method.
    """

    def __init__(self) -> None:
        self.fitted_ = False

    def fit(self, X=None, y=None):  # type: ignore
        self.fitted_ = True
        return self
