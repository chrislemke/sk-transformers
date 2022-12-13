# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator, TransformerMixin

# pylint: disable= missing-function-docstring, unused-argument


class BaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for all transformers. This class inherits from BaseEstimator and TransformerMixin.
    Its main purpose is to provide an implementation of the `fit` method that does nothing.
    Since most custom transformers do not need to implement a fit method, this class
    can be used as a base class for all transformers not needing a `fit` method.
    """

    def fit(self, X=None, y=None):  # type: ignore
        return self
