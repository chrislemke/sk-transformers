# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from feature_reviser.selector import select_with_classifier

# pylint: disable=missing-function-docstring


def test_select_with_classifier_for_fit_attribute(X, y):
    with pytest.raises(AttributeError):
        select_with_classifier(None, X, y)


def test_select_with_classifier_return_types(clf, X, y):
    result = select_with_classifier(clf, X, y)
    assert isinstance(result[0], pd.DataFrame)
    assert isinstance(result[1], pd.Series)


def test_select_with_classifier_shape_without_k_best(clf, X, y):
    result = select_with_classifier(clf, X, y)
    assert result[0].shape == (10, 1)
    assert result[1].shape == (10,)


def test_select_with_classifier_for_missing_parameter(clf, X, y):
    with pytest.raises(ValueError):
        select_with_classifier(clf, X, y, True)


def test_select_with_classifier_column_shape_with_k_best(clf, X, y):
    result = select_with_classifier(
        clf, X, y, True, [("c", 10), ("d", 10), ("e", 10)], 2, 1, 0, 2
    )
    assert result[0].shape == (10, 2)
    assert result[1].shape == (10,)
