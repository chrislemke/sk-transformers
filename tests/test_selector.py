# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from feature_reviser.feature_selection.selector import select_with_classifier

# pylint: disable=missing-function-docstring


def test_select_with_classifier_for_low_threshold(clf, X, y):
    with pytest.raises(ValueError) as error:
        select_with_classifier(
            clf, X, y, True, [("c", 1), ("d", 1), ("e", 1)], 2, 1, 0, 2
        )

    assert (
        "cat_features must be in the dataframe! Check if the threshold is maybe a bit too low."
        == str(error.value)
    )


def test_select_with_classifier_for_fit_attribute(X, y):
    with pytest.raises(AttributeError) as error:
        select_with_classifier(None, X, y)

    assert "Classifier does not have fit method!" == str(error.value)


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
