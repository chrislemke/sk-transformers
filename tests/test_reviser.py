# -*- coding: utf-8 -*-

import pytest

from feature_reviser.reviser import revise_classifier

# pylint: disable=missing-function-docstring


def test_revise_classifier_for_fit_attribute(X, y):
    with pytest.raises(AttributeError):
        revise_classifier(None, X, y, [("c", 10), ("d", 10), ("e", 10)])


def test_revise_classifier_columns(clf, X, y):
    result = revise_classifier(clf, X, y, [("c", 10), ("d", 10), ("e", 10)])
    assert list(result[0].columns) == ["c", "d", "e"]
    assert list(result[1].columns) == ["a", "b"]


def test_revise_classifier_feature_importance(clf, X, y):
    result = revise_classifier(clf, X, y, [("c", 10), ("d", 10), ("e", 10)])
    assert result[0].iloc[3].sum() == 1.0
    assert result[1].iloc[2].sum() == 1.0


def test_revise_classifier_correlation(clf, X, y):
    result = revise_classifier(clf, X, y, [("c", 10), ("d", 10), ("e", 10)])
    assert result[0].iloc[3].sum() == 1.0
    assert result[1].iloc[3].sum() == -1.7056057308448835


def test_revise_classifier_chi2(clf, X, y):
    result = revise_classifier(clf, X, y, [("c", 10), ("d", 10), ("e", 10)])
    assert result[0].iloc[0][0] == 3.333333333333333
    assert result[0].iloc[0][1] == 0.625
    assert result[0].iloc[0][2] == 0.1975308641975312


def test_revise_classifier_f_statistic(clf, X, y):
    result = revise_classifier(clf, X, y, [("c", 10), ("d", 10), ("e", 10)])
    assert result[1].iloc[0][0] == 21.33333396911621
    assert result[1].iloc[0][1] == 23.945640563964844
