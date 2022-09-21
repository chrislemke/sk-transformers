# -*- coding: utf-8 -*-

import numpy as np

from feature_reviser.transformer.column_transformer import transform_df_columns

# pylint: disable=missing-function-docstring


def test_column_transformer_transforms_correctly(
    standard_scaler, ordinal_encoder, X, y
):
    columns = [
        (["a"], standard_scaler),
        (["b"], standard_scaler),
        (["c"], ordinal_encoder),
    ]
    result, _ = transform_df_columns(X, y, columns, drop_columns=["e"])

    expected_a = np.array(
        [
            -1.5666989036012806,
            -1.2185435916898848,
            -0.8703882797784892,
            -0.5222329678670935,
            -0.17407765595569785,
            0.17407765595569785,
            0.5222329678670935,
            0.8703882797784892,
            1.2185435916898848,
            1.5666989036012806,
        ]
    )
    expected_c = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0])

    assert result.shape == (10, 4)
    assert list(result.columns) == ["a", "b", "c", "d"]
    assert np.array_equal(expected_a, result["a"].values)
    assert np.array_equal(expected_c, result["c"].values)


def test_column_transformer_returns_a_copy_x(standard_scaler, ordinal_encoder, X, y):
    columns = [
        (["a"], standard_scaler),
        (["b"], standard_scaler),
        (["c"], ordinal_encoder),
    ]
    result, _ = transform_df_columns(X, y, columns, drop_columns=["e"])

    assert not result["a"].equals(X["a"])
