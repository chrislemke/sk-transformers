# -*- coding: utf-8 -*-

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline

from feature_reviser.transformer.number_transformer import MathExpressionTransformer

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_math_expression_transformer_in_pipeline(X_numbers) -> None:
    pipeline = make_pipeline(
        MathExpressionTransformer(
            [
                ("small_numbers", "add", 1, None),
                ("small_numbers", "np.sum", "small_float_numbers", {"axis": 0}),
                ("big_numbers", "neg", None, None),  # type: ignore
            ]
        )
    )
    result = pipeline.fit_transform(X_numbers)
    expected_add = np.array([8, 13, 83, 2, 1])
    expected_sum = np.array([11.5, 15.5, 88.9, 2.9, 0.6])
    expected_neg = np.array(
        [
            -10_000_000,
            -12_891_207,
            -42_000,
            -11_111_111,
            -99_999_999_999_999,
        ]
    )

    assert pipeline.steps[0][0] == "mathexpressiontransformer"
    assert np.array_equal(result["small_numbers_add_1"].values, expected_add)
    assert np.array_equal(
        result["small_numbers_sum_small_float_numbers"].values, expected_sum
    )
    assert np.array_equal(result["big_numbers_neg"].values, expected_neg)


def test_math_expression_transformer_in_pipeline_with_non_existing_column(
    X_numbers,
) -> None:
    with pytest.raises(ValueError) as error:
        pipeline = make_pipeline(
            MathExpressionTransformer([("non_existing", "add", 1, None)])
        )
        _ = pipeline.fit_transform(X_numbers)

    assert "Not all provided `features` could be found in `X`!" == str(error.value)
