import numpy as np
import pytest
from sklearn.pipeline import make_pipeline

from sk_transformers import MathExpressionTransformer

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_math_expression_transformer_no_np_method(X_numbers) -> None:
    with pytest.raises(AttributeError) as error:
        MathExpressionTransformer(
            [("small_numbers", "np.xxx", None, None)]
        ).fit_transform(X_numbers)
    assert "Operation xxx not found in NumPy!" == str(error.value)


def test_math_expression_transformer_no_operation_method(X_numbers) -> None:
    with pytest.raises(AttributeError) as error:
        MathExpressionTransformer([("small_numbers", "xxx", None, None)]).fit_transform(
            X_numbers
        )
    assert (
        "Invalid operation! `xxx` is not a valid operation! Please refer to the `numpy` and `operator` package."
        == str(error.value)
    )


def test_math_expression_transformer_non_ufunc(X_numbers) -> None:
    with pytest.raises(ValueError) as error:
        MathExpressionTransformer(
            [("small_numbers", "np.sum", "small_numbers", None)]
        ).fit_transform(X_numbers)
    assert (
        "The function `np.sum` is not a NumPy universal function. If you are using `np.sum` or `np.prod`, please use `np.add` or `np.multiply` instead."
        == str(error.value)
    )


def test_math_expression_transformer_in_pipeline(X_numbers) -> None:
    pipeline = make_pipeline(
        MathExpressionTransformer(
            [
                ("small_numbers", "add", 1, None),
                ("small_numbers", "mul", "small_numbers", None),
                (
                    "small_numbers",
                    "np.add",
                    "small_float_numbers",
                    {"where": np.array([True, False, False, True, True])},
                ),
                ("small_numbers", "np.divide", "small_numbers", None),
                ("small_numbers", "numpy.sin", None, None),
                ("small_numbers", "np.subtract", 2, None),
                ("big_numbers", "neg", None, None),
            ]
        )
    )
    result = pipeline.fit_transform(X_numbers)
    expected_add = np.array([8, 13, 83, 2, 1])
    expected_mul = np.array([49, 144, 6724, 1, 0])
    expected_np_add = np.array([11.5, 0, 0, 2.9, 0.6])
    expected_np_divide = np.array([1.0, 1.0, 1.0, 1.0, np.nan])
    expected_np_subtract = np.array([5, 10, 80, -1, -2])
    expected_neg = np.array(
        [
            -10_000_000,
            -12_891_207,
            -42_000,
            -11_111_111,
            -99_999_999_999_999,
        ]
    )
    expected_np_sin = np.array(
        [
            0.6569865987187891,
            -0.5365729180004349,
            0.31322878243308516,
            0.8414709848078965,
            0.0,
        ]
    )

    assert pipeline.steps[0][0] == "mathexpressiontransformer"
    assert np.array_equal(result["small_numbers_add_1"].to_numpy(), expected_add)
    assert np.array_equal(
        result["small_numbers_mul_small_numbers"].to_numpy(), expected_mul
    )
    assert np.allclose(
        result["small_numbers_add_small_float_numbers"].to_numpy(), expected_np_add
    )
    assert np.array_equal(
        result["small_numbers_divide_small_numbers"].to_numpy(),
        expected_np_divide,
        equal_nan=True,
    )
    assert np.array_equal(
        result["small_numbers_sin"].to_numpy().round(3), expected_np_sin.round(3)
    )
    assert np.array_equal(
        result["small_numbers_subtract_2"].to_numpy(), expected_np_subtract
    )
    assert np.array_equal(result["big_numbers_neg"].to_numpy(), expected_neg)


def test_math_expression_transformer_in_pipeline_with_non_existing_column(
    X_numbers,
) -> None:
    with pytest.raises(ValueError) as error:
        pipeline = make_pipeline(
            MathExpressionTransformer([("non_existing", "add", 1, None)])
        )
        _ = pipeline.fit_transform(X_numbers)

    assert """
                MathExpressionTransformer:
                Not all provided `features` could be found in `X`! Following columns were not found in the dataframe: `non_existing`.
                """ == str(
        error.value
    )
