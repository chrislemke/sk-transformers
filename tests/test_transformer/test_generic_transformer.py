import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline

from sk_transformers import (
    AggregateTransformer,
    AllowedValuesTransformer,
    ColumnDropperTransformer,
    DtypeTransformer,
    FunctionsTransformer,
    LeftJoinTransformer,
    MapTransformer,
    NaNTransformer,
    QueryTransformer,
    ValueIndicatorTransformer,
    ValueReplacerTransformer,
)

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_dtype_transformer_in_pipeline(X) -> None:
    pipeline = make_pipeline(DtypeTransformer([("a", np.float32), ("e", "category")]))
    X = pipeline.fit_transform(X)

    assert X["a"].dtype == "float32"
    assert X["e"].dtype == "category"

    assert pipeline.steps[0][0] == "dtypetransformer"


def test_dtype_transformer_in_pipeline_with_nan(X_nan_values) -> None:
    pipeline = make_pipeline(DtypeTransformer([("a", np.float32), ("d", "category")]))
    X = pipeline.fit_transform(X_nan_values)

    assert X["a"].dtype == "float32"
    assert X["d"].dtype == "category"
    assert pipeline.steps[0][0] == "dtypetransformer"


def test_dtype_transformer_raises_error(X) -> None:
    with pytest.raises(ValueError) as error:
        DtypeTransformer(
            [("non_existing", np.float32), ("e", "category")]
        ).fit_transform(X)

    assert (
        "Not all provided `features` could be found in `X`! Following columns were not found in the dataframe: `non_existing`."
        == str(error.value)
    )


def test_aggregate_transformer_in_pipeline(X_group_by) -> None:
    pipeline = make_pipeline(AggregateTransformer([("a", "b", ["mean"])]))
    result = pipeline.fit_transform(X_group_by)
    expected = np.array(
        [
            ["mr", 46, 52.16666793823242],
            ["mr", 32, 52.16666793823242],
            ["ms", 78, 68.75],
            ["ms", 48, 68.75],
            ["ms", 93, 68.75],
            ["mr", 68, 52.16666793823242],
            ["mr", 53, 52.16666793823242],
            ["mr", 38, 52.16666793823242],
            ["mr", 76, 52.16666793823242],
            ["ms", 56, 68.75],
        ],
        dtype=object,
    )
    assert np.array_equal(result.to_numpy(), expected)
    assert list(result.columns) == ["a", "b", "MEAN(a__b)"]
    assert pipeline.steps[0][0] == "aggregatetransformer"


def test_aggregate_transformer_raises_error(X) -> None:
    with pytest.raises(ValueError) as error:
        AggregateTransformer([("e", "non_existing", ["count", "mean"])]).fit_transform(
            X
        )

    assert (
        "Not all provided `features` could be found in `X`! Following columns were not found in the dataframe: `non_existing`."
        == str(error.value)
    )


def test_functions_transformer_in_pipeline(X) -> None:
    pipeline = make_pipeline(
        FunctionsTransformer([("a", np.log1p, None), ("b", np.sqrt, None)])
    )
    result = pipeline.fit_transform(X)
    expected_a = np.array(
        [
            0.6931471805599453,
            1.0986122886681098,
            1.3862943611198906,
            1.6094379124341003,
            1.791759469228055,
            1.9459101490553132,
            2.0794415416798357,
            2.1972245773362196,
            2.302585092994046,
            2.3978952727983707,
        ]
    )

    expected_b = np.array(
        [
            1.0488088481701516,
            1.4832396974191326,
            1.816590212458495,
            2.0976176963403033,
            2.345207879911715,
            2.569046515733026,
            2.7748873851023217,
            2.9664793948382653,
            3.146426544510455,
            3.1780497164141406,
        ]
    )

    assert np.array_equal(result["a"].to_numpy().round(6), expected_a.round(6))
    assert np.array_equal(result["b"].to_numpy().round(6), expected_b.round(6))
    assert pipeline.steps[0][0] == "functionstransformer"


def test_functions_transformer_raises_error(X) -> None:
    with pytest.raises(ValueError) as error:
        FunctionsTransformer([("non_existing", np.sqrt, None)]).fit_transform(X)

    assert (
        "Not all provided `features` could be found in `X`! Following columns were not found in the dataframe: `non_existing`."
        == str(error.value)
    )


def test_map_transformer_in_pipeline(X) -> None:
    pipeline = make_pipeline(MapTransformer([("a", lambda x: x**2)]))
    result = pipeline.fit_transform(X)
    expected = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    assert np.array_equal(result["a"].to_numpy(), expected)
    assert pipeline.steps[0][0] == "maptransformer"


def test_map_transformer_raises_error(X) -> None:
    with pytest.raises(ValueError) as error:
        MapTransformer([("non_existing", lambda x: x**2)]).fit_transform(X)

    assert (
        "Not all provided `features` could be found in `X`! Following columns were not found in the dataframe: `non_existing`."
        == str(error.value)
    )


def test_query_transformer_in_pipeline(X) -> None:
    pipeline = make_pipeline(QueryTransformer(["a > 6"]))
    X = pipeline.fit_transform(X)

    assert X.shape[0] == 4
    assert X.shape[1] == 5
    assert pipeline.steps[0][0] == "querytransformer"


def test_value_indicator_transformer_in_pipeline(X_nan_values) -> None:
    pipeline = make_pipeline(
        ValueIndicatorTransformer([("d", -999), ("e", "-999")], as_int=True)
    )
    X = pipeline.fit_transform(X_nan_values)

    assert X.loc[5, "d_found_indicator"] == 1
    assert X.loc[6, "d_found_indicator"] == 0
    assert X.loc[6, "e_found_indicator"] == 1
    assert X.loc[7, "e_found_indicator"] == 0

    assert pipeline.steps[0][0] == "valueindicatortransformer"


def test_value_indicator_transformer_in_pipeline_with_non_existing_column(
    X_nan_values,
) -> None:
    with pytest.raises(ValueError) as error:
        pipeline = make_pipeline(
            ValueIndicatorTransformer(
                [("d", -999), ("non_existing", "-999")], as_int=True
            )
        )
        _ = pipeline.fit_transform(X_nan_values)

    assert (
        "Not all provided `features` could be found in `X`! Following columns were not found in the dataframe: `non_existing`."
        == str(error.value)
    )


def test_value_replacer_transformer_in_pipeline(X_time_values) -> None:
    values = [
        (["a", "e"], r"^(?:[1-9][0-9]+|9)$", 99),
        (["dd"], "\\N", "-999"),
        (
            ["dd"],
            r"^(?!(19|20)\d\d[-\/.](0[1-9]|1[012]|[1-9])[-\/.](0[1-9]|[12][0-9]|3[01]|[1-9])$).*",
            "1900-01-01",
        ),
        (["f"], "\\N", "-999"),
    ]

    pipeline = make_pipeline(ValueReplacerTransformer(values))
    result = pipeline.fit_transform(X_time_values)
    expected_a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 99, 99])
    expected_dd = np.array(
        [
            "1900-01-01",
            "1900-01-01",
            "1900-01-01",
            "1900-01-01",
            "2022.02.05",
            "1900-01-01",
            "2022/01/08",
            "2022-01-09",
            "1960-01-01",
            "1900-01-01",
        ]
    )
    expected_e = np.array([2, 4, 6, 8, 99, 99, 99, 99, 99, 99])
    expected_f = np.array(["2", "4", "6", "8", "-999", "12", "14", "16", "18", "20"])

    assert np.array_equal(result["a"].to_numpy(), expected_a)
    assert np.array_equal(result["dd"].to_numpy(), expected_dd)
    assert np.array_equal(result["e"].to_numpy(), expected_e)
    assert np.array_equal(result["f"].to_numpy(), expected_f)
    assert pipeline.steps[0][0] == "valuereplacertransformer"


def test_column_dropper_transformer_in_pipeline(X) -> None:
    pipeline = make_pipeline(ColumnDropperTransformer(columns=["a", "b", "c", "d"]))

    result = pipeline.fit_transform(X)
    expected = X.drop(columns=["a", "b", "c", "d"])
    assert result.equals(expected)
    assert pipeline.steps[0][0] == "columndroppertransformer"


def test_column_dropper_transformer_in_pipeline_with_nan(X_nan_values) -> None:
    drop_columns = ["a", "d"]
    pipeline = make_pipeline(ColumnDropperTransformer(columns=drop_columns))

    result = pipeline.fit_transform(X_nan_values)
    expected = X_nan_values.drop(columns=drop_columns)
    assert result.equals(expected)
    assert pipeline.steps[0][0] == "columndroppertransformer"


def test_nan_transformer_in_pipeline(X_nan_values) -> None:
    pipeline = make_pipeline(NaNTransformer([("a", -1), ("b", -1), ("c", "missing")]))
    X = pipeline.fit_transform(X_nan_values)

    assert X.isnull().sum().sum() == 0
    assert X["a"][1] == -1
    assert X["b"][2] == -1
    assert X["c"][6] == "missing"
    assert pipeline.steps[0][0] == "nantransformer"
    assert pipeline.steps[0][1].features[0][1] == -1


def test_left_join_transformer_in_pipeline_for_series(X_categorical) -> None:
    lookup_df = pd.Series([1, 2], index=["A1", "A2"], name="values")
    pipeline = make_pipeline(LeftJoinTransformer([("a", lookup_df)]))
    result = pipeline.fit_transform(X_categorical)
    expected = np.array([1, 2, 2, 1, 1, 2, 1, 1])

    assert "a_values" in result.columns
    assert np.array_equal(result["a_values"].to_numpy(), expected)
    assert pipeline.steps[0][0] == "leftjointransformer"


def test_left_join_transformer_in_pipeline_for_dataframe(X_categorical) -> None:
    lookup_df = pd.DataFrame({"values": [1, 2]}, index=["A1", "A2"])
    pipeline = make_pipeline(LeftJoinTransformer([("a", lookup_df)]))
    result = pipeline.fit_transform(X_categorical)
    expected = np.array([1, 2, 2, 1, 1, 2, 1, 1])

    assert "a_values" in result.columns
    assert np.array_equal(result["a_values"].to_numpy(), expected)
    assert pipeline.steps[0][0] == "leftjointransformer"


def test_left_join_transformer_in_pipeline_with_nan(X_categorical) -> None:
    X_categorical["a"].iloc[5] = np.nan
    lookup_df = pd.Series([1, 2], index=["A1", "A2"], name="values")
    pipeline = make_pipeline(LeftJoinTransformer([("a", lookup_df)]))
    result = pipeline.fit_transform(X_categorical)

    assert "a_values" in result.columns
    assert np.isclose(result["a_values"].iloc[5], np.nan, equal_nan=True)
    assert pipeline.steps[0][0] == "leftjointransformer"


def test_allowed_values_transformer_in_pipeline(X) -> None:
    values = [
        ("a", [1, 2], -999),
        ("c", ["1", "2"], "other"),
    ]
    pipeline = make_pipeline(AllowedValuesTransformer(values))
    result = pipeline.fit_transform(X)
    expected_a = np.array([1, 2, -999, -999, -999, -999, -999, -999, -999, -999])
    expected_c = np.array(
        ["1", "1", "1", "1", "2", "2", "2", "other", "other", "other"]
    )

    assert np.array_equal(result["a"].to_numpy(), expected_a)
    assert np.array_equal(result["c"].to_numpy(), expected_c)
    assert pipeline.steps[0][0] == "allowedvaluestransformer"
