import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline

from sk_transformers import (
    EmailTransformer,
    IPAddressEncoderTransformer,
    PhoneTransformer,
    StringCombinationTransformer,
    StringSimilarityTransformer,
    StringSlicerTransformer,
    StringSplitterTransformer,
)

# pylint: disable=missing-function-docstring, missing-class-docstring


def test_ip_address_encoder_transformer_in_pipeline(X_numbers) -> None:
    pipeline = make_pipeline(IPAddressEncoderTransformer(["ip_address"]))

    X = pipeline.fit_transform(X_numbers)
    expected = np.array(
        [
            0.3405803971,
            0.3232235777,
            4.254076642994478e-11,
            4.254076645264119e-11,
            -999,
        ]
    )
    assert pipeline.steps[0][0] == "ipaddressencodertransformer"
    assert np.allclose(X["ip_address"].to_numpy(), expected)


def test_email_transformer_in_pipeline(X_strings) -> None:
    pipeline = make_pipeline(EmailTransformer(["email"]))
    result = pipeline.fit_transform(X_strings)
    expected = pd.DataFrame(
        {
            "email": {
                0: "test",
                1: "test123",
                2: "test_123$$",
                3: "test_test",
                4: "ttt",
                5: "test_test_test",
            },
            "strings_1": {
                0: "this_is_a_string",
                1: "this_is_another_string",
                2: "this_is_a_third_string",
                3: "this_is_a_fourth_string",
                4: "this_is_a_fifth_string",
                5: "this_is_a_sixth_string",
            },
            "strings_2": {
                0: "this_is_not_a_string",
                1: "this_is_another_string",
                2: "this is a third string",
                3: "this_is_a_fifth_string",
                4: " ",
                5: "!@#$%^&*()_+",
            },
            "email_domain": {
                0: "test1",
                1: "test2",
                2: "test3",
                3: "test4",
                4: "test5",
                5: np.nan,
            },
            "email_num_of_digits": {0: 0, 1: 3, 2: 3, 3: 0, 4: 0, 5: 0},
            "email_num_of_letters": {0: 4, 1: 4, 2: 4, 3: 8, 4: 3, 5: 12},
            "email_num_of_special_chars": {0: 0, 1: 0, 2: 3, 3: 1, 4: 0, 5: 2},
            "email_num_of_repeated_chars": {0: 1, 1: 1, 2: 2, 3: 1, 4: 3, 5: 1},
            "email_num_of_words": {0: 1, 1: 1, 2: 2, 3: 2, 4: 1, 5: 3},
        }
    )
    assert result[expected.columns].equals(expected)
    assert pipeline.steps[0][0] == "emailtransformer"


def test_phone_number_transformer(X_numbers):
    pipeline = make_pipeline(PhoneTransformer(["phone_number"]))

    X = pipeline.fit_transform(X_numbers)
    expected_national_number = np.array(
        [1.763456123, -999.0, 4.045654449, -999.0, -999.0]
    )
    expected_country_code = np.array([0.49, -999.0, 0.49, -999.0, -999.0])

    assert pipeline.steps[0][0] == "phonetransformer"
    assert np.array_equal(
        X["phone_number_national_number"].to_numpy(), expected_national_number
    )
    assert np.array_equal(
        X["phone_number_country_code"].to_numpy(), expected_country_code
    )


def test_string_similarity_transformer_in_pipeline(X_strings):
    pipeline = make_pipeline(StringSimilarityTransformer(("strings_1", "strings_2")))
    result = pipeline.fit_transform(X_strings)
    expected = np.array(
        [
            0.8888888888888888,
            1.0,
            0.8181818181818182,
            0.8888888888888888,
            0.0,
            0.058823529411764705,
        ]
    )
    assert np.array_equal(result["strings_1_strings_2_similarity"].to_numpy(), expected)
    assert pipeline.steps[0][0] == "stringsimilaritytransformer"


def test_string_slicer_transformer_in_pipeline(X_strings):
    pipeline = make_pipeline(
        StringSlicerTransformer(
            [
                ("email", (5,)),
                ("strings_1", (8, 16)),
            ]
        )
    )
    result = pipeline.fit_transform(X_strings)

    expected = pd.DataFrame(
        {
            "email": [
                "test@",
                "test1",
                "test_",
                "test_",
                "ttt@t",
                "test_",
            ],
            "strings_1": [
                "a_string",
                "another_",
                "a_third_",
                "a_fourth",
                "a_fifth_",
                "a_sixth_",
            ],
        }
    )

    assert pipeline.steps[0][0] == "stringslicertransformer"
    assert result[["email", "strings_1"]].equals(expected)


def test_string_slicer_transformer_new_column_name_in_pipeline(X_strings):
    pipeline = make_pipeline(
        StringSlicerTransformer(
            [
                ("email", (5,), "new_email_slice"),
            ]
        )
    )
    result = pipeline.fit_transform(X_strings)

    assert "new_email_slice" in result.columns
    assert pipeline.steps[0][0] == "stringslicertransformer"


def test_string_slicer_transformer_tuple_size_raise_warning(X_strings) -> None:
    with pytest.warns(UserWarning) as warning:
        transformer = StringSlicerTransformer([("email", (5, 10, 2))])
        _ = transformer.fit_transform(X_strings)
    assert str(warning[0].message) == (
        "StringSlicerTransformer currently does not support increments.\n Only the first two elements of the slice tuple will be considered."
    )


def test_string_splitter_transformer_in_pipeline(X_strings):
    pipeline = make_pipeline(
        StringSplitterTransformer(
            [
                ("strings_2", "_", 1),
            ]
        )
    )
    result = pipeline.fit_transform(X_strings)
    expected_part_1 = [
        "this",
        "this",
        "this is a third string",
        "this",
        " ",
        "!@#$%^&*()",
    ]
    expected_part_2 = [
        "is_not_a_string",
        "is_another_string",
        None,
        "is_a_fifth_string",
        None,
        "+",
    ]

    assert np.array_equal(result["strings_2_part_1"], expected_part_1)
    assert np.array_equal(result["strings_2_part_2"], expected_part_2)
    assert pipeline.steps[0][0] == "stringsplittertransformer"


def test_string_splitter_transformer_no_maxsplits_in_pipeline(X_strings):
    pipeline = make_pipeline(
        StringSplitterTransformer(
            [
                ("strings_2", "_"),
            ]
        )
    )
    result = pipeline.fit_transform(X_strings)

    assert "strings_2_part_5" in result.columns
    assert pipeline.steps[0][0] == "stringsplittertransformer"


def test_string_splitter_transformer_zero_maxsplits_in_pipeline(X_strings):
    pipeline = make_pipeline(
        StringSplitterTransformer(
            [
                ("strings_2", "_", 0),
            ]
        )
    )
    result = pipeline.fit_transform(X_strings)

    assert "strings_2_part_5" in result.columns
    assert pipeline.steps[0][0] == "stringsplittertransformer"


def test_string_combination_transformer_in_pipeline(X_categorical):
    pipeline = make_pipeline(StringCombinationTransformer([("a", "e", "_")]))
    result = pipeline.fit_transform(X_categorical)
    expected = pd.Series(
        ["A_A1", "A2_B", "A2_C", "A1_D", "A1_E", "A2_F", "A1_G", "A1_H"],
        name="a_e_combi",
    )
    assert np.array_equal(result["a_e_combi"], expected)
    assert pipeline.steps[0][0] == "stringcombinationtransformer"
