# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline

from feature_reviser import (
    EmailTransformer,
    IPAddressEncoderTransformer,
    PhoneTransformer,
    StringSimilarityTransformer,
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
    assert np.array_equal(X["ip_address"].values, expected)


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
    assert result.equals(expected)
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
        X["phone_number_national_number"].values, expected_national_number
    )
    assert np.array_equal(X["phone_number_country_code"].values, expected_country_code)


def test_string_similarity_transformer_in_pipeline(X_strings):
    pipeline = make_pipeline(StringSimilarityTransformer(["strings_1", "strings_2"]))
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
    assert np.array_equal(result["strings_1_strings_2_similarity"].values, expected)
    assert pipeline.steps[0][0] == "stringsimilaritytransformer"
