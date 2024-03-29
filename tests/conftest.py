import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# pylint: disable=missing-function-docstring


@pytest.fixture()
def clf() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=42)


@pytest.fixture()
def ordinal_encoder() -> OrdinalEncoder:
    return OrdinalEncoder()


@pytest.fixture()
def standard_scaler() -> StandardScaler:
    return StandardScaler()


@pytest.fixture()
def X() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
            "c": ["1", "1", "1", "1", "2", "2", "2", "3", "3", "4"],
            "d": ["5", "5", "5", "6", "6", "6", "6", "7", "7", "7"],
            "e": ["5", "5", "5", "5", "5", "5", "5", "5", "7", "7"],
        }
    )


@pytest.fixture()
def X_group_by() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": ["mr", "mr", "ms", "ms", "ms", "mr", "mr", "mr", "mr", "ms"],
            "b": [46, 32, 78, 48, 93, 68, 53, 38, 76, 56],
            "c": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
            "d": ["foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz", "foo"],
        }
    )


@pytest.fixture()
def X_time_values() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [
                "1960-01-01",
                "1970-01-01",
                "1970-01-02",
                "2022-01-04",
                "2022-01-05",
                "2022-01-06",
                "2022-01-07",
                "2022-01-08",
                "2022-01-09",
                "2022-01-10",
            ],
            "c": [
                "1960-01-01",
                "1970-01-01",
                "1971-01-02",
                "2023-01-04",
                "2022-02-05",
                "2022-02-06",
                "2022-01-08",
                "2022-01-09",
                "1960-01-01",
                "2100-01-01",
            ],
            "dd": [
                "0000-01-01",
                "\\N",
                "1971-01-00",
                "foo",
                "2022.02.05",
                "06-02-2022",
                "2022/01/08",
                "2022-01-09",
                "1960-01-01",
                "10000-01-01",
            ],
            "e": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "f": ["2", "4", "6", "8", "\\N", "12", "14", "16", "18", "20"],
        }
    )


@pytest.fixture()
def tiny_date_df() -> pd.DataFrame:
    return pd.DataFrame({"a": ["2021-01-01", "2022-02-02", "2023-03-03"]})


@pytest.fixture()
def X_nan_values() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1, None, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [1.1, 2.2, None, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
            "c": ["A", "B", "C", "D", "E", "F", None, "H", "I", "J"],
            "d": [1, 2, 3, 4, 5, -999, 7, 8, 9, 10],
            "e": ["A", "B", "C", "D", "E", "F", "-999", "H", "I", "J"],
        }
    )


@pytest.fixture()
def X_numbers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "big_numbers": [
                10_000_000,
                12_891_207,
                42_000,
                11_111_111,
                99_999_999_999_999,
            ],
            "small_numbers": [7, 12, 82, 1, 0],
            "small_float_numbers": [4.5, 3.5, 6.9, 1.9, 0.6],
            "ip_address": [
                "203.0.113.195",
                "192.168.1.1",
                "2001:0db8:3c4d:0015:0000:0000:1a2f:1a2b",
                "2001:0db8:85a3:08d3:1319:8a2e:0370:7344",
                "just_a_string",
            ],
            "phone_number": [
                "+491763456123",
                "030123456",
                "00494045654449",
                "01764547310",
                "+00491764547310",
            ],
            "time_in_seconds": ["917634", "30123", "49404565", "17645", "46787"],
        }
    )


@pytest.fixture()
def X_strings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email": [
                "test@test1.com",
                "test123@test2.com",
                "test_123$$@test3.com",
                "test_test@test4.com",
                "ttt@test5.com",
                "test_test_test",
            ],
            "strings_1": [
                "this_is_a_string",
                "this_is_another_string",
                "this_is_a_third_string",
                "this_is_a_fourth_string",
                "this_is_a_fifth_string",
                "this_is_a_sixth_string",
            ],
            "strings_2": [
                "this_is_not_a_string",
                "this_is_another_string",
                "this is a third string",
                "this_is_a_fifth_string",
                " ",
                "!@#$%^&*()_+",
            ],
        }
    )


@pytest.fixture()
def y() -> pd.Series:
    return pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])


@pytest.fixture()
def X_categorical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": ["A1", "A2", "A2", "A1", "A1", "A2", "A1", "A1"],
            "b": [1, 2, 3, 4, 5, 6, 7, 8],
            "c": [1, 2, 3, 1, 2, 3, 1, 3],
            "d": [1.1, 2, 3, 4, 5, 6, 7, 8],
            "e": ["A", "B", "C", "D", "E", "F", "G", "H"],
        }
    )


@pytest.fixture()
def y_categorical() -> pd.Series:
    return pd.Series([1, 1, 0, 1, 0, 0, 0, 0])


@pytest.fixture()
def X_coordinates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude_1": [
                52.380001,
                50.033333,
                48.353802,
                48.353802,
                51.289501,
                53.63040161,
            ],
            "longitude_1": [
                13.5225,
                8.570556,
                11.7861,
                11.7861,
                6.76678,
                9.988229752,
            ],
            "latitude_2": [
                50.033333,
                52.380001,
                48.353802,
                51.289501,
                53.63040161,
                48.353802,
            ],
            "longitude_2": [
                8.570556,
                13.5225,
                11.7861,
                6.76678,
                9.988229752,
                11.7861,
            ],
        }
    )
