import functools
import ipaddress
import itertools
import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple, Union

import pandas as pd
import phonenumbers

from sk_transformers.base_transformer import BaseTransformer
from sk_transformers.utils import check_ready_to_transform


class IPAddressEncoderTransformer(BaseTransformer):
    """
    Encodes IPv4 and IPv6 strings addresses to a float representation.
    To shrink the values to a reasonable size IPv4 addresses are divided by 2^10 and IPv6 addresses are divided by 2^48.
    Those values can be changed using the `ip4_divisor` and `ip6_divisor` parameters.

    Args:
        features (List[str]): List of features which should be transformed.
        ip4_divisor (float): Divisor for IPv4 addresses.
        ip6_divisor (float): Divisor for IPv6 addresses.
        error_value (Union[int, float]): Value if parsing fails.
    """

    def __init__(
        self,
        features: List[str],
        ip4_divisor: float = 1e10,
        ip6_divisor: float = 1e48,
        error_value: Union[int, float] = -999,
    ) -> None:
        super().__init__()
        self.features = features
        self.ip4_divisor = ip4_divisor
        self.ip6_divisor = ip6_divisor
        self.error_value = error_value

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the column containing the IP addresses to float column.

        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Transformed dataframe.
        """

        X = check_ready_to_transform(self, X, self.features)

        function = functools.partial(
            IPAddressEncoderTransformer.__to_float,
            self.ip4_divisor,
            self.ip6_divisor,
            self.error_value,
        )
        for column in self.features:
            X[column] = X[column].map(function)

        return X

    @staticmethod
    def __to_float(
        ip4_devisor: float,
        ip6_devisor: float,
        error_value: Union[int, float],
        ip_address: str,
    ) -> float:
        try:
            return int(ipaddress.IPv4Address(ip_address)) / int(ip4_devisor)
        except:  # pylint: disable=bare-except
            try:
                return int(ipaddress.IPv6Address(ip_address)) / int(ip6_devisor)
            except:  # pylint: disable=bare-except
                return error_value


class EmailTransformer(BaseTransformer):
    """
    Transforms an email address into multiple features.

    Args:
        features (List[str]): List of features which should be transformed.
    """

    def __init__(self, features: List[str]) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the one column from X, containing the email addresses, into multiple columns.

        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Transformed dataframe containing the extra columns.
        """

        X = check_ready_to_transform(self, X, self.features)

        for column in self.features:

            X[f"{column}_domain"] = (
                X[column].str.split("@").str[1].str.split(".").str[0]
            )

            X[column] = X[column].str.split("@").str[0]

            X[f"{column}_num_of_digits"] = X[column].map(
                EmailTransformer.__num_of_digits
            )
            X[f"{column}_num_of_letters"] = X[column].map(
                EmailTransformer.__num_of_letters
            )
            X[f"{column}_num_of_special_chars"] = X[column].map(
                EmailTransformer.__num_of_special_characters
            )
            X[f"{column}_num_of_repeated_chars"] = X[column].map(
                EmailTransformer.__num_of_repeated_characters
            )
            X[f"{column}_num_of_words"] = X[column].map(EmailTransformer.__num_of_words)
        return X

    @staticmethod
    def __num_of_digits(string: str) -> int:
        return sum(map(str.isdigit, string))

    @staticmethod
    def __num_of_letters(string: str) -> int:
        return sum(map(str.isalpha, string))

    @staticmethod
    def __num_of_special_characters(string: str) -> int:
        return len(re.findall(r"[^A-Za-z0-9]", string))

    @staticmethod
    def __num_of_repeated_characters(string: str) -> int:
        return max(len("".join(g)) for _, g in itertools.groupby(string))

    @staticmethod
    def __num_of_words(string: str) -> int:
        return len(re.findall(r"[.\-_]", string)) + 1


class StringSimilarityTransformer(BaseTransformer):
    """
    Calculates the similarity between two strings using the `gestalt pattern matching` algorithm from the `SequenceMatcher` class.
    Args:
        features (Tuple[str, str]): The two columns that contain the strings for which the similarity should be calculated.
    """

    def __init__(self, features: Tuple[str, str]) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the similarity of two strings provided in `features`.

        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Original dataframe containing the extra column with the calculated similarity.
        """

        X = check_ready_to_transform(self, X, list(self.features))

        X[f"{self.features[0]}_{self.features[1]}_similarity"] = X[
            [self.features[0], self.features[1]]
        ].apply(
            lambda x: StringSimilarityTransformer.__similar(
                StringSimilarityTransformer.__normalize_string(x[self.features[0]]),
                StringSimilarityTransformer.__normalize_string(x[self.features[1]]),
            ),
            axis=1,
        )
        return X

    @staticmethod
    def __similar(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def __normalize_string(string: str) -> str:
        string = str(string).strip().lower()
        return (
            unicodedata.normalize("NFKD", string)
            .encode("utf8", "strict")
            .decode("utf8")
        )


class PhoneTransformer(BaseTransformer):
    """
    Transforms a phone number into multiple features.

    Args:
        features (List[str]): List of features which should be transformed.
        national_number_divisor (float): Divider `national_number`.
        country_code_divisor (flat): Divider for `country_code`.
        error_value (str): Value to use if the phone number is invalid or the parsing fails.
    """

    def __init__(
        self,
        features: List[str],
        national_number_divisor: float = 1e9,
        country_code_divisor: float = 1e2,
        error_value: str = "-999",
    ) -> None:
        super().__init__()
        self.features = features
        self.national_number_divisor = national_number_divisor
        self.country_code_divisor = country_code_divisor
        self.error_value = error_value

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the similarity of two strings provided in `features`.

        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Original dataframe containing the extra column with the calculated similarity.
        """

        X = check_ready_to_transform(self, X, self.features)

        for column in self.features:

            X[f"{column}_national_number"] = X[column].apply(
                lambda x: PhoneTransformer.__phone_to_float(
                    "national_number",
                    x,
                    int(self.national_number_divisor),
                    self.error_value,
                )
            )
            X[f"{column}_country_code"] = X[column].apply(
                lambda x: PhoneTransformer.__phone_to_float(
                    "country_code", x, int(self.country_code_divisor), self.error_value
                )
            )

        return X

    @staticmethod
    def __phone_to_float(
        attribute: str, phone: str, divisor: int, error_value: str
    ) -> float:
        phone = phone.replace(" ", "")
        phone = re.sub(r"[^0-9+-]", "", phone)
        phone = re.sub(r"^00", "+", phone)
        try:
            return float(getattr(phonenumbers.parse(phone, None), attribute)) / divisor
        except:  # pylint: disable=W0702
            try:
                return float(re.sub(r"(?<!^)[^0-9]", "", error_value))
            except:  # pylint: disable=W0702
                return float(error_value)


class StringSlicerTransformer(BaseTransformer):
    """
    Slices all entries of specified string features using the `slice()` function.

    Note: The arguments for the `slice()` function are passed as a tuple. This shares
    the python quirk of writing a tuple with a single argument with the trailing comma.

    Example:
    ```python
    import pandas as pd
    from sk_transformers.string_transformer import StringSlicerTransformer

    X = pd.DataFrame({"foo": ["abc", "def", "ghi"], "bar": ["jkl", "mno", "pqr"]})
    transformer = StringSlicerTransformer([("foo", (0, 3, 2)), ("bar", (2,))])
    transformer.fit_transform(X).to_numpy()
    ```
    ```
    array([['ac', 'jk'],
            ['df', 'mn'],
            ['gi', 'pq']], dtype=object)
    ```

    Args:
        features (List[Tuple[str, Tuple[int, int, int]]]): The arguments to the `slice` function, for each feature.
    """

    def __init__(
        self,
        features: List[
            Tuple[
                str,
                Union[Tuple[int], Tuple[int, int], Tuple[int, int, int]],
            ]
        ],
    ) -> None:
        super().__init__()
        self.features = features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Slices the strings of specified features in the dataframe.

        Args:
            X (pandas.DataFrame): DataFrame to transform.

        Returns:
            pandas.DataFrame: Original dataframe with sliced strings in specified features.
        """

        X = check_ready_to_transform(self, X, [feature[0] for feature in self.features])

        for feature, slice_args in self.features:
            X[feature] = [x[slice(*slice_args)] for x in X[feature]]

        return X
