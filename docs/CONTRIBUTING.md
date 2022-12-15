# How to contribute

First of all, thank you 🙏 for your interest in contributing to this project. This project is open source and is therefore open to contributions from the community. The following document describes how you can contribute to it.

## Scikit-learn pipelines
[Scikit-learn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html?highlight=pipelines) are really handy. the concept is simple: chain multiple preprocessing steps together and apply them to the data. Packages like [Sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) help you even more.
Check out this dummy example of how to create a custom transformer ready for use in a pipeline:

```python
import pandas as pd

from sk_transformers.base_transformer import BaseTransformer
from sk_transformers.utils import check_ready_to_transform

class DummyTransformer(BaseTransformer):
    """
    Replaces all strings in a given column with `dummy`.

    Args:
        string_to_replace (str): The string which should be replaced by `dummy`.
        column (str): The column to replace the strings with dummy.
    """
    def __init__(self, string_to_replace: str, column: str) -> None:
        self.string_to_replace = string_to_replace
        self.column = column

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
         Replaces all occurrences of `string_to_replace`
         in a certain column of X with `DUMMY!`.

        Args:
            X (pd.DataFrame): Dataframe containing the
            column where the replacement should happen.

        Returns:
            pandas.DataFrame: Dataframe with replaced strings.
        """
        X = check_ready_to_transform(self, X, self.column)

        X[self.column] = X[self.column].replace(self.string_to_replace, "DUMMY!")
        return X
```
More documentation than code. You know how it is 🤷‍♂️. This transformer does not need an `fit` method because it does not learn anything from the data. It just replaces a string with another string.

Now you can use it:
```python
import pandas as pd
from sklearn.pipeline import Pipeline

df = pd.DataFrame({
    "cocktail": ["French Connection", "Incredible Hulk", "Tom and Jerry"],
    "bar": ["foo", "Schikaneder", "Futuregarden"]
})

pipeline = Pipeline([
    ("dummy_transformer", DummyTransformer("foo", "bar")),
])

print(pipeline.fit_transform(df).head())
```
```
            cocktail           bar
0  French Connection        DUMMY!
1    Incredible Hulk   Schikaneder
2      Tom and Jerry  Futuregarden
```
For a non-dummy examples check out the [`MathExpressionTransformer`](number_transformer.md#sk-transformers.transformer.number_transformer.MathExpressionTransformer) or the [`ValueIndicatorTransformer`](generic_transformer.md#sk-transformers.transformer.generic_transformer.ValueIndicatorTransformer) for a simpler example.

## Poetry
We are using [Poetry](https://python-poetry.org/) to manage the dependencies and the virtual environment. If you have not used it before please check out the [documentation](https://python-poetry.org/docs/) to get started.

## How to contribute
The following steps will give a short guid on how to contribute to this project:

- Create a personal [fork](https://github.com/chrislemke/feature-reviser/fork) of the project on [GitHub](https://github.com/).
- Clone the fork on your local machine. Your remote repo on [GitHub](https://github.com/) is called `origin`.
- Add the original repository as a remote called `upstream`.
- If you created your fork a while ago be sure to pull upstream changes into your local repository.
- Create a new branch to work on! Branch from `develop` if it exists, else from `main`.
- Implement/fix your feature, and comment your code.
- Follow the code style of the project, including indentation. [Black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort), [Pylint](https://github.com/PyCQA/pylint), and [mypy](https://github.com/python/mypy) can help you with it.
- If the project has tests run them.
- Write or adapt tests as needed.
- Add or change the documentation as needed. Please follow the "[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)".
- Squash your commits into a single commit with git's [interactive rebase](https://help.github.com/articles/interactive-rebase). Create a new branch if necessary.
- We are using pre-commit-hooks to ensure a consistent code style and to avoid common mistakes. Please install the [pre-commit](https://pre-commit.com/#installation) and install the hook with `pre-commit install`.
- Push your branch to your fork on [GitHub](https://github.com/), the remote `origin`.
- From your fork open a pull request in the correct branch. Target the project's `develop` branch if there is one, else go for `main`!
- Once the pull request is approved and merged you can pull the changes from `upstream` to your local repo and delete
your extra branch(es).
