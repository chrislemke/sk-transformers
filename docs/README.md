![The Transformer](https://raw.githubusercontent.com/chrislemke/sk-transformers/master/docs/assets/images/icon.png)

# sk-transformers
***A collection of various pandas & scikit-learn compatible transformers for all kinds of preprocessing and feature engineering steps*** 🛠

[![ChecksAndTesting](https://github.com/chrislemke/sk-transformers/actions/workflows/checks-testing.yml/badge.svg)](https://github.com/chrislemke/sk-transformers/actions/workflows/checks-testing.yml)
[![codecov](https://codecov.io/github/chrislemke/sk-transformers/branch/main/graph/badge.svg?token=LJLXQXX6M8)](https://codecov.io/github/chrislemke/sk-transformers)
[![Release](https://github.com/chrislemke/sk-transformers/actions/workflows/release.yml/badge.svg)](https://github.com/chrislemke/sk-transformers/actions/workflows/release.yml)
[![pypi](https://img.shields.io/pypi/v/sk-transformers)](https://pypi.org/project/sk-transformers/)
[![python version](https://img.shields.io/pypi/pyversions/sk-transformers?logo=python&logoColor=yellow)](https://www.python.org/)
[![downloads](https://img.shields.io/pypi/dm/sk-transformers)](https://pypistats.org/packages/sk-transformers)
[![docs](https://img.shields.io/badge/docs-mkdoks%20material-blue)](https://chrislemke.github.io/sk-transformers/)
[![license](https://img.shields.io/github/license/chrislemke/sk-transformers)](https://github.com/chrislemke/sk-transformers/blob/main/LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](https://github.com/python/mypy)
## Introduction
Every tabular data is different. Every column needs to be treated differently. Pandas is already great! And [scikit-learn](https://scikit-learn.org/stable/index.html) has a nice [collection of dataset transformers](https://scikit-learn.org/stable/data_transforms.html). But the possibilities of data transformation are infinite. This project tries to provide a brought collection of data transformers that can be easily used together with [scikit-learn](https://scikit-learn.org/stable/index.html) - either in a [pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) or just on its own. See the [usage chapter](#usage) for some examples.

The idea is simple. It is like a well-equipped toolbox 🧰: You always find the tool you need and sometimes you get inspired by seeing a tool you did not know before. Please feel free to [contribute](https://chrislemke.github.io/sk-transformers/CONTRIBUTING/) your tools and ideas.

Check out some examples in the [Jupyter notebook](https://github.com/chrislemke/sk-transformers/blob/main/examples/playground.ipynb).<br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chrislemke/sk-transformers/blob/main/examples/playground.ipynb)

## Installation
If you are using [pip](https://pip.pypa.io/en/stable/), you can install the package with the following command:
```bash
pip install sk-transformers
```

If you are using [Poetry](https://python-poetry.org/), you can install the package with the following command:
```bash
poetry add sk-transformers
```

## installing dependencies
With [Poetry](https://python-poetry.org/):
```bash
poetry install
```

## Available transformers
| Module | Transformer | Description |
| ------ | ----------- | ----------- |
|[`Datetime transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/datetime_transformer/)|[`DateColumnsTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/datetime_transformer/#sk_transformers.datetime_transformer.DateColumnsTransformer)|Splits a date column into multiple columns.|
|[`Datetime transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/datetime_transformer/)|[`DurationCalculatorTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/datetime_transformer/#sk_transformers.datetime_transformer.DurationCalculatorTransformer)|Calculates the duration between to given dates.|
|[`Encoder transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/encoder_transformer/)|[`MeanEncoderTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/encoder_transformer/#sk_transformers.encoder_transformer.MeanEncoderTransformer)|Scikit-learn API for the [feature-engine MeanEncoder](https://feature-engine.readthedocs.io/en/latest/api_doc/encoding/MeanEncoder.html).|
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`AggregateTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.AggregateTransformer)|This transformer uses Pandas groupby method and aggregate to apply function on a column grouped by another column.|
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`AllowedValuesTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.AllowedValuesTransformer)|This transformer replaces values that are *not* in a list with another value.|
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`ColumnDropperTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.ColumnDropperTransformer)|Drops columns from a dataframe using Pandas drop method.|
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`ColumnEvalTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.ColumnEvalTransformer)|Provides the possibility to use Pandas methods on columns.|
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`DtypeTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.DtypeTransformer)|Transformer that converts a column to a different dtype.|
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`FunctionsTransformer`]( https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.FunctionsTransformer)|This transformer is a plain wrapper around the [sklearn.preprocessing.FunctionTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html).|
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`LeftJoinTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.LeftJoinTransformer)|Uses Pandas merge function to perform a left-join based on the column of a dataframe and the index of another dataframe. The right dataframe is essentially a lookup table.|
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`MapTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.MapTransformer)|This transformer iterates over all columns in the `features` list and applies the given callback to the column. For this it uses the `pandas.Series.map` method.
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`NaNTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.NaNTransformer)|Replace NaN values with a specified value. Internally Pandas fillna method is used.|
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`QueryTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.QueryTransformer)|Applies a list of queries to a dataframe. If it operates on a dataset used for supervised learning this transformer should be applied on the dataframe containing `X` and `y`.
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`ValueIndicatorTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.ValueIndicatorTransformer)|Adds a column to a dataframe indicating if a value is equal to a specified value.|
|[`Generic transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/)|[`ValueReplacerTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/generic_transformer/#sk_transformers.generic_transformer.ValueReplacerTransformer)|Uses Pandas replace method to replace values in a column.|
[`Number transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/number_transformer/)|[`MathExpressionTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/number_transformer/#sk_transformers.number_transformer.MathExpressionTransformer)|Applies an operation to a column and a given value or column. The operation can be any operation from the `numpy` or `operator` package.
[`String transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/)|[`EmailTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/#sk_transformers.string_transformer.EmailTransformer)|Transforms an email address into multiple features.|
[`String transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/)|[`IPAddressEncoderTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/#sk_transformers.string_transformer.IPAddressEncoderTransformer)|Encodes IPv4 and IPv6 strings addresses to a float representation.|
[`String transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/)|[`PhoneTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/#sk_transformers.string_transformer.PhoneTransformer)|Transforms a phone number into multiple features.|
[`String transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/)|[`StringSimilarityTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/#sk_transformers.string_transformer.StringSimilarityTransformer)|Calculates the similarity between two strings using the `gestalt pattern matching` algorithm from the `SequenceMatcher` class.|
[`String transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/)|[`StringSlicerTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/#sk_transformers.string_transformer.StringSlicerTransformer)|Slices all entries of specified string features using the slice() function.|
[`String transformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/)|[`StringSplitterTransformer`](https://chrislemke.github.io/sk-transformers/API-reference/transformer/string_transformer/#sk_transformers.string_transformer.StringSplitterTransformer)|Splits a string column into multiple columns based on the occurrence of a character.|

## Usage
Let's assume you want to use some method from [NumPy's mathematical functions, to sum up the values of column `foo` and column `bar`. You could
use the [`MathExpressionTransformer`](https://chrislemke.github.io/sk-transformers/number_transformer-reference/#sk-transformers.transformer.number_transformer.MathExpressionTransformer).
```python
import pandas as pd
from sk_transformers import MathExpressionTransformer

X = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
transformer = MathExpressionTransformer([("foo", "np.sum", "bar", {"axis": 0})])
transformer.fit_transform(X).to_numpy()
```
```
array([[1, 4, 5],
       [2, 5, 7],
       [3, 6, 9]])
```
Even if we only pass one tuple to the transformer - in this example. Like with most other transformers the idea is to simplify preprocessing by giving the possibility to operate on multiple columns at the same time. In this case, the [`MathExpressionTransformer`](https://chrislemke.github.io/sk-transformers/number_transformer-reference/#sk-transformers.transformer.number_transformer.MathExpressionTransformer) has created an extra column with the name `foo_sum_bar`.

In the next example, we additionally add the [`MapTransformer`](https://chrislemke.github.io/sk-transformers/generic_transformer-reference/#sk_transformers.transformer.generic_transformer.MapTransformer).
Together with [scikit-learn's pipelines](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) it would look like this:
```python
import pandas as pd
from sk_transformers import MathExpressionTransformer
from sk_transformers import MapTransformer
from sklearn.pipeline import Pipeline

X = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
map_step = MapTransformer([("foo", lambda x: x + 100)])
sum_step = MathExpressionTransformer([("foo", "np.sum", "bar", {"axis": 0})])
pipeline = Pipeline([("map_step", map_step), ("sum_step", sum_step)])
pipeline.fit_transform(X)
```

```
   foo  bar  foo_sum_bar
0  101    4          105
1  102    5          107
2  103    6          109
```

## Contributing
We're all kind of in the same boat. Preprocessing/feature engineering in data science is somehow very individual - every feature is different and must be handled and processed differently. But somehow we all have the same problems: sometimes date columns have to be changed. Sometimes strings have to be formatted, sometimes durations have to be calculated, etc. There is a huge number of preprocessing possibilities but we all use the same tools.

[scikit-learns pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) help to use formalized functions. So why not also share these so-called transformers with others? This open-source project has the goal to collect useful preprocessing pipeline steps. Let us all collect what we used for preprocessing and share it with others. This way we can all benefit from each other's work and save a lot of time. So if you have a preprocessing step that you use regularly, please feel free to contribute it to this project. The idea is that this is not only a toolbox but also an inspiration for what is possible. Maybe you have not thought about this preprocessing step before.

Please check out the [guide](https://chrislemke.github.io/sk-transformers/CONTRIBUTING/) on how to contribute to this project.
