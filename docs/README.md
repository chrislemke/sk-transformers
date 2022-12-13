![The machine](https://raw.githubusercontent.com/chrislemke/sk-transformers/master/docs/assets/images/image.png)

# sk-transformers
### A collection of various scikit-learn transformers for all kinds of preprocessing and feature engineering steps 🛠

[![testing](https://github.com/chrislemke/sk-transformers/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/chrislemke/sk-transformers/actions/workflows/testing.yml)
[![codecov](https://codecov.io/github/chrislemke/sk-transformers/branch/main/graph/badge.svg?token=LJLXQXX6M8)](https://codecov.io/github/chrislemke/sk-transformers)
[![deploy package](https://github.com/chrislemke/sk-transformers/actions/workflows/deploy-package.yml/badge.svg)](https://github.com/chrislemke/sk-transformers/actions/workflows/deploy-package.yml)
[![pypi](https://img.shields.io/pypi/v/sk-transformers)](https://pypi.org/project/sk-transformers/)
![python version](https://img.shields.io/pypi/pyversions/sk-transformers?logo=python&logoColor=yellow)
[![downloads](https://img.shields.io/pypi/dm/sk-transformers)](https://pypistats.org/packages/sk-transformers)
[![docs](https://img.shields.io/badge/docs-mkdoks%20material-blue)](https://chrislemke.github.io/sk-transformers/)
[![license](https://img.shields.io/github/license/chrislemke/sk-transformers)](https://github.com/chrislemke/sk-transformers/blob/main/LICENSE)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
## Introduction
Every data tabular is different. Every column needs to be treated differently. [Scikit-learn](https://scikit-learn.org/stable/index.html) has a nice [collection of dataset transformers](https://scikit-learn.org/stable/data_transforms.html). But the possibilities of data transformation are infinite - one collection is simply not enough. This project provides a brought collection of data transformers. The idea is simple. It is like a well-equipped toolbox 🧰: You always find the tool you need and sometimes you get inspired by seeing a tool you did not know before. Please feel free to [contribute](https://chrislemke.github.io/sk-transformers/CONTRIBUTING/) your tools and ideas.

## Installation
If you are using [Poetry](https://python-poetry.org/), you can install the package with the following command:
```bash
poetry add sk_transformers
```
If you are using [pip](https://pypi.org/project/pip/), you can install the package with the following command:
```bash
pip install sk_transformers
```

## installing dependencies
With [Poetry](https://python-poetry.org/):
```bash
poetry install
```
With [pip](https://pypi.org/project/pip/):
```bash
pip install -r requirements.txt
```

## The transformers
Data preprocessing often involves similar processes. No matter whether it's manipulating strings or numbers, etc. [Scikit-learn's pipeline](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) implementation makes it easy to structure and sequence such preprocessing processes. To take advantage of this, the [`transformers`](https://github.com/chrislemke/sk-transformers/tree/main/sk-transformers/transformer) contain multiple methods that can be easily pipelined to simplify preprocessing. The list of transformers is open and will be extended permanently. Feel free to [contribute](https://chrislemke.github.io/sk-transformers/CONTRIBUTING/)! 🛠

### Usage
Let's assume you want to use some method from [NumPy's mathematical functions](https://numpy.org/doc/stable/reference/routines.math.html), to sum up the values of column `foo` and column `bar`. You could
use the [`MathExpressionTransformer`](https://chrislemke.github.io/sk-transformers/number_transformer-reference/#sk-transformers.transformer.number_transformer.MathExpressionTransformer):
```python
from sk_transformers import MathExpressionTransformer
import pandas as pd
X = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
transformer = MathExpressionTransformer([("foo", "np.sum", "bar", {"axis": 0})])
transformer.fit_transform(X).to_numpy()
```
```
array([[1, 4, 5],
       [2, 5, 7],
       [3, 6, 9]])
```

Together with [Scikit-learn's pipelines](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) it would look like this:
```python
from sklearn.pipeline import Pipeline
from sk_transformers import MapTransformer, MathExpressionTransformer
import pandas as pd
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

Even if we only pass one tuple to the transformer - in this example. Like with most other transformers the idea is to simplify preprocessing by giving the possibility to operate on multiple columns at the same time. In this case, the [`MathExpressionTransformer`](https://chrislemke.github.io/sk-transformers/number_transformer-reference/#sk-transformers.transformer.number_transformer.MathExpressionTransformer) has created an extra column with the name `foo_sum_bar`.

## Contributing
We're all kind of in the same boat. Preprocessing/feature engineering in data science is somehow very individual - every feature is different and must be handled and processed differently. But somehow we all have the same problems: sometimes date columns have to be changed. Sometimes strings have to be formatted, sometimes durations have to be calculated, etc. There is a huge number of preprocessing possibilities but we all use the same tools.

[Scikit-learns pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) help to use formalized functions. So why not also share these so-called transformers with others? This open-source project has the goal to collect useful preprocessing pipeline steps. Let us all collect what we used for preprocessing and share it with others. This way we can all benefit from each other's work and save a lot of time. So if you have a preprocessing step that you use regularly, please feel free to contribute it to this project. The idea is that this is not only a toolbox but also an inspiration for what is possible. Maybe you have not thought about this preprocessing step before.

Please check out the [guide](https://chrislemke.github.io/sk-transformers/CONTRIBUTING/) on how to contribute to this project.

## Further information
For further information, please refer to the [documentation](https://chrislemke.github.io/sk-transformers/).
