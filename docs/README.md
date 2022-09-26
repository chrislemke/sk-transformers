![The machine](https://raw.githubusercontent.com/chrislemke/feature-reviser/master/docs/assets/images/image.png)

# feature-reviser

[![testing](https://github.com/chrislemke/feature-reviser/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/chrislemke/feature-reviser/actions/workflows/testing.yml)
[![codecov](https://codecov.io/github/chrislemke/feature-reviser/branch/main/graph/badge.svg?token=LJLXQXX6M8)](https://codecov.io/github/chrislemke/feature-reviser)
[![deploy package](https://github.com/chrislemke/feature-reviser/actions/workflows/deploy-package.yml/badge.svg)](https://github.com/chrislemke/feature-reviser/actions/workflows/deploy-package.yml)
[![release](https://img.shields.io/github/v/release/chrislemke/feature-reviser?include_prereleases)](https://github.com/chrislemke/feature-reviser/releases)
[![pypi](https://img.shields.io/pypi/v/feature-reviser)](https://pypi.org/project/feature-reviser/)
[![downloads](https://img.shields.io/pypi/dm/feature-reviser)](https://pypistats.org/packages/feature-reviser)

![python version](https://img.shields.io/pypi/pyversions/feature-reviser?logo=python&logoColor=yellow)
[![license](https://img.shields.io/github/license/chrislemke/feature-reviser)](https://github.com/chrislemke/feature-reviser/blob/main/LICENSE)
## Introduction
Every data tabular is different. Every column needs to be treated differently. [Scikit-learn](https://scikit-learn.org/stable/index.html) has a nice [collection of dataset transformers](https://scikit-learn.org/stable/data_transforms.html). But the possibilities of data transformation are infinite - one collection is simply not enough. This project provides a brought collection of data transformers. The idea is simple. It is like a well-equipped toolbox 🧰: You always find the tool you need and sometimes you get inspired by seeing a tool you did not know before. Please feel free to [contribute](https://chrislemke.github.io/feature-reviser/CONTRIBUTING/) your tools and ideas.

## Installation
If you are using [Poetry](https://python-poetry.org/), you can install the package with the following command:
```bash
poetry add feature-reviser
```
If you are using [pip](https://pypi.org/project/pip/), you can install the package with the following command:
```bash
pip install feature-reviser
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
Data preprocessing often involves similar processes. No matter whether it's manipulating strings or numbers, etc. [Scikit-learn's pipeline](https://scikit-learn.org/stable/modules/compose.html#combining-estimators) implementation makes it easy to structure and sequence such preprocessing processes. To take advantage of this, the [`transformer`](https://github.com/chrislemke/feature-reviser/tree/main/feature_reviser/transformer) part of the project contains multiple methods that can be easily pipelined to simplify preprocessing. The list of transformers is open and will be extended permanently. Feel free to [contribute](https://chrislemke.github.io/feature-reviser/CONTRIBUTING/)! 🛠

### Usage
Let's assume you want to use some method from [NumPy's mathematical functions](https://numpy.org/doc/stable/reference/routines.math.html), to sum up the values of column `foo` and column `bar`. You could
use the [`MathExpressionTransformer`](https://chrislemke.github.io/feature-reviser/number_transformer-reference/#feature_reviser.transformer.number_transformer.MathExpressionTransformer):
```python
from feature_reviser import MathExpressionTransformer
import pandas as pd
X = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
transformer = MathExpressionTransformer([("foo", "np.sum", "bar", {"axis": 0})])
print(transformer.fit_transform(X).values)
```
```
array([[1, 4, 5],
       [2, 5, 7],
       [3, 6, 9]])
```
Even if we only pass one tuple to the transformer - in this example. Like with most other transformers the idea is to simplify preprocessing by giving the possibility to operate on multiple columns at the same time. In this case, the [`MathExpressionTransformer`](https://chrislemke.github.io/feature-reviser/number_transformer-reference/#feature_reviser.transformer.number_transformer.MathExpressionTransformer) has created an extra column with the name `foo_sum_bar`.

## The feature reviser (under construction 🚧)
Finding the best features for your model is hard. In the `feature_selection` part of the project, we try to automate this process to make it a bit easier. This part of the project is still in development and is not yet ready for use. If you want to help, you can find more information in the [contributing guide](https://chrislemke.github.io/feature-reviser/CONTRIBUTING/).

## Contributing
We're all kind of in the same boat. Preprocessing/feature engineering in data science is somehow very individual - every feature is different and must be handled and processed differently. But somehow we all have the same problems: sometimes date columns have to be changed. Sometimes strings have to be formatted, sometimes durations have to be calculated, etc. There is a huge number of preprocessing possibilities but we all use the same tools.

[Scikit-learns pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) help to use formalized functions. So why not also share these so-called transformers with others? This open source project has the goal to collect useful preprocessing pipeline steps. Let us all collect what we used for preprocessing and share it with others. This way we can all benefit from each other's work and save a lot of time. So if you have a preprocessing step that you use regularly, please feel free to contribute it to this project. The idea is that this is not only a toolbox but also an inspiration for what is possible. Maybe you have not thought about this preprocessing step before.

Please check out the [guide](https://chrislemke.github.io/feature-reviser/CONTRIBUTING/) on how to contribute to this project.

## Further information
For further information, please refer to the [documentation](https://chrislemke.github.io/feature-reviser/).
